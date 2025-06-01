import pandas as pd
import faiss
from FlagEmbedding import FlagModel
import numpy as np

CSV_PATH = "tuneparam/rag/hyperparam_collection_final_with_unique_en_comment.csv"
INDEX_PATH = "tuneparam/rag/faiss_hyperparam.index"
MODEL_PATH = "bge-finetuned/checkpoint-37500"

model = FlagModel(MODEL_PATH, use_fp16=True, device="cpu")
df = pd.read_csv(CSV_PATH)
index = faiss.read_index(INDEX_PATH)

def faiss_search(query, top_k=1):
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    query_emb /= np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12
    D, I = index.search(query_emb, top_k)
    print(f"\n[Query] {query}")
    print(f"------------------------------ RAG 결과 ------------------------------")
    
    if I.size == 0 or I[0].size == 0:
        print("검색된 결과가 없습니다.")
        return None 

    result_parts = []

    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        row = df.iloc[idx].dropna()  
        
        print(f"\nRank {rank} | Score: {score:.3f}")
        print(row)

        row_str_items = []
        for col_name, value in row.items():
            row_str_items.append(f"{col_name}: {value}")
        
        result_parts.append(f"Rank {rank} | Score: {score:.3f}\n" + "\n".join(row_str_items))
        
    return "\n\n".join(result_parts)

# if __name__ == "__main__":
    # faiss_search("MobilenetV4")
#     faiss_search("lstm best dropout crf classifier")
