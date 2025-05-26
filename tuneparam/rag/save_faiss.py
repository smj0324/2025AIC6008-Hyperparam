import pandas as pd
import faiss
from FlagEmbedding import FlagModel

def load_model(model_path="bge-finetuned/checkpoint-37500"):
    return FlagModel(model_path, use_fp16=True, device="cpu")

model = load_model()

df = pd.read_csv("hyperparam_collection_final_with_unique_en_comment.csv")

def row_to_text(row):
    fields = [
        row["Model"], 
        f"Optimizer={row['Optimizer']}", 
        f"Learning Rate={row['Learning Rate']}",
        f"Dropout={row['Dropout']}",
        f"Batch Size={row['Batch Size']}",
        f"Epochs={row['Epochs']}",
        f"Comment={row['Comment']}"
    ]
    return ", ".join([str(f) for f in fields if pd.notnull(f) and f != ""])

texts = [row_to_text(row) for _, row in df.iterrows()]

embeddings = model.encode(texts)
embeddings = embeddings.astype("float32")

dimension = embeddings.shape[1]
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
faiss.write_index(index, "faiss_hyperparam.index")