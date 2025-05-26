import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from openai import OpenAI
from tkinter import messagebox
from dotenv import load_dotenv
from rag.search_faiss import faiss_search

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class HyperparameterOptimizer:
    """Hyperparameter optimization using LLM or default strategies"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    
    def recommend_params(self, 
                        current_params: Dict[str, Any], 
                        training_results: Dict[str, float],
                        model_name: str, 
                        dataset_type: str, 
                        goal: str = "Accuracy",
                        rag_evidence: str = "") -> Dict[str, Any]:
        """
        Recommend hyperparameters based on current training results
        
        Args:
            current_params: Current hyperparameters
            training_results: Current training metrics
            model_name: Name of the model (MobilenetV4, LSTM, Resnet)
            dataset_type: Type of dataset (Image, Text, Tabular)
            goal: Optimization goal (Accuracy, Speed, Memory)
            
        Returns:
            Dictionary containing recommendations and explanations
        """
        # Try LLM-based recommendation first
        if self.client:
            rag_evidence = faiss_search(f"{model_name}")
            model_prompt = self._get_model_specific_prompt(model_name, current_params, dataset_type, goal, rag_evidence)
            model_parmas = self._get_model_specific_params(model_name)
            response = self._query_llm(current_params, training_results, model_prompt, dataset_type, goal, model_parmas)
            print("*" * 80, "\n", response, "\n", "*" * 80)
            if response:
                return response
            
    def _get_model_specific_params(self, model_name: str):
        model_parameter = {
            "ResNet": {
                "recommendations": {
                    "optimizer": "",
                    "learning_rate": "",
                    "batch_size": "",
                    "epochs": "",
                    "weight_decay": "",
                    "momentum": "",
                    "dropout_rate": "",
                    "label_smoothing": "",
                    "scheduler": "",
                    "data_augmentation": "",
                    "batch_normalization": "",
                    "initialization": ""
                },
                "reasons": {
                    "optimizer": "",
                    "learning_rate": "",
                    "batch_size": "",
                    "epochs": "",
                    "weight_decay": "",
                    "momentum": "",
                    "dropout_rate": "",
                    "label_smoothing": "",
                    "scheduler": "",
                    "data_augmentation": "",
                    "batch_normalization": "",
                    "initialization": ""
                },
                "expected_improvement": ""
            },
            "MobileNetV4": {
                "recommendations": {
                    "optimizer": "",
                    "learning_rate": "",
                    "batch_size": "",
                    "epochs": "",
                    "weight_decay": "",
                    "beta1": "",
                    "beta2": "",
                    "dropout_rate": "",
                    "label_smoothing": "",
                    "stochastic_depth_drop_rate": "",
                    "augmentation": "",
                    "mixup_cutmix": "",
                    "cosine_decay_alpha": "",
                    "warmup_epochs": "",
                    "ema_decay": ""
                },
                "reasons": {
                    "optimizer": "",
                    "learning_rate": "",
                    "batch_size": "",
                    "epochs": "",
                    "weight_decay": "",
                    "beta1": "",
                    "beta2": "",
                    "dropout_rate": "",
                    "label_smoothing": "",
                    "stochastic_depth_drop_rate": "",
                    "augmentation": "",
                    "mixup_cutmix": "",
                    "cosine_decay_alpha": "",
                    "warmup_epochs": "",
                    "ema_decay": ""
                },
                "expected_improvement": ""
            },
            "LSTM": {
                "recommendations": {
                    "optimizer": "",
                    "learning_rate": "",
                    "batch_size": "",
                    "epochs": "",
                    "dropout": "",
                    "hidden_size": "",
                    "num_layers": "",
                    "sequence_length": "",
                    "bidirectional": "",
                    "gradient_clipping": "",
                    "embedding_dim": "",
                    "tagging_scheme": "",
                    "output_classifier": "",
                    "early_stopping_patience": ""
                },
                "reasons": {
                    "optimizer": "",
                    "learning_rate": "",
                    "batch_size": "",
                    "epochs": "",
                    "dropout": "",
                    "hidden_size": "",
                    "num_layers": "",
                    "sequence_length": "",
                    "bidirectional": "",
                    "gradient_clipping": "",
                    "embedding_dim": "",
                    "tagging_scheme": "",
                    "output_classifier": "",
                    "early_stopping_patience": ""
                },
                "expected_improvement": ""
            }
        }
        target_parmas = model_parameter.get(model_name)
        return target_parmas
            
    def _get_model_specific_prompt(self, model_name: str, current_params: dict, dataset_type: str, goal: str, rag_evidence: str = "") -> str:
        """Generate model-specific prompt for LLM, with optional RAG evidence"""
        base_prompts = {
            "MobilenetV4": f"""
                MobilenetV4 model optimization for {dataset_type} dataset.
                Key considerations:
                1. Balance between batch size and memory usage for mobile environments
                2. Learning rate's impact on convergence and accuracy
                3. Optimization goal: {goal}
                """,
                        "LSTM": f"""
                LSTM model optimization for {dataset_type} dataset.
                Key considerations:
                1. Sequence length and batch size impact on memory
                2. Learning rate selection to prevent gradient issues
                3. Optimizer selection (especially Adam vs RMSprop)
                4. Optimization goal: {goal}
                """,
                        "Resnet": f"""
                ResNet model optimization for {dataset_type} dataset.
                Key considerations:
                1. Batch normalization and learning rate scheduling
                2. Batch size impact on GPU memory and generalization
                3. Epoch count and optimizer selection for deep networks
                4. Optimization goal: {goal}
                """
        }
        base_prompt = base_prompts.get(
            model_name, 
            f"Optimize {model_name} model for {dataset_type} dataset with goal: {goal}"
        )

        if rag_evidence:
            base_prompt += f"\n\n[Reference Results from Similar Experiments or Papers]\n{rag_evidence.strip()}\n"

        return base_prompt

    def _query_llm(self, current_params: Dict[str, Any], training_results: Dict[str, float],
                   model_prompt: str, dataset_type: str, goal: str, model_parmas: str) -> Optional[Dict[str, Any]]:
        """Query LLM for hyperparameter recommendations"""
        if not self.client:
            return None
            
        try:
            system_prompt = """You are a machine learning hyperparameter optimization expert.
            Analyze the given model, dataset, current parameters, and training results to recommend optimal hyperparameters.
            Provide response in JSON format only, without additional explanation."""
            
            user_prompt = f"""
            Current hyperparameters:
            {json.dumps(current_params, indent=2)}
            
            Training results:
            {json.dumps(training_results, indent=2)}
            
            Dataset type: {dataset_type}
            Optimization goal: {goal}
            
            {model_prompt}

            Respond in this JSON format only:
            {model_parmas}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            print(user_prompt, content)
            return json.loads(content.split("```json")[1].split("```")[0] if "```json" in content else content)
            
        except Exception as e:
            print(f"LLM query error: {e}")
            return None
        
###################################################################################################################################

# Usage example
if __name__ == "__main__":
    # Test data
    current_hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "optimizer": "Adam"
    }
    
    training_results = {
        "accuracy": 0.85,
        "val_accuracy": 0.82,
        "loss": 0.32,
        "val_loss": 0.40
    }
    
    optimizer = HyperparameterOptimizer()
    recommendations = optimizer.recommend_params(
        current_params=current_hyperparams,
        training_results=training_results,
        model_name="MobileNetV4",
        dataset_type="Image",
        goal="Accuracy"
    )
    
    print("Recommended hyperparameters:")
    print(json.dumps(recommendations, indent=2, ensure_ascii=False))