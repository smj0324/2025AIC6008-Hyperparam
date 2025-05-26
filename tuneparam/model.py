import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv


# OpenAI API 키를 환경 변수에서 가져오거나 직접 설정
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(
    api_key=OPENAI_API_KEY,
)


import json
from typing import Dict, Any, List, Callable

def format_block(keys: List[str], value_fn: Callable[[str], str]) -> str:
    return "{\n" + ",\n".join([f'  "{k}": {value_fn(k)}' for k in keys]) + "\n}"

def value_template(key: str, current_params: Dict[str, Any]) -> str:
    return '"value"' if isinstance(current_params[key], str) else "value"

def reason_template(key: str) -> str:
    return '"reason"'

def create_hparam_prompt(
    current_params: Dict[str, Any],
    training_results: Dict[str, Any],
    dataset_type: str,
    goal: str,
    model_prompt: str
) -> Dict[str, str]:
    # System prompt
    system_prompt = (
        "You are a machine learning hyperparameter optimization expert.\n"
        "Analyze the given model, dataset, current parameters, and training results "
        "to recommend optimal hyperparameters.\n"
        "Provide response in JSON format only, without additional explanation."
    )

    param_keys = list(current_params.keys())

    # Format recommendations and reasons using top-level helpers
    recommendations_block = format_block(
        param_keys, lambda k: value_template(k, current_params)
    )
    reasons_block = format_block(
        param_keys, reason_template
    )

    # Final user prompt
    user_prompt = f"""
Current hyperparameters:
{json.dumps(current_params, indent=2)}

Training results:
{json.dumps(training_results, indent=2)}

Dataset type: {dataset_type}
Optimization goal: {goal}

{model_prompt}

Respond in this JSON format only:
{{
  "recommendations": {recommendations_block},
  "reasons": {reasons_block},
  "expected_improvement": "description"
}}
""".strip()

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }




@dataclass
class TrainingMetrics:
    """Training metrics data class"""
    accuracy: float
    val_accuracy: float
    loss: float
    val_loss: float

@dataclass
class ModelConfig:
    """Model configuration data class"""
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str

class HyperparameterOptimizer:
    """Hyperparameter optimization using LLM or default strategies"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    
    def recommend_params(self, 
                        current_params: Dict[str, Any], 
                        training_results: Dict[str, float],
                        model_name: str, 
                        dataset_type: str, 
                        goal: str = "Accuracy") -> Dict[str, Any]:
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
        try:
            # Try LLM-based recommendation first
            if self.client:
                model_prompt = self._get_model_specific_prompt(model_name, current_params, dataset_type, goal)
                response = self._query_llm(current_params, training_results, model_prompt, dataset_type, goal)
                if response:
                    return response
            
            # Fallback to model-specific default recommendations
            return self._get_default_recommendations(model_name, current_params, training_results, goal)
            
        except Exception as e:
            print(f"Error in parameter recommendation: {e}")
            return self._create_safe_recommendation(current_params)
    
    def _get_model_specific_prompt(self, model_name: str, current_params: Dict[str, Any], 
                                 dataset_type: str, goal: str) -> str:
        """Generate model-specific prompt for LLM"""
        prompts = {
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
                Resnet model optimization for {dataset_type} dataset.
                Key considerations:
                1. Batch normalization and learning rate scheduling
                2. Batch size impact on GPU memory and generalization
                3. Epoch count and optimizer selection for deep networks
                4. Optimization goal: {goal}
                """
        }
        return prompts.get(model_name, f"Optimize {model_name} model for {dataset_type} dataset with goal: {goal}")
    
    def _query_llm(self, current_params: Dict[str, Any], training_results: Dict[str, float],
                   model_prompt: str, dataset_type: str, goal: str) -> Optional[Dict[str, Any]]:
        """Query LLM for hyperparameter recommendations"""


        if not self.client:
            return None
            
        try:
            prompt = create_hparam_prompt(current_params, training_results, dataset_type, goal, model_prompt)

            system_prompt = prompt["system_prompt"]
            user_prompt = prompt["user_prompt"]

            print("system : ", system_prompt)
            print("user : ", user_prompt)

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )

            content = response.choices[0].message.content
            return json.loads(content.split("```json")[1].split("```")[0] if "```json" in content else content)

        except Exception as e:
            print(f"LLM query error: {e}")
            return None
    
    def _get_default_recommendations(self, model_name: str, current_params: Dict[str, Any],
                                   training_results: Dict[str, float], goal: str) -> Dict[str, Any]:
        """Get default recommendations based on model type"""
        model_optimizers = {
            "MobilenetV4": MobileNetOptimizer(),
            "LSTM": LSTMOptimizer(),
            "Resnet": ResNetOptimizer()
        }
        
        optimizer = model_optimizers.get(model_name, DefaultOptimizer())
        return optimizer.get_recommendations(current_params, training_results, goal)
    
    def _create_safe_recommendation(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Create safe fallback recommendations"""
        return {
            "recommendations": {
                "learning_rate": current_params.get("learning_rate", 0.001),
                "batch_size": current_params.get("batch_size", 32),
                "epochs": current_params.get("epochs", 10),
                "optimizer": current_params.get("optimizer", "Adam")
            },
            "reasons": {
                "learning_rate": "Maintaining current learning rate for stability",
                "batch_size": "Keeping current batch size",
                "epochs": "Maintaining current epoch count",
                "optimizer": "Keeping current optimizer"
            },
            "expected_improvement": "Maintaining current configuration for stability"
        }

class BaseOptimizer:
    """Base class for model-specific optimizers"""
    
    def get_recommendations(self, current_params: Dict[str, Any], 
                          training_results: Dict[str, float], goal: str) -> Dict[str, Any]:
        """Get model-specific recommendations"""
        raise NotImplementedError

class MobileNetOptimizer(BaseOptimizer):
    """MobileNet-specific optimization strategies"""
    
    def get_recommendations(self, current_params: Dict[str, Any], 
                          training_results: Dict[str, float], goal: str) -> Dict[str, Any]:
        if goal == "Accuracy":
            return self._optimize_for_accuracy(current_params)
        elif goal == "Speed":
            return self._optimize_for_speed(current_params)
        elif goal == "Memory":
            return self._optimize_for_memory(current_params)
        return self._optimize_default(current_params)
    
    def _optimize_for_accuracy(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "recommendations": {
                "learning_rate": current_params.get("learning_rate", 0.001) * 0.8,
                "batch_size": min(current_params.get("batch_size", 32) * 2, 128),
                "epochs": current_params.get("epochs", 10) + 5,
                "optimizer": "Adam"
            },
            "reasons": {
                "learning_rate": "Lower learning rate for better convergence",
                "batch_size": "Increased batch size for stable gradient estimation",
                "epochs": "Extended training for better convergence",
                "optimizer": "Adam optimizer for optimal accuracy"
            },
            "expected_improvement": "Expected 2-3% accuracy improvement"
        }
    
    def _optimize_for_speed(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "recommendations": {
                "learning_rate": current_params.get("learning_rate", 0.001) * 1.2,
                "batch_size": min(current_params.get("batch_size", 32) * 4, 256),
                "epochs": max(current_params.get("epochs", 10) - 3, 5),
                "optimizer": "RMSprop"
            },
            "reasons": {
                "learning_rate": "Higher learning rate for faster convergence",
                "batch_size": "Larger batch size for improved throughput",
                "epochs": "Reduced epochs for faster training",
                "optimizer": "RMSprop for quick convergence"
            },
            "expected_improvement": "40-50% faster training with ~1% accuracy loss"
        }
    
    def _optimize_for_memory(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "recommendations": {
                "learning_rate": current_params.get("learning_rate", 0.001),
                "batch_size": max(current_params.get("batch_size", 32) // 2, 8),
                "epochs": current_params.get("epochs", 10),
                "optimizer": current_params.get("optimizer", "Adam")
            },
            "reasons": {
                "learning_rate": "Maintained for stability",
                "batch_size": "Reduced batch size for lower memory usage",
                "epochs": "Maintained for consistent accuracy",
                "optimizer": "Current optimizer for stability"
            },
            "expected_improvement": "40-50% reduced memory usage"
        }
    
    def _optimize_default(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "recommendations": {
                "learning_rate": current_params.get("learning_rate", 0.001) * 0.9,
                "batch_size": current_params.get("batch_size", 32),
                "epochs": current_params.get("epochs", 10) + 2,
                "optimizer": current_params.get("optimizer", "Adam")
            },
            "reasons": {
                "learning_rate": "Slightly reduced for stability",
                "batch_size": "Maintained for consistency",
                "epochs": "Slightly increased for better learning",
                "optimizer": "Maintained for stability"
            },
            "expected_improvement": "Modest overall improvement expected"
        }

class LSTMOptimizer(BaseOptimizer):
    """LSTM-specific optimization strategies"""
    
    def get_recommendations(self, current_params: Dict[str, Any], 
                          training_results: Dict[str, float], goal: str) -> Dict[str, Any]:
        if goal == "Accuracy":
            return {
                "recommendations": {
                    "learning_rate": current_params.get("learning_rate", 0.001) * 0.7,
                    "batch_size": current_params.get("batch_size", 32),
                    "epochs": current_params.get("epochs", 10) + 8,
                    "optimizer": "Adam"
                },
                "reasons": {
                    "learning_rate": "Lower learning rate to prevent gradient issues",
                    "batch_size": "Maintained for stable training",
                    "epochs": "Extended training for sequence learning",
                    "optimizer": "Adam for optimal LSTM training"
                },
                "expected_improvement": "5-7% accuracy improvement expected"
            }
        # Add similar methods for Speed and Memory optimization
        return self._optimize_default(current_params)
    
    def _optimize_default(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "recommendations": {
                "learning_rate": current_params.get("learning_rate", 0.001) * 0.8,
                "batch_size": current_params.get("batch_size", 32),
                "epochs": current_params.get("epochs", 10) + 4,
                "optimizer": "Adam"
            },
            "reasons": {
                "learning_rate": "Reduced for LSTM stability",
                "batch_size": "Maintained for consistent training",
                "epochs": "Increased for better sequence learning",
                "optimizer": "Adam for LSTM optimization"
            },
            "expected_improvement": "General improvement in model stability and performance"
        }

class ResNetOptimizer(BaseOptimizer):
    """ResNet-specific optimization strategies"""
    
    def get_recommendations(self, current_params: Dict[str, Any], 
                          training_results: Dict[str, float], goal: str) -> Dict[str, Any]:
        if goal == "Accuracy":
            return {
                "recommendations": {
                    "learning_rate": current_params.get("learning_rate", 0.001) * 0.5,
                    "batch_size": max(current_params.get("batch_size", 32) // 2, 16),
                    "epochs": current_params.get("epochs", 10) + 12,
                    "optimizer": "SGD"
                },
                "reasons": {
                    "learning_rate": "Lower learning rate for better generalization",
                    "batch_size": "Smaller batch size for better generalization",
                    "epochs": "Extended training for deep network",
                    "optimizer": "SGD with momentum for ResNet"
                },
                "expected_improvement": "4-6% accuracy improvement expected"
            }
        # Add similar methods for Speed and Memory optimization
        return self._optimize_default(current_params)
    
    def _optimize_default(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "recommendations": {
                "learning_rate": current_params.get("learning_rate", 0.001) * 0.7,
                "batch_size": current_params.get("batch_size", 32),
                "epochs": current_params.get("epochs", 10) + 6,
                "optimizer": "SGD"
            },
            "reasons": {
                "learning_rate": "Reduced for ResNet stability",
                "batch_size": "Maintained for consistent training",
                "epochs": "Increased for deep network training",
                "optimizer": "SGD for ResNet optimization"
            },
            "expected_improvement": "Improved model accuracy and generalization"
        }

class DefaultOptimizer(BaseOptimizer):
    """Default optimization strategies for unknown models"""
    
    def get_recommendations(self, current_params: Dict[str, Any], 
                          training_results: Dict[str, float], goal: str) -> Dict[str, Any]:
        return {
            "recommendations": {
                "learning_rate": current_params.get("learning_rate", 0.001) * 0.9,
                "batch_size": current_params.get("batch_size", 32),
                "epochs": current_params.get("epochs", 10) + 2,
                "optimizer": current_params.get("optimizer", "Adam")
            },
            "reasons": {
                "learning_rate": "Slightly reduced for stability",
                "batch_size": "Maintained for consistency",
                "epochs": "Slightly increased for better learning",
                "optimizer": "Maintained for stability"
            },
            "expected_improvement": "Modest improvement in model performance"
        }

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
    
    # Create optimizer and get recommendations
    optimizer = HyperparameterOptimizer()
    recommendations = optimizer.recommend_params(
        current_params=current_hyperparams,
        training_results=training_results,
        model_name="MobilenetV4",
        dataset_type="Image",
        goal="Accuracy"
    )
    
    print("Recommended hyperparameters:")
    print(json.dumps(recommendations, indent=2, ensure_ascii=False))