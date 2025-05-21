import json
import os
import requests
from openai import OpenAI
from tkinter import messagebox
from dotenv import load_dotenv


# OpenAI API 키를 환경 변수에서 가져오거나 직접 설정
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(
    api_key=OPENAI_API_KEY,
)

def recommend_params(current_params, training_results, model_name, dataset_type, goal="Accuracy"):
    """
    LLM을 사용하여 하이퍼파라미터 추천
    
    Args:
        current_params (dict): 현재 사용 중인 하이퍼파라미터
        training_results (dict): 현재까지의 학습 결과 (loss, accuracy 등)
        model_name (str): 모델 이름 (MobilenetV4, LSTM, Resnet 등)
        dataset_type (str): 데이터셋 유형 (Image, Text, Tabular 등)
        goal (str): 최적화 목표 (Accuracy, Speed, Memory 등)
        
    Returns:
        dict: 추천된 하이퍼파라미터와 설명
    """
    try:
        # 모델 타입에 따른 특화 처리
        model_specific_prompt = model_type(model_name, current_params, dataset_type, goal)
        
        # LLM에 추천 요청
        response = query_llm(
            current_params=current_params,
            training_results=training_results,
            model_specific_prompt=model_specific_prompt,
            dataset_type=dataset_type,
            goal=goal
        )
        
        if response:
            return response
        else:
            # LLM 연결 실패 시 모델별 기본 추천값 반환
            if model_name == "MobilenetV4":
                return mobilenet(current_params, training_results, goal)
            elif model_name == "LSTM":
                return lstm(current_params, training_results, goal)
            elif model_name == "Resnet":
                return resnet(current_params, training_results, goal)
            else:
                return default_recommendations(current_params)
    except Exception as e:
        print(f"추천 과정 오류: {e}")
        return {
            "recommendations": current_params,
            "reasons": {"error": f"추천 중 오류 발생: {str(e)}"},
            "expected_improvement": "추천을 생성할 수 없습니다."
        }


def query_llm(current_params, training_results, model_specific_prompt, dataset_type, goal):
    """
    LLM API에 요청을 보내 하이퍼파라미터 추천 받기
    
    Args:
        current_params (dict): 현재 하이퍼파라미터
        training_results (dict): 학습 결과
        model_specific_prompt (str): 모델별 특화 프롬프트
        dataset_type (str): 데이터셋 유형
        goal (str): 최적화 목표
        
    Returns:
        dict: 추천 결과 (없으면 None)
    """
    if not OPENAI_API_KEY:
        print("API 키가 설정되지 않았습니다. 기본 추천을 사용합니다.")
        return None
    
    # 기본 시스템 프롬프트
    system_prompt = """당신은 머신러닝 하이퍼파라미터 최적화 전문가입니다. 
주어진 모델, 데이터셋, 현재 하이퍼파라미터, 학습 결과를 분석하여 최적의 하이퍼파라미터를 추천해주세요.
응답은 반드시 JSON 형식으로만 제공하고, 다른 설명은 포함하지 마세요."""
    
    # 사용자 프롬프트 구성
    user_prompt = f"""
현재 하이퍼파라미터:
{json.dumps(current_params, indent=2, ensure_ascii=False)}

학습 결과:
{json.dumps(training_results, indent=2, ensure_ascii=False)}

데이터셋 유형: {dataset_type}
최적화 목표: {goal}

{model_specific_prompt}

다음 JSON 형식으로만 응답해주세요:
```json
{{
  "recommendations": {{
    "learning_rate": 값,
    "batch_size": 값,
    "epochs": 값,
    "optimizer": "값"
  }},
  "reasons": {{
    "learning_rate": "변경 이유",
    "batch_size": "변경 이유",
    "epochs": "변경 이유",
    "optimizer": "변경 이유"
  }},
  "expected_improvement": "예상되는 개선 효과"
}}
```
"""
    
    try:
        # API 요청
        response = client.chat.completions.create(
            model="gpt-4",  # 모델 선택 (gpt-4, gpt-3.5-turbo 등)
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # 일관된 결과를 위해 낮은 temperature 사용
            max_tokens=1000
        )
        
        content = response.choices[0].message.content
        
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        else:
            json_str = content
                
        recommendations = json.loads(json_str)
        return recommendations
    except Exception as e:
        print(f"LLM 쿼리 오류: {e}")
        return None


def model_type(model, current_params, dataset_type, goal):
    """
    모델 유형에 따른 특화된 프롬프트 생성
    
    Args:
        model (str): 모델 이름
        current_params (dict): 현재 하이퍼파라미터
        dataset_type (str): 데이터셋 유형
        goal (str): 최적화 목표
        
    Returns:
        str: 모델별 특화 프롬프트
    """
    if model == "MobilenetV4":
        return f"""
MobilenetV4 모델을 {dataset_type} 데이터셋에 대해 최적화하고 있습니다.
MobilenetV4는 경량 CNN 모델로, 모바일 및 엣지 디바이스에 최적화되어 있습니다.

이 모델에서 중요한 고려사항:
1. 모바일 환경에서는 배치 크기와 메모리 사용량의 균형이 중요합니다.
2. 학습률은 빠른 수렴과 정확도에 큰 영향을 미칩니다.
3. 목표가 '{goal}'이므로 이에 최적화된 파라미터를 추천해주세요.
"""
    
    elif model == "LSTM":
        return f"""
LSTM 모델을 {dataset_type} 데이터셋에 대해 최적화하고 있습니다.
LSTM은 순차 데이터(시계열, 텍스트 등)에 적합한 순환 신경망 모델입니다.

이 모델에서 중요한 고려사항:
1. 시퀀스 길이와 배치 크기는 메모리 사용량에 직접적인 영향을 미칩니다.
2. LSTM의 경우 그래디언트 소실/폭발 문제가 발생할 수 있어 학습률 선택이 중요합니다.
3. 옵티마이저 선택(특히 Adam과 RMSprop)은 수렴 속도에 큰 영향을 줍니다.
4. 목표가 '{goal}'이므로 이에 최적화된 파라미터를 추천해주세요.
"""
    
    elif model == "Resnet":
        return f"""
Resnet 모델을 {dataset_type} 데이터셋에 대해 최적화하고 있습니다.
Resnet은 깊은 레이어를 가진 CNN 모델로, 이미지 분류 및 컴퓨터 비전 작업에 널리 사용됩니다.

이 모델에서 중요한 고려사항:
1. 깊은 네트워크에서는 배치 정규화와 학습률 스케줄링이 중요합니다.
2. 배치 크기는 GPU 메모리와 일반화 성능에 영향을 미칩니다.
3. 깊은 네트워크를 효과적으로 학습하려면 적절한 에포크 수와 옵티마이저가 필요합니다.
4. 목표가 '{goal}'이므로 이에 최적화된 파라미터를 추천해주세요.
"""
    
    else:
        return f"""
{model} 모델을 {dataset_type} 데이터셋에 대해 최적화하고 있습니다.
현재 하이퍼파라미터와 학습 결과를 분석하여, 목표인 '{goal}'에 최적화된 파라미터를 추천해주세요.
"""


def mobilenet(current_params, training_results, goal="Accuracy"):
    """
    MobilenetV4 모델에 대한 기본 추천 (LLM 연결 실패 시)
    
    Args:
        current_params (dict): 현재 하이퍼파라미터
        training_results (dict): 학습 결과
        goal (str): 최적화 목표
        
    Returns:
        dict: 추천 결과
    """
    # 현재 파라미터에서 값 추출 (없으면 기본값 사용)
    current_lr = current_params.get("learning_rate", 0.001)
    current_batch = current_params.get("batch_size", 32)
    current_epochs = current_params.get("epochs", 10)
    current_optimizer = current_params.get("optimizer", "Adam")
    
    # 목표에 따른 추천 로직
    if goal == "Accuracy":
        recommendations = {
            "learning_rate": current_lr * 0.8,  # 학습률 감소
            "batch_size": min(current_batch * 2, 128),  # 배치 크기 증가 (최대 128)
            "epochs": current_epochs + 5,  # 에포크 증가
            "optimizer": "Adam" if current_optimizer != "Adam" else current_optimizer
        }
        
        reasons = {
            "learning_rate": "MobilenetV4는 낮은 학습률에서 더 안정적으로 수렴하며 정확도가 향상됩니다.",
            "batch_size": "더 큰 배치 사이즈는 그래디언트 추정을 안정화시켜 정확도 향상에 도움이 됩니다.",
            "epochs": "더 긴 학습 기간으로 모델이 충분히 수렴할 수 있도록 합니다.",
            "optimizer": "Adam은 MobilenetV4에서 정확도 최적화에 효과적입니다."
        }
        
        expected = "정확도가 2-3% 향상될 것으로 예상됩니다."
        
    elif goal == "Speed":
        recommendations = {
            "learning_rate": current_lr * 1.2,  # 학습률 증가
            "batch_size": min(current_batch * 4, 256),  # 배치 크기 크게 증가
            "epochs": max(current_epochs - 3, 5),  # 에포크 감소 (최소 5)
            "optimizer": "RMSprop"  # 빠른 수렴을 위한 옵티마이저
        }
        
        reasons = {
            "learning_rate": "더 높은 학습률로 빠른 수렴을 유도합니다.",
            "batch_size": "큰 배치 사이즈로 학습 속도를 크게 향상시킵니다.",
            "epochs": "에포크 수를 줄여 전체 학습 시간을 단축합니다.",
            "optimizer": "RMSprop는 MobilenetV4에서 빠른 수렴 속도를 보입니다."
        }
        
        expected = "약간의 정확도 손실(-1%)과 함께 학습 속도가 40-50% 향상될 것으로 예상됩니다."
        
    elif goal == "Memory":
        recommendations = {
            "learning_rate": current_lr,
            "batch_size": max(current_batch // 2, 8),  # 배치 크기 감소 (최소 8)
            "epochs": current_epochs,
            "optimizer": current_optimizer
        }
        
        reasons = {
            "learning_rate": "현재 학습률을 유지하여 학습 안정성을 보장합니다.",
            "batch_size": "배치 크기를 줄여 메모리 사용량을 크게 줄입니다.",
            "epochs": "에포크 수를 유지하여 모델 정확도를 보존합니다.",
            "optimizer": "현재 옵티마이저를 유지하여 안정성을 보장합니다."
        }
        
        expected = "메모리 사용량이 40-50% 감소할 것으로 예상됩니다."
        
    else:
        # 기본 추천
        recommendations = {
            "learning_rate": current_lr * 0.9,
            "batch_size": current_batch,
            "epochs": current_epochs + 2,
            "optimizer": current_optimizer
        }
        
        reasons = {
            "learning_rate": "일반적으로 약간 낮은 학습률이 안정성을 높입니다.",
            "batch_size": "현재 배치 크기를 유지합니다.",
            "epochs": "약간 더 긴 학습으로 성능을 개선합니다.",
            "optimizer": "현재 옵티마이저를 유지합니다."
        }
        
        expected = "전반적인 성능이 소폭 개선될 것으로 예상됩니다."
    
    return {
        "recommendations": recommendations,
        "reasons": reasons,
        "expected_improvement": expected
    }


def lstm(current_params, training_results, goal="Accuracy"):
    """
    LSTM 모델에 대한 기본 추천 (LLM 연결 실패 시)
    
    Args:
        current_params (dict): 현재 하이퍼파라미터
        training_results (dict): 학습 결과
        goal (str): 최적화 목표
        
    Returns:
        dict: 추천 결과
    """
    # 현재 파라미터에서 값 추출 (없으면 기본값 사용)
    current_lr = current_params.get("learning_rate", 0.001)
    current_batch = current_params.get("batch_size", 32)
    current_epochs = current_params.get("epochs", 10)
    current_optimizer = current_params.get("optimizer", "Adam")
    
    # 목표에 따른 추천 로직
    if goal == "Accuracy":
        recommendations = {
            "learning_rate": current_lr * 0.7,  # 학습률 크게 감소
            "batch_size": current_batch,  # 배치 크기 유지
            "epochs": current_epochs + 8,  # 에포크 크게 증가
            "optimizer": "Adam"  # LSTM에 최적화된 옵티마이저
        }
        
        reasons = {
            "learning_rate": "LSTM은 낮은 학습률에서 그래디언트 폭발 문제가 줄어들고 정확도가 향상됩니다.",
            "batch_size": "현재 배치 크기를 유지하여 안정적인 학습을 보장합니다.",
            "epochs": "LSTM은 긴 학습 기간이 필요하며, 더 많은 에포크로 정확도가 크게 향상됩니다.",
            "optimizer": "Adam은 LSTM에서 정확도 최적화에 가장 효과적입니다."
        }
        
        expected = "순차 데이터 모델링 정확도가 5-7% 향상될 것으로 예상됩니다."
        
    elif goal == "Speed":
        recommendations = {
            "learning_rate": current_lr * 1.3,  # 학습률 크게 증가
            "batch_size": min(current_batch * 3, 192),  # 배치 크기 크게 증가
            "epochs": max(current_epochs - 4, 6),  # 에포크 크게 감소 (최소 6)
            "optimizer": "RMSprop"  # 빠른 수렴 옵티마이저
        }
        
        reasons = {
            "learning_rate": "높은 학습률로 빠른 수렴을 유도합니다. LSTM에서는 약간의 불안정성을 주의해야 합니다.",
            "batch_size": "큰 배치 사이즈로 CPU/GPU 활용도를 높이고 학습 속도를 향상시킵니다.",
            "epochs": "에포크를 크게 줄여 전체 학습 시간을 단축합니다.",
            "optimizer": "RMSprop는 LSTM에서 빠른 초기 수렴을 보입니다."
        }
        
        expected = "정확도가 약 3-4% 감소할 수 있지만, 학습 시간이 60% 이상 단축될 것으로 예상됩니다."
        
    elif goal == "Memory":
        recommendations = {
            "learning_rate": current_lr,
            "batch_size": max(current_batch // 4, 4),  # 배치 크기 크게 감소 (최소 4)
            "epochs": current_epochs,
            "optimizer": "Adam"
        }
        
        reasons = {
            "learning_rate": "현재 학습률을 유지하여 학습 안정성을 보장합니다.",
            "batch_size": "LSTM은 메모리 사용량이 많으므로, 배치 크기를 크게 줄여 메모리 효율성을 높입니다.",
            "epochs": "에포크 수를 유지하여 모델 정확도를 보존합니다.",
            "optimizer": "Adam은 메모리 효율성과 정확도의 균형을 제공합니다."
        }
        
        expected = "메모리 사용량이 70% 이상 감소할 것으로 예상됩니다. 특히 긴 시퀀스에서 효과적입니다."
        
    else:
        # 기본 추천
        recommendations = {
            "learning_rate": current_lr * 0.8,
            "batch_size": max(current_batch // 2, 16),
            "epochs": current_epochs + 4,
            "optimizer": "Adam"
        }
        
        reasons = {
            "learning_rate": "LSTM에서는 낮은 학습률이 안정성을 높입니다.",
            "batch_size": "작은 배치 크기는 LSTM의 메모리 효율성을 높입니다.",
            "epochs": "더 긴 학습으로 순차 패턴을 더 잘 포착합니다.",
            "optimizer": "Adam은 LSTM에서 일반적으로 최적의 선택입니다."
        }
        
        expected = "모델의 안정성과 성능이 향상될 것으로 예상됩니다."
    
    return {
        "recommendations": recommendations,
        "reasons": reasons,
        "expected_improvement": expected
    }


def resnet(current_params, training_results, goal="Accuracy"):
    """
    Resnet 모델에 대한 기본 추천 (LLM 연결 실패 시)
    
    Args:
        current_params (dict): 현재 하이퍼파라미터
        training_results (dict): 학습 결과
        goal (str): 최적화 목표
        
    Returns:
        dict: 추천 결과
    """
    # 현재 파라미터에서 값 추출 (없으면 기본값 사용)
    current_lr = current_params.get("learning_rate", 0.001)
    current_batch = current_params.get("batch_size", 32)
    current_epochs = current_params.get("epochs", 10)
    current_optimizer = current_params.get("optimizer", "Adam")
    
    # 목표에 따른 추천 로직
    if goal == "Accuracy":
        recommendations = {
            "learning_rate": current_lr * 0.5,  # 학습률 크게 감소
            "batch_size": max(current_batch // 2, 16),  # 배치 크기 감소
            "epochs": current_epochs + 12,  # 에포크 크게 증가
            "optimizer": "SGD"  # 정확도에 최적화된 옵티마이저
        }
        
        reasons = {
            "learning_rate": "Resnet은 낮은 학습률에서 더 나은 일반화 성능을 보입니다.",
            "batch_size": "작은 배치 크기는 일반화 성능을 향상시키고 local minima를 벗어나는 데 도움이 됩니다.",
            "epochs": "Resnet은 깊은 네트워크로, 충분한 학습 시간이 필요합니다.",
            "optimizer": "SGD with momentum은 Resnet에서 최고의 정확도를 달성하는 데 효과적입니다."
        }
        
        expected = "이미지 분류 정확도가 4-6% 향상될 것으로 예상됩니다."
        
    elif goal == "Speed":
        recommendations = {
            "learning_rate": current_lr * 1.5,  # 학습률 크게 증가
            "batch_size": min(current_batch * 4, 256),  # 배치 크기 크게 증가
            "epochs": max(current_epochs - 5, 5),  # 에포크 크게 감소
            "optimizer": "Adam"  # 빠른 수렴에 적합한 옵티마이저
        }
        
        reasons = {
            "learning_rate": "높은 학습률로 초기 수렴 속도를 높입니다.",
            "batch_size": "큰 배치 크기로, 특히 강력한 GPU에서 처리량을 극대화합니다.",
            "epochs": "에포크 수를 줄여 전체 학습 시간을 크게 단축합니다.",
            "optimizer": "Adam은 빠른 초기 수렴을 제공합니다."
        }
        
        expected = "정확도가 약 2-3% 감소하지만, 학습 시간이 70% 이상 단축될 것으로 예상됩니다."
        
    elif goal == "Memory":
        recommendations = {
            "learning_rate": current_lr,
            "batch_size": max(current_batch // 4, 8),  # 배치 크기 크게 감소
            "epochs": current_epochs,
            "optimizer": current_optimizer
        }
        
        reasons = {
            "learning_rate": "현재 학습률을 유지하여 학습 안정성을 보장합니다.",
            "batch_size": "Resnet은 메모리 사용량이 많으므로 배치 크기를 크게 줄여 메모리 효율성을 높입니다.",
            "epochs": "에포크 수를 유지하여 모델 정확도를 보존합니다.",
            "optimizer": "현재 옵티마이저를 유지하여 일관성을 보장합니다."
        }
        
        expected = "메모리 사용량이 75% 이상 감소할 것으로 예상됩니다. 특히 깊은 모델에서 효과적입니다."
        
    else:
        # 기본 추천
        recommendations = {
            "learning_rate": current_lr * 0.7,
            "batch_size": current_batch,
            "epochs": current_epochs + 6,
            "optimizer": "SGD" if current_optimizer != "SGD" else current_optimizer
        }
        
        reasons = {
            "learning_rate": "Resnet에서는 낮은 학습률이 일반적으로 더 나은 성능을 제공합니다.",
            "batch_size": "현재 배치 크기를 유지합니다.",
            "epochs": "더 긴 학습으로 깊은 네트워크의 잠재력을 최대화합니다.",
            "optimizer": "SGD는 Resnet에서 장기적으로 더 나은 성능을 제공합니다."
        }
        
        expected = "모델 정확도와 일반화 성능이 향상될 것으로 예상됩니다."
    
    return {
        "recommendations": recommendations,
        "reasons": reasons,
        "expected_improvement": expected
    }


def default_recommendations(current_params):
    """
    알 수 없는 모델에 대한 기본 추천
    
    Args:
        current_params (dict): 현재 하이퍼파라미터
        
    Returns:
        dict: 기본 추천
    """
    return {
        "recommendations": {
            "learning_rate": current_params.get("learning_rate", 0.001) * 0.9,
            "batch_size": current_params.get("batch_size", 32),
            "epochs": current_params.get("epochs", 10) + 5,
            "optimizer": current_params.get("optimizer", "Adam")
        },
        "reasons": {
            "learning_rate": "약간 낮은 학습률로 안정성을 높입니다.",
            "batch_size": "현재 배치 크기를 유지합니다.",
            "epochs": "더 긴 학습 기간으로 모델 성능을 향상시킵니다.",
            "optimizer": "현재 옵티마이저를 유지합니다."
        },
        "expected_improvement": "약간의 성능 향상이 예상됩니다."
    }


# 사용 예시:
if __name__ == "__main__":
    # 테스트 데이터
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
    
    # 추천 받기
    recommendations = recommend_params(
        current_params=current_hyperparams,
        training_results=training_results,
        model_name="MobilenetV4",
        dataset_type="Image",
        goal="Accuracy"
    )
    
    print("추천된 하이퍼파라미터:")
    print(json.dumps(recommendations, indent=2, ensure_ascii=False))