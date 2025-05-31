
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy


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


# ✅ GPU 메모리 설정 (optional but recommended)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU 설정 완료:", gpus)
    except RuntimeError as e:
        print("❌ GPU 설정 중 오류 발생:", e)
else:
    print("⚠️ GPU를 찾을 수 없습니다. CPU를 사용합니다.")

def conv_block(x, filters, kernel_size=3, stride=1):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def residual_block(x, filters, stride=1, downsample=False):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if downsample:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet(input_shape, num_classes, block_counts):
    inputs = layers.Input(shape=input_shape)
    x = conv_block(inputs, 64, kernel_size=3, stride=1)
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    filters = 64
    for i, blocks in enumerate(block_counts):
        for j in range(blocks):
            stride = 1
            downsample = False
            if j == 0 and i != 0:
                stride = 2
                downsample = True
            x = residual_block(x, filters, stride=stride, downsample=downsample)
        filters *= 2
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

def get_resnet_model(model_name, input_shape=(32, 32, 3), num_classes=10):
    model_name = model_name.lower()
    if model_name == 'resnet18':
        return build_resnet(input_shape, num_classes, [2, 2, 2, 2])
    elif model_name == 'resnet34':
        return build_resnet(input_shape, num_classes, [3, 4, 6, 3])
    elif model_name == 'resnet50':
        return tf.keras.applications.ResNet50(input_shape=input_shape, weights=None, classes=num_classes)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def build_and_compile_model(training_params: dict) -> Model:
    """
    training_params 딕셔너리 기반으로 ResNet 모델을 생성하고 컴파일.

    Parameters:
        training_params (dict): 하이퍼파라미터 및 모델 구성 딕셔너리

    Returns:
        model (Model): 컴파일된 Keras 모델
    """

    model_name = training_params.get("model_name", "resnet18")
    input_shape = training_params.get("input_shape", (32, 32, 3))
    num_classes = training_params.get("num_classes", 100)

    learning_rate = training_params.get("learning_rate", 0.001)
    optimizer_name = training_params.get("optimizer", "adam").lower()
    momentum = training_params.get("momentum", 0.0)
    weight_decay = training_params.get("weight_decay", 0.0)
    label_smoothing = training_params.get("label_smoothing", 0.0)
    loss_name = training_params.get("loss", "categorical_crossentropy")
    metrics = training_params.get("metrics", ["accuracy"])

    # 모델 생성
    model: Model = get_resnet_model(model_name, input_shape, num_classes)

    # 옵티마이저 설정
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum, decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # 손실 함수 설정
    if loss_name == "categorical_crossentropy":
        loss = CategoricalCrossentropy(label_smoothing=label_smoothing)
    else:
        loss = loss_name

    # 모델 컴파일
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
