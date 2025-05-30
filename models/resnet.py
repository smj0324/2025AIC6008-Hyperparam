
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

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

# ============================
# 👇 ResNet 함수 정의
# ============================

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

# ============================
# 👇 훈련 코드
# ============================


