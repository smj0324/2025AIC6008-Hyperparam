import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10

def preprocess(image, label, img_size=32):
    image = tf.image.resize(image, (img_size, img_size))
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.squeeze(label) 

def test_moblinet(
    mv_base_model_params,
    epochs=1,
    batch_size=32,
    learning_rate=0.05,
    img_size=32,
):
    
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_classes = 10

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.map(lambda x, y: preprocess(x, y, img_size))
    train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_ds = val_ds.map(lambda x, y: preprocess(x, y, img_size))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    mv_base_model_params['input_shape'] = (img_size, img_size, 3)
    base_model = MobileNetV3Small(**mv_base_model_params)
    x = base_model.output
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    training_hyperparams = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "validation_data": val_ds
    }

    return model, train_ds, y_train, training_hyperparams