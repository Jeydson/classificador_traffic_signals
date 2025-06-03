import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class VGGNet(tf.keras.Model):
    def __init__(self, num_classes=43):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Input(shape=(32, 32, 1)),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.3),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.3),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes)
        ])

    def call(self, inputs, training=False):
        return self.seq(inputs, training=training)


def create_model(num_classes=43):
    model = VGGNet(num_classes)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model
