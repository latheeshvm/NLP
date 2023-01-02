# CNN module

import tensorflow as tf
from core.vectorization import text_vectorizer
from core.embedding import embedding

inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)

x = text_vectorizer(inputs)
x = embedding(x)
x = tf.keras.layers.Conv1D(filters=64, kernel_size=5,
                           strides=1, activation="relu", padding="valid")(x)
x = tf.keras.layers.GlobalMaxPool1D()(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model_5 = tf.keras.Model(inputs, outputs, name="model_5_conv1d")


if __name__ == "__main__":
    print(model_5.summary())
