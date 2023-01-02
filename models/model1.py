from core.vectorization import text_vectorizer
from core.embedding import embedding
import tensorflow as tf


inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)  # Create an embed
x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model_1 = tf.keras.Model(inputs, outputs, name="model_1_dense")

if __name__ == "__main__":
    model_1.summary()
