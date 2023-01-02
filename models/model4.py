# tensoflow bidirectional model
import tensorflow as tf
from core.vectorization import text_vectorizer
from core.embedding import embedding

inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)


x = text_vectorizer(inputs)
x = embedding(x)
x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(64, return_sequences=True))(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64))(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model_4 = tf.keras.Model(inputs, outputs, name="model_4_bidirectional")

if __name__ == "__main__":
    print(model_4.summary())
