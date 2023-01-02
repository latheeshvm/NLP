from core.vectorization import text_vectorizer
import tensorflow as tf
layers = tf.keras.layers


embedding = layers.Embedding(
    input_dim=10000, output_dim=128, embeddings_initializer="uniform", input_length=15)


if __name__ == "__main__":
    sample_embed = embedding(text_vectorizer(["this is a test sentence"]))
    print(sample_embed)
    print(sample_embed.shape)
