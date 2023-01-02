import tensorflow as tf
from core.data_sets import train_sentences

TextVectorization = tf.keras.layers.experimental.preprocessing.TextVectorization

text_vectorizer = TextVectorization(max_tokens=10000, ngrams=None, standardize="lower_and_strip_punctuation",
                                    split="whitespace", output_mode="int", output_sequence_length=15, pad_to_max_tokens=True)


# Fit the text vectorizer to the training text
text_vectorizer.adapt(train_sentences)


if __name__ == "__main__":

    words_in_vocab = text_vectorizer.get_vocabulary()
    top_5_words = words_in_vocab[:5]
    bottom_5_words = words_in_vocab[-5:]
    print(f"Number of words in vocab: {len(words_in_vocab)}")
    print(f"5 most common words: {top_5_words}")
    print(f"5 least common words: {bottom_5_words}")
