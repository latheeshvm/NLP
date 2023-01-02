# Using non - DL model

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

model_0 = Pipeline(
    [
        ("tfidf", TfidfVectorizer()),  # covert words to numbers using tfidf
        ("clf", MultinomialNB())  # model the text
    ]
)
