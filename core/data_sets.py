from sklearn.model_selection import train_test_split
import pandas as pd


# import the data
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# shuffle the data
# train_df_shuffled is a shuffled version of train_df
# The shuffle is done using the frac=1 argument, which specifies that the entire dataset should be shuffled
# The random_state=42 argument specifies that the shuffle should be reproducible

train_df_shuffled = train_df.sample(frac=1, random_state=42)

train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    train_df_shuffled["text"].to_numpy(
    ), train_df_shuffled["target"].to_numpy(),
    test_size=0.1, random_state=42,
)


if __name__ == "__main__":
    print(len(train_sentences))
    print(len(val_sentences))
    print(len(train_labels))
    print(len(val_labels))

    print(train_sentences[:10])
    print(train_labels[:10])
