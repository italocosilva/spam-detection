import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
import numpy as np
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def train(data_path: str, train_plot_path: str, model_path: str) -> None:
    # Load data
    df = pd.read_csv(
        os.path.join(data_path), encoding="ISO-8859-1"
    )

    # Preprocessing
    df["is_spam"] = df["v1"].apply(lambda x: 1 if x == "spam" else 0)
    df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4", "v1"], inplace=True)
    df.rename(columns={"v2": "sentence"}, inplace=True)

    # Train test split
    df_train, df_test = train_test_split(df, test_size=0.33, random_state=23)

    # Tokenize
    tokenizer = Tokenizer(num_words=20_000)
    tokenizer.fit_on_texts(df_train["sentence"])
    df_train["sequence"] = tokenizer.texts_to_sequences(df_train["sentence"])
    df_test["sequence"] = tokenizer.texts_to_sequences(df_test["sentence"])

    # Check lengths before padding on train, pad and recheck
    logger.info(
        f"Max seq on train: {df_train['sequence'].apply(len).max()}    Min seq on train: {df_train['sequence'].apply(len).min()}"
    )
    logger.info(
        f"Expected shape lenght after padding: {pad_sequences(df_train['sequence']).shape}"
    )
    logger.info(f"df_train shape: {df_train.shape}")
    data_train = pad_sequences(df_train["sequence"])
    logger.info(f"data_train shape: {np.shape(data_train)}")

    # Set max length based on train
    MAX_LENGTH = df_train["sequence"].apply(len).max()

    # Check lengths before padding on test, pad and recheck
    logger.info(
        f"Max seq on test: {df_test['sequence'].apply(len).max()}    Min seq on test: {df_test['sequence'].apply(len).min()}"
    )
    data_test = pad_sequences(df_test["sequence"], maxlen=MAX_LENGTH)
    logger.info(f"data_test shape: {np.shape(data_test)}")

    # Create LSTM
    i = layers.Input(shape=(MAX_LENGTH,))
    x = layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8)(i)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    model = Model(i, x)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    # Train
    r = model.fit(
        data_train,
        df_train["is_spam"],
        epochs=10,
        validation_data=(data_test, df_test["is_spam"]),
    )

    # Generate historic plot
    plot = pd.DataFrame(r.history).plot()
    plot.get_figure().savefig(
        os.path.join(train_plot_path)
    )

    # Save model
    logger.info(model.summary())
    model.save(model_path)

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)

    # Define paths
    data_path = os.path.join(dirname, "..", "data", "spam.csv")
    train_plot_path = os.path.join(dirname, "..", "artifacts", "training_history.png")
    model_path = os.path.join(dirname, "..", "artifacts", "model.h5")

    # Train model
    train(data_path, train_plot_path, model_path)