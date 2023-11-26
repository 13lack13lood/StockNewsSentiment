# Financial News Sentiment Prediction

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras

data = pd.read_csv("all-data.csv", names=["Label", "Text"], encoding="latin-1")


# Preprocessing for training

def get_sequences(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    print("Vocab Length:", len(tokenizer.word_index) + 1)

    max_seq = np.max(list(map(lambda x: len(x), sequences)))
    sequences = pad_sequences(sequences, maxlen=max_seq, padding="post")

    return sequences


print(get_sequences(data["Text"]).shape)


def preprocess_inputs(dataframe):
    dataframe = dataframe.copy()

    sequences = get_sequences(dataframe["Text"])

    label_map = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    y = dataframe["Label"].replace(label_map)


    train_sequences, test_sequences, y_train, y_test = train_test_split(sequences, y, train_size=0.7, shuffle=True, random_state=1)

    return train_sequences, test_sequences, y_train, y_test


# Training

def create_new_model(min_acc_save):
    # Vocal Length
    input_dimension = 10180

    train_sequences, test_sequences, y_train, y_test = preprocess_inputs(data)

    # Creating the Model

    inputs = keras.Input(shape=(train_sequences.shape[1]))
    x = keras.layers.Embedding(
        input_dim=input_dimension,
        output_dim=128,
        input_length=train_sequences.shape[1]
    )(inputs)
    x = keras.layers.LSTM(256, return_sequences=True, activation="tanh")(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_sequences,
        y_train,
        validation_split=0.2,
        batch_size=32,
        epochs=100,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True
            )
        ]
    )

    accuracy = model.evaluate(test_sequences, y_test)[1]

    if accuracy > min_acc_save:
        model.save("model " + str(accuracy))

    # Results
    print(accuracy)

    print(y_test.value_counts())

    return accuracy


max_acc = 0

while True:
    acc = create_new_model(max_acc)

    if acc > max_acc:
        max_acc = acc

