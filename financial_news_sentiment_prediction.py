# Financial News Sentiment Prediction

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

import keras

data = pd.read_csv("all-data.csv", names=["Label", "Text"], encoding="latin-1")
test_data = pd.read_csv("test_data.csv", names=["Label", "Text"], encoding="latin-1")


# Preprocessing for training

def get_sequences(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    print("Vocab Length:", len(tokenizer.word_index) + 1)

    max_seq = np.max(list(map(lambda x: len(x), sequences)))
    sequences = pad_sequences(sequences, maxlen=max_seq, padding="post")

    return sequences


def get_training_data(dataframe):
    dataframe = dataframe.copy()

    sequences = get_sequences(dataframe["Text"])

    label_map = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    y = dataframe["Label"].replace(label_map)

    return sequences, y


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
    input_dimension = 25000

    train_sequences, test_sequences, y_train, y_test = preprocess_inputs(data)

    # Creating the Model

    inputs = keras.Input(shape=(train_sequences.shape[1]))
    x = keras.layers.Embedding(
        input_dim=input_dimension,
        output_dim=128,
        input_length=train_sequences.shape[1]
    )(inputs)
    x = keras.layers.GRU(256, return_sequences=True, activation="tanh")(x)
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
        model.save("modelGRU " + str(accuracy))

    # Results
    print(accuracy)

    print(y_test.value_counts())

    return accuracy

test_sequence, test_y = get_training_data(test_data)

loaded_model = keras.models.load_model("modelGRU")

loaded_model.summary()

loss, acc = loaded_model.evaluate(test_sequence, test_y, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
