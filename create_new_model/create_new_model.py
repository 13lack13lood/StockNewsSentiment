import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

import keras

import pickle


# get the training data
train_data = pd.read_csv("../training_data/train_dataset.csv", names=["Label", "Text"], encoding="latin-1")
retrain_data = pd.read_csv("../training_data/oov_train_dataset.csv", names=["Label", "Text"], encoding="latin-1")


# get text sequences and create tokenizer
def get_sequences(texts):
    tokenizer = Tokenizer(oov_token=1)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    print("Vocab Length:", len(tokenizer.word_index) + 1)

    max_seq = np.max(list(map(lambda x: len(x), sequences)))
    sequences = pad_sequences(sequences, maxlen=max_seq, padding="post")

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return sequences


# get text sequences for out of vocabulary training
def get_retrain_data(dataframe, tokenizer, max_seq):
    dataframe = dataframe.copy()

    sequences = tokenizer.texts_to_sequences(dataframe["Text"])
    sequences = pad_sequences(sequences, maxlen=max_seq, padding="post")

    label_map = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    y = dataframe["Label"].replace(label_map)

    train_sequences, test_sequences, y_train, y_test = train_test_split(sequences, y, train_size=0.7, shuffle=True, random_state=1)

    return train_sequences, test_sequences, y_train, y_test


# process input for training
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


# create a new model
# takes in the minimum accuracy desired to save the model
def create_new_model(min_acc_save):
    # Vocal Length
    input_dimension = 25000

    train_sequences, test_sequences, y_train, y_test = preprocess_inputs(train_data)

    # Creating the Model

    inputs = keras.Input(shape=(train_sequences.shape[1]))
    x = keras.layers.Embedding(
        input_dim=input_dimension,
        output_dim=256,
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

    accuracy, loss = model.evaluate(test_sequences, y_test)

    if accuracy > min_acc_save:
        model.save("modelGRU(" + str(round(accuracy * 1000)) + ").keras")

    # Results
    print("New model created, accuracy: {:5.2f}%, loss {:5.2f}".format(100 * accuracy, 100 * loss))

    print(y_test.value_counts())

    return accuracy


# trains the model with the out of vocabulary data
def retrain_model(model_path, min_acc_save):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    loaded_model = keras.models.load_model(model_path)

    loaded_model.summary()

    train_sequences, test_sequences, y_train, y_test = get_retrain_data(retrain_data, tokenizer, loaded_model.layers[0].input_shape[0][1])

    loaded_model.fit(
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

    accuracy, loss = loaded_model.evaluate(test_sequences, y_test)

    print("Model Re-trained, accuracy: {:5.2f}%, loss {:5.2f}".format(100 * accuracy, 100 * loss))

    if accuracy > min_acc_save:
        loaded_model.save("modelRetrained" + str(round(accuracy * 1000)) + ".keras")

    return accuracy

