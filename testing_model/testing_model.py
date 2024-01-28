import pandas as pd

from keras.preprocessing.sequence import pad_sequences

import keras

import pickle

test_dataset_1 = pd.read_csv("../training_data/test_dataset_1.csv", names=["Label", "Text"], encoding="latin-1")
test_dataset_2 = pd.read_csv("../training_data/test_dataset_2.csv", names=["Label", "Text"], encoding="latin-1")


# get text sequences for testing
def get_testing_data(dataframe, tokenizer, max_seq):
    dataframe = dataframe.copy()

    sequences = tokenizer.texts_to_sequences(dataframe["Text"])
    sequences = pad_sequences(sequences, maxlen=max_seq, padding="post")

    label_map = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    y = dataframe["Label"].replace(label_map)

    return sequences, y


# test model
def test_model(model_path, tokenizer):
    loaded_model = keras.models.load_model(model_path)

    loaded_model.summary()

    test_sequences, test_y = get_testing_data(test_dataset_1, tokenizer, loaded_model.layers[0].input_shape[0][1])

    accuracy, loss = loaded_model.evaluate(test_sequences, test_y)

    print("Testing model on dataset 1, accuracy: {:5.2f}%, loss {:5.2f}".format(100 * accuracy, 100 * loss))

    test_sequences, test_y = get_testing_data(test_dataset_2, tokenizer, loaded_model.layers[0].input_shape[0][1])

    accuracy, loss = loaded_model.evaluate(test_sequences, test_y)

    print("Testing model on dataset 2, accuracy: {:5.2f}%, loss {:5.2f}".format(100 * accuracy, 100 * loss))
