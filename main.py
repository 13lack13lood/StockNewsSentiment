# Financial News Sentiment Prediction

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

import keras

import pickle

data = pd.read_csv("all-data.csv", names=["Label", "Text"], encoding="latin-1")
test_data = pd.read_csv("test_data.csv", names=["Label", "Text"], encoding="latin-1")
retrain_data = pd.read_csv("train_data.csv", names=["Label", "Text"], encoding="latin-1")
headline_data = pd.read_csv("analyzed_headlines.csv", names=["Label", "Text"], encoding="latin-1")
predict_data = pd.read_csv("oov_data.csv", names=["Text"], encoding="latin-1")


# Preprocessing for training

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


def get_training_data(dataframe, tokenizer, max_seq):
    dataframe = dataframe.copy()

    sequences = tokenizer.texts_to_sequences(dataframe["Text"])

    # max_seq = np.max(list(map(lambda x: len(x), sequences)))
    # print(max_seq)
    sequences = pad_sequences(sequences, maxlen=max_seq, padding="post")

    label_map = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    y = dataframe["Label"].replace(label_map)

    return sequences, y


def get_prediction_input(dataframe, tokenizer, max_seq):
    dataframe = dataframe.copy()

    sequences = tokenizer.texts_to_sequences(dataframe["Text"])

    # max_seq = np.max(list(map(lambda x: len(x), sequences)))
    # print(max_seq)
    sequences = pad_sequences(sequences, maxlen=max_seq, padding="post")

    return sequences


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


# Training

def create_new_model(min_acc_save):
    # Vocal Length
    input_dimension = 25000

    train_sequences, test_sequences, y_train, y_test = preprocess_inputs(data)

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

    accuracy = model.evaluate(test_sequences, y_test)[1]

    if accuracy > min_acc_save:
        model.save("modelGRU(" + str(round(accuracy * 1000)) + ").keras")

    # Results
    print(accuracy)

    print(y_test.value_counts())

    return accuracy


# min_accuracy = 0.76
#
# while True:
#
#     new_acc = create_new_model(min_accuracy)
#
#     if new_acc > min_accuracy:
#         min_accuracy = new_acc

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
#
print(len(tokenizer.word_index) + 1)

test_sequence, test_y = get_training_data(test_data, tokenizer, 71)
predict_sequence = get_prediction_input(retrain_data, tokenizer, 71)
#
loaded_model = keras.models.load_model("model.keras")

# loaded_model.summary()
# loss, acc = loaded_model.evaluate(test_sequence, test_y, verbose=2)
# print('Trained model, accuracy: {:5.2f}%, loss: {:5.2f}%'.format(100 * acc, 100 * loss))

predictions = loaded_model.predict(predict_sequence)

negative = predictions[:, 0].tolist()
neutral = predictions[:, 1].tolist()
positive = predictions[:, 2].tolist()

table = {"Positive": positive,
         "Negative": negative,
         "Neutral": neutral}

df = pd.DataFrame(table).round(5)

prediction_df = df.join(test_data)

# prediction_df["prediction"] = prediction_df.apply(lambda x:
#                                             "neutral" if float(x["Neutral"]) > 0.5
#                                             else "positive" if float(x["Positive"]) > float(x["Negative"])
#                                             else "negative", axis=1)

prediction_df["prediction"] = prediction_df.apply(lambda x:
                                                  "neutral" if float(x["Neutral"]) >= max(float(x["Positive"]), float(x["Negative"]))
                                                  else "positive" if float(x["Positive"]) > max(float(x["Neutral"]), float(x["Negative"]))
                                                  else "negative", axis=1)

prediction_df["error"] = prediction_df.apply(lambda a: 0 if str(a["prediction"]) == str(a["Label"]) else 1, axis=1)

error_count = prediction_df["error"].sum()

print("Error Count:" + str(error_count.round(2)))
print("Error %: " + str(((error_count / prediction_df["error"].shape[0]) * 100).round(3)))

prediction_df.to_csv("output.csv", index=False)

# min_acc_train = 0.79
# min_acc_test = 0.77
#
# while True:
#     loaded_model = keras.models.load_model("modelGRU789.keras")
#
#     loaded_model.summary()
#
#     train_sequences, test_sequences, y_train, y_test = get_retrain_data(retrain_data, tokenizer, 71)
#
#     loaded_model.fit(
#         train_sequences,
#         y_train,
#         validation_split=0.2,
#         batch_size=32,
#         epochs=100,
#         callbacks=[
#             keras.callbacks.EarlyStopping(
#                 monitor='val_loss',
#                 patience=8,
#                 restore_best_weights=True
#             )
#         ]
#     )
#
#     acc = loaded_model.evaluate(test_sequences, y_test, verbose=2)[1]
#
#     print('Training accuracy: {:5.2f}%'.format(100 * acc))
#
#     if acc > min_acc_train:
#         min_acc = acc
#         loaded_model.save("modelV2" + str(round(acc * 1000)) + ".keras")
#
#     loss, acc = loaded_model.evaluate(test_sequence, test_y, verbose=2)
#     print('Testing accuracy: {:5.2f}%'.format(100 * acc))
#
#     if acc > min_acc_test:
#         min_acc = acc
#         loaded_model.save("testmodelV2(" + str(round(acc * 1000)) + ").keras")
