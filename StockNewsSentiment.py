# Financial News Sentiment Prediction

import pandas as pd

from keras.preprocessing.sequence import pad_sequences

import keras

import pickle


# Preprocessing for prediction
def get_prediction_input(dataframe, tokenizer, max_seq):
    dataframe = dataframe.copy()

    sequences = tokenizer.texts_to_sequences(dataframe["Text"])
    sequences = pad_sequences(sequences, maxlen=max_seq, padding="post")

    return sequences


# predict on data
def model_predict(data):
    loaded_model = keras.models.load_model("trained_model/model.keras")

    loaded_model.summary()

    with open('trained_model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("Vocabulary Length: " + str(len(tokenizer.word_index) + 1))

    predict_sequence = get_prediction_input(data, tokenizer, loaded_model.layers[0].input_shape[0][1])

    predictions = loaded_model.predict(predict_sequence)

    negative = predictions[:, 0].tolist()
    neutral = predictions[:, 1].tolist()
    positive = predictions[:, 2].tolist()

    table = {"Positive": positive,
             "Negative": negative,
             "Neutral": neutral}

    df = pd.DataFrame(table).round(5)

    prediction_df = df.join(data)

    prediction_df["prediction"] = prediction_df.apply(lambda x:
                                                      "neutral" if float(x["Neutral"]) >= max(float(x["Positive"]), float(x["Negative"]))
                                                      else "positive" if float(x["Positive"]) > max(float(x["Neutral"]), float(x["Negative"]))
                                                      else "negative", axis=1)
    prediction_df.to_csv("output.csv", index=False)

    return prediction_df
