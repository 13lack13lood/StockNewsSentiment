# Sentimental Analysis of Financial Text

import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import keras
import pickle


class StockNewsSentimentModel:
    def __init__(self, model_path="trained_model/model.keras", tokenizer_path="trained_model/tokenizer.pickle"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        self.model = keras.models.load_model(self.model_path)
        self.model.summary()

        with open('trained_model/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        print("Vocabulary Length: " + str(len(self.tokenizer.word_index) + 1))

        self.max_seq = self.model.layers[0].input_shape[0][1]

        print("_________________________________________________________________\n\n")
        print("Model and Tokenizer Loaded.")
        print("\n\n_________________________________________________________________")

    # Preprocessing for prediction
    def get_prediction_input(self, dataframe):
        dataframe = dataframe.copy()

        sequences = self.tokenizer.texts_to_sequences(dataframe["Text"])
        sequences = pad_sequences(sequences, maxlen=self.max_seq, padding="post")

        return sequences

    # predict on data
    def model_predict(self, data):
        predict_sequence = self.get_prediction_input(data)

        predictions = self.model.predict(predict_sequence)

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
        prediction_df.to_csv("predictions.csv", index=False)

        return prediction_df
