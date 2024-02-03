# Stock News Sentiment

Stock News Sentiment is a natural language processing model in Python to analyze the sentiment of financial text to predict future performance a stock and provide investment insight. It uses deep learning to power the model and a Gated Recurrent Unit neural network was engineered using TensorFlow with over 88% accuracy providing softmaxes for negative, neutral, and positive sentiment. A dataset of with almost 50,000 financial headlines was created through web scraping news on major stock screening sites such as Finviz.com and StockAnalysis.com to train the model.

## Usage
If you decide to run the repository, please run:

```
pip install -r requirements.txt
```

to download the required libraries for the project.

### Performing Predictions

To use the pre-trained model and perform predictions:

```python
import pandas as pd
from StockNewsSentiment import StockNewsSentimentModel

data = pd.read_csv(FILE_PATH, names=["Text"], encoding="latin-1")

model = StockNewsSentimentModel()

model.model_predict(data)
```

This will read a csv file and return a dataframe with the three different softmax values and the final label. It will also save the dataframe as a csv file in ```predictions.csv```. 

### Loading the pre-trained model separately

If you with to load the model manually, the functions to process the input and output can be found in ```predict_model/predict_functions.py```:
```python
import keras
import pickle
import pandas as pd
from predict_model.predict_functions import process_prediction_input, process_prediction_output

data = pd.read_csv(FILE_PATH, names=["Text"], encoding="latin-1")

loaded_model = keras.models.load_model("trained_model/model.keras")

loaded_model.summary()

with open('trained_model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

print("Vocabulary Length: " + str(len(tokenizer.word_index) + 1))
print("_________________________________________________________________\n\n")
print("Model and Tokenizer Loaded.")
print("\n\n_________________________________________________________________")

max_sequence = loaded_model.layers[0].input_shape[0][1]

predict_sequence = process_prediction_input(data, tokenizer, max_sequence)

predictions = loaded_model.predict(predict_sequence)

predictions_df = process_prediction_output(predictions, data)
```

### Creating a new model

If you wish to create a new model, the functions for that can be found in ```create_new_model/create_new_model.py``` and it is a two-step process.

The first step creates the model and trains it with known vocabulary. The function will constantly save the model with the highest accuracy locally.

```python
from create_new_model.create_new_model import create_new_model

min_acc_save = 0

while True:
    acc = create_new_model(min_acc_save)

    if acc > min_acc_save:
        min_acc_save = acc

```

The second step trains the model again with a new dataset with words that are out of vocabulary.

```python
from create_new_model.create_new_model import retrain_model

min_acc_save = 0

while True:
    acc = retrain_model(MODEL_FILE_PATH, min_acc_save)

    if acc > min_acc_save:
        min_acc_save = acc

```

### Testing a model

If you wish to test a model, the functions for that can be found in ```testing_model/testing_model.py```.

```python
import pickle
from testing_model.testing_model import test_model

with open('FILE_PATH/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

test_model(MODEL_FILE_PATH, tokenizer)
```

This will test your model on two different datasets and will print the resulting accuracy and loss.

## Technologies Used
- TensorFlow
- Numpy
- Scikit-learn
- Pandas
- BeautifulSoup (for web scraping)

