# Stock News Sentiment

Stock News Sentiment is a natural language processing model in Python to analyze the sentiment of financial text to predict future performance a stock and provide investment insight. It uses deep learning to power the model and a Gated Recurrent Unit neural network was engineered using TensorFlow with over 88% accuracy providing softmaxes for negative, neutral, and positive sentiment. A dataset of with almost 50,000 financial headlines was created through web scraping news on major stock screening sites such as Finviz.com and StockAnalysis.com to train the model.

If you decide to run the repository, please run:

```python
pip install -r requirements.txt
```

to download the required libraries for the project.

To load the model, first import pickle and load the tokenizer:

```python
import pickle

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
```

<br/>

### Technologies Used
- TensorFlow
- Numpy
- Scikit-learn
- Pandas
- BeautifulSoup (for web scraping)

