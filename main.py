from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd


def stock_news_analysis():
    # URL for news
    finviz_url = "https://finviz.com/quote.ashx?t="

    tickers = ["AAPL"]

    news_tables = {}

    for ticker in tickers:
        url = finviz_url + ticker
        req = Request(url=url, headers={"user-agent": "stock-app"})
        response = urlopen(req)

        html = BeautifulSoup(response, features="html.parser")

        news_table = html.find(id="news-table")
        news_tables[ticker] = news_table

    data = []

    for ticker, news_table in news_tables.items():
        day = ""

        for i, row in enumerate(news_table.findAll("tr")):
            text = row.a.text
            print(text)
            time = row.td.text.strip()
            if " " in time:
                day = time[:time.find(" ")]

                if day == "Today":
                    day = "Dec-20-24"

                time = time[time.find(" ") + 1:]
            data.append([ticker, day, time, text])

    # print(data)

    dataframe = pd.DataFrame(data, columns=['ticker', 'date', 'time', 'text'])

    # initialize vader
    # vader = SentimentIntensityAnalyzer()
    #
     # dataframe["compound"] = dataframe["text"].apply(lambda t: vader.polarity_scores(t)["compound"])
    # dataframe["date"] = pd.to_datetime(dataframe.date).dt.date
    #
    # mean_dataframe = dataframe.groupby(["ticker", "date"]).mean()
    print(type(dataframe["text"]))

