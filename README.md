# CRYPTO TRADER V1

## The API:

- https://docs.kucoin.com

## Algorithmic Trading:

- https://en.wikipedia.org/wiki/Algorithmic_trading
- https://www.investopedia.com/articles/active-trading/101014/basics-algorithmic-trading-concepts-and-examples.asp

## Trading Strategies:

- Strategy of moving averages:

  moving average of 20/30/40...

  -> DROPS BELOW the moving average of 100/200/300 => BUY

  -> RISES ABOVE the moving average of 100/200/300 => SELL

## Installation:

### Requirements:

- [Python 3.6](python.org) or higher

### Instructions:

```bash
$ git clone rep
```

Go to https://www.kucoin.com/ and get your API keys and paste them in [credentials.json](credentials.json)

```bash
$ pip install -r requirements.txt
$ python src/main.py
```
