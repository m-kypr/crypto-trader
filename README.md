# **CRYPTO TRADER V1**

<br>

## ***Algorithmic Trading***

- https://en.wikipedia.org/wiki/Algorithmic_trading
- https://www.investopedia.com/articles/active-trading/101014/basics-algorithmic-trading-concepts-and-examples.asp

<br>

## ***Strategies***

### **Trend-following Strategies**

#### --- Moving averages ---

https://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp

short moving average (sMA)...

- DROPS BELOW the long moving average (lMA) => BUY

- RISES ABOVE the lMA => SELL

Problems with this strategy:

- Short jumps may cause intersection of sMA and lMA -> Almost no profit
- **High Fees** (Currently: 0.1%) are bad for short time trading
  => Solution: Trading API with low fees

#### --- Heikin-Ashi & StochRSI ---

> Got it somewhere from reddit

### **AI Trading**

> Jk. lol

<br>

## ***Installation***

### Requirements

- [Python 3.6](python.org) or higher

### Instructions

```bash
$ git clone git@github.com:m-kypr/crypto-trader.git
```

Go to https://www.kucoin.com/ and get your API keys and paste them in [credentials.json](src/credentials.json)

```bash
$ pip install -r requirements.txt
$ python src/main.py
```

### Notes for DEV:

TODO:

- Implement Heikin-Ashi
- Implement RSI & StochRSI
