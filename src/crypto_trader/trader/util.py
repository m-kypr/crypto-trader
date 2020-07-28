import json
import time
import numpy as np
from abc import ABC
from pandas import DataFrame


class Math():
    """Maths and stuff.
    """
    @classmethod
    def running_mean(cls, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    @classmethod
    def ema(cls, values, window):
        df = DataFrame({'v': values})
        return df.ewm(span=window).mean()

    @classmethod
    def rsi(cls):
        """
        RSI = 100 - (100 / (1 + avrg gain / avrg loss))
        The average gain or loss used in the calculation is the average percentage gain or loss during a look-back period. 
        The formula uses a positive value for the average loss. 
        The standard is to use 14 periods to calculate the initial RSI value. 

        For example, imagine the market closed higher seven out of the past 14 days with an average gain of 1%. 
        The remaining seven days all closed lower with an average loss of -0.8%. 
        The calculation for the first part of the RSI would look like the following expanded calculation: 

        RSI = 100 - (100 / (1 + (1% / 14) / (0.8% / 14))) = 55.55

        Once there are 14 periods of data available, the second part of the RSI formula can be calculated. 
        The second step of the calculation smooths the results. 

        RSI = 100 - (100 / 1 + ( (prev avrg gain * (period - 1) + current gain) / (prev avrg loss * (period - 1) + current loss) ))
        """
        pass

    @classmethod
    def StochRSI(cls):
        """
        Resource: https://www.investopedia.com/terms/s/stochrsi.asp

        (RSI - min(RSI)) / (max(RSI) - min(RSI))
        RSI = Current RSI reading
        min[RSI] = Lowest RSI reading over the last 14 periods( or your chosen lookback interval)
        max[RSI] = Highest RSI reading over the last 14 periods( or your chosen lookback interval)

        How to Calculate the Stochastic RSI

        The StochRSI is based on RSI readings. The RSI has an input value, typically 14, which tells the indicator how many periods of data it is using in its calculation. These RSI levels are then used in the StochRSI formula.
            1. Record RSI levels for 14 periods.
            2. On the 14th period, note the current RSI reading, the highest RSI reading, and lowest RSI reading. It is now possible to fill in all the formula variables for StochRSI.
            3. On the 15th period, note the current RSI reading, highest RSI reading, and lowest reading, but only for the last 14 period (not the last 15). Compute the new StochRSI.
            4. As each period ends compute the new StochRSI value, only using the last 14 RSI values.

        """
        pass

    @classmethod
    def HeikinAshi(cls):
        pass

    # @classmethod
    # def lowest(cls, n, max_buf):
    #     if n > 0:
    #         b = []
    #         for i in argrelextrema(np.array(max_buf), np.less)[0].tolist():
    #             b.append(max_buf[i])
    #         return Util.lowest(n - 1, max_buf=b)
    #     else:
    #         return max_buf

    # @classmethod
    # def greatest(cls, n, max_buf):
    #     if n > 0:
    #         b = []
    #         for i in argrelextrema(np.array(max_buf), np.greater)[0].tolist():
    #             b.append(max_buf[i])
    #         return Util.greatest(n - 1, max_buf=b)
    #     else:
    #         return max_buf


class JsonSerializable(ABC):
    def __repr__(self):
        return json.dumps(self.__dict__)

    @classmethod
    def deserialize(cls, s):
        return json.loads(s)


class T(JsonSerializable, ABC):
    def __init__(self, symbol, price, time):
        self.time = time
        self.symbol = symbol
        self.price = price


class Intersection(JsonSerializable):
    def __init__(self, type, x, y):
        self.type = type
        self.x = float(x)
        self.y = float(y)


class Trade(JsonSerializable):
    def __init__(self, type, symbol, price, time=time.time()):
        self.type = type
        super().__init__(symbol, price, time)


class Candlestick(JsonSerializable):
    def __init__(self, open, close, low, high, time=time.time()):
        self.open = open
        self.close = close
        self.low = low
        self.high = high
        self.time = time

    @classmethod
    def fromJson(cls, time, open, close, low, high):
        return Candlestick(float(open), float(close), float(low), float(high), float(time))
