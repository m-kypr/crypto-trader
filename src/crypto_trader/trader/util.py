import pandas as pd
import json
import time
import numpy as np
from pandas import DataFrame


class JsonSerializable():
    def __repr__(self):
        return json.dumps(self.__dict__)

    @classmethod
    def deserialize(cls, s):
        return json.loads(s)

class MyDataFrame(DataFrame, JsonSerializable):
    """Custom DataFrame for stocks, extends pandas.DataFrame 

    Thanks to https://github.com/arkochhar/Technical-Indicators/blob/master/indicator/indicators.py
    For the indicators

    """

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)

    def calculate(self, do_ha=True, ha_replace=True, rsi_mode='ema', rsi_base='Close', period=14, smoothK=3, smoothD=3, stoch=True, stoch_level=1):
        if do_ha:
            self.HeikenAshi(replace=ha_replace)
        self.RSI(base=rsi_base, mode=rsi_mode, period=period, stoch=stoch,
                 stoch_level=stoch_level, smoothD=smoothD, smoothK=smoothK)

    def RSI2(self, mode='sma', period=14, stoch=True, stoch_level=1, smoothK=3, smoothD=3):
        """Calculate Relative Strength Index

        Args:
            mode (str, optional): sma (Simple moving average), ema (Exponential moving average). Defaults to 'sma'.

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

        delta = self.diff()
        delta = delta[1:]
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        if mode is 'ema':
            roll_up = up.ewm(span=period).mean()
            roll_down = down.abs().ewm(span=period).mean()
        elif mode is 'sma':
            roll_up = up.rolling(period).mean()
            roll_down = down.abs().rolling(period).mean()
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        if stoch:
            rsi = (rsi - rsi.rolling(period).min()) / \
                (rsi.rolling(period).max() - rsi.rolling(period).min())
            if stoch_level >= 1:
                rsi = rsi.rolling(smoothK).mean()
                if stoch_level >= 2:
                    rsi = rsi.rolling(smoothD).mean()
        return rsi

    def SMA(self, base, target, period=14):
        """ Function to compute Simple Moving Average (SMA)

        Args :
            base : String indicating the column name from which the SMA needs to be computed from
            target : String indicates the column name to which the computed data needs to be stored
            period : Integer indicates the period of computation in terms of number of candles (default is 14)

        Returns :
            df : Pandas DataFrame with new column added with name 'target'
        """

        self[target] = self[base].rolling(window=period).mean()
        self[target].fillna(np.nan, inplace=True)

    def EMA(self, base, target, period=14, alpha=False):
        """ Function to compute Exponential Moving Average (EMA)

        Args :
            base : String indicating the column name from which the EMA needs to be computed from
            target : String indicates the column name to which the computed data needs to be stored
            period : Integer indicates the period of computation in terms of number of candles (default is 14)
            alpha : Boolean if True indicates to use the formula for computing EMA using alpha (default is False)
        """

        con = pd.concat([self[:period][base].rolling(
            window=period).mean(), self[period:][base]])
        if (alpha == True):
            self[target] = con.ewm(alpha=1 / period, adjust=False).mean()
        else:
            self[target] = con.ewm(span=period, adjust=False).mean()

        self[target].fillna(np.nan, inplace=True)

    def HeikenAshi(self, ohlc=['Open', 'High', 'Low', 'Close'], replace=True):
        """
        Function to compute Heiken Ashi Candles (HA)

        Args :
            ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
        """

        ha_open = 'HA_' + ohlc[0]
        ha_high = 'HA_' + ohlc[1]
        ha_low = 'HA_' + ohlc[2]
        ha_close = 'HA_' + ohlc[3]
        df = self.copy()
        df[ha_close] = (df[ohlc[0]] + df[ohlc[1]] +
                        df[ohlc[2]] + df[ohlc[3]]) / 4

        df[ha_open] = 0.00
        for i in range(0, len(df)):
            if i == 0:
                df[ha_open].iat[i] = (
                    df[ohlc[0]].iat[i] + df[ohlc[3]].iat[i]) / 2
            else:
                df[ha_open].iat[i] = (
                    df[ha_open].iat[i - 1] + df[ha_close].iat[i - 1]) / 2

        df[ha_high] = df[[ha_open, ha_close, ohlc[1]]].max(axis=1)
        df[ha_low] = df[[ha_open, ha_close, ohlc[2]]].min(axis=1)
        if replace:
            self[ohlc[0]] = df[ha_open]
            self[ohlc[1]] = df[ha_high]
            self[ohlc[2]] = df[ha_low]
            self[ohlc[3]] = df[ha_close]
        else:
            return df

    def RSI(self, base="Close", mode='ema', period=14, stoch=True, stoch_level=1, smoothK=3, smoothD=3):
        """
        Function to compute Relative Strength Index (RSI)

        Args :
            base : String indicating the column name from which the MACD needs to be computed from (default is Close)
            period : Integer indicates the period of computation in terms of number of candles (default is 14)
            stoch: Boolean indicates if result is stooched (default is True)
            stoch_level: Integer indicates stoch level (default is 1)
            smoothK: Integer (default is 3)
            smoothD: Integer (default is 3)
        """

        delta = self[base].diff()
        delta = delta[1:]
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        if mode is 'ema':
            roll_up = up.ewm(span=period).mean()
            roll_down = down.abs().ewm(span=period).mean()
        elif mode is 'sma':
            roll_up = up.rolling(period).mean()
            roll_down = down.abs().rolling(period).mean()
        rsi = 100 - 100 / (1 + roll_up / roll_down)
        if stoch:
            rsi = (rsi - rsi.rolling(period).min()) / \
                (rsi.rolling(period).max() - rsi.rolling(period).min())
            if stoch_level >= 1:
                rsi = rsi.rolling(smoothK).mean()
                if stoch_level >= 2:
                    rsi = rsi.rolling(smoothD).mean()
            self['RSI_' + str(period)] = rsi

    def STDDEV(self, base, target, period):
        """
        Function to compute Standard Deviation (STDDEV)
        
        Args :
            df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
            base : String indicating the column name from which the SMA needs to be computed from
            target : String indicates the column name to which the computed data needs to be stored
            period : Integer indicates the period of computation in terms of number of candles
            
        Returns :
            df : Pandas DataFrame with new column added with name 'target'
        """

        df[target] = df[base].rolling(window=period).std()
        df[target].fillna(0, inplace=True)

        return df

    def ATR(self, period, ohlc=['Open', 'High', 'Low', 'Close']):
        """
        Function to compute Average True Range (ATR)
        
        Args :
            df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
            period : Integer indicates the period of computation in terms of number of candles
            ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
            
        Returns :
            df : Pandas DataFrame with new columns added for 
                True Range (TR)
                ATR (ATR_$period)
        """
        atr = 'ATR_' + str(period)

        # Compute true range only if it is not computed and stored earlier in the df
        if not 'TR' in df.columns:
            df['h-l'] = df[ohlc[1]] - df[ohlc[2]]
            df['h-yc'] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
            df['l-yc'] = abs(df[ohlc[2]] - df[ohlc[3]].shift())

            df['TR'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)

            df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)

        # Compute EMA of true range using ATR formula after ignoring first row
        EMA(df, 'TR', atr, period, alpha=True)

        return df

    def SuperTrend(self, period, multiplier, ohlc=['Open', 'High', 'Low', 'Close']):
        """
        Function to compute SuperTrend
        
        Args :
            df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
            period : Integer indicates the period of computation in terms of number of candles
            multiplier : Integer indicates value to multiply the ATR
            ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
            
        Returns :
            df : Pandas DataFrame with new columns added for 
                True Range (TR), ATR (ATR_$period)
                SuperTrend (ST_$period_$multiplier)
                SuperTrend Direction (STX_$period_$multiplier)
        """

        ATR(df, period, ohlc=ohlc)
        atr = 'ATR_' + str(period)
        st = 'ST_' + str(period) + '_' + str(multiplier)
        stx = 'STX_' + str(period) + '_' + str(multiplier)

        """
        SuperTrend Algorithm :
        
            BASIC UPPERBAND = (HIGH + LOW) / 2 + Multiplier * ATR
            BASIC LOWERBAND = (HIGH + LOW) / 2 - Multiplier * ATR
            
            FINAL UPPERBAND = IF( (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous Close > Previous FINAL UPPERBAND))
                                THEN (Current BASIC UPPERBAND) ELSE Previous FINALUPPERBAND)
            FINAL LOWERBAND = IF( (Current BASIC LOWERBAND > Previous FINAL LOWERBAND) or (Previous Close < Previous FINAL LOWERBAND)) 
                                THEN (Current BASIC LOWERBAND) ELSE Previous FINAL LOWERBAND)
            
            SUPERTREND = IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close <= Current FINAL UPPERBAND)) THEN
                            Current FINAL UPPERBAND
                        ELSE
                            IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close > Current FINAL UPPERBAND)) THEN
                                Current FINAL LOWERBAND
                            ELSE
                                IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close >= Current FINAL LOWERBAND)) THEN
                                    Current FINAL LOWERBAND
                                ELSE
                                    IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close < Current FINAL LOWERBAND)) THEN
                                        Current FINAL UPPERBAND
        """

        # Compute basic upper and lower bands
        df['basic_ub'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 + multiplier * df[atr]
        df['basic_lb'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 - multiplier * df[atr]

        # Compute final upper and lower bands
        df['final_ub'] = 0.00
        df['final_lb'] = 0.00
        for i in range(period, len(df)):
            df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i -
                                                                                                        1] or df[ohlc[3]].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
            df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i -
                                                                                                        1] or df[ohlc[3]].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]

        # Set the Supertrend value
        df[st] = 0.00
        for i in range(period, len(df)):
            df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[i] <= df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[i] > df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] >= df['final_lb'].iat[i] else \
                df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i -
                                                                                1] and df[ohlc[3]].iat[i] < df['final_lb'].iat[i] else 0.00

        # Mark the trend direction up/down
        df[stx] = np.where((df[st] > 0.00), np.where(
            (df[ohlc[3]] < df[st]), 'down',  'up'), np.NaN)

        # Remove basic and final bands from the columns
        df.drop(['basic_ub', 'basic_lb', 'final_ub',
                'final_lb'], inplace=True, axis=1)

        df.fillna(0, inplace=True)

        return df

    def MACD(self, fastEMA=12, slowEMA=26, signal=9, base='Close'):
        """
        Function to compute Moving Average Convergence Divergence (MACD)
        
        Args :
            df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
            fastEMA : Integer indicates faster EMA
            slowEMA : Integer indicates slower EMA
            signal : Integer indicates the signal generator for MACD
            base : String indicating the column name from which the MACD needs to be computed from (Default Close)
            
        Returns :
            df : Pandas DataFrame with new columns added for 
                Fast EMA (ema_$fastEMA)
                Slow EMA (ema_$slowEMA)
                MACD (macd_$fastEMA_$slowEMA_$signal)
                MACD Signal (signal_$fastEMA_$slowEMA_$signal)
                MACD Histogram (MACD (hist_$fastEMA_$slowEMA_$signal)) 
        """

        fE = "ema_" + str(fastEMA)
        sE = "ema_" + str(slowEMA)
        macd = "macd_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)
        sig = "signal_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)
        hist = "hist_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)

        # Compute fast and slow EMA
        EMA(df, base, fE, fastEMA)
        EMA(df, base, sE, slowEMA)

        # Compute MACD
        df[macd] = np.where(np.logical_and(np.logical_not(
            df[fE] == 0), np.logical_not(df[sE] == 0)), df[fE] - df[sE], 0)

        # Compute MACD Signal
        EMA(df, macd, sig, signal)

        # Compute MACD Histogram
        df[hist] = np.where(np.logical_and(np.logical_not(
            df[macd] == 0), np.logical_not(df[sig] == 0)), df[macd] - df[sig], 0)

        return df

    def BBand(self, base='Close', period=20, multiplier=2):
        """
        Function to compute Bollinger Band (BBand)
        
        Args :
            df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
            base : String indicating the column name from which the MACD needs to be computed from (Default Close)
            period : Integer indicates the period of computation in terms of number of candles
            multiplier : Integer indicates value to multiply the SD
            
        Returns :
            df : Pandas DataFrame with new columns added for 
                Upper Band (UpperBB_$period_$multiplier)
                Lower Band (LowerBB_$period_$multiplier)
        """

        upper = 'UpperBB_' + str(period) + '_' + str(multiplier)
        lower = 'LowerBB_' + str(period) + '_' + str(multiplier)

        sma = df[base].rolling(window=period, min_periods=period - 1).mean()
        sd = df[base].rolling(window=period).std()
        df[upper] = sma + (multiplier * sd)
        df[lower] = sma - (multiplier * sd)

        df[upper].fillna(0, inplace=True)
        df[lower].fillna(0, inplace=True)

        return df

    def Ichimoku(self, ohlc=['Open', 'High', 'Low', 'Close'], param=[9, 26, 52, 26]):
        """
        Function to compute Ichimoku Cloud parameter (Ichimoku)
        
        Args :
            df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
            ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
            param: Periods to be used in computation (default [tenkan_sen_period, kijun_sen_period, senkou_span_period, chikou_span_period] = [9, 26, 52, 26])
            
        Returns :
            df : Pandas DataFrame with new columns added for ['Tenkan Sen', 'Kijun Sen', 'Senkou Span A', 'Senkou Span B', 'Chikou Span']
        """

        high = df[ohlc[1]]
        low = df[ohlc[2]]
        close = df[ohlc[3]]

        tenkan_sen_period = param[0]
        kijun_sen_period = param[1]
        senkou_span_period = param[2]
        chikou_span_period = param[3]

        tenkan_sen_column = 'Tenkan Sen'
        kijun_sen_column = 'Kijun Sen'
        senkou_span_a_column = 'Senkou Span A'
        senkou_span_b_column = 'Senkou Span B'
        chikou_span_column = 'Chikou Span'

        # Tenkan-sen (Conversion Line)
        tenkan_sen_high = high.rolling(window=tenkan_sen_period).max()
        tenkan_sen_low = low.rolling(window=tenkan_sen_period).min()
        df[tenkan_sen_column] = (tenkan_sen_high + tenkan_sen_low) / 2

        # Kijun-sen (Base Line)
        kijun_sen_high = high.rolling(window=kijun_sen_period).max()
        kijun_sen_low = low.rolling(window=kijun_sen_period).min()
        df[kijun_sen_column] = (kijun_sen_high + kijun_sen_low) / 2

        # Senkou Span A (Leading Span A)
        df[senkou_span_a_column] = (
            (df[tenkan_sen_column] + df[kijun_sen_column]) / 2).shift(kijun_sen_period)

        # Senkou Span B (Leading Span B)
        senkou_span_high = high.rolling(window=senkou_span_period).max()
        senkou_span_low = low.rolling(window=senkou_span_period).min()
        df[senkou_span_b_column] = (
            (senkou_span_high + senkou_span_low) / 2).shift(kijun_sen_period)

        # The most current closing price plotted chikou_span_period time periods behind
        df[chikou_span_column] = close.shift(-1 * chikou_span_period)

        return df
