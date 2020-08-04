import json
import requests
import pandas
from .util import MyDataFrame
# import logging

# logging.basicConfig(level=logging.DEBUG)


def pretty_print_POST(req):
    """
    At this point it is completely built and ready
    to be fired; it is "prepared".

    However pay attention at the formatting used in 
    this function because it is programmed to be pretty 
    printed and may differ from the actual request.
    """
    print('{}\n{}\r\n{}\r\n\r\n{}'.format(
        '-----------START-----------',
        req.method + ' ' + req.url,
        '\r\n'.join('{}: {}'.format(k, v) for k, v in req.headers.items()),
        req.body,
    ))


class API():
    def __init__(self, credentials, endpoint, intervals, parameters, auth={}):
        self.endpoint = endpoint
        self.session = requests.Session()
        self.intervals = intervals
        self.parameters = parameters
        self.credentials = credentials
        self.auth = auth

    def fetch(self, url, method='GET', headers=None, params=None, body=None):
        r = self.session.request(
            method=method, url=self.endpoint + url, params=params, data=body)
        return r

    def interval_to_millis(self, interval):
        interval = list(interval)
        interval.reverse()
        for c in interval:
            if c is self.intervals['minutes']:
                n = 60000
                continue
            elif c is self.intervals['hours']:
                n = 60000 * 60
                continue
            elif c is self.intervals['days']:
                n = 60000 * 60 * 24
                continue
            elif c is self.intervals['weeks']:
                n = 60000 * 60 * 24 * 7
                continue
            elif c is self.intervals['months']:
                n = 60000 * 60 * 24 * 7 * 4
                continue
            try:
                return int(n * int(c))
            except ValueError as e:
                raise Exception

    def dataframe_from_klines(self, klines):
        df = pandas.DataFrame(klines, columns=self.parameters.values())
        d = {}
        for n, col in df.iteritems():
            try:
                col.astype(float)
            except (ValueError, TypeError) as e:
                try:
                    if col.dtype == pandas.DataFrame.dtypes
                    if not col.dt.tz:
                        #     col = pandas.DatetimeIndex(col.index)
                        col.tz_localize('UTC')
                    col = pandas.to_datetime(col)
                    #     print(col)
                except ValueError as e:
                    print(col)
                    print(e)
                    quit(1)
            d[n] = col

        return MyDataFrame(d)

    def fetch_klines_as_dataframe(self, *args):
        return self.dataframe_from_klines(self.fetch_klines(*args))

    def fetch_price(self):
        raise NotImplementedError

    def fetch_klines(self, *args):
        raise NotImplementedError


class CoinAPI(API):
    def __init__(
            self,
            credentials,
            endpoint='https://rest.coinapi.io',
            intervals={'minutes': 'MIN', 'seconds': 'SEC',
                       'hours': 'HRS', 'days': 'DAY', 'months': 'MTH'},
            parameters={'Time': 'time_period_start', 'Time_end': 'time_period_end', 'Time_open': 'time_open', 'Time_close': 'time_close', 'Open': 'price_open', 'High': 'price_high', 'Low': 'price_low', 'Close': 'price_close', 'Volume': 'volume_traded', 'Trades': 'trades_count'}):
        super().__init__(credentials, endpoint,
                         intervals, parameters, auth={'X-CoinAPI-Key': credentials['apikey']})

    def fetch_klines(self, symbol, interval='1SEC', limit=1000, *args):
        """
        https://docs.coinapi.io/#ohlcv
        """
        start_time = args[0]
        end_time = args[1]
        url = f'/v1/ohlcv/{symbol}/history'
        return json.loads(super().fetch(url, headers=self.auth, params={'period_id': interval, 'time_start': start_time, 'time_end': end_time, 'limit': limit, 'apikey': self.credentials['apikey']}).text)

    def fetch_latest(self, symbol, limit=1000, *args):
        url = f'/v1/quotes/{symbol}/current'
        return json.loads(super().fetch(url, params=self.credentials).text)

    def fetch_symbols(self):
        url = '/v1/symbols'
        return json.loads(super().fetch(url, params=self.credentials).text)


class BinanceAPI(API):

    def __init__(self, credentials={}):
        super().__init__(
            credentials=credentials,
            endpoint='https://api.binance.com',
            intervals={'minutes': 'm', 'hours': 'h',
                       'days': 'd', 'weeks': 'w', 'months': 'M'},
            parameters={'Time': 'Open time', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume', 'Time_close': 'Close time', 'Quote_volume': 'Quote asset volume', 'Trades': 'Number of trades', 'Taker_base_volume': 'Taker buy base asset volume', 'Taker_quote_volume': 'Taker buy quote asset volume', 'Ignore': 'Ignore'})

    def fetch_klines(self, symbol, interval, limit=500):
        url = '/api/v3/klines'
        return json.loads(self.fetch(url, params={'symbol': symbol, 'interval': interval, 'limit': limit}).text)

    def fetch_symbols(self):
        url = '/api/v3/exchangeInfo'
        return json.loads(self.fetch(url).text)

    def fetch_price(self, symbol):
        url = '/api/v3/avgPrice'
        return json.loads(self.fetch(url, params={'symbol': symbol}).text)['price']

    def fetch_ticker(self, symbol):
        url = '/api/v3/ticker/bookTicker'
        # print(self.fetch(url, params={'symbol': symbol}).text)
        return json.loads(self.fetch(url, params={'symbol': symbol}).text)
