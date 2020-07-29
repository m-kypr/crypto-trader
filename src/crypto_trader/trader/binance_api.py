import json
from .api import API


class BinanceAPI(API):

    def __init__(self, credentials={}):
        super().__init__(
            credentials=credentials,
            endpoint='https://api.binance.com',
            intervals={'minutes': 'm', 'hours': 'h',
                       'days': 'd', 'weeks': 'w', 'months': 'M'},
            headers=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])

    def fetch_klines(self, symbol, interval, limit=500):
        url = '/api/v3/klines'
        return json.loads(self.fetch(url, params={'symbol': symbol, 'interval': interval, 'limit': limit}).text)

    def fetch_symbols(self):
        url = '/api/v3/exchangeInfo'
        return json.loads(self.fetch(url).text)

    def fetch_price(self, symbol):
        url = '/api/v3/avgPrice'
        return json.loads(self.fetch(url, params={'symbol': symbol}).text)['price']
