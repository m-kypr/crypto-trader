import json
from .api import API


class BinanceAPI(API):

    def __init__(self, credentials, endpoint='https://api.binance.com'):
        super().__init__(credentials, endpoint)

    def fetch(self, url, method='GET', headers=None, params=None, body=None):
        return self.session.request(method=method, url=self.endpoint + url, params=params, data=body)

    def fetch_klines(self, symbol, interval, limit=500):
        url = '/api/v3/klines'
        return self.fetch(url, params={'symbol': symbol, 'interval': interval, 'limit': limit}).text

    def fetch_symbols(self):
        url = '/api/v3/exchangeInfo'
        return self.fetch(url)

    def fetch_price(self, symbol):
        url = '/api/v3/avgPrice'
        return json.loads(self.fetch(url, params={'symbol': symbol}).text)['price']
