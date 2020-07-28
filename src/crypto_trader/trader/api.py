import requests


class API():
    def __init__(self, credentials, endpoint):
        self.endpoint = endpoint
        self.session = requests.Session()

    def fetch(self):
        raise NotImplementedError

    def fetch_price(self):
        raise NotImplementedError

    def fetch_klines(self):
        raise NotImplementedError
