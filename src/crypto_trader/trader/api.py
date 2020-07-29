import requests
import pandas


class API():
    def __init__(self, credentials, endpoint, intervals, headers):
        self.endpoint = endpoint
        self.session = requests.Session()
        self.intervals = intervals
        self.headers = headers

    def fetch(self, url, method='GET', headers=None, params=None, body=None):
        return self.session.request(
            method=method, url=self.endpoint + url, params=params, data=body)

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
        return pandas.DataFrame(klines, columns=self.headers).astype(float)

    def fetch_klines_as_dataframe(self, *args):
        return self.dataframe_from_klines(self.fetch_klines(*args))

    def fetch_price(self):
        raise NotImplementedError

    def fetch_klines(self, *args):
        raise NotImplementedError
