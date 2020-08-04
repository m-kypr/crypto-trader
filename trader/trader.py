import json
from pandas import DataFrame, to_datetime
import mplfinance as mpl
from ccxt.coinbase import coinbase
from ccxt.crex24 import crex24


class API(crex24):
    def __init__(self, config={}):
        super().__init__(config=config)

    def fetch_symbols(self):
        return super().fetch2('instruments')

    def fetchOHLCV2(self, symbol, timeframe='1m', limit=100, params={}):
        params['instrument'] = symbol
        params['granularity'] = timeframe
        params['limit'] = limit
        print(params)
        return super().fetch2('ohlcv', params=params)


if __name__ == "__main__":
    api = API()
    # sym = api.fetch_symbols()
    # sym = [s for s in sym if s['state']
    #        != 'delisted' and s['state'] != 'suspended']
    # open('cache1.tmp', 'w+').write(json.dumps(sym))
    # print([s['symbol'] for s in sym])
    # sym = api.fetchOHLCV2('BTC-TUSD')
    # open('cache.tmp', 'w+').write(json.dumps(sym))
    sym = json.loads(open('cache.tmp', 'r').read())
    print(sym[0])
    df = DataFrame(sym)
    df.index = to_datetime(df.index)
    print(df)
    mpl.plot(df)
    # a = mpl.plot(df, type='candle', style='binance', volume=True)
    # print(a)
