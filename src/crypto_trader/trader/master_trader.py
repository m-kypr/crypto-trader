import sys
import time
import json
from .trader import *
from .util import MyDataFrame


class MasterTrader(Trader):
    def __init__(self):
        self.binance = BinanceTrader()
        self.coinapi = CoinAPITrader(
            credentials={'apikey': '03323C42-A7FA-482D-9A5B-E1D93CB9CC04'})

    def plot(self):
        # r = self.binance.api.fetch_symbols()['symbols'][0]w
        # self.coinapi.api.fetch_klines
        # r = self.coinapi.api.fetch_symbols()

        # open('klines.tmp', 'w').write(json.dumps(
        #     self.coinapi.api.fetch_klines(
        #         'BITSTAMP_SPOT_BTC_USD',
        #         '1SEC',
        #         1000,
        #         str((datetime.datetime.now() - datetime.timedelta(days=12)
        #              ).replace(microsecond=0).isoformat()),
        #         str(datetime.datetime.now().replace(microsecond=0).isoformat())
        #     )))
        r = json.loads(open('klines.tmp', 'r').read())
        df = MyDataFrame(
            self.coinapi.api.dataframe_from_klines(r),
            parameters=self.coinapi.api.parameters
        )
        t = Thread(target=self.live, args=(df, ))
        t.start()
        # self.plotDf(df, do_ha=True, stoch=False)
        t.join()
        # while True:
        #     r = self.binance.api.fetch_ticker('BTCUSDT')
        #     print(r)
        #     time.sleep(1)

    def live(self, df, interval=1):
        # print(df)
        while True:
            r = self.binance.api.fetch_ticker('BTCUSDT')
            d = df.iloc[-1][self.coinapi.api.parameters['Time_open']]
            # n = datetime.datetime.utcnow()
            n = pd.Timestamp.now()
            if (n-d.tz_convert(None)).total_seconds() > interval:
                newn = n + pd.Timedelta(seconds=interval)
                # newn = n + datetime.timedelta(seconds=interval)
                # newn = newn.isoformat()
                # n = n.isoformat()
                bid = float(r['bidPrice'])
                ask = float(r['askPrice'])
                bidqty = float(r['bidQty'])
                askqty = float(r['askQty'])
                open = (bid + ask) / 2
                newr = {
                    df.parameters['Time']: n,
                    df.parameters['Time_end']: newn,
                    df.parameters['Time_open']: n,
                    df.parameters['Time_close']: newn,
                    df.parameters['Open']: open,
                    df.parameters['High']: max(bid, ask),
                    df.parameters['Low']: min(bid, ask),
                    df.parameters['Close']: open,
                    df.parameters['Volume']: bidqty + askqty,
                    df.parameters['Trades']: 1
                }
                ndf = self.coinapi.api.dataframe_from_klines([newr])
                df.loc[len(df)] = newr
                print(df)
            time.sleep(interval)
