from .api import BinanceAPI, CoinAPI
import os
import sys
import time
import json
import shutil
import datetime
import numpy as np
import pandas as pd
from math import pi
import mplfinance as mpl
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .util import MyDataFrame


class Trader():
    """Algorithmic Trading class
    Originally developed by: https://github.com/m-kypr

    """

    def __init__(self, api=None, credentials={}, config_path='config.json', positions_path='positions.json', log_path='log', demo=True):
        if api:
            self.demo = demo
            # Constants
            self.path = os.path.dirname(os.path.abspath(__file__))
            self.positions_path = os.path.join(self.path, positions_path)
            self.config_path = os.path.join(self.path, config_path)
            self.log_path = os.path.join(self.path, log_path)
            self.trades_log = os.path.join(self.log_path, 'trades.log')
            self.profit_log = os.path.join(self.log_path, 'profit.log')

            # Callable objects
            self.api = api(credentials)
            self.proc = []

            self.load_config()
            self.initfiles()

    def load_config(self):
        self.config = json.loads(open(self.config_path, 'r').read())
        self.fee = self.config['fee']

    def initfiles(self):
        try:
            os.mkdir(self.log_path)
        except Exception as e:
            pass
        try:
            open(self.positions_path, 'w+')
        except Exception as e:
            pass
        if os.stat(self.positions_path).st_size == 0:
            with open(self.positions_path, 'a+') as f:
                f.write('{}')

    def plot(self):
        raise NotImplementedError

    def trade(self):
        raise NotImplementedError

    def candlestick_from_json(self, json):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def clearlogs(self):
        for filename in os.listdir(self.log_path):
            file_path = os.path.join(self.log_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    # def calc(self, css, rsi='sma', stoch=False, period=14, stoch_smoothness=0):
    #     x = np.array([cs.time for cs in css])
    #     y = np.array([cs.open for cs in css])
    #     if rsi:
    #         if rsi is 'sma' or 'ema':
    #             x = x[1:]
    #             y = y[1:]
    #             if stoch:
    #                 StochRSI, StochRSIK, StochRSID = Math.StochRSI(
    #                     css, mode=rsi, period=period)
    #                 if stoch_smoothness == 1:
    #                     StochRSI = StochRSIK
    #                 elif stoch_smoothness == 2:
    #                     StochRSI = StochRSID
    #                 return x, y, StochRSI
    #             else:
    #                 RSI = Math.RSI(css, mode=rsi, period=period)
    #                 return x, y, RSI
    #     return x, y, _

    # def animate(self, _, g, rsi_g, css, symbol, interval, rsi_mode, stoch, stoch_smoothness, *args):
    #     css.append([self.candlestick_from_json(c) for c in json.loads(
    #         self.api.fetch_klines(symbol, interval))][-1])
    #     if stoch:
    #         x, y, StochRSI = self.calc(
    #             css, rsi=rsi_mode, stoch=True, stoch_smoothness=stoch_smoothness)
    #         rsi_g.set_data(x, StochRSI)
    #     else:
    #         x, y, RSI = self.calc(css, rsi=rsi_mode)
    #         rsi_g.set_data(x, RSI)
    #     g.set_data(x, y)

    def plotS(self, symbol, interval, rsi_base, limit=100, do_ha=True, rsi_mode='ema', stoch=True, stoch_smoothness=2, klines_args=()):
        # pause = int(self.api.interval_to_millis(interval) / 1000)
        # fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        print(limit)
        df = MyDataFrame(self.api.fetch_klines_as_dataframe(
            symbol, interval, limit, *klines_args), parameters=self.api.parameters)
        print(df)
        quit()
        df.index = pd.to_datetime(df[self.api.parameters['Time']], unit='ms')
        # print(mpl.available_styles())
        df.calculate(rsi_base=rsi_base, do_ha=do_ha)
        apds = [
            mpl.make_addplot(df['RSI_14'], panel=1)
        ]

        def watcher(self):
            while True:
                ndf = MyDataFrame(
                    self.api.fetch_klines_as_dataframe(symbol, interval, 1))
                if ndf[self.api.parameters['Time']].iloc[0] >= df[self.api.parameters['Time']].iloc[-1] + pause:
                    # ndf.calculate(rsi_base=rsi_base, do_ha=do_ha)
                    print()
                    df.loc[len(df)] = ndf.iloc[0].values
                    df.index = pd.to_datetime(
                        df[self.api.parameters['Time']], unit='ms')
                    df.calculate(rsi_base=rsi_base, do_ha=do_ha)
                    print(df)
                    if self.demo:
                        print('Updating')
                time.sleep(pause)
        # t = Thread(target=watcher, args=(self, ))
        # t.start()
        print(df)
        plotdf = MyDataFrame(df.copy())
        # plotdf.columns = list(self.api.parameters.keys()) + ['RSI_14']
        print(plotdf)
        quit()
        mpl.plot(plotdf, type='candle', style='binance',
                 volume=True, addplot=apds)
        # t.join()

    def plotDf(self, df, rsi_base=None, do_ha=True, ha_replace=True, rsi_mode='ema', period=14, smoothK=3, smoothD=3, stoch=True, stoch_level=1):
        plotdf = MyDataFrame(df.copy(), parameters=df.parameters)
        plotdf.index = pd.to_datetime(
            plotdf[plotdf.parameters['Time']], unit='ms')
        plotdf.calculate(rsi_base, do_ha, ha_replace, rsi_mode,
                         period, smoothK, smoothD, stoch, stoch_level)
        apds = []
        extra = []
        if len(plotdf.columns) > len(plotdf.parameters):
            extra = plotdf.columns[len(plotdf.parameters.keys()):].tolist()
            for col in extra:
                apds.append(mpl.make_addplot(plotdf[col], panel=1))
        plotdf.columns = list(plotdf.parameters.keys()) + extra
        mpl.plot(plotdf, type='candle', style='binance',
                 volume=True, addplot=apds)

    def watchdog(self, pid, symbol, interval, limit=100, do_ha=True, rsi_mode='ema', period=14, stoch=True, stoch_smoothness=1):
        """Watches.

        Args:
            symbol (string): Crypto symbol to watch i.e "BTC"
            pid (int): process id
        """
        stop = False
        log = self.log_path + '/wd' + str(pid) + '.log'
        pause = int(self.api.interval_to_millis(interval) / 1000)

        def print(s, end='\n'):
            open(log,
                 'a').write(str(datetime.datetime.now()) + ' ' + str(s) + end)

        def printtrade(trade):
            open(self.trades_log, 'a').write(str(trade) + '\n')

        def printprofit(profit):
            open(self.profit_log, 'a').write(str(profit) + '\n')

        def sell():
            posjson = json.loads(open(self.positions_path, 'r').read())
            if 'positions' in posjson:
                for pos in posjson['positions']:
                    if pos['symbol'] == symbol:
                        price = pos['price']
                        amount = pos['amount']
                        nprice = self.api.fetch_price(symbol)
                        namount = (amount / price) * \
                            nprice * (1 - self.fee)
                        profit = namount - amount
                        if profit > 0:
                            print('Selling at '+str(nprice))
                            # if self.demo is False:
                            #     self.api.create_market_sell_order(
                            #         symbol, amount)
                            printtrade('SELL ' + symbol + ' ' + str(nprice))
                            printprofit(profit)
                            posjson['positions'].remove(pos)
                            open(self.positions_path, 'w').write(
                                json.dumps(posjson))

        def buy(amount='all'):
            posjson = json.loads(open(self.positions_path, 'r').read())
            if 'positions' in posjson:
                if symbol in [x['symbol'] for x in posjson['positions']]:
                    # print('Symbol is already in positions')
                    return
            else:
                posjson['positions'] = []
            if amount == 'all':
                amount = self.api.fetch_free_balance()
            if amount > 0:
                price = float(self.api.fetch_price(symbol))
                print('Buying at '+str(price))
                # if not self.demo:
                #   self.api.create_market_buy_order(symbol, amount)
                printtrade('BUY ' + symbol + ' ' + str(price))
                posjson['positions'].append({
                    'amount': amount * (1 - self.fee),
                    'symbol': symbol,
                    'Time': time.time(),
                    'price': price
                })
                open(self.positions_path, 'w').write(json.dumps(posjson))

        print('Currently watching: ' + symbol)
        df = MyDataFrame(self.api.fetch_klines_as_dataframe(
            symbol, interval, limit))
        df.calculate(rsi_base=rsi_base, do_ha=do_ha,
                     period=period, rsi_mode=rsi_mode)
        while not stop:
            ndf = MyDataFrame(
                self.api.fetch_klines_as_dataframe(symbol, interval, 1))
            if ndf[self.api.parameters['Time']].iloc[0] >= df[self.api.parameters['Time']].iloc[-1] + pause:
                # ndf.calculate(do_ha=do_ha, rsi_mode=rsi_mode, period=period)
                df.loc[len(df)] = ndf.iloc[0].values
                df.index = pd.to_datetime(
                    df[self.api.parameters['Time']], unit='ms')
                df.calculate(rsi_base=rsi_base, do_ha=do_ha,
                             period=period, rsi_mode=rsi_mode)
                if self.demo:
                    print('Updating')
                print(df['RSI_' + str(period)][-1])
                if df['RSI_' + str(period)][-1] > 0.8:
                    print('BUY')
                elif df['RSI_' + str(period)][-1] < 0.2:
                    print('SELL')
            time.sleep(pause)

    def tradeS(self, symbol, interval, limit=100, do_ha=True, rsi_mode='ema', stoch=True, stoch_smoothness=1):
        self.proc.append(Thread(target=self.watchdog,
                                args=(len(self.proc), symbol, interval, limit, do_ha, rsi_mode, stoch, stoch_smoothness, )))
        self.proc[-1].start()
        print('Spawned ' + symbol + ' dog')
        print('View logs: log/wd'+str(len(self.proc)-1)+'.log')


class CoinAPITrader(Trader):
    """CoinAPI Trader
    https://docs.coinapi.io/
    """

    def __init__(self, api=CoinAPI, credentials={}, config_path='config.json', positions_path='positions.json', log_path='log', demo=True):
        super().__init__(api, credentials=credentials, config_path=config_path,
                         positions_path=positions_path, log_path=log_path, demo=demo)

    def plot(self):
        r = self.api.fetch_latest('BITSTAMP_SPOT_BTC_USD')
        print(r)

        # self.plotS(symbol='BITSTAMP_SPOT_BTC_USD', interval='1DAY', limit=100000, rsi_base=self.api.parameters['Close'],
        #            klines_args=(str((datetime.datetime.now() - datetime.timedelta(days=12)).replace(microsecond=0).isoformat()), str(datetime.datetime.now().replace(microsecond=0).isoformat())))


class BinanceTrader(Trader):
    def __init__(self, credentials={}, api=BinanceAPI, config_path='config.json', positions_path='positions.json', log_path='log', demo=True):
        super().__init__(api, credentials=credentials, config_path=config_path,
                         positions_path=positions_path, log_path=log_path, demo=demo)

    def plot(self):
        self.plotS('BTCUSDT', '1m')

    def trade(self):
        self.tradeS('BTCUSDT', '1m')

    def reset(self):
        self.clearlogs()

# class KTrader():
#     """Kucoin API algorithmic Trader
#     DOCS: https://docs.kucoin.com/#general
#     Using the cctx library as abstraction layer: https://github.com/ccxt/ccxt
#     """

#     def __init__(self, demo):
#         self.config_path = PATH + '/config.json'
#         self.positions_path = PATH + '/../positions.json'
#         self.log_path = PATH + '/log'
#         self.trades_log = self.log_path + '/trades.log'
#         self.profit_log = self.log_path + '/profit.log'

#         self.load_config()
#         self.initfiles()

#         self.demo = demo
#         self.proc = []
#         self.api = KucoinAPI(PATH + '/credentials.json')

#         self.handle_argv()

#         self.deploy_dogs()

#     def initfiles(self):
#         try:
#             os.mkdir(self.log_path)
#             open(self.positions_path, 'w+')
#         except Exception as e:
#             pass

#     def load_config(self):
#         config = json.loads(open(self.config_path, 'r').read())
#         self.symbols = config['symbols']
#         self.fee = config['fee']
#         self.c_short = config['c_short']
#         if config['c_short'] == 0:
#             from math import e
#             self.c_short = int(round(self.fee*((e)*10000)))
#         else:
#             self.c_short = config['c_short']
#         if config['c_long'] == 0:
#             self.c_long = self.c_short * 2
#         else:
#             self.c_long = config['c_long']
#         self.pause = config['pause']

#     def handle_argv(self):
#         """Handle user input.
#         plot -- use matplotlib to plot graph
#         clearlogs -- clear the contents of log/
#         """
#         if len(sys.argv) > 1:
#             if sys.argv[1] == 'plot':
#                 self.hairy_plotter()
#             if sys.argv[1] == 'clearlogs':
#                 self.clearlogs()
#             quit()

#     def clearlogs(self):
#         """Clear the contents of log/
#         """
#         import glob
#         for f in glob.glob(self.log_path + '/*'):
#             os.remove(f)

#     def watchdog(self, symbol, pid):
#         """Watches.

#         Args:
#             symbol (string): Crypto symbol to watch i.e "BTC"
#             pid (int): process id
#         """
#         stop = False
#         log = self.log_path + '/wd' + str(pid) + '.log'

#         def print(s, end='\n'):
#             open(log,
#                  'a').write(str(datetime.datetime.now()) + ' ' + str(s) + end)

#         def printtrade(trade):
#             open(self.trades_log, 'a').write(str(trade) + '\n')

#         def printprofit(profit):
#             open(self.profit_log, 'a').write(str(profit) + '\n')

#         def sell():
#             """Sell.

#             Args:
#                 symbol (string): Crypto symbol to watch i.e "BTC"
#             """
#             with open(self.positions_path, 'r+') as f:
#                 posjson = json.loads(f.read())
#                 if 'positions' in posjson:
#                     for pos in posjson['positions']:
#                         if pos['symbol'] == symbol:
#                             price = pos['price']
#                             amount = pos['amount']
#                             nprice = self.api.fetch_price(symbol)
#                             namount = (amount / price) * \
#                                 nprice * (1 - self.fee)
#                             profit = namount - amount
#                             if profit > 0:
#                                 print('Selling at '+str(nprice))
#                                 # if self.demo is False:
#                                 #     self.api.create_market_sell_order(
#                                 #         symbol, amount)
#                                 printtrade(Trade('SELL', symbol, nprice))
#                                 printprofit(profit)
#                                 posjson['positions'].remove(pos)
#                                 f.seek(0)
#                                 f.write(json.dumps(posjson))
#                                 f.truncate()

#         def buy(amount='all'):
#             """Buy.

#             Args:
#                 symbol (string): Crypto symbol to watch i.e "BTC"
#                 amount (float, optional): Amount of $ to buy with. Defaults to 'all'.
#             """
#             with open(self.positions_path, 'r+') as f:
#                 posjson = json.loads(f.read())
#                 if 'positions' in posjson:
#                     if symbol in [x['symbol'] for x in posjson['positions']]:
#                         # print('Already bought')
#                         return
#                 else:
#                     posjson['positions'] = []
#                 if amount == 'all':
#                     amount = self.api.fetch_free_balance()
#                 if amount > 0:
#                     price = self.api.fetch_price(symbol)
#                     print('Buying at '+str(price))
#                     # if not self.demo:
#                     # self.api.create_market_buy_order(symbol, amount)
#                     printtrade(Trade('BUY', symbol, price))
#                     posjson['positions'].append({
#                         'amount': amount * (1 - self.fee),
#                         'symbol': symbol,
#                         'Time': time.time(),
#                         'price': price
#                     })
#                     f.seek(0)
#                     f.write(json.dumps(posjson))
#                     f.truncate()

#         print('Watching ' + symbol)
#         c_buf = self.api.fetch_candles(symbol + '-USDT')['data'][:11]
#         while not stop:
#             live = self.api.fetch_ticker(symbol)
#             Candlestick(live['price'], )
#             c_buf.append([int(live['Time']/1000), float(live['price'])])
#             x, _, iss, _, _, = self.calc(c_buf)
#             if iss[-1].x > x[-1] - 60 * 3 and len(iss) > 2:
#                 # print(iss[-1].type)
#                 if iss[-1].type == 'ro':
#                     sell()
#                 else:
#                     buy()
#             time.sleep(self.pause)

#     def deploy_dogs(self, symbols=[]):
#         if not symbols:
#             symbols = self.symbols
#         for sym in symbols:
#             self.proc.append(
#                 Thread(target=self.watchdog, args=(sym, len(self.proc))))
#             self.proc[-1].start()
#         print('Spawning ' + str(len(self.proc)) + ' dogs')
#         print('View logs: log/wd<pid>.log')
#         for p in self.proc:
#             p.join()

#     def calc(self, css):
#         x = np.array([cs.time for cs in css])
#         y = np.array([cs.open for cs in css])
#         mas = np.array(Util.ema(y, self.c_short)['v'])
#         mal = np.array(Util.ema(y, self.c_long)['v'])
#         idx = np.argwhere(np.diff(np.sign(mal - mas)))
#         iss = []
#         for ii in idx:
#             isn = Intersection('', x[ii], y[ii])
#             if np.sign(mal-mas)[ii] > 0:
#                 isn.type = 'go'
#             else:
#                 isn.type = 'ro'
#             iss.append(isn)
#         return x, y, iss, mas, mal

#     def animate(self, i, symbol, c_buf, g, s, l, gop, rop):
#         live = self.api.fetch_ticker(symbol)
#         c_buf.append([int(live['Time']/1000), float(live['price'])])
#         x, y, iss, mas, mal = self.calc(c_buf)
#         g.set_data(x, y)
#         s.set_data(x, mas)
#         l.set_data(x, mal)
#         go = [x for x in iss if x.type == 'go']
#         ro = [x for x in iss if x.type == 'ro']
#         gop.set_data([i.x for i in go], [i.y for i in go])
#         rop.set_data([i.x for i in ro], [i.y for i in ro])

#     def plot(self, symbol):
#         css = [Candlestick(i[1], i[2], i[3], i[4], i[0])
#                for i in self.api.fetch_candles(symbol + '-USDT')['data']]
#         css.reverse()
#         x, y, iss, mas, mal = self.calc(css)
#         figure, ax = plt.subplots()
#         g, = ax.plot(x, y, color='black', label='price')
#         s, = ax.plot(x, mas, color='blue', label='short')
#         l, = ax.plot(x, mal, color='gray', label='long')
#         go = [x for x in iss if x.type == 'go']
#         ro = [x for x in iss if x.type == 'ro']
#         gop, = ax.plot([i.x for i in go], [i.y for i in go], 'go')
#         rop, = ax.plot([i.x for i in ro], [i.y for i in ro], 'ro')
#         ax.set_ylim(y[0] * 0.98, y[0] * 1.02)
#         ax.set_xlim(x[0], x[0] + 100000)
#         ax.grid()
#         plt.legend()
#         plt.title(symbol+'-USDT')
#         ani = FuncAnimation(plt.gcf(), self.animate, fargs=(
#             symbol, css, g, s, l, gop, rop), interval=5 * 1000)
#         plt.tight_layout()
#         plt.show()

#     def hairy_plotter(self, symbols=[]):
#         self.plot('BTC')

#     def plot_bondaries(self, symbol):
#         try:
#             data_all = self.api.fetch_candles(symbol + '-USDT',
#                                               type='1min')['data']
#         except:
#             raise Exception
#         times_all = [int(x[0]) for x in data_all]
#         prices = [max(float(x[1]), float(x[2])) for x in data_all]
#         plt.title(symbol)
#         plt.plot(times_all, prices)
#         g = Util.greatest(n=3, max_buf=prices)
#         if not g:
#             g = Util.greatest(n=2, max_buf=prices)
#         times = [int(data_all[x][0]) for x in [prices.index(y) for y in g]]
#         print(g)
#         print(times)
#         upper_m = (g[0] - g[-1]) / (times[0] - times[-1])
#         plt.plot(
#             [times[0], times[-1], times_all[0]],
#             [g[0], g[-1], upper_m * times_all[0] + g[0] - upper_m * times[0]])
#         prices = [min(float(x[1]), float(x[2])) for x in data_all]
#         g = Util.lowest(n=3, max_buf=prices)
#         if not g:
#             g = Util.lowest(n=2, max_buf=prices)
#         lower_m = (g[0] - g[-1]) / (times[0] - times[-1])
#         plt.plot(
#             [times[0], times[-1], times_all[0]],
#             [g[0], g[-1], lower_m * times_all[0] + g[0] - lower_m * times[0]])
#         if upper_m > 0:
#             plt.show()
