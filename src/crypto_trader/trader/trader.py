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

    def __init__(self, api, credentials={}, config_path='config.json', positions_path='positions.json', log_path='log', demo=True):
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
        self.handle_argv()
        self.trade()

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

    def handle_argv(self):
        if len(sys.argv) > 1:
            if sys.argv[1] == 'plot':
                self.plot()
            # if sys.argv[1] == 'clearlogs':
            #     self.clearlogs()
            quit()

    def plot(self):
        pass

    def trade(self):
        pass

    def candlestick_from_json(self, json):
        pass

    def tradeS(self, symbol, interval):
        self.proc.append(Thread(target=self.watchdog,
                                args=(symbol, interval, len(self.proc))))
        self.proc[-1].start()
        print('Spawned ' + symbol + ' dog')
        print('View logs: log/wd'+str(len(self.proc)-1)+'.log')

    def calc(self, css, rsi='sma', stoch=False, period=14, stoch_smoothness=0):
        x = np.array([cs.time for cs in css])
        y = np.array([cs.open for cs in css])
        if rsi:
            if rsi is 'sma' or 'ema':
                x = x[1:]
                y = y[1:]
                if stoch:
                    StochRSI, StochRSIK, StochRSID = Math.StochRSI(
                        css, mode=rsi, period=period)
                    if stoch_smoothness == 1:
                        StochRSI = StochRSIK
                    elif stoch_smoothness == 2:
                        StochRSI = StochRSID
                    return x, y, StochRSI
                else:
                    RSI = Math.RSI(css, mode=rsi, period=period)
                    return x, y, RSI
        return x, y, _

    def animate(self, _, g, rsi_g, css, symbol, interval, rsi_mode, stoch, stoch_smoothness, *args):
        css.append([self.candlestick_from_json(c) for c in json.loads(
            self.api.fetch_klines(symbol, interval))][-1])
        if stoch:
            x, y, StochRSI = self.calc(
                css, rsi=rsi_mode, stoch=True, stoch_smoothness=stoch_smoothness)
            rsi_g.set_data(x, StochRSI)
        else:
            x, y, RSI = self.calc(css, rsi=rsi_mode)
            rsi_g.set_data(x, RSI)
        g.set_data(x, y)

    def plotS(self, symbol, interval, do_ha=True, rsi_mode='ema', stoch=True, stoch_smoothness=2):
        pause = int(self.api.interval_to_millis(interval))
        # fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        df = MyDataFrame(self.api.fetch_klines_as_dataframe(
            symbol, interval, 100))
        df.index = pd.to_datetime(df[self.api.headers[0]], unit='ms')
        # print(mpl.available_styles())
        df.calculate(do_ha=False)
        apds = [
            mpl.make_addplot(df['RSI_14'], panel=1)
        ]

        def watcher(self):
            while True:
                ndf = MyDataFrame(
                    self.api.fetch_klines_as_dataframe(symbol, interval, 1))
                if ndf[self.api.headers[0]].iloc[0] >= df[self.api.headers[0]].iloc[-1] + pause:
                    ndf.calculate(do_ha=do_ha)
                    print()
                    df.loc[len(df)] = ndf.iloc[0].values
                    df.index = pd.to_datetime(
                        df[self.api.headers[0]], unit='ms')
                    df.calculate(do_ha=False)
                    print(df)
                    if self.demo:
                        print('Updating')
                time.sleep(pause / 1000)
        t = Thread(target=watcher, args=(self, ))
        t.start()
        mpl.plot(df, type='candle', style='binance', mav=(
            3, 6, 9), volume=True, addplot=apds)
        t.join()
        quit()
        # if stoch:
        #     x, y, StochRSI = self.calc(
        #         css, rsi=rsi_mode, stoch=True, stoch_smoothness=stoch_smoothness)
        #     rsi_g, = ax2.plot(
        #         x, StochRSI[0], color='blue', label='StochRSI via ' + rsi_mode)
        #     # print(StochRSI[0])
        #     overbought = StochRSI[StochRSI[0] > 0.8]
        #     oversold = StochRSI[StochRSI[0] < 0.2]
        #     print(overbought.shape)
        #     print(oversold.shape)
        # else:
        #     x, y, RSI = self.calc(css, rsi=rsi_mode)
        #     rsi_g, = ax2.plot(x, RSI[0], color='blue',
        #                       label='RSI via ' + rsi_mode)
        # g, = ax1.plot(x, y, color='black', label='price')
        # # go = [x for x in iss if x.type == 'go']
        # # ro = [x for x in iss if x.type == 'ro']
        # # gop, = ax.plot([i.x for i in go], [i.y for i in go], 'go')
        # ax1.set_ylim(min(y) * 0.8, max(y) * 1.2)
        # ax1.set_xlim(x[0], x[-1] + pause / pi)
        # ax1.grid()
        # ax2.grid()
        # ax1.legend()
        # ax2.legend()
        # ax1.set_title(symbol)
        # ax2.set_title('Analytics')
        # ani = FuncAnimation(plt.gcf(), self.animate, fargs=(g, rsi_g, css, symbol, interval, rsi_mode,
        #                                                     stoch, stoch_smoothness, ), interval=pause)
        # plt.tight_layout()
        # plt.show()

    def watchdog(self, symbol, interval, pid):
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
                    'time': time.time(),
                    'price': price
                })
                open(self.positions_path, 'w').write(json.dumps(posjson))

        print('Currently watching: ' + symbol)
        css = [self.candlestick_from_json(c) for c in json.loads(
            self.api.fetch_klines(symbol, interval))]
        while not stop:
            cs = self.candlestick_from_json(json.loads(self.api.fetch_klines(
                symbol, interval, limit=1))[-1])
            if cs.time > css[-1].time:
                css.append(cs)
                self.calc(css)
                # x, y = self.calc(css)
                # if iss[-1].x > x[-1] - 60 * 3 and len(iss) > 2:
                #     if iss[-1].type == 'ro':
                #         sell()
                #     else:
                #         buy()
                buy(10)
            time.sleep(pause)


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
#                         'time': time.time(),
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
#             c_buf.append([int(live['time']/1000), float(live['price'])])
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
#         c_buf.append([int(live['time']/1000), float(live['price'])])
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
