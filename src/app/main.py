from multiprocessing import Process
from threading import Thread
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import time
import json
import os
import datetime
import sys
import shutil
from util import Util
from kucoin_api import KAPI, SYMBOLS
from abc import ABC


APP_PATH = os.path.dirname(os.path.abspath(__file__))


class JsonSerializable(ABC):
    def __repr__(self):
        return json.dumps(self.__dict__)


class T(JsonSerializable, ABC):
    def __init__(self, symbol, price, time):
        self.time = time
        self.symbol = symbol
        self.price = price


class Intersection(JsonSerializable):
    def __init__(self, type, x, y):
        self.type = type
        self.x = float(x)
        self.y = float(y)


class Trade(T):
    def __init__(self, type, symbol, price, time=time.time()):
        self.type = type
        super().__init__(symbol, price, time)


class Position(T):
    def __init__(self, amount, symbol, price, time):
        self.amount = amount
        super().__init__(symbol, price, time)


class KTrader():
    """Kucoin API algorithmic Trader 
    DOCS: https://docs.kucoin.com/#general
    Using the cctx library as abstraction layer: https://github.com/ccxt/ccxt
    """

    def __init__(self, demo):
        self.config_path = APP_PATH + '/../../config.json'
        self.positions_path = APP_PATH + '/../../positions.json'
        self.log_path = APP_PATH + '/../log'
        self.trades_log = self.log_path + '/trades.log'
        self.profit_log = self.log_path + '/profit.log'

        self.load_config()

        self.demo = demo
        self.proc = []
        self.api = KAPI(APP_PATH + '/../../credentials.json')

        self.handle_argv()

        self.deploy_dogs()

    def load_config(self):
        config = json.loads(open(self.config_path, 'r').read())
        self.symbols = config['symbols']
        self.c_short = config['c_short']
        if config['c_long'] == 0:
            self.c_long = self.c_short * 2
        else:
            self.c_long = config['c_long']
        self.fee = config['fee']
        self.pause = config['pause']

    def handle_argv(self):
        """Handle user input.
        plot -- use matplotlib to plot graph
        clearlogs -- clear the contents of log/
        """
        if len(sys.argv) > 1:
            if sys.argv[1] == 'plot':
                self.hairy_plotter()
            if sys.argv[1] == 'clearlogs':
                self.clearlogs()
            quit()

    def clearlogs(self):
        """Clear the contents of log/
        """
        import glob
        for f in glob.glob(self.log_path + '/*'):
            os.remove(f)

    def watchdog(self, symbol, pid):
        """Watches.

        Args:
            symbol (string): Crypto symbol to watch i.e "BTC"
            pid (int): process id
        """
        stop = False
        log = self.log_path + '/wd' + str(pid) + '.log'

        def print(s, end='\n'):
            open(log,
                 'a').write(str(datetime.datetime.now()) + ' ' + str(s) + end)

        def printtrade(trade):
            open(self.trades_log, 'a').write(str(trade))

        def sell():
            """Sell.

            Args:
                symbol (string): Crypto symbol to watch i.e "BTC"
            """
            with open(self.positions_path, 'r+') as f:
                posjson = json.loads(f.read())
                if 'positions' in posjson:
                    for pos in posjson['positions']:
                        if pos['symbol'] == symbol:
                            amount = pos['amount']
                            nprice = self.api.fetch_price(symbol)
                            price = pos['price']
                            namount = (amount * (1 - self.fee) *
                                       (1 - self.fee) * nprice) / price
                            profit = namount - amount
                            if profit > 0:
                                open(self.profit_log, 'a').write(
                                    '\n'+str(profit))
                                print('Selling at '+str(nprice))
                                if self.demo is False:
                                    self.api.create_market_sell_order(
                                        symbol, amount)
                                open(self.trades_log, 'a').write('\n' +
                                                                 str(Trade('SELL', symbol, nprice)))
                                posjson['positions'].remove(pos)
                                f.seek(0)
                                f.write(json.dumps(posjson))
                                f.truncate()

        def buy(amount='all'):
            """Buy.

            Args:
                symbol (string): Crypto symbol to watch i.e "BTC"
                amount (float, optional): Amount of $ to buy with. Defaults to 'all'.
            """
            with open(self.positions_path, 'r+') as f:
                posjson = json.loads(f.read())
                if 'positions' in posjson:
                    if symbol not in [x['symbol'] for x in posjson['positions']]:
                        pass
                    else:
                        return
                else:
                    posjson['positions'] = []
                if amount == 'all':
                    amount = self.api.fetch_free_balance()
                if amount > 0:
                    price = self.api.fetch_price(symbol)
                    print('Buying at '+str(price))
                    # if not self.demo:
                    # self.api.create_market_buy_order(symbol, amount)
                    printtrade(Trade('BUY', symbol, price))
                    posjson['positions'].append({
                        'symbol': symbol,
                        'amount': amount * (1 - self.fee),
                        't': time.time(),
                        'price': price
                    })
                    f.seek(0)
                    f.write(json.dumps(posjson))
                    f.truncate()

        print('Watching ' + symbol)
        c_buf = self.api.fetch_candles(symbol + '-USDT')['data'][:3]
        while not stop:
            live = self.api.fetch_ticker(symbol)
            c_buf.append([int(live['time']/1000), float(live['price'])])
            x, _, iss, _, _, = self.calc(c_buf)
            if iss[-1].x > x[-1] - 60 * 3:
                if iss[-1].type == 'ro':
                    sell()
                else:
                    buy()
            time.sleep(self.pause)

    def deploy_dogs(self, symbols=[]):
        if not symbols:
            symbols = self.symbols
        for sym in symbols:
            self.proc.append(
                Thread(target=self.watchdog, args=(sym, len(self.proc))))
            self.proc[-1].start()
        print('Spawning ' + str(len(self.proc)) + ' dogs')
        print('View logs: log/wd<pid>.log')
        for p in self.proc:
            p.join()

    def calc(self, c_buf):
        x = np.array([float(x[0]) for x in c_buf])
        y = np.array([float(x[1]) for x in c_buf])
        mas = np.array(Util.ema(y, self.c_short)['v'])
        mal = np.array(Util.ema(y, self.c_long)['v'])
        idx = np.argwhere(np.diff(np.sign(mal - mas)))
        iss = []
        for ii in idx:
            isn = Intersection('', x[ii], y[ii])
            if np.sign(mal-mas)[ii] > 0:
                isn.type = 'go'
            else:
                isn.type = 'ro'
            iss.append(isn)
        return x, y, iss, mas, mal

    def animate(self, i, symbol, c_buf, g, s, l, gop, rop):
        live = self.api.fetch_ticker(symbol)
        c_buf.append([int(live['time']/1000), float(live['price'])])
        x, y, iss, mas, mal = self.calc(c_buf)
        g.set_data(x, y)
        s.set_data(x, mas)
        l.set_data(x, mal)
        go = [x for x in iss if x.type == 'go']
        ro = [x for x in iss if x.type == 'ro']
        gop.set_data([i.x for i in go], [i.y for i in go])
        rop.set_data([i.x for i in ro], [i.y for i in ro])

    def plot(self, symbol):
        c_buf = self.api.fetch_candles(symbol + '-USDT')['data'][:3]
        c_buf.reverse()
        x, y, iss, mas, mal = self.calc(c_buf)
        figure, ax = plt.subplots()
        g, = ax.plot(x, y, color='black', label='price')
        s, = ax.plot(x, mas, color='blue', label='short')
        l, = ax.plot(x, mal, color='gray', label='long')
        go = [x for x in iss if x.type == 'go']
        ro = [x for x in iss if x.type == 'ro']
        gop, = ax.plot([i.x for i in go], [i.y for i in go], 'go')
        rop, = ax.plot([i.x for i in ro], [i.y for i in ro], 'ro')
        ax.set_ylim(y[0] * 0.98, y[0] * 1.02)
        ax.set_xlim(x[0], x[0] + 100000)
        ax.grid()
        plt.legend()
        plt.title(symbol+'-USDT')
        ani = FuncAnimation(plt.gcf(), self.animate, fargs=(
            symbol, c_buf, g, s, l, gop, rop), interval=3000)
        plt.tight_layout()
        plt.show()

    def hairy_plotter(self, symbols=[]):
        self.plot('BTC')

    def plot_bondaries(self, symbol):
        try:
            data_all = self.api.fetch_candles(symbol + '-USDT',
                                              type='1min')['data']
        except:
            raise Exception
        times_all = [int(x[0]) for x in data_all]
        prices = [max(float(x[1]), float(x[2])) for x in data_all]
        plt.title(symbol)
        plt.plot(times_all, prices)
        g = Util.greatest(n=3, max_buf=prices)
        if not g:
            g = Util.greatest(n=2, max_buf=prices)
        times = [int(data_all[x][0]) for x in [prices.index(y) for y in g]]
        print(g)
        print(times)
        upper_m = (g[0] - g[-1]) / (times[0] - times[-1])
        plt.plot(
            [times[0], times[-1], times_all[0]],
            [g[0], g[-1], upper_m * times_all[0] + g[0] - upper_m * times[0]])
        prices = [min(float(x[1]), float(x[2])) for x in data_all]
        g = Util.lowest(n=3, max_buf=prices)
        if not g:
            g = Util.lowest(n=2, max_buf=prices)
        lower_m = (g[0] - g[-1]) / (times[0] - times[-1])
        plt.plot(
            [times[0], times[-1], times_all[0]],
            [g[0], g[-1], lower_m * times_all[0] + g[0] - lower_m * times[0]])
        if upper_m > 0:
            plt.show()


def main():
    KTrader(demo=True)

    # TODO:Implement other APIs?


if __name__ == "__main__":
    main()
