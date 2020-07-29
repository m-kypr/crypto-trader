import sys
from trader.binance_trader import BinanceTrader

if __name__ == "__main__":
    trader = BinanceTrader({})
    if len(sys.argv) > 1:
        if sys.argv[1] is 'plot':
            trader.plot()
        elif sys.argv[1] is 'trade':
            trader.trade()
        # if sys.argv[1] == 'clearlogs':
        #     self.clearlogs()
