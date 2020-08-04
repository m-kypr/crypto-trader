import sys
from trader.master_trader import MasterTrader

if __name__ == "__main__":
    trader = MasterTrader()
    # trader = BinanceTrader()
    if len(sys.argv) > 1:
        if sys.argv[1] == 'plot':
            trader.plot()
