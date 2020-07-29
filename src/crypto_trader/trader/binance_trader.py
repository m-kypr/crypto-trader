import os
from .trader import Trader
from .binance_api import BinanceAPI


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