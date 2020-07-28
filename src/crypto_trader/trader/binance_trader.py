import os
from .trader import Trader
from .util import Candlestick
from .binance_api import BinanceAPI


class BinanceTrader(Trader):
    def __init__(self, api=BinanceAPI({}), config_path='config.json', positions_path='positions.json', log_path='log', demo=True):
        super().__init__(api, config_path=config_path,
                         positions_path=positions_path, log_path=log_path, demo=demo)

    def candlestick_from_json(self, json):
        return Candlestick.fromJson(json[0]/1000, json[1], json[2], json[3], json[4])

    def plot(self):
        self.plotS('BTCUSDT', '1d')

    def trade(self):
        self.tradeS('BTCUSDT', '1m')
