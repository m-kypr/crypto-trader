import os
from .util import Candlestick
from .kucoin_api import KucoinAPI


class KucoinTrader(Trader):
    def __init__(self, api=KucoinAPI({}), config_path=os.path.join(PATH, 'config.json'), positions_path=os.path.join(PATH, 'positions.json'), log_path=os.path.join(PATH, 'log'), demo=True):
        super().__init__(api, config_path=config_path,
                         positions_path=positions_path, log_path=log_path, demo=demo)
