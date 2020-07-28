import json
import uuid
from ccxt.kucoin import kucoin
from ccxt.base.errors import BadRequest
from .api import API

STABLE_COINS = ['USDC', 'TUSD', 'USDT', 'TRX', 'DAI', 'KCS', 'PAX']
NOT_PROVIDED = [
    'BTM', 'ADB', 'NRG', 'KEY', 'NEBL', 'KICK', 'FKX', 'NULS', 'DRGN', 'FOTA',
    'POWR', 'ELA', 'PRE', 'REV', 'XYO', 'CHSB', 'MANA', 'LSK', 'CAG', 'CBC',
    'ACAT', 'NIM', 'SPND', 'AOA', 'OPEN', 'AION', 'AMB', 'COV', 'CRPT', 'FTM',
    'CAPP', 'OCEAN', 'CPC', 'DGB', 'TIME', 'SOUL', 'CVC', 'XMR', 'MTC', 'DOCK',
    'IOTX', 'DCR', 'TFD', 'BPT', 'EDN', 'REQ', 'FET', 'ELF', 'DATX', 'MWAT',
    'GVT', 'CXO', 'MAN', 'LOKI', 'RIF', 'GAS', 'ROOBEE', 'MKR', 'QKC', 'CV',
    'FX', 'GMB', 'DACC', 'SOLVE', 'NIX', 'RBTC', 'PLAY', 'MVP', 'TRAC', 'CRO',
    'HPB', 'SNTR', 'TRTL', 'WTC', 'COFI', 'UTK', 'KNC', 'OMG', 'BOS', 'SNT',
    'CHP', 'OPQ', 'DBC', 'DX', 'VRAB', 'EXY', 'CHZ', 'ENJ', 'ZIL', 'NPXS',
    'WAN', 'BAX', 'LOC', 'EGT', 'BSV', 'VIDT', 'AGI', 'OLT', 'HC', 'CS',
    'IOST', 'RHOC', 'ACT', 'AERGO', 'ANKR', 'QTUM', 'AXPR', 'MTN', 'ZRX',
    'TFL', 'BU', 'ZEL', 'PPT', 'PIVX', 'DENT', 'META', 'BCD', 'WAXP', 'CSP',
    'LOOM'
]
SYMBOLS = [
    'BCHABC', 'ETH', 'EOS', 'XRP', 'VET', 'SUSD', 'WOM', 'GRIN', 'SXP', 'ETC',
    'ALGO', 'DAG', 'SENSO', 'BLOC', 'ITC', 'LUNA', 'BOLT', 'VSYS', 'MHC',
    'XLM', 'AVA', 'ARX', 'NEO', 'BTT', 'TOKO', 'AMIO', 'MXW', 'OGO',
    'FORESTPLUS', 'LTC', 'MTV', 'ODE', 'XNS', 'XTZ', 'ADA', 'MAP', 'BEPRO',
    'KSM', 'TKY', 'VI', 'TRY', 'COTI', 'CIX100', 'TOMO', 'ETN', 'EOSC', 'WXT',
    'ZEC', 'XDB', 'SUTER', 'ENQ', 'ATOM', 'JAR', 'ROAD', 'ACOIN', 'SYLO',
    'NWC', 'ONE', 'ONT', 'DASH', 'GGC', 'VOL', 'AKRO', 'PMGT', 'VID', 'CHR',
    'DAPPT', 'NOIA', 'LYM', 'DERO', 'BNB', 'WIN', 'SDT', 'NANO', 'TEL', 'POL',
    'KAT', 'AMPL', 'GO', 'ARPA', 'SNX', 'VRA', 'XEM'
]


class KucoinAPI(kucoin):
    """Improvements and extensions to the cctx library
    """

    def __init__(self, credentials_path):
        CREDENTIALS = json.loads(open(credentials_path, 'r').read())
        if not CREDENTIALS['apiKey']:
            CREDENTIALS['apiKey'] = input('apiKey: ')
        elif not CREDENTIALS['secret']:
            CREDENTIALS['secret'] = input('secret: ')
        elif not CREDENTIALS['password']:
            CREDENTIALS['password'] = input('password: ')
        else:
            print('Using credentials.json...')
        open('../../credentials.json', 'w').write(json.dumps(CREDENTIALS))
        super().__init__(config=CREDENTIALS)

    def create_market_sell_order(self, symbol, amount, params={}):
        return super().fetch2('orders',
                              'private',
                              'POST',
                              params={
                                  'clientOid': str(uuid.uuid4()),
                                  'side': 'sell',
                                  'symbol': symbol,
                                  'funds': amount
                              })
        # return super().create_market_sell_order(symbol, amount, params=params)

    def fetch_tradeable_symbols(self, update=False):
        global SYMBOLS
        if update:
            symbols = set()
            for s in self.fetch_symbols()['data']:
                if s['baseCurrency'] not in STABLE_COINS + NOT_PROVIDED:
                    symbols.add(s['baseCurrency'])
            SYMBOLS = list(symbols)
        return SYMBOLS

    def fetch_symbols(self):
        return super().fetch2('symbols')

    def fetch_accounts(self):
        return super().fetch2(path='accounts', api='private')['data']

    def fetch_free_balance(self, params={}):
        return sum([
            float(x['available']) for x in self.fetch_accounts()
            if x['type'] == 'trade' and x['currency'] in STABLE_COINS
        ])

    def fetch_klines(self, symbol, startAt=0, endAt=0, type='1min'):
        """Fetch candles.

        Args:
            symbol (string): Crypto symbol
            startAt (int, optional): Start of cycle. Defaults to 0.
            endAt (int, optional): End of cycle. Defaults to 0.
            type (string, optional): Type of candle. Defaults to '1min'.

        Raises:
            BadRequest: The KuCoin API probably does not support the currency (SYMBOL NOT PROVIDED).
            Solutions currently not available maybe API switch in the future

        Returns:
            "1545904980",             Start time of the candle cycle
            "0.058",                  opening price
            "0.049",                  closing price
            "0.058",                  highest price
            "0.049",                  lowest price
            "0.018",                  Transaction amount
            "0.000945"                Transaction volume
        """
        return super().fetch2('market/candles',
                              params={
                                  'symbol': symbol,
                                  'startAt': startAt,
                                  'endAt': endAt,
                                  'type': type
                              })

    def fetch_histories(self, symbol):
        return super().fetch2('market/histories', params={'symbol': symbol})

    def fetch_ticker(self, symbol):
        return super().fetch2('market/orderbook/level1',
                              params={'symbol': symbol + '-USDT'})['data']

    def fetch_price(self, symbol):
        return float(self.fetch_ticker(symbol)['price'])
