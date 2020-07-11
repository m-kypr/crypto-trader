import numpy as np
from pandas import DataFrame


class Util():
    """Maths and stuff.
    """
    @classmethod
    def running_mean(cls, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    @classmethod
    def ema(cls, values, window):
        df = DataFrame({'v': values})
        return df.ewm(span=window).mean()

    # @classmethod
    # def lowest(cls, n, max_buf):
    #     if n > 0:
    #         b = []
    #         for i in argrelextrema(np.array(max_buf), np.less)[0].tolist():
    #             b.append(max_buf[i])
    #         return Util.lowest(n - 1, max_buf=b)
    #     else:
    #         return max_buf

    # @classmethod
    # def greatest(cls, n, max_buf):
    #     if n > 0:
    #         b = []
    #         for i in argrelextrema(np.array(max_buf), np.greater)[0].tolist():
    #             b.append(max_buf[i])
    #         return Util.greatest(n - 1, max_buf=b)
    #     else:
    #         return max_buf
