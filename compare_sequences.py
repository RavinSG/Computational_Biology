import numpy as np
from collections import OrderedDict


def min_num_coins(money, coins):
    max_coin = max(coins)
    cache = OrderedDict({0: 0})

    for m in range(1, money + 1):
        try:
            cache[m] = min([cache[m - c] for c in coins if c <= m]) + 1
            while len(cache) > max_coin:
                cache.popitem(False)

        except ValueError:
            cache[m] = np.inf

    return cache
