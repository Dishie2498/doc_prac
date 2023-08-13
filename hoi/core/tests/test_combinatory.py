import pytest
import numpy as np
from math import comb
import jaxlib
import jax
from hoi.core.combinatory import combinations, _combinations
import jax.numpy as jnp
from collections.abc import Iterable


@pytest.mark.parametrize("n", [np.random.randint(5, 10) for _ in range(10)])
@pytest.mark.parametrize("k", [np.random.randint(5, 10) for _ in range(10)])
@pytest.mark.parametrize("order", [True, False])
def test_single_combinations(n, k, order):
    c = list(_combinations(n, k, order))
    assert len(c) == comb(n, k)


# def isiterable(x):
#   try: iter(x)
#   except TypeError: return False
#   else: return True


@pytest.mark.parametrize("n", [np.random.randint(5, 1000) for _ in range(10)])
@pytest.mark.parametrize("min", [np.random.randint(1, 10) for _ in range(10)])
@pytest.mark.parametrize("max", [_ for _ in range(5)])  # addition to minimum size
@pytest.mark.parametrize("astype", ["numpy", "jax", "iterator"])
@pytest.mark.parametrize("order_val", [True, False])
def test_combinations(n, min, max, astype, order_val):
    combs = combinations(n, min, min + max, "iterator", astype, order_val)
    assert (combs, Iterable)
    # # combs = np.asarray(combs)
    # # x = np.fromiter(combs, object)
    # print(type(combs))
    # if (astype == "jax"):
    #     assert isinstance(combs, jaxlib.xla_extension.ArrayImpl)
    # else:
    #     assert isinstance(combs, np.ndarray)
    # assert isiterable(combs)

    # assert combs.ndim == 1
    # assert len(list(combs)) == 1
    # total_comb = 0
    # for i in range(max):
    #     total_comb += comb(n, min + i)

    # assert combs.shape[0] == total_comb
