import pytest
import numpy as np
from math import comb
import jaxlib

from hoi.core.combinatory import combinations

@pytest.mark.parametrize("n", [np.random.randint(10, 1000) for _ in range(10)])
@pytest.mark.parametrize("min", [np.random.randint(1, 10) for _ in range(10)])
@pytest.mark.parametrize("max", [_ for _ in range(5)])  # addition to minimum size
@pytest.mark.parametrize("astype", ["jax", "iterator", "numpy"])
@pytest.mark.parametrize("order_val", [True, False])
def test_combinations(n, min, max, astype, order_val):
    combs = combinations(n, min, min + max, astype, order_val)

    if (astype == "jax"):
        assert isinstance(combs, jaxlib.xla_extension.ArrayImpl)
    else:
        assert isinstance(combs, np.ndarray)
    assert combs.ndim == 1
    total_comb = 0
    for i in range(max):
        total_comb += comb(n, min + i)

    assert combs.shape[0] == total_comb
