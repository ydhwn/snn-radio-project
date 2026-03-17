import numpy as np
from src.encoding import iq_to_vector, iq_to_symbol_vector

def test_shapes():
    x = np.exp(1j * 2 * np.pi * np.arange(1024) / 16)
    v = iq_to_vector(x, n=256)
    s = iq_to_symbol_vector(x, sps=16, n_symbols=256)
    assert v.shape[0] == 512
    assert s.shape[0] == 512
