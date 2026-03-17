import numpy as np

def apply_cfo(x: np.ndarray, freq_offset: float, sps: int):
    n = np.arange(len(x))
    ph = 2 * np.pi * freq_offset * n / sps
    return x * np.exp(1j * ph)

def apply_rayleigh(x: np.ndarray, severity: float = 1.0):
    h = severity * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    return x * h

def impair(x: np.ndarray, sps: int = 16, cfo: float = 0.0, rayleigh: bool = False, rayleigh_severity: float = 1.0):
    y = x
    if rayleigh:
        y = apply_rayleigh(y, severity=rayleigh_severity)
    if cfo != 0.0:
        y = apply_cfo(y, cfo, sps)
    return y
