import numpy as np
from typing import Tuple, Sequence
from .signal_generator import SignalGenerator
from .encoding import iq_to_vector, iq_to_symbol_vector
from .channels import impair

def _augment(sig: np.ndarray):
    theta = float(np.random.uniform(-np.pi/18, np.pi/18))
    gain = float(np.random.uniform(0.9, 1.1))
    return gain * sig * np.exp(1j * theta)

def _gen_mod(gen: SignalGenerator, mod: str, n: int, snr_db: float):
    if mod == "BPSK":
        sig, _ = gen.generate_bpsk(n, snr_db=snr_db)
        return sig
    if mod == "QPSK":
        sig, _ = gen.generate_qpsk(n, snr_db=snr_db)
        return sig
    if mod == "8PSK":
        sig, _ = gen.generate_8psk(n, snr_db=snr_db)
        return sig
    if mod == "16QAM":
        sig, _ = gen.generate_16qam(n, snr_db=snr_db)
        return sig
    if mod == "16PSK":
        sig, _ = gen.generate_16psk(n, snr_db=snr_db)
        return sig
    if mod == "64QAM":
        sig, _ = gen.generate_64qam(n, snr_db=snr_db)
        return sig
    raise ValueError("unknown modulation")

def make_dataset(samples_per_class: int = 512, symbols_per_sample: int = 2048, snr_range: Tuple[float, float] = (0.0, 20.0), vec_len: int = 256, mods: Sequence[str] = ("BPSK","QPSK","8PSK","16QAM","16PSK","64QAM"), use_matched: bool = True, sps: int = 16, augment: bool = True, impair_prob: float = 0.0, cfo: float = 0.0, rayleigh: bool = False, rayleigh_severity: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    gen = SignalGenerator()
    X_list = []
    y_list = []
    low, high = snr_range
    for label, mod in enumerate(mods):
        for _ in range(samples_per_class):
            snr_db = float(np.random.uniform(low, high))
            sig = _gen_mod(gen, mod, symbols_per_sample, snr_db)
            if augment:
                sig = _augment(sig)
            if impair_prob > 0.0 and float(np.random.rand()) < float(impair_prob):
                sig = impair(sig, sps=sps, cfo=float(cfo), rayleigh=bool(rayleigh), rayleigh_severity=float(rayleigh_severity))
            if use_matched:
                feat = iq_to_symbol_vector(sig, sps=sps, n_symbols=vec_len)
            else:
                feat = iq_to_vector(sig, n=vec_len)
            X_list.append(feat)
            y_list.append(label)
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def make_dataset_fixed_snr(samples_per_class: int = 256, symbols_per_sample: int = 2048, snr_db: float = 10.0, vec_len: int = 256, mods: Sequence[str] = ("BPSK","QPSK","8PSK","16QAM","16PSK","64QAM"), use_matched: bool = True, sps: int = 16, augment: bool = False, impair_prob: float = 0.0, cfo: float = 0.0, rayleigh: bool = False, rayleigh_severity: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    gen = SignalGenerator()
    X_list = []
    y_list = []
    for label, mod in enumerate(mods):
        for _ in range(samples_per_class):
            sig = _gen_mod(gen, mod, symbols_per_sample, snr_db)
            if augment:
                sig = _augment(sig)
            if impair_prob > 0.0 and float(np.random.rand()) < float(impair_prob):
                sig = impair(sig, sps=sps, cfo=float(cfo), rayleigh=bool(rayleigh), rayleigh_severity=float(rayleigh_severity))
            if use_matched:
                feat = iq_to_symbol_vector(sig, sps=sps, n_symbols=vec_len)
            else:
                feat = iq_to_vector(sig, n=vec_len)
            X_list.append(feat)
            y_list.append(label)
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y
