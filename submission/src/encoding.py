import numpy as np
import torch

def iq_to_features(x: np.ndarray, bins: int = 128):
    if x is None or x.size == 0:
        return np.zeros(bins, dtype=np.float32)
    amp = np.abs(x)
    amp = amp / (np.max(amp) + 1e-8)
    ph = np.angle(x)
    dph = np.diff(ph, prepend=ph[0])
    dph = (dph + np.pi) % (2 * np.pi) - np.pi
    h_amp, _ = np.histogram(amp, bins=bins // 2, range=(0.0, 1.0), density=True)
    h_dph, _ = np.histogram(dph, bins=bins // 2, range=(-np.pi, np.pi), density=True)
    feat = np.concatenate([h_amp, h_dph]).astype(np.float32)
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat

def features_to_spikes(feat: np.ndarray, scale: float = 10.0):
    if feat is None or feat.size == 0:
        return torch.zeros(1, 1, dtype=torch.float32)
    rates = np.clip(feat * scale, 0.0, scale).astype(np.float32)
    return torch.from_numpy(rates).unsqueeze(0)

def iq_to_vector(x: np.ndarray, n: int = 256):
    if x is None or x.size == 0:
        return np.zeros(n * 2, dtype=np.float32)
    idx = np.linspace(0, len(x) - 1, num=n).astype(int)
    xi = x[idx]
    r = np.real(xi)
    i = np.imag(xi)
    v = np.concatenate([r, i]).astype(np.float32)
    v = (v - np.mean(v)) / (np.std(v) + 1e-8)
    return v

def _rrc(beta: float, span: int, sps: int):
    N = span * sps
    t = np.arange(-N//2, N//2 + 1, dtype=np.float64) / sps
    h = np.zeros_like(t)
    for k, tk in enumerate(t):
        if np.isclose(tk, 0.0):
            h[k] = 1.0 - beta + 4 * beta / np.pi
        elif np.isclose(abs(tk), 1 / (4 * beta)):
            h[k] = (beta / np.sqrt(2)) * (
                ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))) +
                ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
            )
        else:
            num = np.sin(np.pi * tk * (1 - beta)) + 4 * beta * tk * np.cos(np.pi * tk * (1 + beta))
            den = np.pi * tk * (1 - (4 * beta * tk) ** 2)
            h[k] = num / den
    h = h / np.sqrt(np.sum(h ** 2))
    return h.astype(np.float32)

def iq_to_symbol_vector(x: np.ndarray, sps: int = 16, n_symbols: int = 256, beta: float = 0.25, span: int = 8):
    if x is None or x.size == 0:
        return np.zeros(n_symbols * 2, dtype=np.float32)
    h = _rrc(beta, span, sps)
    y = np.convolve(x.astype(np.complex64), h.astype(np.float32), mode="same")
    start = (len(h) // 2) % sps
    samples = y[start::sps]
    if len(samples) < n_symbols:
        pad = np.zeros(n_symbols - len(samples), dtype=np.complex64)
        samples = np.concatenate([samples, pad])
    idx = np.linspace(0, len(samples) - 1, num=n_symbols).astype(int)
    xi = samples[idx]
    r = np.real(xi)
    i = np.imag(xi)
    v = np.concatenate([r, i]).astype(np.float32)
    v = (v - np.mean(v)) / (np.std(v) + 1e-8)
    return v
