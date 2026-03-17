import numpy as np

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
    import torch
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

def estimate_sps(x: np.ndarray, max_sps: int = 32):
    """
    Blindly estimates samples per symbol (SPS) using the FFT of the 
    magnitude squared (Delay-and-Multiply method).
    """
    if x is None or x.size < 128:
        return 16 # Default fallback
    
    # 1. Non-linear operation to generate spectral line at symbol rate
    # For PSK/QAM, |x|^2 contains a strong component at 1/T
    mag_sq = np.abs(x)**2
    mag_sq = mag_sq - np.mean(mag_sq)
    
    # 2. Perform FFT to find the spectral line
    N = len(mag_sq)
    # Use next power of 2 for faster/better FFT
    N_fft = 1 << (N - 1).bit_length()
    f_axis = np.fft.fftfreq(N_fft)
    X_f = np.abs(np.fft.fft(mag_sq, n=N_fft))
    
    # 3. Only look at the positive frequency range corresponding to SPS [4, 32]
    # Frequency f = 1/SPS, so search range is [1/32, 1/4]
    min_f = 1.0 / 32.0
    max_f = 1.0 / 4.0
    
    mask = (f_axis >= min_f) & (f_axis <= max_f)
    if not np.any(mask):
        return 16
        
    search_f = f_axis[mask]
    search_vals = X_f[mask]
    
    # 4. Find the peak frequency
    peak_idx = np.argmax(search_vals)
    best_f = search_f[peak_idx]
    
    # 5. Convert frequency back to SPS
    est_sps = 1.0 / best_f
    
    # Round to nearest integer and bound
    return int(np.round(np.clip(est_sps, 4, 32)))

def blind_sync(x: np.ndarray, sps: int):
    """
    Finds the optimal sampling offset by maximizing the variance 
    of the sampled points (Eye-opening maximization).
    """
    best_offset = 0
    max_var = -1.0
    
    # Try all possible offsets within one symbol period
    for offset in range(sps):
        samples = x[offset::sps]
        if samples.size == 0: continue
        
        # For M-PSK/QAM, the variance of the magnitude is minimized 
        # (or variance of complex points is maximized) at the ideal sample point.
        var = np.var(np.abs(samples)) 
        # Note: In a clean constellation, mag variance is low at symbol centers.
        # However, for 'blind' we often look for the point where the signal 
        # looks most like a constellation.
        
        # Let's use a simpler heuristic: Maximize the mean squared magnitude
        # which often peaks at the symbol center for pulse-shaped signals.
        pwr = np.mean(np.abs(samples)**2)
        
        if pwr > max_var:
            max_var = pwr
            best_offset = offset
            
    return best_offset

def iq_to_symbol_vector(x: np.ndarray, sps: int = 16, n_symbols: int = 256, beta: float = 0.25, span: int = 8, blind: bool = False):
    if x is None or x.size == 0:
        return np.zeros(n_symbols * 2, dtype=np.float32)
    
    actual_sps = sps
    offset = (span * sps // 2) % sps # Default matched filter delay
    
    if blind:
        # 1. Blindly estimate SPS
        actual_sps = estimate_sps(x)
        # 2. Perform Matched Filtering with estimated SPS
        h = _rrc(beta, span, actual_sps)
        y = np.convolve(x.astype(np.complex64), h.astype(np.float32), mode="same")
        # 3. Blindly find best sampling offset
        offset = blind_sync(y, actual_sps)
    else:
        h = _rrc(beta, span, sps)
        y = np.convolve(x.astype(np.complex64), h.astype(np.float32), mode="same")
    
    samples = y[offset::actual_sps]
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
