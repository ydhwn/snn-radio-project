import numpy as np
import matplotlib.pyplot as plt

class SignalGenerator:
    def __init__(self, sample_rate=1e6, samples_per_symbol=16):
        self.fs = sample_rate
        self.sps = samples_per_symbol
        self.beta = 0.25
        self.span = 8

    def _rrc(self, beta: float, span: int, sps: int):
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

    def _shape(self, symbols: np.ndarray, beta: float = None, span: int = None):
        beta = self.beta if beta is None else beta
        span = self.span if span is None else span
        h = self._rrc(beta, span, self.sps)
        up = np.zeros(symbols.size * self.sps, dtype=np.complex64)
        up[::self.sps] = symbols.astype(np.complex64)
        shaped = np.convolve(up, h.astype(np.float32), mode="same")
        return shaped

    def generate_bpsk(self, num_symbols, snr_db=10):
        bits = np.random.randint(0, 2, num_symbols)
        symbols = 2 * bits - 1
        base = self._add_awgn(symbols, snr_db)
        shaped = self._shape(base)
        return shaped, bits

    def generate_qpsk(self, num_symbols, snr_db=10):
        bits = np.random.randint(0, 4, num_symbols)
        constellation = np.array([1+1j, 1-1j, -1-1j, -1+1j]) / np.sqrt(2)
        symbols = constellation[bits]
        base = self._add_awgn(symbols, snr_db)
        shaped = self._shape(base)
        return shaped, bits

    def generate_8psk(self, num_symbols, snr_db=10):
        idx = np.random.randint(0, 8, num_symbols)
        angles = 2 * np.pi * idx / 8.0
        symbols = np.exp(1j * angles)
        base = self._add_awgn(symbols, snr_db)
        shaped = self._shape(base)
        return shaped, idx

    def generate_16qam(self, num_symbols, snr_db=10):
        levels = np.array([-3, -1, 1, 3], dtype=np.float32)
        i = np.random.choice(levels, size=num_symbols)
        q = np.random.choice(levels, size=num_symbols)
        symbols = (i + 1j * q) / np.sqrt(10.0)
        base = self._add_awgn(symbols, snr_db)
        shaped = self._shape(base)
        return shaped, None

    def generate_16psk(self, num_symbols, snr_db=10):
        idx = np.random.randint(0, 16, num_symbols)
        angles = 2 * np.pi * idx / 16.0
        symbols = np.exp(1j * angles)
        base = self._add_awgn(symbols, snr_db)
        shaped = self._shape(base)
        return shaped, idx

    def generate_64qam(self, num_symbols, snr_db=10):
        levels = np.array([-7,-5,-3,-1,1,3,5,7], dtype=np.float32)
        i = np.random.choice(levels, size=num_symbols)
        q = np.random.choice(levels, size=num_symbols)
        symbols = (i + 1j * q) / np.sqrt(42.0)
        base = self._add_awgn(symbols, snr_db)
        shaped = self._shape(base)
        return shaped, None

    def _add_awgn(self, signal, snr_db):
        """Adds Additive White Gaussian Noise to the signal."""
        sig_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise = (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)) * np.sqrt(noise_power / 2)
        return signal + noise

    def plot_constellation(self, signal, title="Constellation Diagram"):
        plt.figure(figsize=(6, 6))
        plt.scatter(np.real(signal), np.imag(signal), alpha=0.5)
        plt.title(title)
        plt.xlabel("In-Phase (I)")
        plt.ylabel("Quadrature (Q)")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    gen = SignalGenerator()
    sig, _ = gen.generate_qpsk(1000, snr_db=15)
    gen.plot_constellation(sig, "QPSK (SNR=15dB)")
