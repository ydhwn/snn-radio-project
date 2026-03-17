from src.signal_generator import SignalGenerator

def test_generators():
    gen = SignalGenerator()
    b,_ = gen.generate_bpsk(256, snr_db=10)
    q,_ = gen.generate_qpsk(256, snr_db=10)
    p,_ = gen.generate_8psk(256, snr_db=10)
    qa,_ = gen.generate_16qam(256, snr_db=10)
    assert b.size > 0 and q.size > 0 and p.size > 0 and qa.size > 0
