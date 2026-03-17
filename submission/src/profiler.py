import math

def estimate_energy(spikes: int, bit_precision: int = 8):
    e_bit = 0.5e-12
    return spikes * bit_precision * e_bit

def hardware_summary(spikes: int):
    targets = [
        ("Intel Loihi", 9, 128, 1024),
        ("BrainChip Akida", 4, 384, 4096),
        ("SpiNNaker", 16, 864, 1024),
    ]
    lines = []
    for name, bits, cores, neurons_per_core in targets:
        energy = estimate_energy(spikes, bits)
        lines.append(f"{name}: bits={bits}, cores={cores}, neurons/core={neurons_per_core}, est_energy_nJ={energy*1e9:.2f}")
    return "\n".join(lines)
