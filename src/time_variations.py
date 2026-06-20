import numpy as np

class ring_mod:
    """First-order DC blocker with pole radius R."""

    def __init__(self, mod_freq: float, mod_amp: float, fs: int):
        self.mod_freq = mod_freq
        self.mod_amp = mod_amp
        self.fs = fs
        self.sample_count = 0

    def __call__(self, x):
        mod_signal = self.mod_amp * np.sin(2 * np.pi * self.mod_freq * self.sample_count / self.fs)
        self.sample_count += 1
        return x * mod_signal



def get_tv_function(tv_type: str, N: int, positions: list = None, extra_arg = None):
    tv_matrix = [lambda x: x for _ in range(N)]

    for pos in positions or range(N):
        if tv_type == "ring_mod":
            tv_matrix[pos] = ring_mod(extra_arg['mod_freq'], extra_arg['mod_amp'], extra_arg['fs'])

    return tv_matrix
