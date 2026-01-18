import numpy as np

class Normalize:
    def __init__(self, eps=1e-9):
        self.eps = eps

    def peak_per_channel(self, channels):
        # x_norm = x / max(|x|)
        out = []
        for ch in channels:
            m = np.max(np.abs(ch))
            out.append(ch / (m + self.eps))
        return out
