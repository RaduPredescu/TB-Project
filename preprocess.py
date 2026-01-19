import numpy as np
from normalize import Normalize

normalizer = Normalize()



def remove_mean(channels):

    return [ch - np.mean(ch) for ch in channels]


def clip_outliers_mad(channels, k=6.0):

    clipped = []
    for ch in channels:
        med = np.median(ch)
        mad = np.median(np.abs(ch - med)) + 1e-9
        z = 0.6745 * (ch - med) / mad  # robust z-score

        upper = med + (k * mad / 0.6745)
        lower = med - (k * mad / 0.6745)

        ch2 = ch.copy()
        ch2[z > k] = upper
        ch2[z < -k] = lower
        clipped.append(ch2)

    return clipped


def zscore_normalize(channels):

    normed = []
    for ch in channels:
        mu = np.mean(ch)
        sigma = np.std(ch) + 1e-9
        normed.append((ch - mu) / sigma)
    return normed


def preprocess_channels(channels, clip_k=6.0):
    ch = remove_mean(channels)
    ch = clip_outliers_mad(ch, k=clip_k)
    ch = normalizer.peak_per_channel(ch)   # <-- aici e apelul
    return ch

def make_windows(channels, win_size=512, overlap=0.5):

    step = int(win_size * (1 - overlap))
    if step <= 0:
        raise ValueError("Overlap too large: step <= 0")

    n = min(len(ch) for ch in channels)
    if n < win_size:
        return []

    windows = []
    for start in range(0, n - win_size + 1, step):
        w = np.stack([ch[start:start + win_size] for ch in channels], axis=0)
        windows.append(w.astype(np.float32))

    return windows


def build_window_dataset(dataStore, labels, win_size=512, overlap=0.5,
                         clip_k=6.0, max_windows_per_example=None):

    X_list, y_list = [], []

    for ex, lab in zip(dataStore, labels):
        ch = preprocess_channels(ex, clip_k=clip_k)
        ws = make_windows(ch, win_size=win_size, overlap=overlap)

        if max_windows_per_example is not None:
            ws = ws[:max_windows_per_example]

        for w in ws:
            X_list.append(w)
            y_list.append(lab)

    if len(X_list) == 0:
        return np.empty((0, 0, 0), dtype=np.float32), np.array([], dtype=np.int64)

    Xw = np.stack(X_list, axis=0)
    yw = np.array(y_list, dtype=np.int64)
    return Xw, yw

def build_window_dataset_with_subjects(dataStore, labels, subjects,
                                       win_size=512, overlap=0.5,
                                       clip_k=6.0, max_windows_per_example=None):
    X_list, y_list, s_list = [], [], []

    for ex, lab, subj in zip(dataStore, labels, subjects):
        ch = preprocess_channels(ex, clip_k=clip_k)
        ws = make_windows(ch, win_size=win_size, overlap=overlap)

        if max_windows_per_example is not None:
            ws = ws[:max_windows_per_example]

        for w in ws:
            X_list.append(w)
            y_list.append(lab)
            s_list.append(subj)

    if len(X_list) == 0:
        return (
            np.empty((0, 0, 0), dtype=np.float32),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64)
        )

    Xw = np.stack(X_list, axis=0)
    yw = np.array(y_list, dtype=np.int64)
    sw = np.array(s_list, dtype=np.int64)
    return Xw, yw, sw
