import numpy as np
from descriptors import MAV, RMS, WL, ZCR, SSC, Skewness, ISEMG, HjorthActivity


def get_feature_names(n_channels, add_relational=False):
    """
    Build a list of feature names in the exact order in which features are extracted.

    Parameters
    ----------
    n_channels : int
        Number of channels in the window.
    add_relational : bool
        If True, append relational features (only meaningful for >=2 channels).

    Returns
    -------
    list[str]
        Feature names.
    """
    base = ["MAV", "RMS", "WL", "ZCR", "SSC", "Skewness", "ISEMG", "HjorthActivity"]
    names = []
    for c in range(n_channels):
        for f in base:
            names.append(f"ch{c}_{f}")

    if add_relational and n_channels >= 2:
        names += ["rel_absdiff_RMS_ch0_ch1", "rel_ratio_RMS_ch0_over_ch1", "rel_corr_ch0_ch1"]

    return names


def extract_features_per_window(window, alpha=0.0, add_relational=False):
    """
    Extract time-domain features for a single multi-channel window.

    Features per channel (in order):
      MAV, RMS, WL, ZCR, SSC, Skewness, ISEMG, HjorthActivity

    Optionally adds relational features between channel 0 and channel 1:
      |RMS0 - RMS1|, RMS0/(RMS1+eps), corr(ch0, ch1)

    Parameters
    ----------
    window : np.ndarray
        Window array of shape (n_channels, win_size).
    alpha : float
        Threshold used for ZCR and SSC (noise suppression).
    add_relational : bool
        Whether to add relational features (requires at least 2 channels).

    Returns
    -------
    np.ndarray
        1D feature vector.
    """
    feats = []
    for ch in window:
        feats.extend([
            MAV(ch),
            RMS(ch),
            WL(ch),
            ZCR(ch, alpha=alpha),
            SSC(ch, alpha=alpha),
            Skewness(ch),
            ISEMG(ch),
            HjorthActivity(ch),
        ])

    if add_relational and window.shape[0] >= 2:
        ch0, ch1 = window[0], window[1]
        rms0, rms1 = RMS(ch0), RMS(ch1)

        diff_rms = abs(rms0 - rms1)
        ratio_rms = rms0 / (rms1 + 1e-9)

        corr = np.corrcoef(ch0, ch1)[0, 1]
        if not np.isfinite(corr):
            corr = 0.0

        feats.extend([diff_rms, ratio_rms, float(corr)])

    return np.array(feats, dtype=np.float32)


def build_feature_matrix(Xw, alpha=0.0, add_relational=False, return_names=True):
    """
    Convert window-level EMG signals into a feature matrix.

    Parameters
    ----------
    Xw : np.ndarray
        Array of shape (num_windows, n_channels, win_size).
    alpha : float
        Threshold used for ZCR and SSC.
    add_relational : bool
        Add relational features between channel 0 and 1 (requires >=2 channels).
    return_names : bool
        If True, also return feature_names.

    Returns
    -------
    Xf : np.ndarray
        Feature matrix of shape (num_windows, num_features).
    feature_names : list[str] (optional)
        Names for each feature column, returned if return_names=True.
    """
    if Xw.ndim != 3:
        raise ValueError("Xw must have shape (num_windows, n_channels, win_size)")

    n_channels = Xw.shape[1]
    Xf = np.stack(
        [extract_features_per_window(w, alpha=alpha, add_relational=add_relational) for w in Xw],
        axis=0
    ).astype(np.float32)

    if return_names:
        names = get_feature_names(n_channels, add_relational=add_relational)
        return Xf, names
    return Xf
