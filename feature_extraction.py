import numpy as np
from descriptors import MAV, RMS, WL, ZCR, SSC, Skewness, ISEMG, HjorthActivity, AsymmetryCoefficient


def get_feature_names(n_channels, add_relational=False, biceps_idx = 2, triceps_idx = 5):

    base = ["MAV", "RMS", "WL", "ZCR", "SSC", "Skewness", "ISEMG", "HjorthActivity"]
    names = []
    for c in range(n_channels):
        for f in base:
            names.append(f"ch{c}_{f}")

    if add_relational and n_channels > max(biceps_idx, triceps_idx):
        names += [
            f"KAs_RMS_ch{biceps_idx}_ch{triceps_idx}",
            f"KAs_Skew_ch{biceps_idx}_ch{triceps_idx}",
        ]
    return names


def extract_features_per_window(window, alpha=0.0, add_relational=False, biceps_idx=2, triceps_idx=5):

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

    if add_relational and window.shape[0] > max(biceps_idx, triceps_idx):
        biceps = window[biceps_idx]
        triceps = window[triceps_idx]

        rms_b = RMS(biceps)
        rms_t = RMS(triceps)
        skew_b = Skewness(biceps)
        skew_t = Skewness(triceps)

        kas_rms = AsymmetryCoefficient(rms_b, rms_t)
        kas_skew = AsymmetryCoefficient(skew_b, skew_t)

        feats.extend([kas_rms, kas_skew])

    return np.array(feats, dtype=np.float32)

def build_feature_matrix(Xw, alpha=0.0, add_relational=False, return_names=True, biceps_idx=2, triceps_idx=5):
    
    if Xw.ndim != 3:
        raise ValueError("Xw must have shape (num_windows, n_channels, win_size)")

    nW = Xw.shape[0]

    # calculeaza primul vector ca sa afli dimensiunea
    first = extract_features_per_window(
        Xw[0],
        alpha=alpha,
        add_relational=add_relational,
        biceps_idx=biceps_idx,
        triceps_idx=triceps_idx
    )

    if not hasattr(first, "shape") or first.ndim != 1:
        raise ValueError(f"extract_features_per_window must return 1D array, got: {type(first)} shape={getattr(first,'shape',None)}")

    nF = first.shape[0]
    Xf = np.empty((nW, nF), dtype=np.float32)
    Xf[0] = first

    for i in range(1, nW):
        feats = extract_features_per_window(
            Xw[i],
            alpha=alpha,
            add_relational=add_relational,
            biceps_idx=biceps_idx,
            triceps_idx=triceps_idx
        )
        if feats.shape != (nF,):
            raise ValueError(f"Inconsistent feature length at window {i}: got {feats.shape}, expected {(nF,)}")
        Xf[i] = feats

    if return_names:
        names = get_feature_names(Xw.shape[1], add_relational=add_relational, biceps_idx=biceps_idx, triceps_idx=triceps_idx)
        return Xf, names
    return Xf

