import numpy as np


def MAV(x):
    """
    Mean Absolute Value (MAV).

    Computes the average of the absolute values of the signal samples
    in a window. MAV reflects the overall muscle activation level.

    Params
    x : np.ndarray
        1D array containing EMG samples of a window.

    Returns
    float
        Mean Absolute Value of the signal.
    """
    return float(np.mean(np.abs(x)))


def RMS(x):
    """
    Root Mean Square (RMS).

    Measures the energy of the EMG signal within a window.
    Higher RMS values usually correspond to stronger muscle contractions.

    Params
    x : np.ndarray
        1D array containing EMG samples of a window.

    Returns
    float
        Root Mean Square of the signal.
    """
    return float(np.sqrt(np.mean(x ** 2)))


def WL(x):
    """
    Waveform Length (WL).

    Quantifies the cumulative absolute difference between consecutive
    samples. WL captures both amplitude and frequency characteristics
    of the signal.

    Params
    x : np.ndarray
        1D array containing EMG samples of a window.

    Returns
    float
        Waveform Length of the signal.
    """
    return float(np.sum(np.abs(np.diff(x))))


def ZCR(x, alpha=0.02):
    """
    Zero Crossing Rate (ZCR).

    Counts how many times the signal changes sign within a window.
    A threshold alpha can be used to reduce the influence of noise.

    A zero-crossing is counted when:
      - x[k] * x[k-1] < 0 (sign change)
      - |x[k] - x[k-1]| >= alpha

    Params
    x : np.ndarray
        1D array containing EMG samples of a window.
    alpha : float, optional
        Threshold to reduce noise influence.

    Returns
    float
        Number of zero crossings in the window.
    """
    dx = np.abs(x[1:] - x[:-1])
    sign_change = (x[1:] * x[:-1]) < 0
    thresh = dx >= alpha
    return float(np.sum(sign_change & thresh))


def SSC(x, alpha=0.02):
    """
    Slope Sign Changes (SSC).

    Counts the number of times the slope of the signal changes sign
    within a window. A threshold alpha can be used to reduce noise.

    SSC detects abrupt changes in muscle activation.

    Params
    x : np.ndarray
        1D array containing EMG samples of a window.
    alpha : float, optional
        Threshold to reduce noise influence.

    Returns
    float
        Number of slope sign changes in the window.
    """
    if len(x) < 3:
        return 0.0
    left = x[1:-1] - x[:-2]
    right = x[1:-1] - x[2:]
    return float(np.sum((left * right) >= alpha))


def Skewness(x):
    """
    Skewness.

    Measures the asymmetry of the amplitude distribution of the EMG
    signal within a window.

    Params
    x : np.ndarray
        1D array containing EMG samples of a window.

    Returns
    float
        Skewness of the signal distribution.
    """
    mu = np.mean(x)
    sigma = np.std(x) + 1e-9
    return float(np.mean(((x - mu) / sigma) ** 3))


def ISEMG(x):
    """
    Integrated Square-root EMG (ISEMG).

    Provides a measure of the total muscular activity in a window
    by integrating the square root of the absolute signal.

    Params
    x : np.ndarray
        1D array containing EMG samples of a window.

    Returns
    float
        ISEMG value of the signal.
    """
    return float(np.sum(np.sqrt(np.abs(x))))


def HjorthActivity(x):
    """
    Hjorth Activity parameter.

    Corresponds to the variance of the signal and represents the
    overall signal power within a window.

    Params
    x : np.ndarray
        1D array containing EMG samples of a window.

    Returns
    float
        Hjorth Activity (variance) of the signal.
    """
    mu = np.mean(x)
    return float(np.mean((x - mu) ** 2))
