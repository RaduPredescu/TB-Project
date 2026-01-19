import numpy as np


def MAV(x):

    return float(np.mean(np.abs(x)))


def RMS(x):

    return float(np.sqrt(np.mean(x ** 2)))


def WL(x):

    return float(np.sum(np.abs(np.diff(x))))


def ZCR(x, alpha=0.02):

    dx = np.abs(x[1:] - x[:-1])
    sign_change = (x[1:] * x[:-1]) < 0
    thresh = dx >= alpha
    return float(np.sum(sign_change & thresh))


def SSC(x, alpha=0.02):

    if len(x) < 3:
        return 0.0
    left = x[1:-1] - x[:-2]
    right = x[1:-1] - x[2:]
    return float(np.sum((left * right) >= alpha))


def Skewness(x):

    mu = np.mean(x)
    sigma = np.std(x) + 1e-9
    return float(np.mean(((x - mu) / sigma) ** 3))


def ISEMG(x):

    return float(np.sum(np.sqrt(np.abs(x))))


def HjorthActivity(x):

    mu = np.mean(x)
    return float(np.mean((x - mu) ** 2))

def AsymmetryCoefficient(X1, X2):

    denom = max(abs(X1), abs(X2)) + 1e-9
    return float(abs(X1 - X2) / denom * 100.0)
