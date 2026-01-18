import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


def train_svm(X_tr, y_tr, X_va=None, y_va=None, *, C=1.0, kernel="rbf"):
    clf = SVC(C=C, kernel=kernel, probability=False)
    clf.fit(X_tr, y_tr)

    stats = {}
    if X_va is not None and y_va is not None:
        yva = clf.predict(X_va)
        stats["val_acc"] = accuracy_score(y_va, yva)

    return clf, stats


def evaluate_svm(clf, X_te, y_te):
    ypred = clf.predict(X_te)
    acc = accuracy_score(y_te, ypred)
    cm = confusion_matrix(y_te, ypred)
    return acc, cm
