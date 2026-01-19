import yaml
import numpy as np
from sklearn.model_selection import train_test_split

from dataloader import LoadData
from preprocess import build_window_dataset_with_subjects
from feature_extraction import build_feature_matrix
from model import *
from svm import *
import time
from metrics import compute_classification_metrics, print_metrics


def print_split_stats(name, y):
    u, c = np.unique(y, return_counts=True)
    print(f"{name}: n={len(y)}  per class:", dict(zip(u.tolist(), c.tolist())))


def main():
    # --- load config ---
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    alpha = float(cfg["features"]["alpha"])
    p_train = float(cfg["split"]["train"])
    p_val = float(cfg["split"]["val"])
    p_test = float(cfg["split"]["test"])

    print("Config:")
    print(" alpha =", alpha)
    print(" split =", p_train, p_val, p_test)

    # --- load data ---
    loader = LoadData("sEmg_databases", dc_offset=128)
    dataStore, labels, subjects = loader.load_three_classes(n_channels=8)

    # import numpy as np

    ex = dataStore[0]  # primul exemplu
    for i, ch in enumerate(ex):
        rms = np.sqrt(np.mean(ch**2))
        print(f"Channel {i}: RMS = {rms:.3f}")


    # windows + subjects
    Xw, yw, sw = build_window_dataset_with_subjects(
        dataStore, labels, subjects,
        win_size=512, overlap=0.5, clip_k=6.0
    )

    print("\nALL windows:", Xw.shape)
    print_split_stats("ALL", yw)

    # --- subject-wise split ---
    uniq_subj = np.unique(sw)

    # train vs temp
    subj_train, subj_temp = train_test_split(
        uniq_subj, test_size=(1.0 - p_train), random_state=42, shuffle=True
    )

    # val vs test from temp
    val_ratio_in_temp = p_val / (p_val + p_test)
    subj_val, subj_test = train_test_split(
        subj_temp, test_size=(1.0 - val_ratio_in_temp), random_state=42, shuffle=True
    )

    train_mask = np.isin(sw, subj_train)
    val_mask   = np.isin(sw, subj_val)
    test_mask  = np.isin(sw, subj_test)

    Xw_tr, yw_tr = Xw[train_mask], yw[train_mask]
    Xw_va, yw_va = Xw[val_mask],   yw[val_mask]
    Xw_te, yw_te = Xw[test_mask],  yw[test_mask]

    print("\n=== SUBJECT-WISE SPLIT ===")
    print("Subjects:", len(subj_train), len(subj_val), len(subj_test))
    print_split_stats("TRAIN", yw_tr)
    print_split_stats("VAL",   yw_va)
    print_split_stats("TEST",  yw_te)

    # --- feature extraction with alpha from config ---
    Xf_tr, feat_names = build_feature_matrix(Xw_tr, alpha=alpha, add_relational=True, return_names=True)
    Xf_va = build_feature_matrix(Xw_va, alpha=alpha, add_relational=True, return_names=False)
    Xf_te = build_feature_matrix(Xw_te, alpha=alpha, add_relational=True, return_names=False)

    print("\n=== FEATURE MATRICES ===")
    print("Xf_tr:", Xf_tr.shape)
    print("Xf_va:", Xf_va.shape)
    print("Xf_te:", Xf_te.shape)
    print("num_features:", Xf_tr.shape[1])
    print("first 5 feature names:", feat_names[:5])


    model_type = cfg["model"]["type"].lower()
    epochs = cfg["model"]["mlp"]["epochs"]
    batch_size = cfg["model"]["mlp"]["batch_size"]
    lr = cfg["model"]["mlp"]["lr"]
    experiment_name = cfg["model"]["mlp"]["experiment_name"]

    initial_time = time.time()

    if model_type == "mlp":
        mlp_cfg = cfg["model"]["mlp"]
        sched_cfg = mlp_cfg["scheduler"]
        experiment_name = mlp_cfg["experiment_name"]
        model, device = train_mlp(
            Xf_tr, yw_tr,
            Xf_va, yw_va,
            epochs=int(mlp_cfg["epochs"]),
            batch_size=int(mlp_cfg["batch_size"]),
            lr=float(mlp_cfg["lr"]),
            experiment_name=experiment_name,
            sched_step=int(sched_cfg["step"]),
            sched_factor=float(sched_cfg["factor"]),
        )

        final_time = time.time()
        print(f"\nTraining time (MLP): {final_time - initial_time:.2f} seconds")
        ds_te = TorchDataset(Xf_te, yw_te)
        dl_te = DataLoader(ds_te, batch_size=512, shuffle=False)

        y_true, y_pred = predict(model, dl_te, device)
        metrics = compute_classification_metrics(y_true, y_pred)
        print_metrics("MLP – TEST", metrics)


    elif model_type == "svm":
        C = float(cfg["model"]["svm"]["C"])
        kernel = cfg["model"]["svm"]["kernel"]

        clf, stats = train_svm(Xf_tr, yw_tr, Xf_va, yw_va, C=C, kernel=kernel)
        final_time = time.time()
        print(f"\nTraining time (SVM): {final_time - initial_time:.2f} seconds")
        if "val_acc" in stats:
            print(f"VAL accuracy (SVM): {stats['val_acc'] * 100:.2f} %")

        y_pred = clf.predict(Xf_te)
        metrics = compute_classification_metrics(yw_te, y_pred)
        print_metrics("SVM – TEST", metrics)


if __name__ == "__main__":
    main()
