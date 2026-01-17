from dataloader import LoadData
from preprocess import build_window_dataset

def main():
    data_dir = "sEmg_databases"
    loader = LoadData(data_dir, dc_offset=128)

    dataStore, labels, subjects = loader.load_three_classes(n_channels=8)

    print("=== FILE-LEVEL ===")
    print("Numar fisiere:", len(labels))
    print("Clase unice:", sorted(set(labels)))
    print("Primele 10 labels:", labels[:10])

    if len(labels) == 0:
        print("Nu s-a incarcat nimic. Verifica path-ul si numele fisierelor.")
        return

    Xw, yw = build_window_dataset(
        dataStore, labels,
        win_size=512,
        overlap=0.5,
        clip_k=6.0,
        max_windows_per_example=5  # doar pentru debug
    )

    print("\n=== WINDOW-LEVEL ===")
    print("Xw shape:", Xw.shape)  # (num_windows, n_channels, win_size)
    print("yw shape:", yw.shape)
    print("Primele 20 etichete ferestre:", yw[:20])

    if Xw.shape[0] > 0:
        print("\nPrima fereastra:")
        print(" shape:", Xw[0].shape)
        print(" label:", yw[0])
        print(" canal 0 primele 10 valori:", Xw[0, 0, :10])

if __name__ == "__main__":
    main()
