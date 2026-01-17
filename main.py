import numpy as np
from sklearn.model_selection import train_test_split

from dataloader import LoadData

def stack_examples(dataStore):
    """
    dataStore = list of examples
      each example = list of channels (len = n_channels)
      each channel = 1D array (samples)
    return: X shape (N, n_channels, samples)
    """
    X = np.array([np.stack(ex, axis=0) for ex in dataStore], dtype=np.float32)
    return X

def main():
    data_dir = "PATH_CĂTRE_FOLDERUL_CU_NPY"  # <- schimba aici
    loader = LoadData(data_dir, dc_offset=128)

    dataStore, labels = loader.loadData_armthreeClasses(n_channels=8, class_index_in_name=2)

    X = stack_examples(dataStore)
    y = np.array(labels, dtype=np.int64)

    print("X:", X.shape, "y:", y.shape, "classes:", np.unique(y))

    # stratify ca sa pastrezi proportiile claselor in split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Train:", X_train.shape, y_train.shape)
    print("Test:",  X_test.shape,  y_test.shape)

if __name__ == "__main__":
    main()
