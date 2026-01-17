import os
import re
import numpy as np


class LoadData:
    """
    Data loader for EMG Database 1 with flat folder structure.
    """

    def __init__(self, dataDirectory, dc_offset=128):
        """
        Initialize the data loader.

        Params
        dataDirectory : str
            Path to the folder containing the .npy EMG files.
        dc_offset : int, optional
            Constant offset to subtract from raw samples in order
            to center the signal around zero.
        """
        self.dataDirectory = dataDirectory
        self.dc_offset = dc_offset
        self.pattern = re.compile(r"^Subiect_(\d+)_(\d+)_r\.npy$", re.IGNORECASE)

    def load_data(self, filepath):
        """
        Load a NumPy file from disk.

        Params
        filepath : str
            Full path to the .npy file.

        Returns
        np.ndarray
            Array containing the raw EMG data.
        """
        return np.load(filepath)

    def _get_channels(self, file_data, n_channels):
        """
        Extract and preprocess EMG channels from a loaded file.

        The function selects the first `n_channels` channels,
        converts them to integers, and removes the DC offset.

        Params
        file_data : np.ndarray
            Raw EMG data with shape (num_channels, num_samples).
        n_channels : int
            Number of channels to extract.

        Returns
        list of np.ndarray
            List of 1D arrays, one per channel.
        """
        n_channels = min(n_channels, file_data.shape[0])
        return [(file_data[i].astype(np.int32) - self.dc_offset)
                for i in range(n_channels)]

    def iter_files(self):
        """
        Iterate over all valid EMG files in the data directory.

        Yields
        ------
        tuple
            (filename, subject_id, class_id, file_data)
        """
        for fn in os.listdir(self.dataDirectory):
            if not fn.lower().endswith(".npy"):
                continue

            m = self.pattern.match(fn)
            if not m:
                continue

            subject_id = int(m.group(1))
            class_id = int(m.group(2))

            full_path = os.path.join(self.dataDirectory, fn)
            file_data = self.load_data(full_path)

            yield fn, subject_id, class_id, file_data

    def load_three_classes(self, n_channels=8):
        """
        Load the dataset for three-class movement classification.

        Only files with class IDs 0, 1, and 2 are considered.
        Each file becomes one example consisting of multiple channels.

        Params
        n_channels : int, optional
            Number of EMG channels to load from each file.

        Returns
        dataStore : list
            List of examples; each example is a list of channels.
        labels : list
            List of integer class labels (0, 1, or 2).
        subjects : list
            List of subject IDs corresponding to each example.
        """
        dataStore, labels, subjects = [], [], []

        for fn, subject_id, class_id, file_data in self.iter_files():
            if class_id not in (0, 1, 2):
                continue

            channels = self._get_channels(file_data, n_channels=n_channels)
            dataStore.append(channels)
            labels.append(class_id)
            subjects.append(subject_id)

        return dataStore, labels, subjects
