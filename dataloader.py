import os
import re
import numpy as np


class LoadData:
    """
    Data loader for EMG Database 1 with flat folder structure.
    """

    def __init__(self, dataDirectory, dc_offset=128):

        self.dataDirectory = dataDirectory
        self.dc_offset = dc_offset
        self.pattern = re.compile(r"^Subiect_(\d+)_(\d+)_r\.npy$", re.IGNORECASE)

    def load_data(self, filepath):

        return np.load(filepath)

    def _get_channels(self, file_data, n_channels):

        n_channels = min(n_channels, file_data.shape[0])
        return [(file_data[i].astype(np.int32) - self.dc_offset)
                for i in range(n_channels)]

    def iter_files(self):

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

        dataStore, labels, subjects = [], [], []

        for fn, subject_id, class_id, file_data in self.iter_files():
            if class_id not in (0, 1, 2):
                continue

            channels = self._get_channels(file_data, n_channels=n_channels)
            dataStore.append(channels)
            labels.append(class_id)
            subjects.append(subject_id)

        return dataStore, labels, subjects
