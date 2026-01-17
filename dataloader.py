import os
import numpy as np


class LoadData:
    def __init__(self, dataDirectory, dc_offset=128):
        self.dataDirectory = dataDirectory
        self.dc_offset = dc_offset

    # ----------------- Utils -----------------
    def load_data(self, filename):
        return np.load(filename)

    def _get_channels(self, file_data, n_channels=None):
        if n_channels is None:
            n_channels = file_data.shape[0]

        channels = [
            file_data[i].astype(int) - self.dc_offset
            for i in range(n_channels)
        ]
        return channels

    def _iter_npy_files(self):
        for filename in os.listdir(self.dataDirectory):
            if filename.endswith(".npy"):
                full_path = os.path.join(self.dataDirectory, filename)
                file_data = self.load_data(full_path)
                parts = filename.split("_")
                # ai deja acest print, îl păstrez pentru debug
                print(parts)
                yield filename, parts, file_data

    # ----------------- 3 classes, full arm -----------------
    def loadData_armthreeClasses(self, n_channels=8, class_index_in_name=2):
       
        dataStore = []
        labels = []

        class_map = {
            "0": 0,
            "1": 1,
            "2": 2
        }

        for filename, parts, file_data in self._iter_npy_files():
            if len(parts) <= class_index_in_name:
                continue

            cl = parts[class_index_in_name]
            if cl in class_map:
                channels = self._get_channels(file_data, n_channels=n_channels)
                dataStore.append(channels)
                labels.append(class_map[cl])

        return dataStore, labels

    # ----------------- 2 classes, leg movement -----------------
    def loadData_twoClasses_leg(
        self,
        n_channels=4,
        class_id="3",
        class_index_in_name=2,
        start_rest=0,
        end_rest=20480,
        start_fatigue=20480,
        end_fatigue=40960
    ):
        
        dataStore = []
        labels = []

        for filename, parts, file_data in self._iter_npy_files():
            if len(parts) <= class_index_in_name:
                continue

            cl = parts[class_index_in_name]
            if cl == class_id:
                channels = self._get_channels(file_data, n_channels=n_channels)

                
                seg1 = [ch[start_rest:end_rest] for ch in channels]
                dataStore.append(seg1)
                labels.append(0)

                
                seg2 = [ch[start_fatigue:end_fatigue] for ch in channels]
                dataStore.append(seg2)
                labels.append(1)

        return dataStore, labels

    # ----------------- 2 classes, first arm movement -----------------
    def loadData_twoClasses_firstarmmovement(
        self,
        n_channels=8,
        class_id="0",
        class_index_in_name=2,
        start_class0=0,
        end_class0=15360,
        start_class1=15360,
        end_class1=None
    ):
        
        dataStore = []
        labels = []

        for filename, parts, file_data in self._iter_npy_files():
            if len(parts) <= class_index_in_name:
                continue

            cl = parts[class_index_in_name]
            if cl == class_id:
                channels = self._get_channels(file_data, n_channels=n_channels)

                # Class 0 segment
                seg0 = [ch[start_class0:end_class0] for ch in channels]
                dataStore.append(seg0)
                labels.append(0)

                # Class 1 segment (până la end_class1 sau până la final)
                if end_class1 is None:
                    seg1 = [ch[start_class1:] for ch in channels]
                else:
                    seg1 = [ch[start_class1:end_class1] for ch in channels]
                dataStore.append(seg1)
                labels.append(1)

        return dataStore, labels

    # ----------------- 2 classes, second arm movement -----------------
    def loadData_twoClasses_secondarmmovement(
        self,
        n_channels=8,
        class_id="1",
        class_index_in_name=2,
        # implicit: 10s–30s (5120–15360) vs 40s–60s (20480–end)
        start_class0=5120,
        end_class0=15360,
        start_class1=20480,
        end_class1=None
    ):
        
        dataStore = []
        labels = []

        for filename, parts, file_data in self._iter_npy_files():
            if len(parts) <= class_index_in_name:
                continue

            cl = parts[class_index_in_name]
            if cl == class_id:
                channels = self._get_channels(file_data, n_channels=n_channels)

                
                seg0 = [ch[start_class0:end_class0] for ch in channels]
                dataStore.append(seg0)
                labels.append(0)

                
                if end_class1 is None:
                    seg1 = [ch[start_class1:] for ch in channels]
                else:
                    seg1 = [ch[start_class1:end_class1] for ch in channels]
                dataStore.append(seg1)
                labels.append(1)

        return dataStore, labels

    # ----------------- 2 classes, third arm movement -----------------
    def loadData_twoClasses_thirdarmmovement(
        self,
        n_channels=8,
        class_id="2",
        class_index_in_name=2,
        start_class0=5120,
        end_class0=15360,
        start_class1=20480,
        end_class1=None
    ):
        
        dataStore = []
        labels = []

        for filename, parts, file_data in self._iter_npy_files():
            if len(parts) <= class_index_in_name:
                continue

            cl = parts[class_index_in_name]
            if cl == class_id:
                channels = self._get_channels(file_data, n_channels=n_channels)

                # Class 0 segment
                seg0 = [ch[start_class0:end_class0] for ch in channels]
                dataStore.append(seg0)
                labels.append(0)

                # Class 1 segment
                if end_class1 is None:
                    seg1 = [ch[start_class1:] for ch in channels]
                else:
                    seg1 = [ch[start_class1:end_class1] for ch in channels]
                dataStore.append(seg1)
                labels.append(1)

        return dataStore, labels