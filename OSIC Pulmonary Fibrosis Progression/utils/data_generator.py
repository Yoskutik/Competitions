from tensorflow.keras.utils import Sequence
from joblib import Parallel, delayed
import multiprocessing
import numpy as np


num_cores = multiprocessing.cpu_count()

class DataGenerator(Sequence):
    def __init__(self, patients, read_func, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.patients = patients
        self.read_func = read_func
        np.random.shuffle(self.patients)
        self.shuffle = shuffle
        self.on_epoch_end()
        self._labels = np.zeros([batch_size, 5])

    def __len__(self):
        return int(np.floor(len(self.patients) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        patients = [self.patients[k] for k in indexes]
        
        results = Parallel(n_jobs=num_cores)(
            delayed(self.read_func)(path) for path in patients
        )
        
        images = []
        ns = []
        outputs = []
        for img, n, output in results:
            images.append(img)
            ns.append(n)
            outputs.append(output)
        
        return (np.array(images), np.array(ns)), np.array(outputs)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.patients))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)