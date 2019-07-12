from keras.utils import Sequence
import math
import numpy as np

class SequenceData(Sequence):
    def __init__(self, x_path, y_path, batch_size):
        self.x_path = x_path
        self.y_path = y_path
        self.batch_size = batch_size

    def __len__(self):
        num_imgs = len(self.x_path)
        return math.ceil(num_imgs / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x_path[idx * self.batch_size: (idx + 1) * self.batch_size]
        x_arrays = np.array([np.load(filename) for filename in batch_x])
        batch_y = self.y_path[idx * self.batch_size: (idx + 1) * self.batch_size]
        y_arrays = np.array([np.load(filename) for filename in batch_y])   # load some pictures

        return x_arrays, y_arrays
