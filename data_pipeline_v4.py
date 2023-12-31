import numpy as np
import tensorflow as tf

'''
This class allows preprocessing of the data to be used with
multiprocessing later during training.
Idea from: https://medium.com/@acordier/tf-data-dataset-generators-with-parallelization-the-easy-way-b5c5f7d2a18
'''
class DataGen():
    def __init__(self, sample_rate=259/3., seconds=3):
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.dim = int(self.sample_rate * self.seconds)
    
    def get_sample(self, fpath, label):
        fpath = fpath.numpy()
        audio = np.load(fpath, mmap_mode='r', allow_pickle=True)
        shape = audio.shape
        start_idx = tf.random.uniform(shape=[], minval=0, maxval=shape[0] - self.dim, dtype=tf.dtypes.int32)
        audio = audio[start_idx:start_idx+self.dim]
        return audio, label