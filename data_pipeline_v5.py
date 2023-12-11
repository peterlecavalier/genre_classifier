import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from soundfile import read as sf_read

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
        self.total_dim = self.dim * 9
    
    def get_sample(self, fpath, label):
        fpath = fpath.numpy()
        audio = np.load(fpath, allow_pickle=True)
        audio = audio[:self.total_dim]
        audio = tf.reshape(audio, (9, self.dim, 128))
        #output = tf.cast(audio, tf.dtypes.float32)
        #return output, label
        return audio, label