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
    def __init__(self, df, sample_rate=259/3., seconds=3, batch_size=32, shuffle=True):
        self.df = np.array(df.loc[:, ['parent_genre_id', 'fpath']])
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.dim = int(self.sample_rate * self.seconds)
        self.data_size = self.df.shape[0]
        self.batch_size = batch_size
    
    def __len__(self):
        return self.df.shape[0]//self.batch_size
    
    def get_sample(self, x):
        # x = x.numpy()
        fpath = './data/fma_medium' + self.df[x][-1]
        #print(fpath)
        audio = np.load(fpath, mmap_mode='r', allow_pickle=True)
        shape = audio.shape
        start_idx = tf.random.uniform(shape=[], minval=0, maxval=shape[0] - self.dim, dtype=tf.dtypes.int32)
        audio = audio[start_idx:start_idx+self.dim]
        output = tf.cast(audio, tf.dtypes.float32)
        return output, self.df[x][0]