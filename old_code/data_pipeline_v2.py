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
    def __init__(self, df, sample_rate=44100, seconds=3, batch_size=32, shuffle=True):
        self.df = np.array(df.loc[:, ['parent_genre_id', 'fpath']])
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.dim = self.sample_rate * self.seconds
        self.data_size = self.df.shape[0]
        self.indices = tf.range(0, self.data_size, delta=1)
        if shuffle:
            self.indices = tf.random.shuffle(self.indices)
        self.batch_size = batch_size
    
    def __len__(self):
        return self.df.shape[0]//self.batch_size
    
    def get_sample(self, x):
        x = x.numpy()
        fpath = './data/fma_medium' + self.df[x][-1]
        #print(fpath)
        audio = tfio.audio.AudioIOTensor(fpath)
        shape = tf.cast(audio.shape, tf.dtypes.int32)
        start_idx = tf.random.uniform(shape=[], minval=0, maxval=shape[0] - self.dim, dtype=tf.dtypes.int32)
        audio_tensor, _ = sf_read(fpath, frames=self.dim, start=start_idx)
        audio_tensor = tf.cast(audio_tensor, tf.dtypes.float32)
        # Convert to one channel by averaging stereo channels
        if tf.equal(audio_tensor.shape[-1], 2):
            audio_tensor = tf.reduce_mean(audio_tensor, axis=[-1])
        
        # Make a spectrogram and clip off values to get 512x512 image
        # Original image will be 517x513 so this doesn't affect much.
        spectrogram = tfio.audio.spectrogram(audio_tensor, nfft=1024, window=512, stride=256)[:512, :512] # 86 stride for 1sec
        spectrogram = tf.math.log(spectrogram)

        ##### Replace all infinite values with the mean of non-infinite values
        non_infinite_mask = tf.math.is_finite(spectrogram)
        non_infinite_values = tf.boolean_mask(spectrogram, non_infinite_mask)

        # Calculate the mean of non-infinite values
        mean_non_infinite = tf.reduce_mean(non_infinite_values)

        # Replace infinite values with the mean
        spectrogram = tf.where(tf.math.is_inf(spectrogram), tf.fill(spectrogram.shape, mean_non_infinite), spectrogram)
        #####
        
        ##### Normalize the spectrogram
        # Calculate min and max values across the tensor
        min_values = tf.reduce_min(spectrogram)
        max_values = tf.reduce_max(spectrogram)

        # Min-max normalization
        spectrogram = (spectrogram - min_values) / (max_values - min_values)
        #####
        return spectrogram, self.df[x][0]
    
    def __call__(self):
        for idx in self.indices:
            yield self.get_sample(idx)