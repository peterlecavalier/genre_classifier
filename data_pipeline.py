import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

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
        fpath = './data/fma_medium' + self.df[x][-1]
        #print(fpath)
        audio = tfio.audio.AudioIOTensor(fpath)
        shape = tf.cast(audio.shape, tf.dtypes.int32)
        start_idx = tf.random.uniform(shape=[], minval=0, maxval=shape[0] - self.dim, dtype=tf.dtypes.int32)
        audio_slice = audio[start_idx:start_idx + self.dim]
        # Convert to one channel
        # either by averaging stereo channels or removing extra dim on mono
        if tf.equal(shape[-1], 2):
            audio_tensor = tf.reduce_mean(audio_slice, axis=[-1])
        else:
            audio_tensor = tf.squeeze(audio_slice, axis=[-1])
        
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