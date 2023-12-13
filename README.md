# Music Genre Classifier
## by Peter LeCavalier

Welcome! This project sets to classify music audio files into genres based purely on the raw audio data.

This is done by converting the files into [mel spectrogram](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53), and running these through a [Long Short-Term Memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) network. The music data is a subset of the [Free Music Archive (FMA)](https://github.com/mdeff/fma) which, in it's entirety, contains 879 GB of 106,574 tracks, part of 161 genres. Due to time and storage constraints, this model uses a subset containing 24,922 ~30 second song clips.

Currently, this model achieves an accuracy of 69.50% across 16 genres.

See [my website](https://sites.google.com/colorado.edu/musicgenreclassification/) if you're interested in the project as a whole.

## Brief file descriptions - in order of when they should be run

### Main files:
- **get_data.py**: Script for downloading the necessary project data.
- **genres_processing.ipynb**: Notebook for initial processing of the data (before generating Mel spectrograms).
- **mel_spectrogram_gen.ipynb**: Notebook for generating mel spectrograms and updating necessary CSV. 
- **train_v4.ipynb**: Main training notebook. Run all cells in this notebook to start training!
- **visualize_loss_acc.ipynb**: Notebook for visualizing training loss and accuracy plots.

### Other files:
- **audio_visualize.ipynb**: Notebook for visualizing audio soundwave and Mel spectrogram.
- **make_subset.ipynb**: Notebook to make a smaller subset of the data as a sample.
- **test.ipynb**: Notebook for testing various things. Not clean/intuitive.
- **audio_load_testing.ipynb**: Notebook for testing audio loading. Not clean/intuitive.
- **data_pipeline_test.ipynb**: Notebook for testing the initial data pipeline. Deprecated. Not clean/intuitive.
