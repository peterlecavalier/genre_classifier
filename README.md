# Music Genre Classifier
## by Peter LeCavalier

Welcome! This project sets to classify music audio files into genres based purely on the raw audio data.

This is done by converting the files into [mel spectrogram](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53), and running these through a [Long Short-Term Memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) network. The music data is a subset of the [Free Music Archive (FMA)](https://github.com/mdeff/fma) which, in it's entirety, contains 879 GB of 106,574 tracks, part of 161 genres. Due to time and storage constraints, this model uses a subset containing 24,922 ~30 second song clips.

Currently, this model achieves an accuracy of X% across 16 genres.

See [my website]() if you're interested in the project as a whole.