import librosa as libr
import numpy as np
import random
import matplotlib.pyplot as plt


def get_spectral_features(y, sr):
    return [
        libr.feature.chroma_stft(y=y, sr=sr),
        libr.feature.spectral_centroid(y=y, sr=sr),
        libr.feature.spectral_bandwidth(y=y, sr=sr),
        libr.feature.spectral_rolloff(y=y, sr=sr),
        libr.feature.tempo(y=y, sr=sr)
    ]


def get_spectral_feature_means(y, sr):
    features = get_spectral_features(y, sr)
    return np.array(
        [
            np.mean(i) for i in features
        ]
    )


def create_mel_spectrogram(normalized_wave, sampling_rate):
    spectrum = libr.feature.melspectrogram(y=normalized_wave, sr=sampling_rate)
    spectrum_decibals = libr.power_to_db(spectrum, ref=np.max)
    return spectrum_decibals

def display_spectrum(spectrum, sampling_rate):
    fig, ax = plt.subplots()
    img = libr.display.specshow(spectrum, x_axis='time',
                                   y_axis='mel', sr=sampling_rate,
                                   fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.savefig("figure.png")