from loader import *
from preprocess import *
import cnn
import numpy as np


if __name__ == "__main__":
    print("loading raw dataset")
    dataset = load_dataset()

    print("loading audio data")
    preprocess = ZFinchDataProcessor(dataset)
    preprocess.load_audio_data()

    print("Normalizing audio data")
    preprocess.noramlize_audio_data()

    print("Training CNN!")
    cnn.train_and_evaluate(preprocess)
    print()
    display_spectrum(create_mel_spectrogram(preprocess.normalized_waves[111], 22050), 22050)