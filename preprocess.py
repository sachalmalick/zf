from loader import ZFinchDataset
import librosa as libr
import numpy as np
import constants as const
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
import util
from scipy.io import wavfile
import tensorflow as tf
import audioutils as audio

class ZFinchDataProcessor:
    def __init__(self, dataset: ZFinchDataset):
        self.dataset = dataset
        self.class_ids = np.array([util.get_call_id(i) for i in self.dataset.examples])
    def load_audio_data(self):
        self.audio_data = [libr.load(example.path) for example in self.dataset.examples]
        return self.audio_data
    def noramlize_audio_data(self):
        self.normalized_waves = np.array([normalize_sample_length(i[0]) for i in self.audio_data])
        return self.normalized_waves
    def write_recorings(self, folder):
        x = np.array(self.normalized_waves)
        print(x.shape)
        for i in range(0, len(self.normalized_waves)):
            wavfile.write(folder + "/" + str(i), 22050, self.normalized_waves[i])
    def get_full_spectral_feature_matrix(self):
        print(self.audio_data[0][1])
        x = np.array([audio.get_spectral_feature_means(normalize_sample_length(i[0]), 22050) for i in self.audio_data])
        y = self.class_ids
        return x, y
    def get_spectral_feature_matrix(self, j):
        print(self.audio_data[0][1])
        x = np.array([audio.get_spectral_features(normalize_sample_length(i[0]), 22050)[j] for i in self.audio_data])
        y = self.class_ids
        return x, y
    def get_mel_coefficients_matrix(self):
        x = np.array([audio.get_mel_coeffs(normalize_sample_length(i[0]), 22050) for i in self.audio_data])
        y = self.class_ids
        return x, y

def normalize_sample_length(sample):
    sample_length = int((const.SAMPLE_LENGTH_MS/1000)*const.SAMPLING_RATE)
    if(sample.shape[0] == sample_length):
        return sample
    if(sample.shape[0] < sample_length):
        #pad with zeros
        delta = (sample_length - sample.shape[0])
        beginning_pad = delta//2
        end_pad = delta - beginning_pad
        return np.pad(sample, (beginning_pad, end_pad), 'constant')
    else:
        #default to choosing middle of the sample
        delta = sample.shape[0] - sample_length
        start_index = delta//2
        return sample[start_index:start_index + sample_length]
    
class ZFinchTorchset(Dataset):
    def __init__(self, data_processor: ZFinchDataProcessor):
        self.data_processor = data_processor
    def __len__(self):
        return self.data_processor.normalized_waves.shape[0]
    def __getitem__(self, index):
        spectogram = audio.create_mel_spectrogram(self.data_processor.normalized_waves[index],
                                            const.SAMPLING_RATE)
        class_id = self.data_processor.class_ids[index]
        spectogram = np.array([spectogram])
        return spectogram, class_id