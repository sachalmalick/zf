import pathlib
import re
import librosa as libr
import numpy as np
import itertools
import audioutils as audio
import matplotlib.pyplot as plt
import cnn
from joblib import dump, load
import preprocess as proc
import constants
import torch
import util

GENERATED_WAVES_DIR = "generated_waves"

DOSE_PATTERN = r'^gz_(\w+)-(\d+)__dose_(-?\d+)_(\d+)\.wav$'
BASE_PATTERN = r'^gz_(\w+)-(\d+)__(\d+)\.wav$'


def compare_class_predictive_qualities():
    predictions = load("predictions_on_generators.pkl")
    #we want to see if the classes are the same for
    # the same latent codes with no doses
    latent_correlations = {}
    latent_totals = {}
    features_correlations = {}
    features_totals = {}
    #print([(filename, int(predictions[filename]["ramdom_c_pred"][0])) for filename in predictions])
    for filename in predictions:
        #for model in predictions[filename]:
        call_id = int(predictions[filename]["logistic_pred"][0])
        call_id_name = constants.CALL_NAMES[call_id]
        if("dose" in filename):
            match = re.match(DOSE_PATTERN, filename)
            latent, features, dose, sample_num = match.groups()
        else:
            if(latent in latent_correlations):
                latent_totals[latent]+=1
                if(call_id_name in latent_correlations[latent]):
                    latent_correlations[latent][call_id_name]+=1
                else:
                    latent_correlations[latent][call_id_name] = 1
            else:
                latent_correlations[latent] = {call_id_name : 1}
                latent_totals[latent] = 1
            if(features in features_correlations):
                features_totals[features] +=1
                if(call_id_name in features_correlations[features]):
                    features_correlations[features][call_id_name]+=1
                else:
                    features_correlations[features][call_id_name] = 1
            else:
                features_correlations[features] = {call_id_name : 1}
                features_totals[features] = 1
    distributions_of_id_given_features = {}
    for features in features_correlations:
        totals = features_totals[features]
        distributions_of_id_given_features[features] = {}
        for call_id_name in features_correlations[features]:
            call_id_freq = features_correlations[features][call_id_name]
            prob = call_id_freq / totals
            distributions_of_id_given_features[features][call_id_freq] = prob
            print(features, call_id_name, prob)
    #dump("p_call_type_given_x", distributions_of_id_given_features)
    distributions_of_id_given_latents= {}
    for latents in latent_correlations:
        totals = latent_totals[latents]
        distributions_of_id_given_latents[latents] = {}
        for call_id_name in latent_correlations[latents]:
            call_id_freq = latent_correlations[latents][call_id_name]
            prob = call_id_freq / totals
            distributions_of_id_given_latents[latents][call_id_freq] = prob
            print(latent, call_id_name, prob)
def load_data():
    filepaths = list(pathlib.Path(GENERATED_WAVES_DIR).iterdir())
    wave_data = []
    ids = 0
    for path in filepaths:
        if("dose" in path.name):
            match = re.match(DOSE_PATTERN, path.name)
            latent, features, dose, sample_num = match.groups()
            gw = GeneratedWave(ids,
                               path,
                               latent,
                               features,
                               sample_num,
                               is_dose=True,
                               dose=dose)
            wave_data.append(gw)
        else:
            latent, features, sample_num = re.match(BASE_PATTERN, path.name).groups()
            gw = GeneratedWave(ids, path, latent, features, sample_num)
            wave_data.append(gw)
        ids+=1
    return wave_data

class GeneratedWave():
    def __init__(self, id, path, latent, features, sample_num, is_dose=False, dose=None):
        self.id = id
        self.latent = latent
        self.features = features
        self.sample_num = sample_num
        self.is_dose = is_dose
        self.dose = dose
        self.path = path
        self.wave, self.sr = libr.load(path)
        self.spectral_features = audio.get_spectral_feature_means(self.wave, self.sr)
        self.mel_coeffs = libr.feature.mfcc(y=self.wave, sr=self.sr, n_mfcc=30)


def l2_norm(w1, w2):
    l2_norm = np.linalg.norm(w1 - w2)
    return l2_norm

def plot_and_save_latent_data(data):
    for latent, features in data.items():
        # Preparing data for each feature code
        feature_data_mel = {}
        feature_data_spec = {}
        for feature_code, doses in features.items():
            doses_int = sorted(int(dose) for dose in doses.keys())
            feature_data_mel[feature_code] = [doses[str(dose)]['mel'] for dose in doses_int]
            feature_data_spec[feature_code] = [doses[str(dose)]['spec'] for dose in doses_int]

        plt.figure(figsize=(10, 5))
        for feature_code, values in feature_data_mel.items():
            plt.plot(doses_int, values, label=f'Feature {feature_code}', marker='o')
        plt.title(f"Latent: {latent} - Mel Values")
        plt.xlabel('Dose')
        plt.ylabel('Mel Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"artifacts/{latent}_mel.png") 
        plt.close()

        plt.figure(figsize=(10, 5))
        for feature_code, values in feature_data_spec.items():
            plt.plot(doses_int, values, label=f'Feature {feature_code}', marker='o')
        plt.title(f"Latent: {latent} - Spec Values")
        plt.xlabel('Dose')
        plt.ylabel('Spec Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"artifacts/{latent}_spec.png") 
        plt.close()

def define_color_map(features):
    colors = itertools.cycle(["b", "g", "r", "c", "m", "y", "k"])  # basic color cycle
    color_map = {}
    for feature_code in features:
        color_map[feature_code] = next(colors)
    return color_map



def compare_classifiers(data_set):
    print("Loading CNN")
    cnn_model = cnn.load_model()
    lda = load('lda.joblib')
    ramdom_c = load('random_c.joblib')
    logistic = load('logistic.joblib')

    predictions = {}

    for example in data_set:
        normalized = proc.normalize_sample_length(example.wave)
        spectogram = audio.create_mel_spectrogram(normalized, constants.SAMPLING_RATE)
        if(example.latent == 'lat2'):
            audio.display_spectrum(spectogram, example.sr, name=example.path.name + '_lat2cnn.png')
        cnn_pred = cnn_model(torch.tensor(np.array([[spectogram]])))
        _, cnn_pred = torch.max(cnn_pred,1)
        spectral_features = np.reshape(example.spectral_features, (1, -1))
        lda_pred = lda.predict(spectral_features)
        ramdom_c_pred = ramdom_c.predict(spectral_features)
        # log_features = audio.get_mel_coeffs(normalized, constants.SAMPLING_RATE)
        # log_features = np.reshape(log_features, (1, -1))
        logistic_pred = logistic.predict(spectral_features)
        predictions[example.path.name] = {
            "cnn" : cnn_pred, 
            "lda_pred" : lda_pred, 
            "ramdom_c_pred" : ramdom_c_pred, 
            "logistic_pred" : logistic_pred, 
        }
    print(predictions)
    util.save_obj(predictions, "predictions_on_generators")
    return predictions

def impact_of_dose_on_audio_features(data_set):
    results = {}
    for example in data_set:
        if(not example.latent in results):
            results[example.latent] = {}
        if(not example.features in results[example.latent]):
            results[example.latent][example.features] = {}
        if(not example.sample_num in results[example.latent][example.features]):
            results[example.latent][example.features][example.sample_num] = {"doses":{}}
        if(example.is_dose):
            results[example.latent][example.features][example.sample_num]["doses"][example.dose] = example
        else:
            results[example.latent][example.features][example.sample_num]["base"] = example
    data = results
    results = {}
    for latent in data:
        results[latent] = {}
        for feature in data[latent]:
            feature_values = {}
            count = 0
            for sample in data[latent][feature]:
                base = data[latent][feature][sample]["base"]
                doses = data[latent][feature][sample]["doses"]
                for dose in doses:
                    mel_dist = l2_norm(base.mel_coeffs, doses[dose].mel_coeffs)
                    spec_dist = l2_norm(base.spectral_features, doses[dose].spectral_features)
                    if(dose in feature_values):
                        feature_values[dose]["mel"]+=mel_dist
                        feature_values[dose]["spec"]+=spec_dist
                    else:
                        feature_values[dose] = {"mel" : mel_dist, "spec" : spec_dist}
                count+=1
            for dose in feature_values:
                feature_values[dose]["mel"] = feature_values[dose]["mel"] / count
                feature_values[dose]["spec"] = feature_values[dose]["spec"] / count
            results[latent][feature] = feature_values
    plot_and_save_latent_data(results)
    return results

def experiment():
    data_set = load_data()
    #results = impact_of_dose_on_audio_features(data_set)
    compare_classifiers(data_set)

    compare_class_predictive_qualities()
#here is what I want to do:

'''

For every sample:
compare it with every other sample

We know the latent is likelu


'''

experiment()