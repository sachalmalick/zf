from loader import *
from preprocess import *
import audioutils as audio
import cnn
import numpy as np
import classic_ml as cml
from sklearn.metrics import accuracy_score
import util
from joblib import dump, load
import constants

def train_and_save_classifiers(x, y):
    # print("Running lda")
    # scores = cml.lda(x, y)
    # lda = scores["model"]
    # dump(lda, 'lda_mel.joblib') 

    print("Running logistic regression")
    scores = cml.logistic_regression(x, y, name="mel_coef_300")
    logistic = scores["model"]
    print(scores)
    dump(logistic, 'logistic_mel.joblib') 

    # print("Running random forrest")
    # scores = cml.random_forest(x, y)
    # random_c = scores["model"]
    # dump(random_c, 'random_c_mel.joblib')

def generate_call_type_graphs(data_set):
    visited = []
    for i in range(0, len(data_set.dataset.examples)):
        example = data_set.dataset.examples[i]
        if(not example.call_type in visited):
            wave = data_set.normalized_waves[i]
            sg = audio.create_mel_spectrogram(wave, constants.SAMPLING_RATE)
            audio.display_spectrum(sg, constants.SAMPLING_RATE, name=str(example.call_type))
            visited.append(example.call_type)
    return visited

def effect_of_randomization_on_classification(x, y, logistic, lda, random_c, cnn_model):
    for i in range(0, x.shape[1]):
        print("Testing feature ", i)
        print()
        print("Logistic regression")
        print("Baseline")
        
        x_rand = util.randomize_feature(x, i)
        y_pred = logistic.predict(x_rand)
        accuracy = accuracy_score(y, y_pred)
        print("logistic accuracy", accuracy)

        y_pred = lda.predict(x_rand)
        accuracy = accuracy_score(y, y_pred)
        print("lda accuracy", accuracy)

        y_pred = random_c.predict(x_rand)
        accuracy = accuracy_score(y, y_pred)
        print("random forrest accuracy", accuracy)

        # model = cnn_model
        # y_pred = model(x_rand)
        # accuracy = accuracy_score(y, y_pred)
        # print("cnn accuracy", cnn.evaluate_model(model, [x_rand, y]))


def experiment(dataset):
    features_list = [
        "chroma_stft",
        "chroma_cens",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_contrast",
        "spectral_rolloff",
        "spectral_flatness",
        "tempo"
    ]

    x, y = dataset.get_mel_coefficients_matrix()
    print(x.shape)
    train_and_save_classifiers(x, y)

    #print("Loading CNN")
    # print("CNN Scores", cnn.evaluate_model(cnn_model, [waves, y]))


    # print("Running random forrest")
    # scores = cml.random_forest(x, y, name = feature + "random_forrest.png")
    # random_c = scores["model"]
    # print("random forrest scores", scores)
    # x = dataset.spectral_means
    # train_and_save_classifiers()


    # for index, feature in enumerate(features_list):
    #     print("Runnign tests for feature", feature)
    #     x, y = dataset.get_spectral_feature_matrix(index)
    #     train_and_save_classifiers(x, y)

    #     print("Loading CNN")
    #     # print("CNN Scores", cnn.evaluate_model(cnn_model, [waves, y]))


    #     print("Running random forrest")
    #     scores = cml.random_forest(x, y, name = feature + "random_forrest.png")
    #     random_c = scores["model"]
    #     print("random forrest scores", scores)

    #     print()

    #effect_of_randomization_on_classification(x, y, logistic, lda, random_c, cnn_model)
    # print("Running kmeans")
    # kmeans_x = cml.kmeans(waves)

    # print("Running pca")
    # pca_x= cml.pca(waves)

    # print("Running logistic regression with pca")
    # scores = cml.logistic_regression(pca_x, y)
    # print(scores)

    # print("Running logistic regression with kmeans")
    # scores = cml.logistic_regression(kmeans_x, y)
    # print(scores)
    # print("Running random forrest with pca")
    # scores = cml.random_forest(pca_x, y)
    # print(scores)

    # print("Running random forrest with kmeans")
    # scores = cml.random_forest(kmeans_x, y)
    # print(scores)



    # print("Random forrest with grid search")
    # scores = cml.random_forest(x, y)
    # print(scores)

if __name__ == "__main__":
    print("loading raw dataset")
    dataset = load_dataset()

    print("loading audio data")
    dataset = ZFinchDataProcessor(dataset)
    dataset.load_audio_data()
    
    print("Normalizing audio data")
    dataset.noramlize_audio_data()
    #generate_call_type_graphs(dataset)

    print(audio.get_spectral_feature_means(dataset.normalized_waves[111], 22050))
    experiment(dataset)


    # print("Training CNN!")
    # cnn.train_and_evaluate(dataset)
    # #print()
    # audio.display_spectrum(audio.create_mel_spectrogram(dataset.normalized_waves[111], 22050), 22050)