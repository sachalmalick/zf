from loader import *
from preprocess import *
import audioutils as audio
import cnn
import numpy as np
import classic_ml as cml
from sklearn.metrics import accuracy_score
import util

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
    x, y = dataset.get_spectral_feature_matrix()
    waves = dataset.normalized_waves

    print("Loading CNN")
    cnn_model = cnn.load_model()
    # print("CNN Scores", cnn.evaluate_model(cnn_model, [waves, y]))

    print("Running lda")
    scores = cml.lda(x, y)
    lda = scores["model"]
    print("lda scores", scores)

    print("Running logistic regression")
    scores = cml.logistic_regression(x, y)
    logistic = scores["model"]
    print("logistic scores", scores)

    print("Running random forrest")
    scores = cml.random_forest(x, y)
    random_c = scores["model"]
    print("random forrest scores", scores)

    print()

    effect_of_randomization_on_classification(x, y, logistic, lda, random_c, cnn_model)
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

    print(audio.get_spectral_feature_means(dataset.normalized_waves[111], 22050))
    experiment(dataset)


    print("Training CNN!")
    #cnn.train_and_evaluate(preprocess)
    #print()
    #display_spectrum(create_mel_spectrogram(preprocess.normalized_waves[111], 22050), 22050)