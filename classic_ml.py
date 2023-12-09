from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def logistic_regression(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    print(X_train)
    hp = {
        "C" : 1.0
    }
    model = LogisticRegression(random_state=0, max_iter=8000, C=hp["C"], penalty='l2').fit(X_train, y_train)
    cross_validation_score = cross_val_score(model, X, y, cv=3)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    # Create a learning curve display
    lc_display = LearningCurveDisplay.from_estimator(
        model, X_train, y_train, cv=cv, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5), ax=ax)
    
    # Plot the learning curve
    lc_display.plot()

    # Set plot labels and title
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve for Logistic Regression")

    # Display the learning curve
    #plt.show()

    return {
        "accuracy_score" : accuracy,
        "cross_validation_score" : cross_validation_score,
        "model" : model
    }

def pca(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA()
    param_grid = {
        'n_components': [5, 20, 100, 1000]
    }
    grid_search = GridSearchCV(pca, param_grid, cv=5)
    grid_search.fit(X)
    
    best_num_components = grid_search.best_params_['n_components']
    print(f"Best number of components: {best_num_components}")
    pca = grid_search.best_estimator_
    X_transformed = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)
    print(pca.get_covariance())
    print(X_transformed.shape)
    return X_transformed

def kmeans(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    param_grid = {
        'n_clusters': [5, 20, 100]
    }
    kmeans = KMeans()
    grid_search = GridSearchCV(kmeans, param_grid, cv=5)
    grid_search.fit(X)
    best_num_components = grid_search.best_params_['n_clusters']
    print(f"Best number of clusters: {best_num_components}")

    kmeans = grid_search.best_estimator_
    X_transformed = kmeans.fit_transform(X)
    print(X_transformed.shape)
    return X_transformed

def lda(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    print(X_train)
    model = LinearDiscriminantAnalysis().fit(X_train, y_train)
    cross_validation_score = cross_val_score(model, X, y, cv=3)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    # Create a learning curve display
    lc_display = LearningCurveDisplay.from_estimator(
        model, X_train, y_train, cv=cv, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5), ax=ax)
    
    # Plot the learning curve
    lc_display.plot()

    # Set plot labels and title
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve for Linear Descriminant Analysis")

    # Display the learning curve
    #plt.show()

    return {
        "accuracy_score" : accuracy,
        "cross_validation_score" : cross_validation_score,
        "model" : model
    }



def random_forest(X, y):
    param_grid = {'max_depth': [3, 5, 10],
                 'min_samples_split': [2, 5, 10]}
    base_estimator = RandomForestClassifier(random_state=0)
    sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                          factor=2, resource='n_estimators',
                          max_resources=30).fit(X, y)
    model = sh.best_estimator_

    cross_validation_score = cross_val_score(model, X, y, cv=3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    # Create a learning curve display
    lc_display = LearningCurveDisplay.from_estimator(
        model, X_train, y_train, cv=cv, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5), ax=ax)
    
    # Plot the learning curve
    lc_display.plot()

    # Set plot labels and title
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve for Random Forrest Classifier")

    # Display the learning curve
    plt.show()

    return {
        "accuracy_score" : accuracy,
        "cross_validation_score" : cross_validation_score,
        "model" : model
    }
