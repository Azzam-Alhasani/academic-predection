"""
The candidate models, with the hyper-parameter grids used for tuning.

One place, shared by the offline model-selection study (app/ml/model_selection.py) and the
live application (app/ml/predictor.py), so the app always uses a model that was actually
evaluated.  `complexity` is used by the parsimony rule: among models that are statistically
tied, the simplest one is chosen.
"""

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42

# key: (label, factory, tuning grid, complexity rank, one-line description)
CLASSIFIERS = {
    "majority": ("Majority-class baseline", lambda: DummyClassifier(strategy="prior"), {}, 0,
                 "Always predicts the most common outcome; the bar every real model must beat."),
    "logistic": ("Logistic Regression", lambda: LogisticRegression(max_iter=3000),
                 {"C": [0.03, 0.1, 0.3, 1, 3, 10]}, 1,
                 "Linear model; coefficients are easy to explain to students and staff."),
    "naive_bayes": ("Naive Bayes", lambda: GaussianNB(), {"var_smoothing": [1e-9, 1e-6, 1e-3]}, 1,
                    "Probabilistic model that assumes independent features."),
    "decision_tree": ("Decision Tree", lambda: DecisionTreeClassifier(random_state=SEED),
                      {"max_depth": [3, 4, 5, 7], "min_samples_leaf": [5, 20, 50]}, 2,
                      "A readable set of if-then rules."),
    "knn": ("k-Nearest Neighbours", lambda: KNeighborsClassifier(),
            {"n_neighbors": [11, 25, 51, 101], "weights": ["uniform", "distance"]}, 3,
            "Predicts from the most similar past students."),
    "svm_rbf": ("Support Vector Machine (RBF)", lambda: SVC(kernel="rbf"),
                {"C": [0.3, 1, 3], "gamma": ["scale", 0.01]}, 4,
                "Non-linear boundary; strong on small tables, slower to train."),
    "random_forest": ("Random Forest", lambda: RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
                      {"max_depth": [6, 10, None], "min_samples_leaf": [2, 5, 10]}, 5,
                      "Many decision trees voting together."),
    "extra_trees": ("Extra Trees", lambda: ExtraTreesClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
                    {"max_depth": [8, None], "min_samples_leaf": [2, 5, 10]}, 5,
                    "Like a random forest with extra randomness."),
    "gradient_boosting": ("Gradient Boosting", lambda: HistGradientBoostingClassifier(random_state=SEED),
                          {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, None],
                           "l2_regularization": [0.0, 1.0]}, 6,
                          "Trees built one after another, each fixing the last one's errors."),
    "neural_net": ("Neural Network (MLP)", lambda: MLPClassifier(max_iter=600, early_stopping=True, random_state=SEED),
                   {"hidden_layer_sizes": [(32,), (64, 32)], "alpha": [1e-4, 1e-2]}, 7,
                   "Small multi-layer network."),
}

REGRESSORS = {
    "mean": ("Mean baseline", lambda: DummyRegressor(strategy="mean"), {}, 0,
             "Always predicts the average mark."),
    "linear": ("Linear Regression", lambda: LinearRegression(), {}, 1, "Straight-line relationship."),
    "ridge": ("Ridge Regression", lambda: Ridge(), {"alpha": [0.1, 1, 10, 100]}, 1,
              "Linear regression with a penalty against over-fitting."),
    "knn": ("k-Nearest Neighbours", lambda: KNeighborsRegressor(),
            {"n_neighbors": [11, 25, 51], "weights": ["uniform", "distance"]}, 3, "Average of similar students."),
    "random_forest": ("Random Forest", lambda: RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
                      {"max_depth": [6, 10, None], "min_samples_leaf": [2, 5, 10]}, 5, "Many trees averaged."),
    "gradient_boosting": ("Gradient Boosting", lambda: HistGradientBoostingRegressor(random_state=SEED),
                          {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, None],
                           "l2_regularization": [0.0, 1.0]}, 6, "Trees built one after another."),
}


def build_estimator(registry: dict, key: str, params: dict | None = None):
    """A fresh estimator for `key`, with tuned parameters applied if given."""
    estimator = registry[key][1]()
    if params:
        estimator.set_params(**params)
    return estimator
