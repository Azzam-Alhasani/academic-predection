"""
The candidate models, with the hyper-parameter grids used for tuning.

One place, shared by the offline model-selection study (app/ml/model_selection.py) and the
live application (app/ml/predictor.py), so the app always uses a model that was actually
evaluated.  `complexity` is used by the parsimony rule: among models that are statistically
tied, the simplest one is chosen.

The project deliberately compares only FOUR model families, the same four for both tasks, going
from simplest to most powerful:

  1. a straight-line model   (Logistic Regression for pass/fail, Linear Regression for the mark)
  2. one Decision Tree       (a readable list of if-then rules)
  3. a Random Forest         (many trees that vote)
  4. Gradient Boosting       (trees built one after another, each fixing the last one's mistakes)

The "baseline" entry is not a model: it always gives the most common answer and is the bar every
real model has to beat.
"""

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 42

# key: (label, factory, tuning grid, complexity rank, one-line plain-language description)
CLASSIFIERS = {
    "majority": ("Baseline: always guess \"pass\"", lambda: DummyClassifier(strategy="prior"), {}, 0,
                 "Not a real model. Always predicts the most common outcome; every real model must beat it."),
    "logistic": ("Logistic Regression", lambda: LogisticRegression(max_iter=3000),
                 {"C": [0.03, 0.1, 0.3, 1, 3, 10]}, 1,
                 "Gives each input a weight (for example, low CGPA adds risk) and adds them up into a chance of failing."),
    "decision_tree": ("Decision Tree", lambda: DecisionTreeClassifier(random_state=SEED),
                      {"max_depth": [3, 4, 5, 7], "min_samples_leaf": [5, 20, 50]}, 2,
                      "Asks a short list of yes/no questions, e.g. \"Is CGPA below 2.0? Is attendance below 75%?\""),
    "random_forest": ("Random Forest", lambda: RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
                      {"max_depth": [6, 10, None], "min_samples_leaf": [2, 5, 10]}, 3,
                      "Builds 200 different decision trees and lets them vote."),
    "gradient_boosting": ("Gradient Boosting", lambda: HistGradientBoostingClassifier(random_state=SEED),
                          {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, None],
                           "l2_regularization": [0.0, 1.0]}, 4,
                          "Builds trees one after another; each new tree fixes the mistakes of the ones before."),
}

REGRESSORS = {
    "mean": ("Baseline: always guess the average", lambda: DummyRegressor(strategy="mean"), {}, 0,
             "Not a real model. Always predicts the average mark."),
    "linear": ("Linear Regression", lambda: LinearRegression(), {}, 1,
               "Draws the best straight-line relationship between the inputs and the final mark."),
    "decision_tree": ("Decision Tree", lambda: DecisionTreeRegressor(random_state=SEED),
                      {"max_depth": [3, 4, 5, 7], "min_samples_leaf": [5, 20, 50]}, 2,
                      "Splits students into groups with yes/no questions and predicts each group's average mark."),
    "random_forest": ("Random Forest", lambda: RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
                      {"max_depth": [6, 10, None], "min_samples_leaf": [2, 5, 10]}, 3,
                      "Averages the marks predicted by 200 different trees."),
    "gradient_boosting": ("Gradient Boosting", lambda: HistGradientBoostingRegressor(random_state=SEED),
                          {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, None],
                           "l2_regularization": [0.0, 1.0]}, 4,
                          "Trees built one after another, each correcting the last one's errors."),
}


def build_estimator(registry: dict, key: str, params: dict | None = None):
    """A fresh estimator for `key`, with tuned parameters applied if given."""
    estimator = registry[key][1]()
    if params:
        estimator.set_params(**params)
    return estimator
