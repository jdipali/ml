# Practical No. 7
# Aim: Model Evaluation and Hyperparameter Tuning OC3, OC4, OC5.

# ======== Cross-validation techniques for robust model evaluation ========

# ---------------- K-Fold Cross-Validation ----------------
import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier()
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
accuracies = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)
print(f'K-Fold Cross-Validation Accuracies: {accuracies}')
print(f'Mean Accuracy: {np.mean(accuracies)}')

# ---------------- Stratified K-Fold Cross-Validation ----------------
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
stratified_accuracies = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    stratified_accuracies.append(accuracy)
print(f'Stratified K-Fold Cross-Validation Accuracies: {stratified_accuracies}')
print(f'Mean Accuracy: {np.mean(stratified_accuracies)}')

# ---------------- Hyperparameter Tuning with Cross-Validation ----------------
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 
10]}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=skf, 
n_jobs=-1)
grid_search.fit(X, y)
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Cross-Validation Accuracy: {grid_search.best_score_}')

# ======== Combinations of hyperparameters to optimise model performance ========

# ---------------- Grid Search ----------------
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-
1)
grid_search.fit(X, y)
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Cross-Validation Accuracy: {grid_search.best_score_}')

# ---------------- Randomized Search ----------------
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 20)
}
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, 
scoring='accuracy', cv=5, n_jobs=-1, random_state=42)
random_search.fit(X, y)
print(f'Best Parameters: {random_search.best_params_}')
print(f'Best Cross-Validation Accuracy: {random_search.best_score_}')
