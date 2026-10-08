# Practical No. 8
# Aim: Bayesian Learning using inferences.

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def objective_function(n_estimators, max_depth):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators), max_depth=int(max_depth)
    )
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()


pbounds = {"n_estimators": (10, 200), "max_depth": (1, 30)}
optimizer = BayesianOptimization(f=objective_function, pbounds=pbounds, random_state=1)
optimizer.maximize(init_points=5, n_iter=25)
best_params = optimizer.max["params"]
print(f"Best Hyperparameters: {best_params}")
