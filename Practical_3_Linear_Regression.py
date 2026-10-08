# Practical No. 3
# Aim: Linear Regression.

# ---------------- Simple Linear Regression ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("D:/MSc IT/Sem 3/ML/datasets/Iris.csv")
print(data.head())
X = data[["SepalLengthCm"]]
y = data["SepalWidthCm"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
intercept = model.intercept_
slope = model.coef_[0]
print(f"Intercept: {intercept}")
print(f"Slope: {slope}")
y_pred = model.predict(X_test)
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R-squared: {r_squared}")
print(f"Mean Squared Error: {mse}")
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", label="Predicted")
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.title("Linear Regression Results")
plt.legend()
plt.show()

# ---------------- Multiple Linear Regression ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

iris = sns.load_dataset("iris")
print(iris.head())
X = iris[["sepal_length", "sepal_width", "petal_width"]]
y = iris["petal_length"]


def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    return vif_data


vif_data = calculate_vif(X)
print(vif_data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
intercept = model.intercept_
coefficients = model.coef_
print(f"Intercept: {intercept}")
for feature, coef in zip(X.columns, coefficients):
    print(f"Coefficient for {feature}: {coef}")
y_pred = model.predict(X_test)
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R-squared: {r_squared}")
print(f"Mean Squared Error: {mse}")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="blue", label="Predicted vs Actual")
plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    color="red",
    linestyle="--",
    label="Perfect Prediction",
)
plt.xlabel("Actual Petal Length")
plt.ylabel("Predicted Petal Length")
plt.title("Actual vs Predicted Petal Length")
plt.legend()
plt.show()

# ---------------- Regularized Linear Models ----------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

iris = sns.load_dataset("iris")
print(iris.head())
X = iris[["sepal_length", "sepal_width", "petal_width"]]
y = iris["petal_length"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)
print(f"Ridge Regression - Mean Squared Error: {ridge_mse}")
print(f"Ridge Regression - R-squared: {ridge_r2}")
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_r2 = r2_score(y_test, y_pred_lasso)
print(f"Lasso Regression - Mean Squared Error: {lasso_mse}")
print(f"Lasso Regression - R-squared: {lasso_r2}")
print("\nComparison of Ridge and Lasso Regression:")
print(f"Ridge MSE: {ridge_mse}, R-squared: {ridge_r2}")
print(f"Lasso MSE: {lasso_mse}, R-squared: {lasso_r2}")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_ridge, color="blue", label="Predicted vs Actual")
plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    color="red",
    linestyle="--",
    label="Perfect Prediction",
)
plt.xlabel("Actual Petal Length")
plt.ylabel("Predicted Petal Length")
plt.title("Ridge Regression: Actual vs Predicted")
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_lasso, color="green", label="Predicted vs Actual")
plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    color="red",
    linestyle="--",
    label="Perfect Prediction",
)
plt.xlabel("Actual Petal Length")
plt.ylabel("Predicted Petal Length")
plt.title("Lasso Regression: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()
