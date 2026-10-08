# Practical No. 1
# Aim: Data Pre-processing and Exploration.

# Step 1: Load the Iris Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv("E:/78008/ML/datasets/iris.csv", header=None, names=column_names, skiprows=1)
print(iris_data.head())

# Step 2: Handle Missing Values and Inconsistent Formatting
print(iris_data.isnull().sum())

# Step 3: Calculate Descriptive Summary Statistics
print(iris_data.describe())

# Step 4: Create Visualizations
iris_data.hist(bins=10, figsize=(10, 8))
plt.suptitle('Histograms of Iris Dataset Features')
plt.show()
sns.pairplot(iris_data, hue='species')
plt.title('Pairplot of Iris Dataset')
plt.show()
plt.figure(figsize=(12, 6))
for i, feature in enumerate(column_names[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='species', y=feature, data=iris_data)
    plt.title(f'Boxplot of {feature} by Species')
plt.tight_layout()
plt.show()

# Step 5: Pre-processing Routines
from sklearn.preprocessing import LabelEncoder, StandardScaler, Binarizer
label_encoder = LabelEncoder()
iris_data['species'] = label_encoder.fit_transform(iris_data['species'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(iris_data.iloc[:, :-1]) # Exclude target variable
scaled_iris_data = pd.DataFrame(scaled_features, columns=column_names[:-1])
scaled_iris_data['species'] = iris_data['species']
binarizer = Binarizer(threshold=0.5)
binarized_features = binarizer.fit_transform(scaled_iris_data.iloc[:, :-1])
binarized_iris_data = pd.DataFrame(binarized_features, columns=column_names[:-1])
binarized_iris_data['species'] = scaled_iris_data['species']
print("Scaled Iris Data:")
print(scaled_iris_data.head())
print("\nBinarized Iris Data:")
print(binarized_iris_data.head())
