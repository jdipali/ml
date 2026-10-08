# Practical No. 5
# Aim: Generative Models OC2, OC6.

# ---------------- Naive Bayesian classifier ----------------
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
new_sample = np.array([[5.0, 3.5, 1.5, 0.2]])
predicted_class = nb_classifier.predict(new_sample)
predicted_class_name = iris.target_names[predicted_class][0]
print(f"\nPredicted class for the new sample {new_sample[0]}: {predicted_class_name}")

# ---------------- Hidden Markov Models using hmmlearn ----------------
import numpy as np
from hmmlearn import hmm
states = ["Rainy", "Sunny"]
n_states = len(states)
observations = ["Walk", "Shop", "Clean"]
n_observations = len(observations)
obs_map = {obs: i for i, obs in enumerate(observations)}
X = np.array([[obs_map["Walk"], obs_map["Shop"], obs_map["Clean"]], [obs_map["Walk"], 
obs_map["Walk"], obs_map["Shop"]], [obs_map["Clean"], obs_map["Walk"], obs_map["Walk"]], 
[obs_map["Shop"], obs_map["Clean"], obs_map["Walk"]]])
X = np.concatenate([X[i].reshape(-1, 1) for i in range(X.shape[0])])
model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, random_state=42)
model.startprob_ = np.array([0.6, 0.4]) 
model.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]]) 
model.emissionprob_ = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
model.fit(X)
hidden_states = model.predict(X)
print("Observed Activities:")
print([observations[i] for i in X.flatten()])
print("\nPredicted Hidden States:")
print([states[i] for i in hidden_states])
