# Practical No. 6
# Aim: Probabilistic Models.

# ---------------- Bayesian Linear Regression ----------------
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 100 
D = 1
X = np.random.randn(N, D)
true_theta = np.array([2.0])
sigma = 1.0
y = X @ true_theta + sigma * np.random.randn(N)
mu_0 = np.zeros(D) 
Sigma_0 = np.eye(D) * 10 
sigma_sq = sigma**2 
Sigma_p = np.linalg.inv((1 / sigma_sq) * X.T @ X + np.linalg.inv(Sigma_0))
mu_p = Sigma_p @ ((1 / sigma_sq) * X.T @ y + np.linalg.inv(Sigma_0) @ mu_0)
prior_samples = np.random.multivariate_normal(mu_0, Sigma_0, 1000)
posterior_samples = np.random.multivariate_normal(mu_p.flatten(), Sigma_p, 1000)
plt.figure(figsize=(10, 6))
plt.hist(prior_samples, bins=50, alpha=0.5, label="Prior", density=True)
plt.hist(posterior_samples, bins=50, alpha=0.5, label="Posterior", density=True)
plt.axvline(true_theta, color='red', linestyle='--', label="True Theta")
plt.xlabel("Theta")
plt.ylabel("Density")
plt.title("Prior and Posterior Distributions of Theta")
plt.legend()
plt.show()
print("Posterior Mean:", mu_p)
print("Posterior Covariance:", Sigma_p)

# ---------------- Gaussian Mixture Models for density estimation and unsupervised clustering ----------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-6):
        self.n_components = n_components 
        self.max_iter = max_iter 
        self.tol = tol 
        self.weights = None 
        self.means = None 
        self.covariances = None 
        self.responsibilities = None 
    def fit(self, X):
        n_samples, n_features = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = [np.eye(n_features) for _ in range(self.n_components)]
        log_likelihood = 0
        for iteration in range(self.max_iter):
            responsibilities = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(X, mean=self.means[k], 
cov=self.covariances[k])
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)
            Nk = responsibilities.sum(axis=0)
            self.weights = Nk / n_samples
            self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]
            new_log_likelihood = 0
            for k in range(self.n_components):
                new_log_likelihood += self.weights[k] * multivariate_normal.pdf(X, mean=self.means[k], 
cov=self.covariances[k])
            new_log_likelihood = np.log(new_log_likelihood).sum()
            if np.abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood
        self.responsibilities = responsibilities
    def predict(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(X, mean=self.means[k], 
cov=self.covariances[k])
        return np.argmax(responsibilities, axis=1)
np.random.seed(42)
n_samples = 500
X1 = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=n_samples)
X2 = np.random.multivariate_normal(mean=[5, 5], cov=[[1, 0], [0, 1]], size=n_samples)
X = np.vstack((X1, X2))
gmm = GaussianMixtureModel(n_components=2)
gmm.fit(X)
labels = gmm.predict(X)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.scatter(gmm.means[:, 0], gmm.means[:, 1], c='red', marker='x', s=200, label="Centers")
plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
