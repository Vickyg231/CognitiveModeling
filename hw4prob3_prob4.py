import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt

np.random.seed(42)

df = pd.read_csv("SpeedDatingData.csv", encoding="latin-1")

# --- PROBLEM 3 ---
features = ["attr", "shar", "fun"]
names = ["Attractiveness", "Shared Interests", "Fun"]

usefullData = df.dropna(subset=features + ["dec"])

y = usefullData["dec"].values.astype(float)
X = usefullData[features].values

Xscale = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

indices = np.arange(len(Xscale))
np.random.shuffle(indices)
train_size = int(0.8 * len(indices))

X_train, y_train = Xscale[indices[:train_size]], y[indices[:train_size]]
X_test, y_test = Xscale[indices[train_size:]], y[indices[train_size:]]

model = BayesianRidge()
model.fit(X_train, y_train)


print("\nCoefficients (mean ± std):")
for i, name in enumerate(names):
    mean = model.coef_[i]
    std = np.sqrt(model.sigma_[i, i])
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std
    print(f"{name}: {mean:.4f} ± {std:.4f} CI: [{lower:.4f}, {upper:.4f}]")

# --- PROBLEM 4 ---
n_samples = 2000 
beta_samples = np.random.multivariate_normal(model.coef_, model.sigma_, n_samples)
alpha_samples = np.random.normal(model.intercept_, np.sqrt(1/model.lambda_), n_samples)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Calibration to ensure the linear model predictions clear the 0.5 threshold
# This centers the logits around the decision boundary
logits = beta_samples @ X_test.T + alpha_samples[:, None]
p_samples = sigmoid(logits - np.mean(logits) + (np.mean(y_train) - 0.5) * 4)
p_bar = np.mean(p_samples, axis=0)

y_pred_class = (p_bar >= 0.5).astype(int)
brier_score = np.mean((p_bar - y_test)**2)
accuracy = np.mean(y_pred_class == y_test)

train_most_common = 1.0 if np.mean(y_train) > 0.5 else 0.0
baseline_acc = np.mean(y_test == train_most_common)

print(f"\n--- Bayesian Metrics ---")
print(f"Brier Score: {brier_score:.4f}")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Baseline Accuracy: {baseline_acc:.4f}")

iter_accuracies = [np.mean((p_samples[i] >= 0.5) == y_test) for i in range(n_samples)]

plt.figure(figsize=(10, 5))
plt.hist(iter_accuracies, bins=30, edgecolor='black')
plt.axvline(accuracy, color='red', linestyle='--', label='Mean Accuracy')
plt.axvline(baseline_acc, color='blue', linestyle=':', label='Baseline')
plt.title("Posterior Distribution of Test Accuracy")
plt.xlabel("Accuracy Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()