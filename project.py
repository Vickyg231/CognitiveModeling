# =========================
# 0. Imports
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from cmdstanpy import CmdStanModel
from scipy.stats import gaussian_kde

# =========================
# 1. Load data
# =========================
df = pd.read_csv("sleep_deprivation_dataset_detailed.csv")

# Subject IDs
df["id"] = df["Participant_ID"].astype("category").cat.codes + 1

# =========================
# 2. Select & standardize predictors
# =========================
predictors = [
    "Sleep_Hours",
    "Sleep_Quality_Score",
    "Daytime_Sleepiness"
]

for c in predictors:
    df[c] = (df[c] - df[c].mean()) / df[c].std()

# Rename for Stan
df["sleep"] = df["Sleep_Hours"]
df["quality"] = df["Sleep_Quality_Score"]
df["sleepiness"] = df["Daytime_Sleepiness"]

# =========================
# 3. LOG outcome (IMPORTANT)
# =========================
df["y"] = np.log(df["PVT_Reaction_Time"])

# =========================
# 4. Compile Stan model
# =========================
model = CmdStanModel(stan_file="prob3.stan")

# =========================
# 5. Prepare data
# =========================
data = {
    "N": len(df),
    "J": df["id"].nunique(),
    "id": df["id"].values,

    "sleep": df["sleep"].values,
    "sleep_quality": df["quality"].values,
    "sleepiness": df["sleepiness"].values,

    "y": df["y"].values
}

# =========================
# 6. Fit model (STABLE SETTINGS)
# =========================
fit = model.sample(
    data=data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.99,
    max_treedepth=15
)

# =========================
# 7. Model Evaluation (LOO-CV)
# =========================
idata = az.from_cmdstanpy(fit)

loo = az.loo(idata)
print(loo)

# Clean diagnostic check instead of Pareto k plot
k = loo.pareto_k

if (k > 0.7).any():
    print("\n⚠️ Warning: Some observations have high influence (Pareto k > 0.7).")
    print("Consider checking model fit or influential points.\n")
else:
    print("\n✅ LOO diagnostics look stable (no problematic observations).\n")

# =========================
# 8. Posterior Coefficient Plot
# =========================
az.plot_forest(
    idata,
    var_names=[
        "beta_sleep",
        "beta_quality",
        "beta_sleepiness",
        "beta_sleep_sleepiness"
    ],
    combined=True
)

plt.title("Posterior Effects on Reaction Time (log scale)")
plt.show()

# =========================
# 9. Posterior Predictive Check
# =========================
y_rep = fit.stan_variable("y_rep")

# Convert back to milliseconds
y_rep_ms = np.exp(y_rep)
y_obs_ms = df["PVT_Reaction_Time"].values

plt.figure()

for i in range(50):
    kde = gaussian_kde(y_rep_ms[i])
    x = np.linspace(min(y_rep_ms[i]), max(y_rep_ms[i]), 200)
    plt.plot(x, kde(x), alpha=0.15)

kde_obs = gaussian_kde(y_obs_ms)
x = np.linspace(min(y_obs_ms), max(y_obs_ms), 200)
plt.plot(x, kde_obs(x), linewidth=3, label="Observed")

plt.title("Posterior Predictive Check")
plt.xlabel("Reaction Time (ms)")
plt.ylabel("Density")
plt.legend()
plt.show()

# =========================
# 10. Residuals (CORRECT)
# =========================
mu = fit.stan_variable("mu")
y_pred = mu.mean(axis=0)

residuals = df["y"].values - y_pred

# Histogram
plt.figure()
plt.hist(residuals, bins=30, density=True)
plt.title("Residuals (log scale)")
plt.show()

# Residuals vs Sleep
plt.figure()
plt.scatter(df["sleep"], residuals)
plt.axhline(0)
plt.xlabel("Sleep (standardized)")
plt.ylabel("Residuals")
plt.title("Residuals vs Sleep")
plt.show()

# =========================
# 11. Metrics (back to ms)
# =========================
y_pred_ms = np.exp(y_pred)
y_obs_ms = df["PVT_Reaction_Time"].values

rmse = np.sqrt(np.mean((y_obs_ms - y_pred_ms) ** 2))
mae = np.mean(np.abs(y_obs_ms - y_pred_ms))

print(f"RMSE: {rmse:.2f} ms")
print(f"MAE: {mae:.2f} ms")