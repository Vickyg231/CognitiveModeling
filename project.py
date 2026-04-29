# =========================
# 0. Imports
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from cmdstanpy import CmdStanModel

# =========================
# 1. Load + preprocess
# =========================
df = pd.read_csv("sleep_deprivation_dataset_detailed.csv")

df["id"] = df["Participant_ID"].astype("category").cat.codes + 1

df["sleep"] = (df["Sleep_Hours"] - df["Sleep_Hours"].mean()) / df["Sleep_Hours"].std()
df["stress"] = (df["Stress_Level"] - df["Stress_Level"].mean()) / df["Stress_Level"].std()
df["y"] = (df["PVT_Reaction_Time"] - df["PVT_Reaction_Time"].mean()) / df["PVT_Reaction_Time"].std()

# =========================
# 2. Stan data
# =========================
base_data = {
    "N": len(df),
    "J": df["id"].nunique(),
    "id": df["id"].values,
    "sleep": df["sleep"].values,
    "stress": df["stress"].values,
    "y": df["y"].values,
}

# =========================
# 3. Compile model
# =========================
model = CmdStanModel(stan_file="prob3.stan")

# =========================
# 4. Fit all models
# =========================
fits = {}
idatas = {}

for m in [1, 2, 3]:
    print(f"\nFitting model {m}...")

    data = base_data.copy()
    data["model_type"] = m

    fit = model.sample(
        data=data,
        chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        parallel_chains=1,
        show_console=False
    )

    fits[m] = fit

    idatas[m] = az.from_cmdstanpy(
        fit,
        log_likelihood="log_lik"
    )

# =========================
# 5. Model comparison (LOO)
# =========================
comparison = az.compare({
    "baseline": idatas[1],
    "nonlinear": idatas[2],
    "interaction": idatas[3],
})

print("\n===== MODEL COMPARISON =====")
print(comparison)

az.plot_compare(comparison)
plt.title("Model Comparison (LOO)")
plt.show()

# =========================
# 6. Posterior predictive checks
# =========================
y_rep = {m: fits[m].stan_variable("y_rep") for m in fits}

plt.figure()

plt.hist(df["y"], bins=30, density=True, alpha=0.5, label="Observed")

for m, name in zip([1, 2, 3], ["Baseline", "Nonlinear", "Interaction"]):
    plt.hist(
        y_rep[m].mean(axis=0),
        bins=30,
        alpha=0.3,
        density=True,
        label=name
    )

plt.legend()
plt.title("Posterior Predictive Comparison")
plt.show()

# =========================
# 7. Residual comparison
# =========================
plt.figure()

obs = df["y"].values

for m, name in zip([1, 2, 3], ["Baseline", "Nonlinear", "Interaction"]):
    pred = y_rep[m].mean(axis=0)
    residuals = obs - pred

    plt.hist(residuals, bins=30, alpha=0.4, density=True, label=name)

plt.legend()
plt.title("Residual Distributions")
plt.show()

# =========================
# 8. RMSE comparison
# =========================
print("\n===== RMSE =====")

for m, name in zip([1, 2, 3], ["Baseline", "Nonlinear", "Interaction"]):
    pred = y_rep[m].mean(axis=0)
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    print(f"{name}: {rmse:.4f}")

# =========================
# 9. Final best model
# =========================
best = comparison.index[0]
print("\nBEST MODEL:", best)