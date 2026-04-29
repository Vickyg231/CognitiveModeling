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

df["id"] = df["Participant_ID"].astype("category").cat.codes + 1

# =========================
# 2. Standardize predictors
# =========================
predictors = [
    "Sleep_Hours",
    "Sleep_Quality_Score",
    "Daytime_Sleepiness",
    "Stress_Level",
    "Caffeine_Intake",
    "Physical_Activity_Level",
    "Age",
    "BMI"
]

for c in predictors:
    df[c] = (df[c] - df[c].mean()) / df[c].std()

# rename for Stan
df["sleep"] = df["Sleep_Hours"]
df["quality"] = df["Sleep_Quality_Score"]
df["sleepiness"] = df["Daytime_Sleepiness"]
df["stress"] = df["Stress_Level"]
df["caffeine"] = df["Caffeine_Intake"]
df["activity"] = df["Physical_Activity_Level"]
df["age"] = df["Age"]
df["bmi"] = df["BMI"]

# =========================
# 3. Outcome scaling
# =========================
y_mean = df["PVT_Reaction_Time"].mean()
y_std = df["PVT_Reaction_Time"].std()

df["y"] = (df["PVT_Reaction_Time"] - y_mean) / y_std

# =========================
# 4. Compile Stan model (single file)
# =========================
model = CmdStanModel(stan_file="prob3.stan")

# =========================================================
# 5. BASELINE MODEL
# =========================================================
data_base = {
    "N": len(df),
    "J": df["id"].nunique(),
    "id": df["id"].values,

    "sleep": df["sleep"].values,
    "sleep_quality": df["quality"].values,
    "sleepiness": df["sleepiness"].values,
    "stress": df["stress"].values,
    "caffeine": df["caffeine"].values,
    "activity": df["activity"].values,
    "age": df["age"].values,
    "bmi": df["bmi"].values,

    "y": df["y"].values
}

fit_base = model.sample(data=data_base)

# =========================================================
# 6. NONLINEAR MODEL
# =========================================================
df["sleep_sq"] = df["sleep"] ** 2
df["stress_sq"] = df["stress"] ** 2
df["caffeine_sq"] = df["caffeine"] ** 2

data_nonlin = {
    "N": len(df),
    "J": df["id"].nunique(),
    "id": df["id"].values,

    "sleep": df["sleep_sq"].values,
    "sleep_quality": df["quality"].values,
    "sleepiness": df["sleepiness"].values,
    "stress": df["stress_sq"].values,
    "caffeine": df["caffeine_sq"].values,
    "activity": df["activity"].values,
    "age": df["age"].values,
    "bmi": df["bmi"].values,

    "y": df["y"].values
}

fit_nonlin = model.sample(data=data_nonlin)

# =========================================================
# 7. INTERACTION MODEL
# =========================================================
df["sleep_stress"] = df["sleep"] * df["stress"]
df["sleep_caffeine"] = df["sleep"] * df["caffeine"]

data_inter = {
    "N": len(df),
    "J": df["id"].nunique(),
    "id": df["id"].values,

    "sleep": df["sleep"].values,
    "sleep_quality": df["quality"].values,
    "sleepiness": df["sleepiness"].values,
    "stress": df["stress"].values,
    "caffeine": df["caffeine"].values,

    "activity": df["sleep_stress"].values,
    "age": df["sleep_caffeine"].values,

    "bmi": df["bmi"].values,

    "y": df["y"].values
}

fit_inter = model.sample(data=data_inter)

# =========================================================
# 8. LOO comparison
# =========================================================
idata_base = az.from_cmdstanpy(fit_base)
idata_nonlin = az.from_cmdstanpy(fit_nonlin)
idata_inter = az.from_cmdstanpy(fit_inter)

loo_base = az.loo(idata_base)
loo_nonlin = az.loo(idata_nonlin)
loo_inter = az.loo(idata_inter)

loo_scores = {
    "Baseline": loo_base.elpd_loo,
    "Nonlinear": loo_nonlin.elpd_loo,
    "Interaction": loo_inter.elpd_loo
}

# =========================================================
# 9. LOO LINE PLOT
# =========================================================
plt.figure()

models = list(loo_scores.keys())
values = list(loo_scores.values())

plt.plot(models, values, marker="o", linewidth=2)

plt.title("Sleep Deprivation Models: LOO Comparison")
plt.ylabel("ELPD-LOO (higher = better)")

for i, v in enumerate(values):
    plt.text(i, v, f"{v:.1f}", ha="center", va="bottom")

plt.grid(alpha=0.3)
plt.show()

print("Best model:", max(loo_scores, key=loo_scores.get))

# =========================================================
# 10. Posterior Predictive (use interaction model)
# =========================================================
y_rep = fit_inter.stan_variable("y_rep")

y_pred_z = y_rep.mean(axis=0)

y_obs_ms = df["PVT_Reaction_Time"].values
y_pred_ms = y_pred_z * y_std + y_mean
y_rep_ms = y_rep * y_std + y_mean

# =========================
# Posterior Predictive (NO OBSERVED DATA)
# =========================
plt.figure()

# Sort posterior predictive samples
y_rep_sorted = np.sort(y_rep_ms, axis=0)

# Credible intervals
lower = np.percentile(y_rep_sorted, 5, axis=0)
upper = np.percentile(y_rep_sorted, 95, axis=0)
median = np.percentile(y_rep_sorted, 50, axis=0)

# X-axis (reaction time scale)
x_grid = np.linspace(
    np.min(y_rep_ms),
    np.max(y_rep_ms),
    len(median)
)

# Median prediction
plt.plot(
    x_grid,
    median,
    color="blue",
    linewidth=2.5,
    label="Posterior Median"
)

# Uncertainty band
plt.fill_between(
    x_grid,
    lower,
    upper,
    alpha=0.25,
    color="blue",
    label="90% Credible Interval"
)

# Labels
plt.title("Posterior Predictive Distribution")
plt.xlabel("Reaction Time (ms)")
plt.ylabel("Density")

plt.legend()
plt.show()

# =========================================================
# 12. Residuals
# =========================================================
residuals = y_obs_ms - y_pred_ms

plt.figure()
plt.hist(residuals, bins=30, density=True)
plt.title("Residuals (ms)")
plt.show()

# =========================================================
# 13. Metrics
# =========================================================
rmse = np.sqrt(np.mean(residuals ** 2))
mae = np.mean(np.abs(residuals))

print(f"RMSE: {rmse:.2f} ms")
print(f"MAE: {mae:.2f} ms")