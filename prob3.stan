data {
  int<lower=1> N;
  int<lower=1> J;

  array[N] int<lower=1, upper=J> id;

  vector[N] sleep;
  vector[N] sleep_quality;
  vector[N] sleepiness;
  vector[N] stress;
  vector[N] caffeine;
  vector[N] activity;
  vector[N] age;
  vector[N] bmi;

  vector[N] y;
}

parameters {
  real alpha;

  // fixed effects
  real beta_sleep;
  real beta_quality;
  real beta_sleepiness;
  real beta_stress;
  real beta_caffeine;
  real beta_activity;
  real beta_age;
  real beta_bmi;

  // hierarchical sleep effect (non-centered)
  vector[J] z_sleep;
  real<lower=0> tau_sleep;

  // noise (FIXED: cannot collapse to 0)
  real<lower=0.05> sigma_y;
}

transformed parameters {
  vector[J] u_sleep = tau_sleep * z_sleep;
}

model {
  // --------------------
  // Priors
  // --------------------
  alpha ~ normal(0, 1);

  beta_sleep ~ normal(-0.2, 0.3);
  beta_quality ~ normal(-0.2, 0.3);
  beta_sleepiness ~ normal(0.3, 0.3);
  beta_stress ~ normal(0.2, 0.3);
  beta_caffeine ~ normal(-0.1, 0.3);
  beta_activity ~ normal(-0.1, 0.3);
  beta_age ~ normal(0, 0.3);
  beta_bmi ~ normal(0, 0.3);

  z_sleep ~ normal(0, 1);

  tau_sleep ~ exponential(1);

  // FIX: stable noise prior (prevents zero collapse)
  sigma_y ~ exponential(1);

  // --------------------
  // Likelihood
  // --------------------
  for (n in 1:N) {

    real mu =
      alpha
      + (beta_sleep + u_sleep[id[n]]) * sleep[n]
      + beta_quality * sleep_quality[n]
      + beta_sleepiness * sleepiness[n]
      + beta_stress * stress[n]
      + beta_caffeine * caffeine[n]
      + beta_activity * activity[n]
      + beta_age * age[n]
      + beta_bmi * bmi[n];

    y[n] ~ normal(mu, sigma_y);
  }
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;

  for (n in 1:N) {

    real mu =
      alpha
      + (beta_sleep + u_sleep[id[n]]) * sleep[n]
      + beta_quality * sleep_quality[n]
      + beta_sleepiness * sleepiness[n]
      + beta_stress * stress[n]
      + beta_caffeine * caffeine[n]
      + beta_activity * activity[n]
      + beta_age * age[n]
      + beta_bmi * bmi[n];

    y_rep[n] = normal_rng(mu, sigma_y);
    log_lik[n] = normal_lpdf(y[n] | mu, sigma_y);
  }
}