data {
  int<lower=1> N;
  int<lower=1> J;

  array[N] int<lower=1, upper=J> id;

  vector[N] sleep;
  vector[N] sleep_quality;
  vector[N] sleepiness;

  vector[N] y;  // log reaction time
}

parameters {
  real alpha;

  // fixed effects
  real beta_sleep;
  real beta_quality;
  real beta_sleepiness;

  // optional interaction
  real beta_sleep_sleepiness;

  // RANDOM INTERCEPT ONLY
  vector[J] z_intercept;
  real<lower=0> tau_intercept;

  real<lower=0> sigma_y;
}

transformed parameters {
  vector[J] u_intercept = tau_intercept * z_intercept;
}

model {
  // --------------------
  // Priors (STRONGER)
  // --------------------
  alpha ~ normal(0, 1);

  beta_sleep ~ normal(-0.2, 0.2);
  beta_quality ~ normal(-0.2, 0.2);
  beta_sleepiness ~ normal(0.3, 0.2);

  beta_sleep_sleepiness ~ normal(0, 0.2);

  z_intercept ~ normal(0, 1);
  tau_intercept ~ exponential(1);

  sigma_y ~ exponential(1);

  // --------------------
  // Likelihood
  // --------------------
  for (n in 1:N) {

    real mu =
      alpha
      + u_intercept[id[n]]
      + beta_sleep * sleep[n]
      + beta_quality * sleep_quality[n]
      + beta_sleepiness * sleepiness[n]
      + beta_sleep_sleepiness * sleep[n] * sleepiness[n];

    y[n] ~ normal(mu, sigma_y);
  }
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  vector[N] mu;

  for (n in 1:N) {

    mu[n] =
      alpha
      + u_intercept[id[n]]
      + beta_sleep * sleep[n]
      + beta_quality * sleep_quality[n]
      + beta_sleepiness * sleepiness[n]
      + beta_sleep_sleepiness * sleep[n] * sleepiness[n];

    y_rep[n] = normal_rng(mu[n], sigma_y);
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma_y);
  }
}
