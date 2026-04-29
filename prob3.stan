data {
  int<lower=1> N;
  int<lower=1> J;

  array[N] int<lower=1, upper=J> id;

  vector[N] sleep;
  vector[N] stress;
  vector[N] y;

  int<lower=1, upper=3> model_type;
}

parameters {
  // population effects
  real alpha;
  real beta_sleep;
  real beta_stress;

  // nonlinear + interaction
  real beta_sleep2;
  real beta_inter;

  // hierarchical (non-centered)
  vector[J] z_sleep;
  real<lower=0> tau_sleep;

  real<lower=0> sigma_y;
}

transformed parameters {
  vector[J] u_sleep;

  for (j in 1:J) {
    u_sleep[j] = tau_sleep * z_sleep[j];
  }
}

model {
  // --------------------
  // Priors (stable)
  // --------------------
  alpha ~ normal(0, 1);
  beta_sleep ~ normal(0, 0.5);
  beta_stress ~ normal(0, 0.5);

  beta_sleep2 ~ normal(0, 0.2);
  beta_inter ~ normal(0, 0.2);

  tau_sleep ~ lognormal(0, 0.3);
  z_sleep ~ normal(0, 1);

  sigma_y ~ exponential(1);

  // --------------------
  // Likelihood
  // --------------------
  for (n in 1:N) {

    real mu;

    if (model_type == 1) {

      // baseline linear
      mu =
        alpha
        + (beta_sleep + u_sleep[id[n]]) * sleep[n]
        + beta_stress * stress[n];

    } else if (model_type == 2) {

      // nonlinear
      mu =
        alpha
        + (beta_sleep + u_sleep[id[n]]) * sleep[n]
        + beta_sleep2 * square(sleep[n])
        + beta_stress * stress[n];

    } else {

      // interaction
      mu =
        alpha
        + (beta_sleep + u_sleep[id[n]]) * sleep[n]
        + beta_stress * stress[n]
        + beta_inter * sleep[n] * stress[n];
    }

    // safety clamp (prevents inf)
    mu = fmin(fmax(mu, -50), 50);

    y[n] ~ normal(mu, sigma_y);
  }
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;

  for (n in 1:N) {

    real mu;

    if (model_type == 1) {
      mu =
        alpha
        + (beta_sleep + u_sleep[id[n]]) * sleep[n]
        + beta_stress * stress[n];

    } else if (model_type == 2) {
      mu =
        alpha
        + (beta_sleep + u_sleep[id[n]]) * sleep[n]
        + beta_sleep2 * square(sleep[n])
        + beta_stress * stress[n];

    } else {
      mu =
        alpha
        + (beta_sleep + u_sleep[id[n]]) * sleep[n]
        + beta_stress * stress[n]
        + beta_inter * sleep[n] * stress[n];
    }

    mu = fmin(fmax(mu, -50), 50);

    y_rep[n] = normal_rng(mu, sigma_y);
    log_lik[n] = normal_lpdf(y[n] | mu, sigma_y);
  }
}