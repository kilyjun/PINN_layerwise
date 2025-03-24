import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from flax import linen as nn
import optax
from jax.random import PRNGKey
import matplotlib.pyplot as plt

# Define a deterministic neural network using Flax
class MLP(nn.Module):
    hidden_dims: list

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze()

# Define Bayesian neural network using NumPyro
def bayesian_nn(x, y=None, hidden_dims=[10]):
    in_features = x.shape[1]
    params = {}
    
    prev_dim = in_features
    for i, dim in enumerate(hidden_dims):
        params[f"w{i+1}"] = numpyro.sample(f"w{i+1}", dist.Normal(jnp.zeros((prev_dim, dim)), 5 * jnp.ones((prev_dim, dim))))
        params[f"b{i+1}"] = numpyro.sample(f"b{i+1}", dist.Normal(jnp.zeros(dim), 5 * jnp.ones(dim)))
        prev_dim = dim
    
    params[f"w_out"] = numpyro.sample("w_out", dist.Normal(jnp.zeros((prev_dim, 1)), 5 * jnp.ones((prev_dim, 1))))
    params[f"b_out"] = numpyro.sample("b_out", dist.Normal(0.0, 5.0))
    
    # Forward pass
    h = x
    for i in range(len(hidden_dims)):
        h = jnp.tanh(jnp.dot(h, params[f"w{i+1}"]) + params[f"b{i+1}"])
    y_pred = jnp.dot(h, params["w_out"]) + params["b_out"]
    
    # Likelihood
    sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
    numpyro.sample("obs", dist.Normal(y_pred, sigma), obs=y)

# Generate synthetic dataset
key = PRNGKey(0)
x_data = jax.random.normal(key, (100, 2))
y_data = jnp.sin(x_data[:, 0]) + jnp.cos(x_data[:, 1]) + 0.1 * jax.random.normal(key, (100,))

# Run inference using HMC
nuts_kernel = NUTS(lambda x, y: bayesian_nn(x, y, hidden_dims=[10, 10]))
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000, num_chains=1)
mcmc.run(PRNGKey(1), x_data, y_data)

# Get posterior samples
posterior_samples = mcmc.get_samples()
print(posterior_samples)

# Compute mean and uncertainty from posterior samples
def predict_samples(x, samples):
    h = jnp.tanh(jnp.dot(x, samples["w1"]) + samples["b1"])
    h = jnp.tanh(jnp.dot(h, samples["w2"]) + samples["b2"])
    return jnp.dot(h, samples["w_out"]) + samples["b_out"]

# Get predictions for multiple posterior samples
y_preds_samples = jax.vmap(lambda x: predict_samples(x, posterior_samples))(x_data)
y_preds_mean = y_preds_samples.mean(axis=1)
y_preds_std = y_preds_samples.std(axis=1)

# Plot results with uncertainty
plt.figure(figsize=(8, 6))
plt.scatter(x_data[:, 0], y_data, label="True Data", alpha=0.6)
plt.scatter(x_data[:, 0], y_preds_mean, label="Posterior Mean Prediction", alpha=0.6)
plt.fill_between(x_data[:, 0], y_preds_mean - 2 * y_preds_std, y_preds_mean + 2 * y_preds_std, color='orange', alpha=0.3, label="Uncertainty (2σ)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Bayesian Neural Network Predictions with Uncertainty")
plt.savefig("bnn_results.pdf")
plt.show()

