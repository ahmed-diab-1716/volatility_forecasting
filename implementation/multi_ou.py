import numpy as np


class MultiOU:

    def __init__(self, mean, mean_reversions, weights):
        self.mean = mean
        self.mean_reversions = np.asarray(mean_reversions, dtype=float)
        self.weights = np.asarray(weights, dtype=float)

        if self.mean_reversions.ndim != 1 or self.weights.ndim != 1:
            raise ValueError("mean_reversions and weights must be one-dimensional")

        if len(self.mean_reversions) != len(self.weights):
            raise ValueError("mean_reversions and weights must have the same length")

        if len(self.weights) == 0:
            raise ValueError("At least one OU factor is required")

        if np.any(self.mean_reversions <= 0):
            raise ValueError("Mean-reversion rates must be positive")

        if np.any(self.weights < 0):
            raise ValueError("Weights must be non-negative")

        self.n_factors = len(self.weights)
        self.sigmas = np.sqrt(2 * self.mean_reversions * self.weights)


    def covariance(self, lag):
        lag = np.asarray(lag)
        return np.sum(self.weights * np.exp(-self.mean_reversions * lag[..., None]), axis=-1)


    def correlation(self, lag):
        variance = np.sum(self.weights)

        if variance == 0:
            raise ValueError("Correlation is undefined when total variance is zero")

        return self.covariance(lag) / variance


    def conditional_mean(self, factors, delta_t):
        factors = np.asarray(factors, dtype=float)

        if len(factors) != self.n_factors:
            raise ValueError("factors must contain one value per OU factor")

        if delta_t < 0:
            raise ValueError("delta_t must be non-negative")

        decay = np.exp(-self.mean_reversions * delta_t)

        return decay * factors


    def transition(self, factors, delta_t, rng=None):
        factors = np.asarray(factors, dtype=float)

        if len(factors) != self.n_factors:
            raise ValueError("factors must contain one value per OU factor")

        if delta_t < 0:
            raise ValueError("delta_t must be non-negative")

        if rng is None:
            rng = np.random.default_rng()

        decay = np.exp(-self.mean_reversions * delta_t)
        innovation_std = np.sqrt(self.weights * (1 - decay ** 2))
        innovations = rng.normal(size=self.n_factors)

        return decay * factors + innovation_std * innovations


    def log_variance(self, factors):
        factors = np.asarray(factors, dtype=float)

        if len(factors) != self.n_factors:
            raise ValueError("factors must contain one value per OU factor")

        return self.mean + np.sum(factors)


    def variance(self, factors):
        return np.exp(self.log_variance(factors))


    def stationary_sample(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        return rng.normal(0, np.sqrt(self.weights), size=self.n_factors)