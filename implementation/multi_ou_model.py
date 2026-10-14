import numpy as np

from scipy.optimize import least_squares, nnls, minimize_scalar

from implementation.multi_ou import MultiOU
from implementation.multi_ou_filter import MultiOUFilter
from implementation.market_clock import MarketClock


class MultiOUModel:

    def __init__(self, n_factors, c_night):
        if n_factors < 1:
            raise ValueError("n_factors must be at least 1")

        if c_night < 0 or c_night > 1:
            raise ValueError("c_night must be between 0 and 1")

        self.n_factors = n_factors
        self.c_night = c_night

        self.process = None
        self.clock = MarketClock(c_night)
        self.filter = None
        self.observation_variance = None


    def covariance(self, open_hours, closed_hours, mean_reversions, weights):
        effective_time = open_hours + self.c_night * closed_hours

        return np.sum(
            weights * np.exp(-effective_time[:, None] * mean_reversions),
            axis=1
        )


    def fit(self, covariance_data, observed_variance, initial_timescales=None):
        open_hours = covariance_data["open_hours"].to_numpy()
        closed_hours = covariance_data["closed_hours"].to_numpy()
        observed_covariance = covariance_data["covariance"].to_numpy()

        if initial_timescales is None:
            initial_timescales = np.logspace(np.log10(2), np.log10(500), self.n_factors)

        initial_timescales = np.asarray(initial_timescales, dtype=float)

        if len(initial_timescales) != self.n_factors:
            raise ValueError("initial_timescales must contain one value per factor")

        initial_mean_reversions = 1 / initial_timescales
        effective_time = open_hours + self.c_night * closed_hours

        def residuals(log_mean_reversions):
            mean_reversions = np.exp(log_mean_reversions)
            basis = np.exp(-np.outer(effective_time, mean_reversions))

            weights, _ = nnls(basis, observed_covariance)

            return basis @ weights - observed_covariance

        result = least_squares(
            residuals,
            np.log(initial_mean_reversions),
            max_nfev=100000
        )

        mean_reversions = np.exp(result.x)

        basis = np.exp(-np.outer(effective_time, mean_reversions))
        weights, _ = nnls(basis, observed_covariance)

        order = np.argsort(mean_reversions)[::-1]

        mean_reversions = mean_reversions[order]
        weights = weights[order]

        latent_variance = np.sum(weights)
        observation_variance = observed_variance - latent_variance

        if observation_variance <= 0:
            raise ValueError("Estimated latent variance exceeds observed variance")

        self.observation_variance = observation_variance

        self.process = MultiOU(
            mean=0,
            mean_reversions=mean_reversions,
            weights=weights
        )

        self.filter = MultiOUFilter(
            self.process,
            observation_variance
        )

        return self

    def fit_fixed_slow(self, covariance_data, observed_variance, slow_timescale):
        if self.n_factors != 2:
            raise ValueError("fit_fixed_slow requires exactly two factors")

        if slow_timescale <= 0:
            raise ValueError("slow_timescale must be positive")

        open_hours = covariance_data["open_hours"].to_numpy()
        closed_hours = covariance_data["closed_hours"].to_numpy()
        observed_covariance = covariance_data["covariance"].to_numpy()

        effective_time = open_hours + self.c_night * closed_hours
        slow_mean_reversion = 1 / slow_timescale

        def loss(log_fast_timescale):
            fast_timescale = np.exp(log_fast_timescale)
            fast_mean_reversion = 1 / fast_timescale

            mean_reversions = np.array([fast_mean_reversion, slow_mean_reversion])
            basis = np.exp(-np.outer(effective_time, mean_reversions))

            weights, _ = nnls(basis, observed_covariance)
            fitted = basis @ weights

            return np.mean((observed_covariance - fitted) ** 2)

        result = minimize_scalar(
            loss,
            bounds=(np.log(10 / 60), np.log(slow_timescale)),
            method="bounded"
        )

        fast_timescale = np.exp(result.x)
        mean_reversions = np.array([1 / fast_timescale, slow_mean_reversion])

        basis = np.exp(-np.outer(effective_time, mean_reversions))
        weights, _ = nnls(basis, observed_covariance)

        latent_variance = np.sum(weights)
        observation_variance = observed_variance - latent_variance

        if observation_variance <= 0:
            raise ValueError("Estimated latent variance exceeds observed variance")

        self.observation_variance = observation_variance
        self.process = MultiOU(mean=0, mean_reversions=mean_reversions, weights=weights)
        self.filter = MultiOUFilter(self.process, observation_variance)

        return self

    def fitted_covariance(self, covariance_data):
        if self.process is None:
            raise ValueError("Model must be fitted first")

        open_hours = covariance_data["open_hours"].to_numpy()
        closed_hours = covariance_data["closed_hours"].to_numpy()

        return self.covariance(
            open_hours,
            closed_hours,
            self.process.mean_reversions,
            self.process.weights
        )


    def summary(self):
        if self.process is None:
            raise ValueError("Model must be fitted first")

        print("Multi-OU Model")
        print("Factors:", self.n_factors)
        print("c_night:", self.c_night)
        print("Observation variance:", self.observation_variance)
        print()

        for i in range(self.n_factors):
            print("Factor", i + 1)
            print("Weight:", self.process.weights[i])
            print("Mean reversion:", self.process.mean_reversions[i])
            print("Timescale:", 1 / self.process.mean_reversions[i], "effective hours")
            print("Sigma:", self.process.sigmas[i])
            print()