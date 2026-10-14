import numpy as np
import pandas as pd


class VolatilityForecaster:

    def __init__(self, filter_model, clock, seasonal_mean):
        self.filter = filter_model
        self.clock = clock
        self.seasonal_mean = seasonal_mean


    def expected_block_rv(self, forecast_origin, block_time):
        clock_time = block_time.strftime("%H:%M")
        seasonal_level = self.seasonal_mean.loc[clock_time]

        delta_t = self.clock.effective_time(forecast_origin, block_time)
        state_mean, state_covariance = self.filter.forecast_state(delta_t)

        log_rv_mean = seasonal_level + np.sum(state_mean)
        log_rv_variance = np.sum(state_covariance) + self.filter.observation_variance

        return np.exp(log_rv_mean + 0.5 * log_rv_variance)


    def forecast_day(self, effective_start):
        block_numbers = np.arange(len(self.seasonal_mean))
        delta_t = effective_start + block_numbers / 6

        mean_reversions = self.filter.model.mean_reversions
        weights = self.filter.model.weights

        decay = np.exp(-np.outer(delta_t, mean_reversions))
        future_means = decay * self.filter.state_mean

        stationary_covariance = np.diag(weights)

        aggregate_variances = []

        for row in decay:
            transition = np.diag(row)

            future_covariance = (
                transition @ self.filter.state_covariance @ transition.T
                + stationary_covariance
                - transition @ stationary_covariance @ transition.T
            )

            aggregate_variances.append(np.sum(future_covariance))

        aggregate_variances = np.asarray(aggregate_variances)

        log_rv_means = self.seasonal_mean.to_numpy() + np.sum(future_means, axis=1)

        log_rv_variances = (
            aggregate_variances
            + self.filter.observation_variance
        )

        expected_rv = np.exp(log_rv_means + 0.5 * log_rv_variances)

        return np.sum(expected_rv)


    def forecast_horizons(self, forecast_origin, future_dates, horizons):
        maximum_horizon = max(horizons)

        if len(future_dates) < maximum_horizon:
            raise ValueError("Not enough future trading dates for requested horizon")

        future_dates = future_dates[:maximum_horizon]

        first_open = future_dates[0].normalize() + pd.Timedelta(hours=9, minutes=30)
        effective_start = self.clock.effective_time(forecast_origin, first_open)

        daily_forecasts = []

        for i, forecast_date in enumerate(future_dates):
            daily_forecasts.append(self.forecast_day(effective_start))

            if i == len(future_dates) - 1:
                break

            next_date = future_dates[i + 1]

            calendar_hours = (next_date - forecast_date).total_seconds() / 3600
            closed_hours = calendar_hours - 6.5

            effective_start += 6.5 + self.clock.c_night * closed_hours

        forecasts = {}

        for horizon in horizons:
            forecasts[horizon] = np.mean(daily_forecasts[:horizon])

        return forecasts