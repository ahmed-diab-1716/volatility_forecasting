import numpy as np


class MultiOUFilter:

    def __init__(self, model, observation_variance):
        self.model = model
        self.observation_variance = observation_variance

        self.state_mean = np.zeros(model.n_factors)
        self.state_covariance = np.diag(model.weights)


    def predict(self, delta_t):
        decay = np.exp(-self.model.mean_reversions * delta_t)

        transition = np.diag(decay)
        process_variance = self.model.weights * (1 - decay ** 2)
        process_covariance = np.diag(process_variance)

        self.state_mean = transition @ self.state_mean
        self.state_covariance = (
            transition @ self.state_covariance @ transition.T
            + process_covariance
        )


    def update(self, observation):
        observation_matrix = np.ones(self.model.n_factors)

        predicted_observation = observation_matrix @ self.state_mean
        innovation = observation - predicted_observation

        innovation_variance = (
            observation_matrix @ self.state_covariance @ observation_matrix
            + self.observation_variance
        )

        kalman_gain = self.state_covariance @ observation_matrix / innovation_variance

        self.state_mean = self.state_mean + kalman_gain * innovation

        identity = np.eye(self.model.n_factors)
        update_matrix = identity - np.outer(kalman_gain, observation_matrix)

        self.state_covariance = (
            update_matrix @ self.state_covariance @ update_matrix.T
            + self.observation_variance * np.outer(kalman_gain, kalman_gain)
        )

        return self.state_mean.copy()

    def predicted_observation(self):
        return np.sum(self.state_mean)

    def forecast_state(self, delta_t):
        decay = np.exp(-self.model.mean_reversions * delta_t)
        transition = np.diag(decay)

        process_variance = self.model.weights * (1 - decay ** 2)
        process_covariance = np.diag(process_variance)

        state_mean = transition @ self.state_mean
        state_covariance = (
            transition @ self.state_covariance @ transition.T
            + process_covariance
        )

        return state_mean, state_covariance

    def forecast_state(self, delta_t):
        decay = np.exp(-self.model.mean_reversions * delta_t)
        transition = np.diag(decay)

        process_variance = self.model.weights * (1 - decay ** 2)
        process_covariance = np.diag(process_variance)

        state_mean = transition @ self.state_mean
        state_covariance = (
            transition @ self.state_covariance @ transition.T
            + process_covariance
        )

        return state_mean, state_covariance