import numpy as np


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class GaussianNoise:
    def __init__(self, action_dimension, scale=0.1):
        self.action_dimension = action_dimension
        self.scale = scale
        self.reset()

    def reset(self):
        pass

    def noise(self):
        return np.random.randn(self.action_dimension) * self.scale


class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


# class RandomProcess:
#     def reset_states(self):
#         pass
#
#
# class GaussianProcess(RandomProcess):
#     def __init__(self, dim_act, var_low, var_decay):
#         self.var = 1.0
#         self.dim_act = dim_act
#         self.var_low = var_low
#         self.var_decay = var_decay
#
#     def reset(self):
#
#
#     def reset_states(self):
#         self.x_prev = self.var if self.var is not None else 1.0
#
#     @property
#     def current_var(self):
#         return self.var
#
#     def sample(self, episode_done, episodes_before_train):
#         return_state = np.random.randn(self.dim_act) * self.var
#         if episode_done > episodes_before_train and self.var > self.var_low:
#             self.var *= self.var_decay
#         return return_state
#
#
# class AnnealedGaussianProcess(RandomProcess):
#     def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
#         self.mu = mu
#         self.sigma = sigma
#         self.n_steps = 0
#
#         if sigma_min is not None:
#             self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
#             self.c = sigma
#             self.sigma_min = sigma_min
#         else:
#             self.m = 0.
#             self.c = sigma
#             self.sigma_min = sigma
#
#     @property
#     def current_sigma(self):
#         sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
#         return sigma
