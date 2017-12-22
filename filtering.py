import math
import numpy as np


class KalmanFilter:
    def __init__(self, x, y, vx=0.0, vy=0.0, e=0, timestamp=0):
        self.mu = np.array([x, vx, y, vy])
        self.sigma = np.diag([e, 0.15, e, 0.15])
        self.Q = np.diag([0.01, 0.01])
        self.H = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0]])
        self.I = np.eye(4)
        self.timestamp = timestamp
        self.believes = []

    def F(self, dt):
        return np.array([
            [1.0,  dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0,  dt],
            [0.0, 0.0, 0.0, 1.0]
        ])

    def L(self, dt):
        return np.array([[0, 0],
                        [1, 0],
                        [0, 0],
                        [0, 1]]) * dt

    def R(self, e):
            return np.diag([0.1, 0.1]) # * e

    def G(self, dt):
        return np.array([
            [0.5 * dt ** 2, 0],
            [dt,              0],
            [0.0,   0.5 * dt ** 2],
            [0.0,              dt]
        ])

    def hypothesize(self, t, u=np.array([0.0, 0.0])):
        dt = math.fabs(self.timestamp - t)
        F = self.F(dt)
        L = self.L(dt)
        G = self.G(dt)
        mu = np.matmul(F, self.mu.T)
        print mu
        return mu + np.matmul(G, u.T), \
               np.matmul(F, np.matmul(self.sigma, F.T)) + np.matmul(L, np.matmul(self.Q, L.T))

    def predict(self, t, u=np.array([0.0, 0.0])):
        self.mu, self.sigma = self.hypothesize(t, u)
        self.timestamp = t
        self.believes.append([self.mu[0], self.mu[2], self.mu[1], self.mu[3]])

    def update(self, point, e):
        z = np.array([point[0], point[1]])
        R = self.R(e)
        S = np.matmul(self.H, np.matmul(self.sigma, self.H.T)) + R
        S_i = np.linalg.inv(S)
        K = np.matmul(self.sigma, np.matmul(self.H.T, S_i))
        y = z - np.matmul(self.H, self.mu)
        self.mu = self.mu + np.matmul(K, y)
        self.sigma = np.matmul(self.I - np.matmul(K, self.H), self.sigma)


