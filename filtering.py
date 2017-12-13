import numpy as np


class KalmanFilter:
    def __init__(self, x, y, e, timestamp):
        self.mu = [x, 0.0, y, 0.0]
        self.sigma = np.diag([e, 0.15, e, 0.15])
        self.Q = np.diag([e, e])
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
            return np.diag([0.9, .9]) * e

    def G(self, dt):
        return np.array([
            [0.5 * dt ** 2, 0],
            [dt,              0],
            [0.0,   0.5 * dt ** 2],
            [0.0,              dt]
        ])

    def predict(self, t, u=np.array([0.0, 0.0])):
        dt = (self.timestamp - t)
        F = self.F(dt)
        L = self.L(dt)
        G = self.G(dt)
        self.mu = np.matmul(F, self.mu) + np.matmul(G, u.T) # times u ?
        self.sigma = np.matmul(F, np.matmul(self.sigma, F.T)) + np.matmul(L, np.matmul(self.Q, L.T))
        self.timestamp = t
        self.believes.append([self.mu[0], self.mu[2]])

    def update(self, point, e):
        z = np.array([point[0], point[1]])
        S = np.matmul(self.H, np.matmul(self.sigma, self.H.T)) + self.R(e)
        K = np.matmul(self.sigma, np.matmul(self.H.T, np.linalg.inv(S)))
        y = z - np.matmul(self.H, self.mu)
        self.mu = self.mu + np.matmul(K, y)
        self.sigma = np.matmul(self.I - np.matmul(K, self.H), self.sigma)

