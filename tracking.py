import colorsys

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock, mahalanobis, euclidean
from filtering import KalmanFilter
import math

INPUT_FILE = 'data/detections2.npy'
MAX_NEAREST_NEIGHBOR = 0.2
SIMULATION_DELTA = 0.01


class Tracker:

    def __init__(self):
        self.targets = {}

    def add_detection(self, detection):
        x, y, timestamp, discrete_time, confidence = detection
        min_similarity = MAX_NEAREST_NEIGHBOR
        nearest_neighbor = None
        for tracklet in self.targets:
            target = self.targets.get(tracklet)
            tracker = target['tracker']
            tracker.predict(timestamp)
            similarity = euclidean((x, y), (tracker.mu[0], tracker.mu[2]))
            if similarity <= min_similarity:
                min_similarity = similarity
                nearest_neighbor = tracklet

        if nearest_neighbor is not None:
            tracklet = self.targets[nearest_neighbor]
            tracker = tracklet['tracker']
            tracker.update((x, y), 1 - confidence)
            trajectory = tracklet['trajectory']
            trajectory.append((tracker.mu[0], tracker.mu[2], tracker.mu[1], tracker.mu[3], timestamp, tracker.sigma[0][0], tracker.sigma[2][2]))
        else:
            self.targets[hash(timestamp)] = {
                'tracker': KalmanFilter(x, y, e=1 - confidence, timestamp=timestamp),
                'trajectory': [(x, y, 0, 0, timestamp, 1 - confidence, 1 - confidence)]
            }

    def predict(self, t):
        for key in self.targets:
            tracklet = self.targets[key]
            tracker = tracklet['tracker']
            tracker.predict(t)
            print(key, math.degrees(math.atan2(tracker.mu[3], tracker.mu[1])))


def tracking():
    data = np.load(INPUT_FILE)

    last_detection = data[0]
    last_timestamp = last_detection[2]

    tracker = Tracker()
    tracker.add_detection(last_detection)
    for i in range(1, len(data)):
        detection = data[i]
        rate = (detection[2] - last_timestamp) / SIMULATION_DELTA
        for j in range(1, int(math.floor(rate))):
            tracker.predict(last_timestamp + j * SIMULATION_DELTA)
        tracker.add_detection(detection)
        last_timestamp = detection[2]

    return tracker.targets


def generate_colors(N):
    HSV_tuples = [(x * 0.5 / N, 0.6, 0.6) for x in range(N)]
    return map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)


def plot_trajectories(plot_data=False):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    targets = tracking()
    colors = generate_colors(len(targets.keys()))

    for index, target in enumerate(targets):
        tracklet = targets[target]
        trajectory = tracklet['trajectory']
        np_trajectory = np.array(trajectory)
        ax1.scatter(np_trajectory[:, 0], np_trajectory[:, 1], color=colors[index])

        tracker = tracklet['tracker']
        estimate = tracker.mu
        uncertainty = tracker.sigma
        ax1.scatter(estimate[0], estimate[2], s=(5, 5))
        ax1.annotate('u: ' + str(index), (estimate[0], estimate[2]))

        believes = np.array(tracklet['tracker'].believes)
        ax2.scatter(believes[:, 0], believes[:, 1], color=colors[index])
        final = len(believes)
        ax2.scatter(believes[final-1][0], believes[final-1][1], color=colors[index], s=(4, 2))
        ax2.annotate('mu: ' + str(index), (believes[final-1][0], believes[final-1][1]))
        # ax.annotate('b', (believes[:, 0], believes[:, 1]))
        print('Trajectory {}'.format(index))
        print('============================')
        print('Estimate:')
        print(estimate)
        print('Uncertainty:')
        print(uncertainty)
        print("============================")


    if plot_data:
        data = np.load(INPUT_FILE)
        ax1.scatter(data[:, 0], data[:, 1], color=(0.2, 0.5, 0.5), alpha=0.3)


plot_trajectories(plot_data=True)
plt.show()