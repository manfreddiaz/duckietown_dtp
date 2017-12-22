import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from filtering import KalmanFilter

np.random.seed(777)

fig = plt.figure()
ax = fig.add_subplot(111)


fig2 = plt.figure()
ax1 = fig2.add_subplot(221)
ax2 = fig2.add_subplot(222)
ax21 = fig2.add_subplot(223)
ax22 = fig2.add_subplot(224)

t = np.linspace(0, 5, 500)

x = np.linspace(-2, 2, 500)
y = np.linspace(-2, 2, 500)

# t_m = np.linspace(0, 5, 100)
x_m = np.linspace(-2, 2, 100) + np.random.normal(0, 0.3, 100)
y_m = np.linspace(-2, 2, 100) + np.random.normal(0, 0.3, 100)

ax.scatter(x, y)
ax.scatter(x_m, y_m)

count = 0
p_errors = []
v_errors = []
p_covariance = []
v_covariance = []


kf = None

for index, tt in enumerate(t):
    if kf is not None:
        kf.predict(tt)
    if index % 5 == 0:
        if kf is None:
            kf = KalmanFilter(x_m[count], y_m[count], e=1.0, timestamp=tt)
            kf.predict(tt)
        kf.update((x_m[count], y_m[count]), 0.01)
        count = count + 1
    if kf is not None:
        p_errors.append((x[index] - kf.mu[0], y[index] - kf.mu[2]))
        v_errors.append((0.8 - kf.mu[1], 0.8 - kf.mu[3]))
        p_covariance.append((math.sqrt(kf.sigma[0][0]) * 3, math.sqrt(kf.sigma[2][2]) * 3))
        v_covariance.append((math.sqrt(kf.sigma[1][1]) * 3, math.sqrt(kf.sigma[3][3]) * 3))
    else:
        p_errors.append((0.0, 0.0))
        v_errors.append((0.0, 0.0))
        p_covariance.append((1.0 * 3, 1.0 * 3))
        v_covariance.append((1.0 * 3, 1.0 * 3))

believes = np.array(kf.believes)
p_errors = np.array(p_errors)
p_covariance = np.array(p_covariance)
v_covariance = np.array(v_covariance)
v_errors = np.array(v_errors)

ax.scatter(believes[:, 0], believes[:, 1])
ax1.plot(t, p_errors[:, 0])
ax1.plot(t, p_covariance[:, 0])
ax1.plot(t, p_covariance[:, 0] * -1)
ax1.set_title('e_x vs t')

ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')

ax2.plot(t, p_errors[:, 1])
ax2.plot(t, p_covariance[:, 1])
ax2.plot(t, p_covariance[:, 1] * -1)
ax2.set_title('e_y vs t')
ax2.axhline(y=0, color='k')
ax2.axvline(x=0, color='k')

ax21.set_title('e_vx vs t')
ax21.plot(t, v_errors[:, 0])
ax21.plot(t, v_covariance[:, 0])
ax21.plot(t, v_covariance[:, 0] * -1)

ax22.set_title('e_vy vs t')
ax22.plot(t, v_errors[:, 1])
ax22.plot(t, v_covariance[:, 1])
ax22.plot(t, v_covariance[:, 1] * -1)

print count
print(believes[:, 0][-1], believes[:, 1][-1])
print(believes[:, 2][-1], believes[:, 3][-1])
print(p_errors[-1])
#
# for i in range(1, 500):
#     kf.predict(0.01)
#     if time[i] < len(x_m):
#         kf.update((x_m[i], y_m[i]), 0.01)
#
# ax3d.set_xlim(-1, 1)
# ax3d.set_ylim(-1, 1)
plt.show()

