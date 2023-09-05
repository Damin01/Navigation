import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

N = 100
Q = np.matrix([[1, 0], [0, 1]])
R = np.matrix([[1, 0], [0, 1]])
P0 = np.matrix([[10, 0], [0, 10]])

F = np.matrix([[0.5, 0], [0.2, 0.0001]])
G = np.matrix([[1, 0], [0, 1]])
H = np.matrix([[1, 0], [0, 0.5]])
J = np.matrix([[1, 0], [0, 0]])

rng = np.random.default_rng()
w = np.random.uniform(-1,1, size=(2,N))*np.sqrt(np.linalg.norm(Q))
v = np.random.uniform(-1,1, size=(2,N))*np.sqrt(np.linalg.norm(R))

x0 = np.random.uniform(-1,1, size=(2,1))*np.sqrt(np.linalg.norm(P0))

x = x0
P = P0
xhat = np.matrix([[0], [0]])
xhistory = np.zeros((2,N-1))
yhistory = np.zeros((2,N-1))
xhathistory = np.zeros((2,N-1))
Phistory =np.zeros((2,N-1))

for i in range(N-1):
    xhistory[0,i] = x[0]
    xhistory[1, i] = x[1]
    y = np.matmul(H,x) + np.matrix([[v[0,i]], [v[1,i]]])
    # xhatkgivenk-1 for Kalman Filter
    xhatkgivenk = np.matmul(F,xhat)
    xhathistory[0,i] = xhatkgivenk[0]
    xhathistory[1, i] = xhatkgivenk[1]
    x = np.matmul(F,x) + np.matmul(G, np.matrix([[w[0,i]], [w[1,i]]]))
    # K for Kalman Filter
    K = np.matmul(np.matmul(P,H), np.linalg.inv(np.matmul(H.transpose(),np.matmul(P,H)) + R))
    # Predicted Covariance for Kalman Filter
    P = np.matmul(F, np.matmul(P,F.transpose())) + Q
    # Update State Estimate for Kalman Filter
    xhat = xhat + np.matmul(K, (y - np.matmul(H.transpose(), xhat)))

fig, axs = plt.subplots(1, 2, figsize = (12, 8))
axs[0].set_title('x - axis Kalman Filter')
axs[0].plot(xhistory[0,:], label = r'$x_{k}$')
axs[0].plot(xhathistory[0, :],'--', label = '$\hat{x}$')
axs[0].legend()
axs[0].grid()

axs[1].set_title('y - axis Kalman Filter')
axs[1].plot(xhistory[1,:], label = r'$y_{k}$')
axs[1].plot(xhathistory[1, :],'--', label = '$\hat{y}$')
axs[1].legend()
axs[1].grid()

#Tight Layout making it easier to stop the overlap in the graphs above
plt.tight_layout()
plt.show()