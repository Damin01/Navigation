import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

N = 100
Q = 1
R = 1
F = 0.2
G = 1
H = 1
J = 0
P0 = 10

rng = np.random.default_rng()
w = np.random.uniform(-1,1, size=N)*np.sqrt(Q)
v = np.random.uniform(-1,1, size=N)*np.sqrt(R)

# System Equations
system = signal.StateSpace(F, G, H, J)
x0 = np.random.uniform(-1,1, size=1)*np.sqrt(P0)
tout, y, x = signal.lsim(system,U = w,T=np.linspace(0, N-1, N), X0 = x0) + v

# Simulating explicitly in parallel in Kalman Filter
x = x0
P = P0
xhat = 0
xhistory = np.zeros(N-1)
yhistory = np.zeros(N-1)
xhathistory = np.zeros(N-1)
Phistory =np.zeros(N-1)

for i in range(N-1):
    xhistory[i] = x
    y = np.matmul(np.matrix(H),np.matrix(x)) + np.matrix(v[i])
    yhistory[i] = y
    Phistory[i] = P
    # xhatkgivenk-1 for Kalman Filter
    xhatkgivenk = np.matmul(np.matrix(F),np.matrix(xhat))
    xhathistory[i] = xhatkgivenk
    x = np.matmul(np.matrix(F),np.matrix(x)) + np.matmul(np.matrix(G), np.matrix(w[i]))
    # K for Kalman Filter
    K = np.matmul(np.matmul(np.matrix(P),np.matrix(H)), np.linalg.inv(np.matmul(np.matrix(H).transpose(),np.matmul(np.matrix(P),np.matrix(H))) + np.matrix(R)))
    # Predicted Covariance for Kalman Filter
    P = np.matmul(np.matrix(F), np.matmul(np.matrix(P),np.matrix(F).transpose())) + np.matrix(Q)
    # Update State Estimate for Kalman Filter
    xhat = xhat + np.matmul(K, (y - np.matmul(np.matrix(H).transpose(), np.matrix(xhat))))

fig, ax = plt.subplots()
ax.set_title('1D Kalman Filter')
ax.plot(xhistory, label = r'$x_{k}$')
ax.plot(xhathistory,'--', label = '$\hat{x}$')
ax.grid()
ax.legend()
plt.show()
