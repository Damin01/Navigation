import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def deg2rad(a):
    b = a*np.pi/180
    return b

def eulr2dcm(vec):
    roll = vec[0]
    pitch = vec[1]
    yaw = vec[2]
    A = np.matrix([[np.cos(yaw), np.sin(yaw), 0],
                  [-np.sin(yaw), np.cos(yaw), 0],
                  [0,0,1]])
    B = np.matrix([[np.cos(pitch), 0, -np.sin(pitch)],
                  [0, 1, 0],
                  [np.sin(pitch), 0, np.cos(pitch)]])
    C = np.matrix([[1, 0, 0],
                  [0, np.cos(roll), np.sin(roll)],
                  [0, -np.sin(roll), np.cos(pitch)]])
    D = np.matmul(C,np.matmul(B,A))
    return D

def cross(a, b):
    result = np.matrix([[a[1,0]*b[2,0] - a[2,0]*b[1,0]],
            [a[2,0]*b[0,0] - a[0,0]*b[2,0]],
            [a[0,0]*b[1,0] - a[1,0]*b[0,0]]])
    return result

df = pd.read_csv(r'C:\Users\Dawood\Downloads\Chico_trial.csv')
t = df['time'].values
ax = df['ax (m/s^2)'].values
ay = df['ay(m/s^2)'].values
az = df['az(m/s^2)'].values

wx = df['wx (deg/s)'].values
wy = df['wy (deg/s)'].values
wz = df['wz (deg/s)'].values

fig, axs = plt.subplots(2, 3, figsize = (12, 8))

axs[0, 0].plot(t,ax)
axs[0, 0].set_xlabel('time (s)')
axs[0, 0].set_ylabel(r'$a_{x}  \left ( g \right )$')
axs[0, 0].set_title('Accelerometer Reading')
axs[0, 0].grid()

axs[0, 1].plot(t,ay)
axs[0, 1].set_xlabel('time (s)')
axs[0, 1].set_ylabel(r'$a_{y}  \left ( g \right )$')
axs[0, 1].set_title('Accelerometer Reading')
axs[0, 1].grid()

axs[0, 2].plot(t,az)
axs[0, 2].set_xlabel('time (s)')
axs[0, 2].set_ylabel(r'$a_{z}  \left ( g \right )$')
axs[0, 2].set_title('Accelerometer Reading')
axs[0, 2].grid()

axs[1, 0].plot(t,wx)
axs[1, 0].set_xlabel('time (s)')
axs[1, 0].set_ylabel(r'$\omega_{x}  \left ( \frac{\degree}{s} \right )$')
axs[1, 0].set_title('Gyroscope Reading')
axs[1, 0].grid()

axs[1, 1].plot(t,wy)
axs[1, 1].set_xlabel('time (s)')
axs[1, 1].set_ylabel(r'$\omega_{y}  \left ( \frac{\degree}{s} \right )$')
axs[1, 1].set_title('Gyroscope Reading')
axs[1, 1].grid()

axs[1, 2].plot(t,wz)
axs[1, 2].set_xlabel('time (s)')
axs[1, 2].set_ylabel(r'$\omega_{z}  \left ( \frac{\degree}{s} \right )$')
axs[1, 2].set_title('Gyroscope Reading')
axs[1, 2].grid()

#Tight Layout making it easier to stop the overlap in the graphs above
plt.tight_layout()
plt.show()

# cutting off excess noise by only focusing on data with more impact
t = t[31:88]
ax = ax[31:88]
ay = ay[31:88]
az = az[31:88]
wx = deg2rad(wx[31:88])
wy = deg2rad(wy[31:88])
wz = deg2rad(wz[31:88])

dt = t[1]-t[0]

# Radius of the Earth in m:
R = 6.36*pow(10,6)
# Earth's rotation rate in rad/s
we = 7.2921159*pow(10,-5)
#w_ie
wie = np.matrix([[0], [0], [we]])

# Latitude (deg)
L = deg2rad(33.643148)
# Longitude (deg)
l = deg2rad(-117.839637)
# Elevation (m)
h = 60
# Gravity (m/s^2)
IGF = 9.780327*(1 + 0.0053024*pow(np.sin(L),2) - 0.0000058*pow(np.sin(2*L),2))
FAC = -3.086*pow(10,-6)*h
g = (IGF + FAC)

# Velocity
v = np.matrix([[0], [0], [0]])

# Position
x = R*np.matrix([[np.cos(L)*np.cos(l)], [np.cos(L)*np.sin(l)], [np.sin(L)]])

# Gravity Vector (Converting g into m/s^2)
G = g*np.matrix([[np.cos(L)*np.cos(l)], [np.cos(L)*np.sin(l)], [np.sin(L)]])
R0 = R*np.matrix([[np.cos(L)*np.cos(l)], [np.cos(L)*np.sin(l)], [np.sin(L)]])

# First Loop
Fb = np.matrix([[ax[0]], [ay[0]], [az[0]]])*g
# Roll, Pitch, Yaw
Pitch = np.arcsin(ax[0])
Roll = np.arcsin(ay[0] / np.cos(Pitch))
Yaw = 0

# Transformation on C via Latitude and Longitude
T = np.matrix([[-np.sin(L)*np.cos(l), -np.sin(L)*np.sin(l), np.cos(L)],
               [-np.sin(l), np.cos(l), 0],
               [-np.cos(L)*np.cos(l), -np.cos(L)*np.sin(l), -np.sin(L)]]).transpose()
C = np.matmul(T,eulr2dcm([Roll,Pitch,Yaw]).transpose())
Omega = np.matrix([[0, -wz[0], wy[0]],
                   [wz[0], 0, -wx[0]],
                   [-wy[0], wx[0], 0]])
a = np.matmul(C,Fb) - cross(wie,v)+ G - cross(wie,cross(wie,R0))
v = a*dt + v
x = v*dt + x

# L and l updates
Ldot = v[0,0] / (h+R)
ldot = v[1,0]/(np.cos(L)*(h+R))
hdot = -v[2,0]

L = L + Ldot*dt
l = l + ldot*dt
h = h + hdot*dt

# Gravity (m/s^2)
IGF = 9.780327*(1 + 0.0053024*pow(np.sin(L),2) - 0.0000058*pow(np.sin(2*L),2))
FAC = -3.086*pow(10,-6)*h
g = (IGF + FAC)

# Defining variables before the first loop
axx = np.zeros(len(ax))
ayy = np.zeros(len(ax))
azz = np.zeros(len(ax))
vxx = np.zeros(len(ax))
vyy = np.zeros(len(ax))
vzz = np.zeros(len(ax))
xx = np.zeros(len(ax))
yy = np.zeros(len(ax))
zz = np.zeros(len(ax))

axx[0] = a[0,0]
ayy[0] = a[1,0]
azz[0] = a[2,0]

vxx[0] = v[0,0]
vyy[0] = v[1,0]
vzz[0] = v[2,0]

xx[0] = x[0,0]
yy[0] = x[1,0]
zz[0] = x[2,0]

# Second loop
for i in range(1,len(ax)):
    Fb = np.matrix([[ax[i]], [ay[i]], [az[i]]]) * g
    # Normalizing the vectors
    G = g * (x / np.sqrt(pow(x[0,0],2) + pow(x[1,0],2) + pow(x[2,0],2)))
    R0 = R * (x / np.sqrt(pow(x[0,0],2) + pow(x[1,0],2) + pow(x[2,0],2)))
    Omega = np.matrix([[0, -wz[i], wy[i]],
                       [wz[i], 0, -wx[i]],
                       [-wy[i], wx[i], 0]])
    C = np.matmul(C, (Omega*dt + np.matrix([[1,0,0], [0,1,0], [0,0,1]])))
    a = np.matmul(C,Fb) - cross(wie,v) + G
    v = a * dt + v
    x = v * dt + x

    # L and l updates
    Ldot = v[0, 0] / (h + R)
    ldot = v[1, 0] / (np.cos(L) * (h + R))
    hdot = -v[2, 0]

    L = L + Ldot * dt
    l = l + ldot * dt
    h = h + hdot * dt

    # Gravity (m/s^2)
    IGF = 9.780327 * (1 + 0.0053024 * pow(np.sin(L), 2) - 0.0000058 * pow(np.sin(2 * L), 2))
    FAC = -3.086 * pow(10, -6) * h
    g = (IGF + FAC)

    axx[i] = a[0, 0]
    ayy[i] = a[1, 0]
    azz[i] = a[2, 0]

    vxx[i] = v[0, 0]
    vyy[i] = v[1, 0]
    vzz[i] = v[2, 0]

    xx[i] = x[0, 0]
    yy[i] = x[1, 0]
    zz[i] = x[2, 0]

fig, axs = plt.subplots(3, 3, figsize = (12, 8))

axs[0, 0].plot(t,axx)
axs[0, 0].set_xlabel('time (s)')
axs[0, 0].set_ylabel(r'$a_{x}  \left ( \frac{m}{s^{2}} \right )$')
axs[0, 0].set_title('Acceleration vs. Time')
axs[0, 0].grid()

axs[0, 1].plot(t,ayy)
axs[0, 1].set_xlabel('time (s)')
axs[0, 1].set_ylabel(r'$a_{y}  \left ( \frac{m}{s^{2}} \right )$')
axs[0, 1].set_title('Acceleration vs. Time')
axs[0, 1].grid()

axs[0, 2].plot(t,azz)
axs[0, 2].set_xlabel('time (s)')
axs[0, 2].set_ylabel(r'$a_{z}  \left ( \frac{m}{s^{2}} \right )$')
axs[0, 2].set_title('Acceleration vs. Time')
axs[0, 2].grid()

axs[1, 0].plot(t,vxx)
axs[1, 0].set_xlabel('time (s)')
axs[1, 0].set_ylabel(r'$v_{x}  \left ( \frac{m}{s} \right )$')
axs[1, 0].set_title('Velocity vs. Time')
axs[1, 0].grid()

axs[1, 1].plot(t,vyy)
axs[1, 1].set_xlabel('time (s)')
axs[1, 1].set_ylabel(r'$v_{y}  \left ( \frac{m}{s} \right )$')
axs[1, 1].set_title('Velocity vs. Time')
axs[1, 1].grid()

axs[1, 2].plot(t,vzz)
axs[1, 2].set_xlabel('time (s)')
axs[1, 2].set_ylabel(r'$v_{z}  \left ( \frac{m}{s} \right )$')
axs[1, 2].set_title('Velocity vs. Time')
axs[1, 2].grid()

axs[2, 0].plot(t,xx-xx[0])
axs[2, 0].set_xlabel('time (s)')
axs[2, 0].set_ylabel('x (m)')
axs[2, 0].set_title('Position vs. Time')
axs[2, 0].grid()

axs[2, 1].plot(t,yy-yy[0])
axs[2, 1].set_xlabel('time (s)')
axs[2, 1].set_ylabel('y (m)')
axs[2, 1].set_title('Position vs. Time')
axs[2, 1].grid()

axs[2, 2].plot(t,zz-zz[0])
axs[2, 2].set_xlabel('time (s)')
axs[2, 2].set_ylabel('z (m)')
axs[2, 2].set_title('Position vs. Time')
axs[2, 2].grid()

#Tight Layout making it easier to stop the overlap in the graphs above
plt.tight_layout()
plt.show()

ax = plt.axes(projection='3d')
ax.plot3D(xx-xx[0], yy-yy[0], zz-zz[0])
ax.set_title('3-D Position data')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()