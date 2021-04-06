import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits import mplot3d
from celluloid import Camera
import time
import datetime

start_time = time.time()
#---------------- constants --------------------
r0 = np.array([0, 2, 0]).T      # starting height (m)
g = np.array([0, -9.81, 0]).T   # gravity (m/s^2)
t_tot = 1                        # simulation time (s)
dt = 1e-3                       # time step (s)
V0 = 30                         # magnitude of cough velocity (m / s)
rho = 1000                      # density of particles (kg / m^3)
mTot = 0.0005                   # total mass of droplets (kg)

vf = np.array([0, 0, 0]).T      # surrounding fluid velocity (m/s)
muf = 1.8e-5                    # surrounding fluid viscosity (Pa/s)
rhoF = 1.225                    # density of air (kg / m^3)

#---------------- particle generation --------------------
R_ave = 0.0001                  # average particle radius (m)
R = []                          # vector containing particle radii
Mtot = 0                        # track total number of particles
mi = []                         # mass of each particle (kg)
Aci = []                        # cross sectional area of each particle (m^2)
A = 0.9975                      # deviatiatoric constant
i = 0                           # index
counter = True                  # counter

while counter:
    xi = np.random.randint(low=-1000, high=1000, size=1)/1000;
    R.append(R_ave * (1 + A * xi))
    Mtot = Mtot + rho*(4/3)*np.pi*R[-1]**3
    mi.append(rho*(4/3)*np.pi*R[-1]**3)
    Aci.append(np.pi * R[-1]**2)
    if Mtot >= mTot:
        counter = False
        pTot = i + 1
        print('Total # particles = ', pTot)
        print('rMin = ', np.amin(R))
        print('rMax = ', np.amax(R))
    else:
        i += 1
Aci = np.asarray(Aci)
R = np.asarray(R).T
mi = np.asarray(mi)

# initial trajectories
Nc = np.array([1., 0, 0]).T     # direction of cough [x, y, z]
NPart = np.zeros((pTot, 3))     # perturbed direction for each particle [xi, yi, zi]
nPart = np.zeros((pTot, 3))     # normalized vector for each particle [nx, ny, nz]
Ac = np.array([0.5, 1., 1.]).T  # deviatoric constants [Ax, Ay, Az]

for i in range(pTot):
    eta = [np.random.randint(low=-1000, high=1000, size=1)/1000,
           np.random.randint(low=-1000, high=1000, size=1)/1000,
           np.random.randint(low=-1000, high=1000, size=1)/1000]

    NPart[i, 0] = Nc[0] + Ac[0]*eta[0]
    NPart[i, 1] = Nc[1] + Ac[1]*eta[1]
    NPart[i, 2] = Nc[2] + Ac[2]*eta[2]

    nMag = np.linalg.norm(NPart[i, :], 2)
    nPart[i, 0] = NPart[i, 0] / nMag
    nPart[i, 1] = NPart[i, 1] / nMag
    nPart[i, 2] = NPart[i, 2] / nMag

v0 = V0 * nPart

#---------------- solution arrays --------------------
rSol = []
vSol = []

# ---------------------------------------------
# ----------- Begin time stepping -------------
# ---------------------------------------------
tspace = np.arange(0, t_tot + dt, dt)
ones = np.ones(pTot)
vi = v0
ri = np.outer(r0, ones).T
rSol.append(ri)
fGrav = np.outer(mi, g)

for i in range(len(tspace)):
    print('i = {} / {}'.format(i, len(tspace)))
    Cd = np.zeros(pTot)
    # compute forces

    vdiff = np.linalg.norm(vf - vi, 2, 1)

    Re = (2 * R * rhoF * vdiff / muf).T
    cond1 = np.where(np.logical_and(Re > 0, Re <= 1.0))[0]
    cond2 = np.where(np.logical_and(Re > 1.0, Re <= 400))[0]
    cond3 = np.where(np.logical_and(Re > 400, Re <= 3e5))[0]
    cond4 = np.where(np.logical_and(Re > 3e5, Re <= 2e6))[0]
    cond5 = np.where(np.logical_and(Re > 2e6, Re > 1.0))[0]

    if len(cond1) > 0:
        Cd[cond1] = 24. / np.squeeze(Re[cond1])
    if len(cond2) > 0:
        Cd[cond2] = 24. / np.squeeze(np.power(Re[cond2], 0.646))
    if len(cond3) > 0:
        Cd[cond3] = 0.5
    if len(cond4) > 0:
        Cd[cond4] = 0.000366 * np.squeeze(np.power(Re[cond4], 0.4275))
    if len(cond5) > 0:
        Cd[cond5] = 0.18

    Cd = np.reshape(Cd, (len(Cd), 1))
    vdiff = np.reshape(Cd, (len(vdiff), 1))

    # https://stackoverflow.com/questions/18522216/multiplying-across-in-a-numpy-array
    fDrag_constant = np.squeeze(0.5 * Cd * rhoF * Aci * vdiff)
    fDrag = ((vf - vi).T * fDrag_constant).T

    fTot = fDrag + fGrav

    ri = ri + dt*vi
    vi = vi + dt*(fTot / mi)

    if i % 10 == 0:
        rSol.append(ri)


counter1 = 0
Rplot = R * 1e4
for j in range(len(rSol)):
    print('j = {} / {}'.format(j, len(rSol)))
    xt = rSol[j]
    save_path1 = "/Users/bhopro/Desktop/Berkeley/MSOL/COVID19/output/state_matrix.txt.{}".format(counter1)

    f = open(save_path1, "w+")
    f.write('time , ID , X-coord , Y-coord , Z-coord \n')

    # iterate through number of particles
    time_elapsed = tspace[j]

    counter2 = 0

    for k in range(len(xt[:, 0])):
        x = xt[k, 0]
        y = xt[k, 1]
        z = xt[k, 2]
        mass = mi[k]
        f.write('{} , {}, {}, {}, {} \n'.format(time_elapsed, Rplot, x, y, z))
        counter2 += 1

    f.close()

    counter1 += 1

print('')
print('--- total time: {} min ---'.format((time.time() - start_time) / 60))

























