import numpy as np
import matplotlib.pyplot as plt

#---------------- constants --------------------
h0 = np.array([0, 2, 0]).T      # starting height (m)
g = np.array([0, -9.81, 0]).T   # gravity (m/s^2)
time = 4                        # simulation time (s)
dt = 1e-6                       # time step (s)
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
mi = np.asarray(mi).T

# initial trajectories
Nc = np.array([0, 1, 0]).T      # direction of cough [x, y, z]
NPart = np.zeros((pTot, 3))     # perturbed direction for each particle [xi, yi, zi]
nPart = np.zeros((pTot, 3))     # normalized vector for each particle [nx, ny, nz]
Ac = np.array([1, 0.5, 1]).T    # deviatoric constants [Ax, Ay, Az]

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
print('v0 = ', v0.shape)

# ---------------------------------------------
# ----------- Begin time stepping -------------
# ---------------------------------------------

tspace = np.arange(0, time + dt, dt)
ones = np.ones(pTot)
Cd = np.zeros(pTot)
vi = v0
for i in range(1):

    # compute forces
    fGrav = np.outer(mi, g)
    print(fGrav.shape)

    vf_ = np.outer(vf, ones).T
    vdiff = np.linalg.norm(vf - vi, 2, 1)
    # vdiff2 = np.linalg.norm(vf_-vi, 2, 1)



    Re = (2 * R * rhoF * vdiff / muf).T

    cond1 = np.where(np.logical_and(Re > 0, Re <= 1.0))[0]
    cond2 = np.where(np.logical_and(Re > 1.0, Re <= 400))[0]
    cond3 = np.where(np.logical_and(Re > 400, Re <= 3e5))[0]
    cond4 = np.where(np.logical_and(Re > 3e5, Re <= 2e6))[0]
    cond5 = np.where(np.logical_and(Re > 2e6, Re > 1.0))[0]

    if len(cond1) > 0:
        Cd[cond1] = 24. / Re[cond1]
    if len(cond2) > 0:
        Cd[cond2] = 24. / np.power(Re[cond2], 0.646)
    if len(cond3) > 0:
        Cd[cond3] = 0.5
    if len(cond4) > 0:
        Cd[cond4] = 0.000366 * np.power(Re[cond3], 0.4275)
    if len(cond5) > 0:
        Cd[cond5] = 0.18

    print('shape Cd: ', Cd.shape)
    print('shape Aci: ', Aci.shape)
    print('shape vdiff: ', vdiff.shape)
    print('shape vi: ', vi.shape)
    print('shape vf: ', (vi-vf).shape)
    # fDrag = 0.5 * Cd * rhoF * Aci * vdiff * (vi - vf)
    # print('fDrag = ', fDrag)















