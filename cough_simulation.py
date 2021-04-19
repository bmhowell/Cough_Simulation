from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import time
import datetime

start_time = time.time()

#---------------- constants --------------------
r0 = np.array([0, 0, 2]).T      # starting height (m)
g = np.array([0, 0, -9.81]).T   # gravity (m/s^2)

t_tot = 4                        # simulation time (s)
dt = 1e-4                       # time step (s)
V0 = 30                         # magnitude of cough velocity (m / s)
rho = 1000                      # density of particles (kg / m^3)
mTot = 0.00005                   # total mass of droplets (kg)

vf = np.array([0, 0, 0]).T      # surrounding fluid velocity (m/s)
muf = 1.8e-5                    # surrounding fluid viscosity (Pa/s)
rhoF = 1.225                    # density of air (kg / m^3)

# ---------------- particle generation --------------------
R_ave = 0.0001  # average particle radius (m)
R = []  # vector containing particle radii
Mtot = 0  # track total number of particles
mi = []  # mass of each particle (kg)
Aci = []  # cross sectional area of each particle (m^2)
A = 0.9975  # deviatiatoric constant
i = 0  # index
counter = True  # counter

while counter:
    xi = np.random.randint(low=-1000, high=1000, size=1) / 1000;
    R.append(R_ave * (1 + A * xi))
    Mtot = Mtot + rho * (4 / 3) * np.pi * R[i] ** 3
    mi.append(rho * (4 / 3) * np.pi * R[i] ** 3)
    Aci.append(np.pi * R[i] ** 2)
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

np.random.seed(10)
sample = np.random.choice(pTot, 1000, replace=False)  # select random particles to plot

# initial trajectories
Nc = np.array([0, 1., 0]).T     # direction of cough [x, y, z]
NPart = np.zeros((pTot, 3))     # perturbed direction for each particle [xi, yi, zi]
nPart = np.zeros((pTot, 3))     # normalized vector for each particle [nx, ny, nz]
Ac = np.array([1.0, 0.5, 1.]).T  # deviatoric constants [Ax, Ay, Az]


for i in range(pTot):
    eta = [np.random.randint(low=-1000, high=1000, size=1)/1000,
           np.random.randint(low=0, high=1000, size=1)/1000,
           np.random.randint(low=-1000, high=1000, size=1)/1000]

    NPart[i, 0] = Nc[0] + Ac[0]*eta[0]
    NPart[i, 1] = Nc[1] + Ac[1]*eta[1]
    NPart[i, 2] = Nc[2] + Ac[2]*eta[2]

    nMag = np.linalg.norm(NPart[i, :], 2)

    nPart[i, 0] = NPart[i, 0] / nMag
    nPart[i, 1] = NPart[i, 1] / nMag
    nPart[i, 2] = NPart[i, 2] / nMag

v0 = V0 * nPart

from celluloid import Camera

# ---------------- solution arrays --------------------
v0 = V0 * nPart
rSol = []
time_arr = []
fig = plt.figure()
camera = Camera(fig)
plot_R = R[0, sample]
# ---------------------------------------------
# ----------- Begin time stepping -------------
# ---------------------------------------------

ones = np.ones(pTot)
vi = v0
ri = np.outer(r0, ones).T
rSol.append(ri)
fGrav = np.outer(mi, g)

inAir = np.ones(len(ri[:, 0]), dtype=bool)
timeFlight = np.zeros(len(inAir))

t_step = 0
time_arr.append(t_step)
i = 0
print('vi = ', vi.shape)
while np.any(inAir):
    if i % 10 == 0:
        print('time = ', t_step)
    if t_step > t_tot:
        print("simulation finished")
        break

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

    # https://stackoverflow.com/questions/18522216/multiplying-across-in-a-numpy-array
    dragC = np.array([(Cd * np.squeeze(Aci) * vdiff),
                      (Cd * np.squeeze(Aci) * vdiff),
                      (Cd * np.squeeze(Aci) * vdiff)]).T
    fDrag = 1 / 2 * rhoF * dragC * (vf - vi)
    fTot = fDrag + fGrav

    ri[inAir, :] = ri[inAir, :] + dt * vi[inAir, :]
    vi[inAir, :] = vi[inAir, :] + dt * (fTot[inAir, :] / mi[inAir])

    contactSubstrate = np.where(ri[:, 2] <= 0)
    inAir[contactSubstrate] = False
    ri[contactSubstrate, 2] = 0

    # animate
    if i % 10 == 0:
        x2d = ri[sample, 1]
        y2d = ri[sample, 2]
        plt.scatter(r0[1], r0[2], color='k', s=500)
        plt.scatter(x2d, y2d, c=plot_R, cmap='viridis')
        plt.xlim(-0.02, 6)
        plt.ylim(0, 4)
        camera.snap()

    rSol.append(ri)

    t_step += dt
    time_arr.append(t_step)
    i += 1

# --------------------------------------------
print('compiling video ...')
animation = camera.animate()
animation.save('cough_simulation.mp4')
print('____ compile complete ____')

print('')
print('--- total time: {} min ---\n'.format((time.time() - start_time) / 60))

# --------------------------------------------

# find furthest particle in x direction
xMax = np.max(rSol[-1][:,0])
print('furthest particle in x-direction = ', xMax)
zMax = np.max(rSol[-1][:,2])
print('furthest particle in z-direction = ', zMax)
yMax = np.min(rSol[-1][:,1])
print('furthest particle in y-direction = ', yMax)

print('finalSol.shape: ', finalSol.shape)
np.random.seed(10)
sample = np.random.choice(pTot, 1000, replace=False)     # select random particles to plot
plot_sol = finalSol[sample, :]
plot_R = R[0,sample]

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection='3d')

ax.scatter3D(r0[0], r0[1], r0[2], c='k', marker='o', s=500)
output_fig = ax.scatter3D(plot_sol[:, 0], plot_sol[:, 1], plot_sol[:, 2],
                          marker='o', s=50, c=plot_R)
cbar = plt.colorbar(output_fig)
cbar.set_label('Particle radius (m)')
ax.set_title('Cough at time = {}s'.format(t_tot),y = 1.08)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim(-0.02, 6)
ax.set_ylim(0, 6)
ax.set_zlim(-0.2, 6)
ax.view_init(180,45)
plt.savefig('cough.png')
