import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------
# constants

g = [0, 0, 9.81];            # gravity (m/s^2)
time = 4;                    # simulation time (s)
dt = 1e-6;                   # time step (s)
V0 = 30;                     # magnitude of cough velocity (m / s)
rho = 1000;                  # density of particles (kg / m^3)
rhoF = 1.225;                # density of air (kg / m^3)
mTot = 0.0005;               # total mass of droplets (kg)

# particle generation
R_ave = 0.0001;              # average particle radius (m)
R = [];                      # vector containing particle radii
Mtot = 0;                    # track total number of particles
A = 0.9975;                  # deviatiatoric constant
i = 0;                       # index
counter = True;              # counter

while counter:
    xi = np.random.randint(low=-1000, high=1000, size=1)/1000;
    R.append(R_ave * (1 + A * xi));
    Mtot = Mtot + rho*(4/3)*np.pi*R[-1]**3;
    if Mtot >= mTot:
        counter = False;
        print('Total # particles = ', i + 1)
        print('rMin = ', np.amin(R))
        print('rMax = ', np.amax(R))
    else:
        i += 1;

# initial trajectories







