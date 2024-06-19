import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# Constants
n0 = 1.48
dn1 = 4e-3
n1 = n0 + dn1

dn2 = 12e-3
n2 = n0 + dn2

dn3 = 25e-3
n3 = n0 + dn3

a = 3e-6
wavelength = 730e-9
k0 = 2 * np.pi / wavelength

R1 = k0*a*np.sqrt(n1**2-n0**2)
R2 = k0*a*np.sqrt(n2**2-n0**2)
R3 = k0*a*np.sqrt(n3**2-n0**2)

x = np.linspace(0, 6*np.pi/2, 2000)

y1 = x * np.tan(x)
y2 = - x / np.tan(x)
y1[:-1][np.diff(y1) < 0] = np.nan
y2[:-1][np.diff(y2) < 0] = np.nan

plt.style.use(['science'])
fig, ax = plt.subplots(dpi=300)
ax.plot(x, y1, label='pares')
ax.plot(x, y2, label='impares')



ax.plot(x, np.sqrt(R1**2-x**2), label=r'$V_1$')
ax.plot(x, np.sqrt(R2**2-x**2), label=r'$V_2$')
ax.plot(x, np.sqrt(R3**2-x**2), label=r'$V_3$')

for i in range(len(x)):
    if x[i] < R1:
        if np.abs(np.sqrt(R1**2-x[i]**2) - y1[i]) < 1e-2:
            ax.plot(x[i], y1[i], 'ro')
        if np.abs(np.sqrt(R1**2-x[i]**2) - y2[i]) < 1e-2:
            ax.plot(x[i], y2[i], 'ro')
    if x[i] < R2:
        if np.abs(np.sqrt(R2**2-x[i]**2) - y1[i]) < 1e-2:
            ax.plot(x[i], y1[i], 'ro')
        if np.abs(np.sqrt(R2**2-x[i]**2) - y2[i]) < 1e-2:
            ax.plot(x[i], y2[i], 'ro')
    if x[i] < R3:
        if np.abs(np.sqrt(R3**2-x[i]**2) - y1[i]) < 1e-2:
            ax.plot(x[i], y1[i], 'ro')
        if np.abs(np.sqrt(R3**2-x[i]**2) - y2[i]) < 1e-2:
            ax.plot(x[i], y2[i], 'ro')

ax.set_ylim(0.0, 1.1*R3)
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, 5*np.pi/2, 3*np.pi])
ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$', r'$5\pi/2$', r'$3\pi$'])

ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, 5*np.pi/2, 3*np.pi])
ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$', r'$5\pi/2$', r'$3\pi$'])

ax.set_xlabel(r'$\alpha a$')
ax.set_ylabel(r'$\beta a$')

fig.legend()
fig.savefig('../media/slabgraphical.png')

plt.close('all')