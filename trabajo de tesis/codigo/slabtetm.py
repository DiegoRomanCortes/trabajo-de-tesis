import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# Constants
n0 = 1.48
dn1 = 1e-3
n1 = n0 + dn1

dn2 = 15e-3
n2 = n0 + dn2

dn3 = 30e-3
n3 = n0 + dn3

a = 3e-6
wavelength = 730e-9
k0 = 2 * np.pi / wavelength

R1 = k0*a*np.sqrt(n1**2-n0**2)
R2 = k0*a*np.sqrt(n2**2-n0**2)
R3 = k0*a*np.sqrt(n3**2-n0**2)

x = np.linspace(k0*n0, k0*n3, 100)

alpha1 = np.sqrt((k0*n1)**2-x**2)
lhs1 = alpha1 * np.tan(alpha1 * a)
rhs1 = np.sqrt(x**2 - (k0*n0)**2)

alpha2 = np.sqrt((k0*n2)**2-x**2)
lhs2 = alpha2 * np.tan(alpha2 * a)
rhs2 = np.sqrt(x**2 - (k0*n0)**2)




plt.style.use(['science'])
fig, ax = plt.subplots(dpi=300)

colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']

ax.plot(x, lhs1, color=colors[0])
ax.plot(x, rhs1, color=colors[0])

# ax.plot(x, lhs2, color=colors[1])
# ax.plot(x, rhs2, color=colors[1])



ax.set_ylim(0.0, 1e6)
# ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
# ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

# ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
# ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

# ax.set_xlabel(r'$\alpha a$')
# ax.set_ylabel(r'$\beta a$')

fig.legend()
fig.savefig('../media/slabgraphicalTETM.png')

plt.close('all')