import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# Constants
n0 = 1.48
dn1 = 1e-3
n1 = n0 + dn1

dn2 = 5e-3
n2 = n0 + dn2

dn3 = 30e-3
n3 = n0 + dn3

a = 3e-6
wavelength = 730e-9
k0 = 2 * np.pi / wavelength

R1 = k0*a*np.sqrt(n1**2-n0**2)
R2 = k0*a*np.sqrt(n2**2-n0**2)
R3 = k0*a*np.sqrt(n3**2-n0**2)


x1 = np.logspace(np.log10(k0*n0), np.log10(k0*n1), 400)
x2 = np.logspace(np.log10(k0*n0), np.log10(k0*n2), 400)
x3 = np.logspace(np.log10(k0*n0), np.log10(k0*n3), 400)

alpha1 = np.sqrt((k0*n1)**2-x1**2)

lhs1p = alpha1* a * np.tan(alpha1 * a)
lhs1i = -alpha1 * a / np.tan(alpha1 * a)
beta = np.sqrt(x3**2 - (k0*n0)**2)

alpha2 = np.sqrt((k0*n2)**2-x2**2)
lhs2p = alpha2 * a * np.tan(alpha2 * a)
lhs2i = -alpha2 * a / np.tan(alpha2 * a)

alpha3 = np.sqrt((k0*n3)**2-x3**2)
lhs3p = alpha3 * a * np.tan(alpha3 * a)
lhs3i = -alpha3 *a / np.tan(alpha3 * a)

# lhs1[np.abs(lhs1-1e6) > 1e8] = np.nan
lhs2p[np.abs(lhs2p-5) > 20] = np.nan
lhs2i[np.abs(lhs2i-5) > 20] = np.nan

lhs3p[np.abs(lhs3p-5) > 20] = np.nan
lhs3i[np.abs(lhs3i-5) > 20] = np.nan

plt.style.use(['science'])
fig, ax = plt.subplots(dpi=300)

colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']




# ax.plot(x1, lhs1p, color=colors[1])
# ax.plot(x2, lhs2i, "--", color=colors[2])
# ax.plot(x2, lhs2p, color=colors[2])
ax.plot(x3*a, lhs3p, color=colors[0], label=r'$\alpha a \tan(\alpha a)$')
ax.plot(x3*a, lhs3i, color=colors[1], label=r'$-\alpha a \cot(\alpha a)$')


ax.plot(x3*a, beta * a, color=colors[2], label=r'$\beta a$')

# ax.plot(x3, beta*(n1/n0)**2, "--", color=colors[1])
# ax.plot(x3, beta*(n2/n0)**2, "--", color=colors[2])
ax.plot(x3*a, beta*(n3/n0)**2 * a, "--", color=colors[3], label=r'$\beta a (n_1/n_0)^2$')


# ax.set_xscale('log')

ax.set_ylim(0, 9)

ax.set_xlabel(r'$k_z a$')

fig.legend()
fig.savefig('../media/slabgraphicalTETM1.png')

plt.close('all')