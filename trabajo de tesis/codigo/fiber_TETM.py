import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.special import jv, kn

# Constants
n0 = 1.48
dn1 = 1e-3
n1 = n0 + dn1

dn2 = 15e-3
n2 = n0 + dn2

dn3 = 20e-3
n3 = n0 + dn3

n1 = n2

a = 6e-6
wavelength = 730e-9
k0 = 2 * np.pi / wavelength

kz = np.linspace(k0*n0, k0*n1, 1000)
l = 0

alpha = np.sqrt((k0*n1)**2 - kz**2)

beta = np.sqrt((kz)**2 - (k0*n0)**2)

plt.style.use(['science'])
fig,ax = plt.subplots(dpi=300)

def j_prime(l, x):
    return 0.5 * (jv(l-1, x) - jv(l+1, x))

def k_prime(l, x):
    return 0.5 * (kn(l-1, x) + kn(l+1, x))


lhs_TE = - j_prime(l, alpha*a) / (jv(l, alpha*a)* (alpha*a))
rhs_TE =  k_prime(l, beta*a) /( kn(l, beta*a)*beta*a)

rhs_TM = ((n0/n1)**2  *  k_prime(l, beta*a) /( kn(l, beta*a)*beta*a))

kz /= k0

lhs_TE[:-1][np.diff(lhs_TE) > 0] = np.nan
rhs_TE[:-1][np.diff(rhs_TE) > 0] = np.nan

# lhs_TM[:-1][np.diff(lhs_TM) < 0] = np.nan
rhs_TM[:-1][np.diff(rhs_TM) > 0] = np.nan

colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']


ax.plot(kz, lhs_TE, label=r'$J_1(\alpha a)/J_0(\alpha a)$', color=colors[0])
ax.plot(kz, rhs_TE, "--", label='TE', color=colors[1])

# ax.plot(kz, lhs_TM, label='TM', color=colors[1])
ax.plot(kz, rhs_TM, "--", label='TM', color=colors[2])
# ax.set_xscale('log')
ax.set_ylim(0, 0.2)
ax.set_xlabel(r'$k_z/k_0$')
fig.legend()
fig.savefig('../media/fibergraphical.png')

plt.close('all')