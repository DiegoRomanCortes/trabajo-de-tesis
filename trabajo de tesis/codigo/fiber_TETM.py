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

dn3 = 30e-3
n3 = n0 + dn3

n1 = n3

a = 5e-6
wavelength = 730e-9
k0 = 2 * np.pi / wavelength

kz = np.linspace(k0*n0, k0*n1, 1000)
l = 0

alpha = np.sqrt((k0*n1)**2 - kz**2)

beta = np.sqrt((kz)**2 - (k0*n0)**2)

plt.style.use(['science'])
fig, ax = plt.subplots(dpi=300)

def j_prime(l, x):
    return 0.5 * (jv(l-1, x) - jv(l+1, x))

def k_prime(l, x):
    return 0.5 * (kn(l-1, x) + kn(l+1, x))


lhs_TE = (j_prime(l, alpha*a) / (jv(l, alpha*a)* (alpha*a)) + k_prime(l, beta*a) /( kn(l, beta*a)*beta*a)) 
lhs_TM = (n1**2 * j_prime(l, alpha*a) / (jv(l, alpha*a)* (alpha*a)) + n0**2 * k_prime(l, beta*a) /( kn(l, beta*a)*beta*a))

kz /= k0

ax.plot(kz, lhs_TE, label='TE')
ax.plot(kz, lhs_TM, label='TM')
ax.plot(kz, np.zeros_like(kz), 'k--')
# ax.set_yscale('log')
ax.set_ylim(-6, 6)

fig.legend()
fig.savefig('../media/fibergraphical.png')

plt.close('all')