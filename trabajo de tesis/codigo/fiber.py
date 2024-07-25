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

a = 3e-6
wavelength = 730e-9
k0 = 2 * np.pi / wavelength

kz = np.linspace(k0*n0, k0*n1, 400)
l = 0

alpha = np.sqrt((k0*n0)**2 - kz**2)
beta = np.sqrt((kz)**2 - (k0*n1)**2)

plt.style.use(['science'])
fig, ax = plt.subplots(dpi=300)

def j_prime(l, x):
    return 0.5 * (jv(l-1, x) - jv(l+1, x))

def k_prime(l, x):
    return 0.5 * (kn(l-1, x) + kn(l+1, x))

lhs = (j_prime(l, alpha*a) / jv(l, alpha*a) + k_prime(l, beta*a) / kn(l, beta*a)) * (n1 * j_prime(l, alpha*a) / jv(l, alpha*a) + n0 * k_prime(l, beta*a) / kn(l, beta*a))
rhs = l**2 * ((1/(alpha*a))**2 + (1/(beta*a))**2)**2 * (kz/k0)**2  

ax.plot(kz, lhs, label="lhs")
ax.plot(kz, rhs, label="rhs")


fig.legend()
fig.savefig('../media/fibergraphical.png')

plt.close('all')