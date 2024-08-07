import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.special import jv, kn, jvp, kvp

# Constants
n0 = 1.48
dn1 = 40e-3
n1 = n0 + dn1

wavelength = 730e-9

k0 = 2*np.pi/wavelength

a = 3e-6

V = k0*a*np.sqrt(n1**2-n0**2)

ell = 0

alpha = 3.5/a
beta = np.sqrt(V**2 - (alpha*a)**2)/a
kz = np.sqrt(beta**2 + k0**2*n0**2)


x = np.linspace(-4*a, 4*a, 200)
y = np.linspace(-4*a, 4*a, 200)
z = 0

X,Y = np.meshgrid(x, y)

R = np.sqrt(X**2 + Y**2)
PHI = np.arctan2(Y, X)

# Field components
E_z = np.zeros_like(R, dtype=complex)
for i in range(len(x)):
    for j in range(len(y)):
        if R[i, j] < a:
            E_z[i, j] = jv(ell, alpha*R[i, j])*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)
        else:
            E_z[i, j] = kn(ell, beta*R[i, j])*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z) * jv(ell, alpha*a)/kn(ell, beta*a)

plt.imshow(np.abs(E_z)**2, extent=[x[0], x[-1], y[0], y[-1]], cmap='hot', origin='lower')
plt.show()