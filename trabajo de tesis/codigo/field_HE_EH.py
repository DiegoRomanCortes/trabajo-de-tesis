import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.special import jv, kn, jvp, kvp
from scipy.constants import c, mu_0, epsilon_0

# Constants
n0 = 1.48
dn1 = 40e-3
n1 = n0 + dn1

wavelength = 730e-9

k0 = 2*np.pi/wavelength
omega = 2*np.pi*c/wavelength
a = 3e-6

V = k0*a*np.sqrt(n1**2-n0**2)

ell = 0

alpha = 3.437776/a #TE = 3.438
beta = np.sqrt(V**2 - (alpha*a)**2)/a
kz = np.sqrt(beta**2 + k0**2*n0**2)


x = np.linspace(-2*a, 2*a, 200)
y = np.linspace(-2*a, 2*a, 200)
z = 0

X,Y = np.meshgrid(x, y)

R = np.sqrt(X**2 + Y**2)
PHI = np.arctan2(Y, X)

TE = jvp(ell, alpha*a)/(alpha*a*jv(ell, alpha*a)) + kvp(ell, beta*a)/(beta*a*kn(ell, beta*a))
TM = n1**2*jvp(ell, alpha*a)/(alpha*a*jv(ell, alpha*a)) + n0**2*kvp(ell, beta*a)/(beta*a*kn(ell, beta*a))
print(TE, TM)
atol = 1e-6
if np.abs(TE) < atol
E1 = 1
H1 = E1 *1j * kz *ell * (1/(alpha*a)**2+1/(beta*a)**2)/(omega*mu_0) / TE
E0 = E1 * jv(ell, alpha*a)/kn(ell, beta*a)
H0 = H1* jv(ell, alpha*a)/kn(ell, beta*a)
# Field components
E_z = np.zeros_like(R, dtype=complex)
E_rho = np.zeros_like(R, dtype=complex)
E_phi = np.zeros_like(R, dtype=complex)
for i in range(len(x)):
    for j in range(len(y)):
        if R[i, j] < a:
            E_z[i, j] = E1*jv(ell, alpha*R[i, j])*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)

            E_rho[i, j] = kz*alpha*E1*jvp(ell, alpha*R[i, j]) + 1j*omega*mu_0*ell*H1*jv(ell, alpha*R[i, j])/R[i, j]
            E_phi[i, j] = 1j*kz*ell*E1*jv(ell, alpha*R[i, j])/R[i, j] - omega*mu_0*alpha*H1*jvp(ell, alpha*R[i, j])
            
            E_rho[i, j] *= 1j*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)/(k0**2*n1**2 - kz**2)
            E_phi[i, j] *= 1j*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)/(k0**2*n1**2 - kz**2)

        else:
            E_z[i, j] = E0 * kn(ell, beta*R[i, j])*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)

            E_rho[i, j] = kz*beta*E1*kvp(ell, beta*R[i, j]) + 1j*omega*mu_0*ell*H0*kn(ell, beta*R[i, j])/R[i, j]
            E_phi[i, j] = 1j*kz*ell*E1*kn(ell, beta*R[i, j])/R[i, j] - omega*mu_0*beta*H0*kvp(ell, alpha*R[i, j])

            E_phi[i, j] *= 1j*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)/(k0**2*n0**2 - kz**2)
            E_rho[i, j] *= 1j*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)/(k0**2*n0**2 - kz**2)

plt.style.use(['science'])
fig, ax = plt.subplots(dpi=400)
ax.imshow(np.abs(E_rho)**2+np.abs(E_phi)**2, extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6], cmap='hot', origin='lower')
fig.show()