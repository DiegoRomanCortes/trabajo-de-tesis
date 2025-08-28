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
omega = k0 * c
a = 3e-6

V = k0*a*np.sqrt(n1**2-n0**2)

ell = 1

alpha =   4.607987148/a #TE1 = 3.437777 TM1 = 3.453521 HE11 = 2.166564 HE21 = 3.447657 EH11 = 4.60798715 HE12 = 4.947020 H31 = 4.613618
beta = np.sqrt(V**2 - (alpha*a)**2)/a
kz = np.sqrt(k0**2*n1**2-alpha**2)


x = np.linspace(-2*a, 2*a, 200)
y = np.linspace(-2*a, 2*a, 200)
z = 0

X,Y = np.meshgrid(x, y)

R = np.sqrt(X**2 + Y**2)
PHI = np.arctan2(Y, X)

TE = jvp(ell, alpha*a)/(alpha*a*jv(ell, alpha*a)) + kvp(ell, beta*a)/(beta*a*kn(ell, beta*a))
TM = n1**2*jvp(ell, alpha*a)/(alpha*a*jv(ell, alpha*a)) + n0**2*kvp(ell, beta*a)/(beta*a*kn(ell, beta*a))
print(V)
print(TE*TM - ell**2*(1/(alpha*a)**2 + 1/(beta*a)**2)**2 * (kz/k0)**2)
atol = 1e-6
if np.abs(TE) < atol:
    E1 = 0+0j
    H1 = 1j
    print('TE')
elif np.abs(TM) < atol:
    E1 = 1j
    H1 = 0+0j
    print('TM')
else:
    E1 = -1j
    H1 = -E1 * 1j * np.sqrt(epsilon_0/mu_0) * np.sqrt(TM/TE)
    lhs = jvp(ell, alpha*a)/(alpha*a*jv(ell, alpha*a)) + (n1**2 + n0**2)/(2*n1**2) * kvp(ell, beta*a)/(beta*a*kn(ell, beta*a))
    rhs = np.sqrt(((n1**2 - n0**2)/(2*n1**2))**2 * (kvp(ell, beta*a)/(beta*a*kn(ell, beta*a)))**2 + (kz*ell/(k0*n1))**2*(1/((alpha*a)**2)+1/((beta*a)**2))**2)
    if np.abs(lhs - rhs) < atol:
        print('EH')
    elif np.abs(lhs + rhs) < atol:
        print('HE')
    else:
        print('Unknown mode')
E0 = E1 * jv(ell, alpha*a)/kn(ell, beta*a)
H0 = H1 * jv(ell, alpha*a)/kn(ell, beta*a)
# Field components
E_z = np.zeros_like(R, dtype=complex)
E_rho = np.zeros_like(R, dtype=complex)
E_phi = np.zeros_like(R, dtype=complex)

H_z = np.zeros_like(R, dtype=complex)
H_rho = np.zeros_like(R, dtype=complex)
H_phi = np.zeros_like(R, dtype=complex)
for i in range(len(x)):
    for j in range(len(y)):
        if R[i, j] <= a:
            E_z[i, j] = E1*jv(ell, alpha*R[i, j])*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)
            H_z[i, j] = H1*jv(ell, alpha*R[i, j])*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)

            E_rho[i, j] = kz*alpha*E1*jvp(ell, alpha*R[i, j]) + 1j*omega*mu_0*ell*H1*jv(ell, alpha*R[i, j])/R[i, j]
            H_rho[i, j] = kz*alpha*H1*jvp(ell, alpha*R[i, j]) - 1j*omega*epsilon_0*n1**2*ell*E1*jv(ell, alpha*R[i, j])/R[i, j]

            E_phi[i, j] = 1j*kz*ell*E1*jv(ell, alpha*R[i, j])/R[i, j] - omega*mu_0*alpha*H1*jvp(ell, alpha*R[i, j])
            H_phi[i, j] = 1j*kz*ell*H1*jv(ell, alpha*R[i, j])/R[i, j] + omega*epsilon_0*n1**2*alpha*E1*jvp(ell, alpha*R[i, j])
            
            E_rho[i, j] *= 1j*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)/(alpha**2)
            H_rho[i, j] *= 1j*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)/(alpha**2)
            E_phi[i, j] *= 1j*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)/(alpha**2)
            H_phi[i, j] *= 1j*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)/(alpha**2)

        else:
            E_z[i, j] = E0 * kn(ell, beta*R[i, j])*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)
            H_z[i, j] = H0 * kn(ell, beta*R[i, j])*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)

            E_rho[i, j] = kz*beta*E0*kvp(ell, beta*R[i, j]) + 1j*omega*mu_0*ell*H0*kn(ell, beta*R[i, j])/R[i, j]
            H_rho[i, j] = kz*beta*H0*kvp(ell, beta*R[i, j]) - 1j*omega*epsilon_0*n0**2*ell*E0*kn(ell, beta*R[i, j])/R[i, j]

            E_phi[i, j] = 1j*kz*ell*E0*kn(ell, beta*R[i, j])/R[i, j] - omega*mu_0*beta*H0*kvp(ell, beta*R[i, j])
            H_phi[i, j] = 1j*kz*ell*H0*kn(ell, beta*R[i, j])/R[i, j] + omega*epsilon_0*n0**2*beta*E0*kvp(ell, beta*R[i, j])

            E_phi[i, j] *= 1j*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)/(-beta**2)
            H_phi[i, j] *= 1j*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)/(-beta**2)
            E_rho[i, j] *= 1j*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)/(-beta**2)
            H_rho[i, j] *= 1j*np.exp(1j*ell*PHI[i, j])*np.exp(1j*kz*z)/(-beta**2)

plt.style.use(['science'])
fig, axs = plt.subplots(2, 2, dpi=400, sharey=True)
Ex = E_rho*np.cos(PHI)-E_phi*np.sin(PHI)
Hx = H_rho*np.cos(PHI)-H_phi*np.sin(PHI)

Ey = E_rho*np.sin(PHI)+E_phi*np.cos(PHI)
Hy = H_rho*np.sin(PHI)+H_phi*np.cos(PHI)

step = 20
print(E0, H0, E1, H1)
scaleE = 2e2
scaleH = scaleE * np.abs(H1)

axs[0, 0].contourf(X*1e6, Y*1e6, np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2), cmap='hot')
axs[0, 0].quiver(X[::step, ::step]*1e6, Y[::step, ::step]*1e6, np.real(Ex[::step, ::step]), np.real(Ey[::step, ::step]), scale=scaleE)
axs[0, 0].set_aspect('equal')
axs[0, 0].set_xlabel(r'$x$ ($\mu$m)')
axs[0, 0].set_ylabel(r'$y$ ($\mu$m)')

axs[0, 1].contourf(X*1e6, Y*1e6, np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2), cmap='hot')
axs[0, 1].quiver(X[::step, ::step]*1e6, Y[::step, ::step]*1e6, np.imag(Ex[::step, ::step]), np.imag(Ey[::step, ::step]), scale=scaleE)
axs[0, 1].set_aspect('equal')
axs[0, 1].set_xlabel(r'$x$ ($\mu$m)')
axs[0, 1].set_ylabel(r'$y$ ($\mu$m)')

axs[1, 0].contourf(X*1e6, Y*1e6, np.sqrt(np.abs(Hx)**2 + np.abs(Hy)**2), cmap='hot')
axs[1, 0].quiver(X[::step, ::step]*1e6, Y[::step, ::step]*1e6, np.real(Hx[::step, ::step]), np.real(Hy[::step, ::step]), scale=scaleH)
axs[1, 0].set_xlabel(r'$x$ ($\mu$m)')
axs[1, 0].set_aspect('equal')

axs[1, 1].contourf(X*1e6, Y*1e6, np.sqrt(np.abs(Hx)**2 + np.abs(Hy)**2), cmap='hot')
axs[1, 1].quiver(X[::step, ::step]*1e6, Y[::step, ::step]*1e6, np.imag(Hx[::step, ::step]), np.imag(Hy[::step, ::step]), scale=scaleH)
axs[1, 1].set_xlabel(r'$x$ ($\mu$m)')
axs[1, 1].set_aspect('equal')

# ax.imshow(np.abs(E_rho)**2, extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6], cmap='hot', origin='lower')
# fig.show()
# fig.savefig('../media/fiberEH11.pdf')
plt.close("all")