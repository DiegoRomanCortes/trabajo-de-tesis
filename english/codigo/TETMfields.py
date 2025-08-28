import numpy as np
import matplotlib.pyplot as plt
import scienceplots
# Constants
n0 = 1.48

dn3 = 15e-3
n3 = n0 + dn3

a = 3e-6
wavelength = 730e-9
k0 = 2 * np.pi / wavelength

betas = np.array([5.292, 4.775, 3.8, 1.974])/a
kz = np.sqrt(betas**2 + k0**2*n0**2)

def E_field(x, n):
    alpha = np.sqrt((k0*n3)**2-kz[n]**2)
    output = np.zeros_like(x)
    if n % 2 == 0:
        for idx, xi in enumerate(x):
            if np.abs(xi) < a:
                output[idx] = np.cos(alpha*xi)
            else:
                output[idx] = np.exp(-betas[n]*np.abs(xi))*np.exp(betas[n]*a)*np.cos(alpha*a)
    else:
        for idx, xi in enumerate(x):
            if np.abs(xi) < a:
                output[idx] = np.sin(alpha*xi)
            else:
                output[idx] = np.abs(xi)/xi*np.exp(-betas[n]*np.abs(xi))*np.exp(betas[n]*a)*np.sin(alpha*a)
    return output


plt.style.use(['science'])
fig, ax = plt.subplots(dpi=300)

scale = 0.001
x_to_plot = np.linspace(-a*3, a*3, 1000)

n_shape = np.zeros_like(x_to_plot)
for idx, xi in enumerate(x_to_plot):
    if np.abs(xi) < a:
        n_shape[idx] = n3
    else:
        n_shape[idx] = n0


colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
ax.plot(x_to_plot*1e6, n_shape, color=colors[4])
for n in range(4):
    E = E_field(x_to_plot, n)
    ax.plot(x_to_plot*1e6, scale*E+kz[n]/k0, color=colors[n])
    ax.plot(x_to_plot*1e6, kz[n]/k0*np.ones_like(x_to_plot), "--", color=colors[n])

ax.set_xlabel(r'$x$ ($\mu$m)')
ax.set_ylabel(r'$n_{\text{eff}}+H_z(x)$')
ax.set_yticks([n0, n3])
ax.set_yticklabels(["$n_0$", "$n_1$"])
fig.savefig('../media/TETMfields.pdf', bbox_inches='tight')