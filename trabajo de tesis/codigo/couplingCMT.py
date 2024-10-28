import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.constants import epsilon_0, mu_0, c

# Constants
n0 = 1.48

dn3 = 1e-3
n3 = n0 + dn3

a = 3e-6
d = 12e-6
wavelength = 730e-9
k0 = 2 * np.pi / wavelength

betas = np.array([1.0895])/a
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

scale = 1
x_to_plot = np.linspace(-30e-6, 50e-6, 10000)

n_shape_0 = np.zeros_like(x_to_plot)
n_shape_1 = np.zeros_like(x_to_plot)
n_shape_2 = np.zeros_like(x_to_plot)
n_shape_3 = np.zeros_like(x_to_plot)
n_shape_4 = np.zeros_like(x_to_plot)

n1 = n0
n2 = n3
for idx, xi in enumerate(x_to_plot):
    if np.abs(xi+d) < a:
        n_shape_1[idx] = n2
    else:
        n_shape_1[idx] = n1
    if np.abs(xi) < a:
        n_shape_2[idx] = n2
    else:
        n_shape_2[idx] = n1
    if np.abs(xi-d) < a:
        n_shape_3[idx] = n2
    else:
        n_shape_3[idx] = n1
    if np.abs(xi-2*d) < a:
        n_shape_4[idx] = n2
    else:
        n_shape_4[idx] = n1

n_shape = n_shape_1 + n_shape_2 + n_shape_3 + n_shape_4
n_shape[n_shape < n1] = n1

n_shape2 = n_shape_1**2 + n_shape_2**2 + n_shape_3**2 + n_shape_4**2 -3*n1**2
colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
ax.plot((x_to_plot)*1e6, (n_shape2 - n_shape_3**2), color=colors[0], label=r"$n^2-n_m^2$")

ax2 = ax.twinx()
for n in range(1):
    E1 = E_field(x_to_plot+d, n)
    E2 = E_field(x_to_plot, n)
    E3 = E_field(x_to_plot-d, n)
    E4 = E_field(x_to_plot-2*d, n)

    ax2.plot((x_to_plot)*1e6, E2, color=colors[1], label=r"$E_{m\prime}$")
    # ax.plot((x_to_plot)*1e6, scale*E2, color=colors[2])
    ax2.plot((x_to_plot)*1e6, E3, color=colors[3], label=r"$E_{m}$")
    # ax.plot((x_to_plot)*1e6, scale*E4, color=colors[4])
    
    # ax.plot(x_to_plot*1e6, kz[n]/k0*np.ones_like(x_to_plot), "--", color=colors[n])
    ax2.plot((x_to_plot)*1e6,  E2*E3, color="black", label=r"$E_{m^\prime}\cdot E_{m}$")

ax.set_xlabel(r'$x$ ($\mu$m)')
ax.set_ylabel(r'$n^2-n_m^2$')
ax2.set_ylabel(r'$E_{m\prime}, E_{m}$')
# ax.set_yticks([])
# ax.set_yticklabels(["$n_0$", "$n_1$"])
fig.legend(loc='upper center', ncol=4, fontsize='x-small')
fig.savefig('../media/coupling.pdf', bbox_inches='tight')



constant = epsilon_0 * c * k0**2 / 2 * 1e-2

coupling11 = np.trapz(E1*E1*(n_shape2 - n_shape_1**2), x_to_plot) * constant
coupling12 = np.trapz(E1*E2*(n_shape2 - n_shape_2**2), x_to_plot) * constant
coupling13 = np.trapz(E1*E3*(n_shape2 - n_shape_3**2), x_to_plot)* constant
coupling14 = np.trapz(E1*E4*(n_shape2 - n_shape_4**2), x_to_plot)* constant

coupling21 = np.trapz(E1*E2*(n_shape2 - n_shape_1**2), x_to_plot)* constant
coupling23 = np.trapz(E2*E3*(n_shape2 - n_shape_3**2), x_to_plot)* constant
coupling24 = np.trapz(E2*E4*(n_shape2 - n_shape_4**2), x_to_plot)* constant

coupling31 = np.trapz(E1*E3*(n_shape2 - n_shape_1**2), x_to_plot)* constant
coupling32 = np.trapz(E2*E3*(n_shape2 - n_shape_2**2), x_to_plot)* constant
coupling34 = np.trapz(E3*E4*(n_shape2 - n_shape_4**2), x_to_plot)* constant

coupling41 = np.trapz(E1*E4*(n_shape2 - n_shape_1**2), x_to_plot)* constant
coupling42 = np.trapz(E2*E4*(n_shape2 - n_shape_2**2), x_to_plot)* constant
coupling43 = np.trapz(E3*E4*(n_shape2 - n_shape_3**2), x_to_plot)* constant

print(coupling12, coupling13, coupling14)
print(coupling21, coupling23, coupling24)
print(coupling31, coupling32, coupling34)
print(coupling41, coupling42, coupling43)

print(coupling11)