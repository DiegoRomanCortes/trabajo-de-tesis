import numpy as np
import matplotlib.pyplot as plt
import scienceplots

n0 = 1.48  # Refractive index of the background
wavelength = 730E-9  # Wavelength of the light
k0 = 2 * np.pi / wavelength  # Wavenumber

angles = np.load("angles.npy")
couplings = np.zeros(len(angles))
plt.style.use('science')
fig, ax = plt.subplots(1, 1, dpi=400)

coupling_kappa_sigma = -np.load('coupling_kappa_sigma.npy')

coupling_kappa_theta = [np.load(f'coupling_kappa_theta_{i}.npy') for i in range(len(angles))]
coupling_kappa_theta = np.array(coupling_kappa_theta)

idx_magic = np.argmin(np.abs(coupling_kappa_theta))+1
print(f"idx_magic: {idx_magic}")
coupling_kappa_theta[idx_magic:] *= -1

coupling_t_pi = [np.load(f'coupling_t_pi_{i}.npy') for i in range(len(angles))]
coupling_t_pi = np.array(coupling_t_pi)

coupling_t_theta_1 = [np.load(f'coupling_t_theta_1_{i}.npy') for i in range(len(angles))]
coupling_t_theta_1 = -np.array(coupling_t_theta_1)

coupling_t_theta_2 = [np.load(f'coupling_t_theta_2_{i}.npy') for i in range(len(angles))]
coupling_t_theta_2 = -np.array(coupling_t_theta_2)

coupling_t_theta_3 = [np.load(f'coupling_t_theta_3_{i}.npy') for i in range(len(angles))]
coupling_t_theta_3 = -np.array(coupling_t_theta_3)


ax.plot(angles, np.abs(coupling_kappa_sigma)*np.ones(len(angles)), "b-", label=r'$\varkappa_\sigma$')
ax.plot(angles, np.abs(coupling_kappa_theta), "b--", label=r'$\varkappa_\theta$')
ax.plot(angles, np.abs(coupling_t_pi), "r", label=r'$t_\pi$')
ax.plot(angles, np.abs(coupling_t_theta_1), "#FF00FF", label=r'$t_{\theta 1}$')
ax.plot(angles, np.abs(coupling_t_theta_2), "gray", label=r'$t_{\theta 2}$')
ax.plot(angles, np.abs(coupling_t_theta_3), "lightgreen", label=r'$t_{\theta 3}$')

ax.set_yscale('log')

# ax.hlines(y=0, xmin=angles[0], xmax=angles[-1], color='k', linestyle='--', linewidth=0.5)
# ax.vlines(angles[idx_magic], ymin=couplings_exp.min(), ymax=couplings_exp.max(), color='k', linestyle='--', linewidth=0.5)

# ax.text(angles[idx_magic-1], couplings.min(), r"$\theta_m$", ha="center", va="bottom", fontsize=14)

ax.set_ylabel(r'Coupling (cm$^{-1}$)')
# ax.ticklabel_format(axis='y', style='sci')
ax.set_xlabel('Angle (rad)')
ax.set_ylabel(r'Coupling (cm$^{-1}$)')
ax.set_title('Coupling vs Angle')
fig.legend(loc=(0.64, 0.22), fontsize=6, fancybox=True, ncols=2)

fig.savefig('./NNNN_coupling_vs_angle.png')
plt.close("all")

