import numpy as np
import matplotlib.pyplot as plt
import scienceplots

n0 = 1.48  # Refractive index of the background
wavelength = 730E-9  # Wavelength of the light
k0 = 2 * np.pi / wavelength  # Wavenumber

angles = np.load("angles.npy")
plt.style.use('science')
fig, axs = plt.subplots(2, 1, dpi=500, sharex='col', figsize=(5, 5))

ax = axs[1]

overlap_kappa_sigma = -np.load('overlap_kappa_sigma.npy')

overlap_kappa_theta = [np.load(f'overlap_kappa_theta_{i}.npy') for i in range(len(angles))]
overlap_kappa_theta = np.array(overlap_kappa_theta)

idx_magic = np.argmin(np.abs(overlap_kappa_theta))+1
print(f"idx_magic: {idx_magic}")
overlap_kappa_theta[idx_magic:] *= -1

overlap_t_pi = [np.load(f'overlap_t_pi_{i}.npy') for i in range(len(angles))]
overlap_t_pi = np.array(overlap_t_pi)

overlap_t_theta_1 = [np.load(f'overlap_t_theta_1_{i}.npy') for i in range(len(angles))]
overlap_t_theta_1 = -np.array(overlap_t_theta_1)

overlap_t_theta_2 = [np.load(f'overlap_t_theta_2_{i}.npy') for i in range(len(angles))]
overlap_t_theta_2 = -np.array(overlap_t_theta_2)

overlap_t_theta_3 = [np.load(f'overlap_t_theta_3_{i}.npy') for i in range(len(angles))]
overlap_t_theta_3 = -np.array(overlap_t_theta_3)

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

# Adjust the vertical space between subplots
fig.subplots_adjust(hspace=0.0)

axs[0].plot(angles, np.abs(coupling_kappa_sigma)*np.ones(len(angles)), "b-", label=r'$\varkappa_\sigma$')
axs[0].plot(angles, np.abs(coupling_kappa_theta), "b--", label=r'$\varkappa_\theta$')
axs[0].plot(angles, np.abs(coupling_t_pi), "r", label=r'$t_\pi$')
axs[0].plot(angles, np.abs(coupling_t_theta_1), "#FF00FF", label=r'$t_{\theta 1}$')
axs[0].plot(angles, np.abs(coupling_t_theta_2), "gray", label=r'$t_{\theta 2}$')
axs[0].plot(angles, np.abs(coupling_t_theta_3), "lightgreen", label=r'$t_{\theta 3}$')

axs[0].set_yscale('log')

axs[0].set_ylabel(r'Coupling (cm$^{-1}$)')
# axs[0].set_xlabel('Angle (rad)')

axs[0].legend(loc='lower right', fontsize=10, fancybox=True, ncol=2)

ax.plot(angles, np.abs(overlap_kappa_sigma)*np.ones(len(angles)), "b-", label=r'$c_\sigma$')
ax.plot(angles, np.abs(overlap_kappa_theta), "b--", label=r'$c_\theta$')
ax.plot(angles, np.abs(overlap_t_pi), "r", label=r'$c_\pi$')
ax.plot(angles, np.abs(overlap_t_theta_1), "#FF00FF", label=r'$c_{\theta 1}$')
ax.plot(angles, np.abs(overlap_t_theta_2), "gray", label=r'$c_{\theta 2}$')
ax.plot(angles, np.abs(overlap_t_theta_3), "lightgreen", label=r'$c_{\theta 3}$')

ax.set_yscale('log')

ax.set_xlabel('Angle (rad)')
ax.set_ylabel(r'Overlap (arb. units)')

ax.legend(loc='lower right', fontsize=10, fancybox=True, ncol=2)

fig.savefig('./NNNN_overlap_vs_angle.png')
plt.close("all")

