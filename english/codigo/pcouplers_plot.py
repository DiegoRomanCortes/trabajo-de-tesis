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
ax2 = ax.twinx()

eigenvalues_all = [np.load(f'eigenvalues_{i}.npy')[:-2] for i in range(len(angles))]
eigenvalues_all = np.array(eigenvalues_all)

sign = 1

kz_minus = (np.sqrt(eigenvalues_all[:, (1-sign)//2]) - k0*n0) * 1e-2
kz_plus = (np.sqrt(eigenvalues_all[:, (1+sign)//2]) - k0*n0) * 1e-2
couplings = np.abs(np.sqrt(eigenvalues_all[:, 0]) - np.sqrt(eigenvalues_all[:, 1])) * 1e-2 / 2
idx_magic = np.argmin(np.abs(couplings))
print(f"idx_magic: {idx_magic}")
couplings[idx_magic:] *= -1

kz_temp = kz_plus[idx_magic:].copy()
kz_plus[idx_magic:] = kz_minus[idx_magic:]
kz_minus[idx_magic:] = kz_temp

angles_exp = np.array([ 0,  1,  2,  3,  4,  5,  6,  7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])*np.pi/40
couplings_exp = np.array( [ 0.18982468,  0.19759136,  0.19167419,  0.18454876,  
                           0.18354335, 0.14111277,  0.11345177,  0.07334573, -0.08385819, -0.13810501, 
                           -0.22390854, -0.32961958, -0.44581551, -0.59109267, -0.6538189, -0.78539816, 
                           -0.86393798, -0.93583845, -1.05301696, -1.13995956, -1.21760956])


ax.plot(angles, kz_minus, 'r', label=r'$k_z^{+-}$')
ax.plot(angles, kz_plus, 'g', label=r'$k_z^{++}$')
ax2.plot(angles, couplings, 'k-', label=r'$\Delta k_z / 2$')

ax2.hlines(y=0, xmin=angles[0], xmax=angles[-1], color='k', linestyle='--', linewidth=0.5)
ax2.vlines(angles[idx_magic], ymin=couplings_exp.min(), ymax=couplings_exp.max(), color='k', linestyle='--', linewidth=0.5)

ax2.text(angles[idx_magic-1], couplings.min(), r"$\theta_m$", ha="center", va="bottom", fontsize=14)

ax2.set_ylabel(r'Acoplamiento (cm$^{-1}$)')


ax2.plot(angles_exp, couplings_exp, 'k.', label='Datos exp.')
ax.ticklabel_format(axis='y', style='sci')
ax.set_xlabel('Ángulo (rad)')
ax.set_ylabel(r'$k_z - k_0 n_0$ (cm$^{-1}$)')
ax.set_title('Constantes de propagación vs Ángulo')
fig.legend(loc=(0.6, 0.55), fontsize=5)

fig.savefig('./eigenvalues_vs_angle.png')
plt.close("all")

