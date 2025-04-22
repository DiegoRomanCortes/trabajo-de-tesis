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
sign = 1

eigenvalues = np.load(f'eigenvalues_{0}.npy')[:-2]
ax.plot(0, (np.sqrt(eigenvalues[(1-sign)//2])- k0*n0)*1e-2, 'r.-', label=r'$k_z^{+-}$')
ax.plot(0, (np.sqrt(eigenvalues[(1+sign)//2])- k0*n0)*1e-2, 'g.-', label=r'$k_z^{++}$')

couplings[0] = sign*np.abs(np.sqrt(eigenvalues)[0]-np.sqrt(eigenvalues)[1])*1e-2/2

ax2.plot(0, couplings[0], 'k.-', label=r'$\Delta k_z / 2$')

for i in range(1, len(angles)):
    angle = angles[i]
    eigenvalues = np.load(f'eigenvalues_{i}.npy')[:-2]
    # print(eigenvalues)
    ax.plot((angle), (np.sqrt(eigenvalues[(1-sign)//2])- k0*n0)*1e-2, 'r.-')
    ax.plot((angle), (np.sqrt(eigenvalues[(1+sign)//2])- k0*n0)*1e-2, 'g.-')

    if(np.abs(np.sqrt(eigenvalues)[0]-np.sqrt(eigenvalues)[1])*1e-2 < 0.05):
        idx_magic = i
        print(i)
        sign = -1
    couplings[i] = sign*np.abs(np.sqrt(eigenvalues)[0]-np.sqrt(eigenvalues)[1])*1e-2/2
    ax2.plot((angle), couplings[i], 'k.-')

ax2.hlines(y=0, xmin=angles[0], xmax=angles[-1], color='k', linestyle='--', linewidth=0.5)
ax2.vlines(angles[idx_magic], ymin=couplings.min(), ymax=couplings.max(), color='k', linestyle='--', linewidth=0.5)

ax2.text(angles[idx_magic-1], couplings.min(), r"$\theta_m$", ha="center", va="bottom", fontsize=14)

ax2.set_ylabel(r'Coupling (cm$^{-1}$)')
ax.ticklabel_format(axis='y', style='sci')
ax.set_xlabel('Angle (rad)')
ax.set_ylabel(r'$k_z - k_0 n_0$ (cm$^{-1}$)')
ax.set_title('Propagation constants vs Angle')
fig.legend(loc=(0.6, 0.5), fontsize=7)

fig.savefig('./eigenvalues_vs_angle.png')
plt.close("all")

