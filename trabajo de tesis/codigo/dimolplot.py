import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.constants import c, epsilon_0, mu_0

# Load single data
eigenvalues1 = np.load('dimol/eigenvalues1.npy')
eigenvalues2 = np.load('dimol/eigenvalues2.npy')

x = np.load('dimol/x.npy')
distances = np.load('dimol/distances.npy')
wavelength = 730E-9
k0 = 2 * np.pi / wavelength
n0 = 1.48
omega = k0 * c   # Frequency of the light

beta1 = np.sqrt(eigenvalues1[-1])
beta2 = np.sqrt(eigenvalues2[-1])

E1 = np.abs(np.load('dimol/eigenvectors1.npy')[:, -1])
E2 = np.abs(np.load('dimol/eigenvectors2.npy')[:, -1])

# E1 /= (np.sum(np.abs(E1)**2)*(x[1]-x[0]))
# E2 /= (np.sum(np.abs(E2)**2)*(x[1]-x[0]))
# Load coupler data

eigenvalues_couplers = np.zeros((len(distances), 3))
C_mat = np.zeros((len(distances), 2, 2))
eigenvalues_couplers_CMT = np.zeros((len(distances), 2))
for idx, dist in enumerate(distances):
    eigenvalues_couplers[idx, :] = np.load(f'dimol/eigenvalues_{idx}.npy')
    dnmdn1s = np.load(f'dimol/dn1_array_{idx}.npy')
    dnmdn2s = np.load(f'dimol/dn2_array_{idx}.npy')

    p11 = np.sum(np.roll(E1, -int(dist/2/(x[1]-x[0]))) * np.roll(E1, -int(dist/2/(x[1]-x[0])))) * (x[1]-x[0])
    p12 = np.sum(np.roll(E1, -int(dist/2/(x[1]-x[0]))) * np.roll(E2, int(dist/2/(x[1]-x[0])))) * (x[1]-x[0])
    p21 = np.sum(np.roll(E2, int(dist/2/(x[1]-x[0]))) * np.roll(E1, -int(dist/2/(x[1]-x[0])))) * (x[1]-x[0])
    p22 = np.sum(np.roll(E2, int(dist/2/(x[1]-x[0]))) * np.roll(E2, int(dist/2/(x[1]-x[0])))) * (x[1]-x[0])

    p11 = (beta1 + beta1)/(4*omega*mu_0) * p11
    p12 = (beta1 + beta2)/(4*omega*mu_0) * p12
    p21 = (beta2 + beta1)/(4*omega*mu_0) * p21
    p22 = (beta2 + beta2)/(4*omega*mu_0) * p22
    
    h11 = p11*beta1 + (omega*epsilon_0/4) * np.sum(dnmdn1s*np.roll(E1, -int(dist/2/(x[1]-x[0]))) * np.roll(E1, -int(dist/2/(x[1]-x[0])))) * (x[1]-x[0])
    h12 = p12*beta2 + (omega*epsilon_0/4) * np.sum(dnmdn2s*np.roll(E1, -int(dist/2/(x[1]-x[0]))) * np.roll(E2, int(dist/2/(x[1]-x[0])))) * (x[1]-x[0])
    h21 = p21*beta1 + (omega*epsilon_0/4) * np.sum(dnmdn1s*np.roll(E2, int(dist/2/(x[1]-x[0]))) * np.roll(E1, -int(dist/2/(x[1]-x[0])))) * (x[1]-x[0])
    h22 = p22*beta2 + (omega*epsilon_0/4) * np.sum(dnmdn2s*np.roll(E2, int(dist/2/(x[1]-x[0]))) * np.roll(E2, int(dist/2/(x[1]-x[0])))) * (x[1]-x[0])

    print(f"p11: {p11}, p12: {p12}, p21: {p21}, p22: {p22}")
    print(f"h11: {h11}, h12: {h12}, h21: {h21}, h22: {h22}")

    P = np.array([[p11, p12], [p21, p22]])
    H = np.array([[h11, h12], [h21, h22]])

    C_mat[idx, :, :] = np.linalg.inv(P) @ H
    eigenvalues_couplers_CMT[idx, :] = np.linalg.eigvals(C_mat[idx, :, :])
    print(f"c11: {C_mat[idx, 0, 0]}, c12: {C_mat[idx, 0, 1]}, c21: {C_mat[idx, 1, 0]}, c22: {C_mat[idx, 1, 1]}")
    print(f"eigenvalues_couplers_CMT: {eigenvalues_couplers_CMT[idx, :]/k0}")
    print(f"eigenvalues_couplers: {np.sqrt(eigenvalues_couplers[idx, :-1])/k0}")

plt.style.use('science')
fig, ax = plt.subplots(1, 1, dpi=400)
distances *= 1E6    

ax.plot(distances, np.sqrt(eigenvalues_couplers[:, -1])/(k0) - n0, '-', color='blue', label='EME')
ax.plot(distances, np.sqrt(eigenvalues_couplers[:, -2])/(k0) - n0, '-', color='blue')

ax.plot(distances, (eigenvalues_couplers_CMT[:, 0]/(k0))- n0, '--', color='red', label='CMT')
ax.plot(distances, (eigenvalues_couplers_CMT[:, 1]/(k0))- n0, '--', color='red')

ax2 = ax.twinx()
ax2.set_yscale('log')
ax2.plot(distances, np.abs(np.sqrt(eigenvalues_couplers[:, -1])/(k0)  - ( (eigenvalues_couplers_CMT[:, 0]/(k0))))/np.sqrt(eigenvalues_couplers[:, -1])/(k0), '-', color='green', label='Error')
# ax2.plot(distances, (np.sqrt(eigenvalues_couplers[:, -2])/(k0)  - ( (eigenvalues_couplers_CMT[:, 1]/(k0))))/np.sqrt(eigenvalues_couplers[:, -2])/(k0), '-', color='green')
ax2.set_ylabel('Relative error')
ax.set_xlabel('Separation distance ($\mu$m)')
ax.set_ylabel(r'$\beta/k_0 - n_0$')

# Format y-axis tick labels to scientific notation
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax.yaxis.offsetText.set_fontsize(7)  # Adjust the font size of the offset text

fig.legend(fontsize=7, loc=(0.60, 0.7))
ax.set_title('Coupler eigenvalues')
plt.tight_layout()
# fig.show()

fig.savefig('dimol/coupler.png', dpi=500)
plt.close("all")

# # Plot eigenvectors

# for idx, dist in enumerate(distances):
#     eigenvalues_couplers[idx, :] = np.load(f'dimol/eigenvalues_{idx}.npy')
#     print(eigenvalues_couplers[idx, :])
#     fig, ax = plt.subplots(1, 1)
#     E1 = np.load(f'dimol/eigenvectors_{idx}.npy')[:, -1]

#     E2 = np.load(f'dimol/eigenvectors_{idx}.npy')[:, -2]

#     ax.plot(x*1E6,  np.sqrt(eigenvalues_couplers[idx, -1])*np.ones(len(E1))+np.abs(E1*100)**2)
#     ax.plot(x*1E6,  np.sqrt(eigenvalues_couplers[idx, -2])*np.ones(len(E2))+np.abs(E2*100)**2)

#     ax.set_xlabel(r'$x$ ($\mu$m)')
#     ax.set_ylabel('Amplitude')
#     fig.savefig(f'dimol/eigenvectors_{idx}.png', dpi=500)

#     plt.close("all")