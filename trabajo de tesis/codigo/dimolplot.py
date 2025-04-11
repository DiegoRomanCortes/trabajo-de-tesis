import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.constants import c, epsilon_0

# Load single data
eigenvalues1 = np.load('dimol/eigenvalues1.npy')
eigenvalues2 = np.load('dimol/eigenvalues2.npy')

x = np.load('dimol/x.npy')
distances = np.load('dimol/distances.npy')
wavelength = 730E-9
k0 = 2 * np.pi / wavelength
omega = k0 * c    # Frequency of the light

beta1 = np.sqrt(eigenvalues1[-1])
beta2 = np.sqrt(eigenvalues2[-1])

E1 = np.abs(np.load('dimol/eigenvectors1.npy')[:, -1])
E2 = np.abs(np.load('dimol/eigenvectors2.npy')[:, -1])

E1 /= (np.sum(np.abs(E1)**2)*(x[1]-x[0]))
E2 /= (np.sum(np.abs(E2)**2)*(x[1]-x[0]))
# Load coupler data
plt.style.use('science')
fig, ax = plt.subplots(1, 1)

eigenvalues_couplers = np.zeros((len(distances), 3))
dnmdn1s = np.zeros((len(distances), len(x)))
dnmdn2s = np.zeros((len(distances), len(x)))
betasp = np.zeros(len(distances))
betasm = np.zeros(len(distances))
for idx, dist in enumerate(distances):
    eigenvalues_couplers[idx, :] = np.load(f'dimol/eigenvalues_{idx}.npy')
    dnmdn1s[idx, :] = np.load(f'dimol/dn1_array_{idx}.npy')
    dnmdn2s[idx, :] = np.load(f'dimol/dn2_array_{idx}.npy')
    
    k11 = np.sum(dnmdn1s[idx, :] * np.roll(E1, -int(dist/2/(x[1]-x[0]))) * np.roll(E1, -int(dist/2/(x[1]-x[0])))) * (x[1]-x[0])
    k12 = np.sum(dnmdn2s[idx, :] * np.roll(E1, -int(dist/2/(x[1]-x[0]))) * np.roll(E2, int(dist/2/(x[1]-x[0])))) * (x[1]-x[0])
    k21 = np.sum(dnmdn1s[idx, :] * np.roll(E1, -int(dist/2/(x[1]-x[0]))) * np.roll(E2, int(dist/2/(x[1]-x[0])))) * (x[1]-x[0])
    k22 = np.sum(dnmdn2s[idx, :] * np.roll(E2, int(dist/2/(x[1]-x[0]))) * np.roll(E2, int(dist/2/(x[1]-x[0])))) * (x[1]-x[0])

    cte = 1/4
    # cte = 1
    k11 *= cte
    k12 *= cte
    k21 *= cte
    k22 *= cte

    xs = np.sum(np.roll(E1, -int(dist/2/(x[1]-x[0]))) * np.roll(E2, int(dist/2/(x[1]-x[0])))) /np.sqrt(np.sum(np.roll(E1, -int(dist/2/(x[1]-x[0]))) * np.roll(E1, -int(dist/2/(x[1]-x[0])))) * np.sum(np.roll(E2, int(dist/2/(x[1]-x[0]))) * np.roll(E2, int(dist/2/(x[1]-x[0])))) )
    print(k12*1e-3, k21*1e-3, xs*1e-3, sep='\n', end='\n\n')
    beta1prime = beta1 + k11
    beta2prime = beta2 + k22
    k12prime = k12 + beta2 * xs 
    k21prime = k21 + beta1 * xs

    det = np.sqrt((1 / (1-xs**2))*(xs * (beta1prime+beta2prime)/2 - k12prime)**2 + ((beta2prime-beta1prime)**2/4)/(1-xs**2))

    betasp[idx] = ((beta1prime+beta2prime)/2 - xs*k12prime)/(1-xs**2) + det
    betasm[idx] = ((beta1prime+beta2prime)/2 - xs*k12prime)/(1-xs**2) - det
    print(det)

distances *= 1E6

ax.plot(distances, np.sqrt(eigenvalues_couplers[:, -1])/k0, '-', color='blue')
ax.plot(distances, np.sqrt(eigenvalues_couplers[:, -2])/k0, '-', color='blue')



ax.plot(distances, betasp/k0, '.', color='red')
ax.plot(distances, betasm/k0, '.', color='red')

ax.set_xlabel('Separation distance ($\mu$m)')
ax.set_ylabel(r'$\beta/k_0$')

fig.savefig('dimol/coupler.png', dpi=500)
plt.close("all")

# Plot eigenvectors

for idx, dist in enumerate(distances):
    eigenvalues_couplers[idx, :] = np.load(f'dimol/eigenvalues_{idx}.npy')
    print(eigenvalues_couplers[idx, :])
    fig, ax = plt.subplots(1, 1)
    E1 = np.load(f'dimol/eigenvectors_{idx}.npy')[:, -1]

    E2 = np.load(f'dimol/eigenvectors_{idx}.npy')[:, -2]

    ax.plot(x*1E6,  np.sqrt(eigenvalues_couplers[idx, -1])*np.ones(len(E1))+np.abs(E1*100)**2)
    ax.plot(x*1E6,  np.sqrt(eigenvalues_couplers[idx, -2])*np.ones(len(E2))+np.abs(E2*100)**2)

    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel('Amplitude')
    fig.savefig(f'dimol/eigenvectors_{idx}.png', dpi=500)

    plt.close("all")