import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.constants import c, epsilon_0, mu_0

# Load single data
eigenvalue_CMT = np.load('1darraycmt/eigenvalues_CMT.npy')
eigenvalues_EME = np.load('1darraycmt/eigenvalues_EME.npy')

x = np.load('1darraycmt/x.npy')
distance = 17E-6
wavelength = 730E-9
k0 = 2 * np.pi / wavelength
n0 = 1.48
omega = k0 * c   # Frequency of the light

n = np.load(f'1darraycmt/n_array_EME.npy')
    
def e_j(j):
    dist = distance * (j-10)
    E = np.abs(np.load(f'1darraycmt/eigenvectors_CMT.npy')[:, -1])
    E = np.roll(E, int(dist/(x[1]-x[0])))
    return E

def pij(i, j):
    pij = np.sqrt(eigenvalue_CMT)/(2*omega*mu_0) * np.sum(e_j(i) * e_j(j)) * (x[1]-x[0])
    return pij

def hij(i, j):
    dist = distance * (j-10)
    nj = np.load(f'1darraycmt/n_array_CMT.npy')
    dnmdnj = n**2 - np.roll(nj, int(dist/(x[1]-x[0])))**2

    hij = pij(i, j) * np.sqrt(eigenvalue_CMT) + (omega*epsilon_0/4) * np.sum(dnmdnj * e_j(i) * e_j(j)) * (x[1]-x[0])
    return hij

# Load coupler data

P = np.zeros((len(eigenvalues_EME), len(eigenvalues_EME)))
H = np.zeros((len(eigenvalues_EME), len(eigenvalues_EME)))

for i in range(len(eigenvalues_EME)):
    for j in range(len(eigenvalues_EME)):
        P[i, j] = pij(i, j)[0]
        H[i, j] = hij(i, j)[0]
        # print(f"p{i+1}{j+1}: {P[i, j]}, h{i+1}{j+1}: {H[i, j]}")

C_mat = np.linalg.inv(P) @ H
eigenvalues_CMT_calc, eigenvecs_CMT_calc = np.linalg.eig(C_mat)


plt.style.use('science')
fig, ax = plt.subplots(1, 1, dpi=600, figsize=(10, 6))

ini_EME = np.exp(-x**2/(2*3E-6**2), dtype=complex)
ini_CMT = np.zeros(len(eigenvalues_EME), dtype=complex)
ini_CMT[10] = 1

sol_EME = np.zeros(len(x), dtype=complex)
sol_CMT = np.zeros(len(eigenvalues_EME), dtype=complex)

z = 18e-3
for i in range(len(eigenvalues_EME)):
    eigenvecs_EME_i = np.load(f'1darraycmt/eigenvectors_EME.npy')[:, i]
    sol_EME += eigenvecs_EME_i * np.exp(1j * np.sqrt(eigenvalues_EME[i]) * z) * np.dot(eigenvecs_EME_i, ini_EME)
    sol_CMT += eigenvecs_CMT_calc[:, i] * np.exp(1j * (eigenvalues_CMT_calc[i]) * z) * np.dot(eigenvecs_CMT_calc[:, i], ini_CMT)

sol_CMT /= np.max(np.abs(sol_CMT))
sol_EME /= np.max(np.abs(sol_EME))
ax.plot(x*1e6, np.abs(sol_EME)**2, label='EME')
ax.set_xlabel(r'$x$ ($\mu$m)', fontsize=18)
ax.set_ylabel('Intensidad (un. arb.)', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=18)

ax.plot(np.arange(-10, 11)*distance*1e6, np.abs(sol_CMT)**2, ".--",label='CMT')
ax.legend(loc=(0.7, 0.8), fontsize=18)
# ax.set_title('Output field comparison between EME and CMT', fontsize=10)
fig.savefig('1darraycmt/1darraycmt.png', dpi=600)
# fig.show()
print()