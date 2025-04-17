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

n = n0 + np.load(f'1darraycmt/n_array_EME.npy')
    
def e_j(j):
    dist = distance * (j+1)//2 * (-1)**j
    E = np.abs(np.load(f'1darraycmt/eigenvectors_CMT.npy')[:, 0])
    E = np.roll(E, int(dist/(x[1]-x[0])))
    return E

def pij(i, j):
    pij = eigenvalue_CMT/(4*omega*mu_0) * np.sum(e_j(i) * e_j(j)) * (x[1]-x[0])
    return pij

def hij(i, j):
    nj = np.load(f'1darraycmt/n_array_CMT.npy')
    dnmdnj = n**2 - nj**2

    hij = pij(i, j) * eigenvalue_CMT + (omega*epsilon_0/4) * np.sum(dnmdnj * e_j(i) * e_j(j)) * (x[1]-x[0])
    return hij

# Load coupler data

P = np.zeros((len(eigenvalues_EME), len(eigenvalues_EME)))
H = np.zeros((len(eigenvalues_EME), len(eigenvalues_EME)))

for i in range(len(eigenvalues_EME)):
    for j in range(len(eigenvalues_EME)):
        P[i, j] = pij(i, j)
        H[i, j] = hij(i, j)

C_mat = np.linalg.inv(P) @ H
eigenvalues_CMT_calc = np.linalg.eigvals(C_mat)


plt.style.use('science')
fig, ax = plt.subplots(1, 1, dpi=400)
distance *= 1E6    


