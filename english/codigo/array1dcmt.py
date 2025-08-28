 
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from cupyx.scipy.sparse import diags, linalg

# Define the parameters
N = 5000  # Number of grid points
L = 500e-6  # Length of the domain

dx = L / (N - 1)
x = cp.linspace(-L/2, L/2, num=N)
np.save('1darraycmt/x.npy', x)

n0 = 1.48  # Refractive index of the background
dn1 = 4.00E-4  # Amplitude of the refractive index modulation

wavelength = 730E-9  # Wavelength of the light
k0 = 2 * cp.pi / wavelength  # Wavenumber

n_eigen = 21  # Number of eigenvalues to compute

wx = 3E-6  # Width of the refractive index modulation

a1 = 17E-6

def dn_func(x):
    output = cp.tanh(33.0*cp.exp(-(x/wx)**2))
    return output

# isolated waveguide
dn1_array = cp.zeros(N)
dn1_array += dn1 * dn_func(x)

n1 = n0 + dn1_array
k1 = k0 * n1

H1 = diags([1, -2, 1], [-1, 0, 1], shape=(N, N)) / dx**2 + diags(k1**2)

# Solve the eigenvalue problem
eigenvalues1, eigenvectors1 = linalg.eigsh(H1, k=1, which='LA')

np.save('1darraycmt/eigenvalues_CMT.npy', eigenvalues1)
np.save('1darraycmt/eigenvectors_CMT.npy', eigenvectors1)


# Coupled waveguides

dn_array = cp.zeros(N)
dn_array += dn1 * dn_func(x)

for i in range(1, 11):
    dn_array += dn1 * dn_func(x-i*a1)
    dn_array += dn1 * dn_func(x+i*a1)

np.save(f'1darraycmt/n_array_EME.npy', (n0+dn_array))
np.save(f'1darraycmt/n_array_CMT.npy', (n0+dn1*dn_func(x)))

n = n0 + dn_array
k = k0 * n

H = diags([1, -2, 1], [-1, 0, 1], shape=(N, N)) / dx**2 + diags(k**2)

# Solve the eigenvalue problem
eigenvalues, eigenvectors = linalg.eigsh(H, k=n_eigen, which='LA')
np.save(f'1darraycmt/eigenvalues_EME.npy', eigenvalues)
np.save(f'1darraycmt/eigenvectors_EME.npy', eigenvectors)
