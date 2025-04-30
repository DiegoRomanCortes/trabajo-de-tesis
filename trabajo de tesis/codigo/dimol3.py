import numpy as np
from scipy.sparse import diags, linalg, spdiags, kronsum
import matplotlib.pyplot as plt
# Define the parameters
Nx = 500
Ny = 500
Lx = 80E-6
Ly = 80E-6

x, dx = np.linspace(-Lx/2, Lx/2, Nx, retstep=True)
y, dy = np.linspace(-Ly/2, Ly/2, Ny, retstep=True)

print(f"dx: {dx*1e6:.2f} um")
print(f"dy: {dy*1e6:.2f} um")

X, Y = np.meshgrid(x, y, indexing="xy")


n0 = 1.48  # Refractive index of the background
dn1 = 5.2E-4  # 4.1E-4 Amplitude of the refractive index modulation
dn2 = 10.423E-4  # 4.1E-4 Amplitude of the refractive index modulation

distance = 7e-6  # Wavelength of the light


wavelengths = np.linspace(600e-9, 800e-9, 21)

# print(f"Distances: {distances*1e6:.2f} um")

np.save("dimol3/wavelengths.npy", wavelengths)

n_eigen = 2

wx1 = 0.9E-6  # Width of the refractive index modulation
wy1 = 3.4E-6


# Create the 2D Laplacian matrix for dx != dy using the 5-point stencil
main_diag = -2 * (1 / dx**2 + 1 / dy**2) * np.ones(Nx * Ny)
side_diag_x = 1 / dx**2 * np.ones(Nx * Ny)
side_diag_y = 1 / dy**2 * np.ones(Nx * Ny)

# Adjust for boundaries
for i in range(1, Ny):
    side_diag_x[i * Nx - 1] = 0  # Remove connections at the right boundary
    side_diag_x[i * Nx] = 0      # Remove connections at the left boundary

# Create the sparse matrix
diagonals = [main_diag, side_diag_x, side_diag_x, side_diag_y, side_diag_y]
offsets = [0, -1, 1, -Nx, Nx]
T = diags(diagonals, offsets, format="csr")

def dn_func(X, Y, wx, wy):
    output = np.tanh(33.0*np.exp(-(X/wx)**2- ((Y)/wy)**2))
    return output

dn1_array = np.zeros(X.shape)

dn1_array += dn1 * dn_func(X+(distance/2), Y, wx1, wy1)
dn1_array += dn1 * dn_func(X-(distance/2), Y, wx1, wy1)

# Create the sparse matrix for the Helmholtz problem
n1 = n0 + dn1_array
n1 = diags(n1.reshape(Nx*Ny),(0))

dn2_array = np.zeros(X.shape)

dn2_array += dn2 * dn_func(X+(distance/2), Y, wx1, wy1)
dn2_array += dn2 * dn_func(X-(distance/2), Y, wx1, wy1)

# Create the sparse matrix for the Helmholtz problem
n2 = n0 + dn2_array
n2 = diags(n2.reshape(Nx*Ny),(0))


for idx, wavelength in enumerate(wavelengths):
    print(f"Wavelength: {wavelength*1e9:.0f} nm")

    k0 = 2 * np.pi / wavelength  # Wavenumber    
    H1 = T + (k0**2 * n1**2)
    H2 = T + (k0**2 * n2**2)

    # Solve the eigenvalue problem
    eigenvalues1, eigenvectors1 = linalg.eigsh(H1, k=1, which='LM', sigma=(k0*(n0+dn1))**2)
    eigenvalues2, eigenvectors2 = linalg.eigsh(H2, k=2, which='LM', sigma=(k0*(n0+dn2))**2)

    print(f"Eigenvalues 1: {np.sqrt(eigenvalues1[-1])/k0 - n0:.2e}")
    print(f"Eigenvalues 2: {np.sqrt(eigenvalues2[-2])/k0 - n0:.2e}")

    np.save(f"dimol3/eigenvalues_S_{idx}.npy", eigenvalues1[-1])
    np.save(f"dimol3/eigenvectors_S_{idx}.npy", eigenvectors1[:, -1])

    np.save(f"dimol3/eigenvalues_P_{idx}.npy", eigenvalues2[-2])
    np.save(f"dimol3/eigenvectors_P_{idx}.npy", eigenvectors2[:, -2])