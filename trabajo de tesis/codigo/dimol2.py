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
wavelength = 730E-9  # Wavelength of the light
k0 = 2 * np.pi / wavelength  # Wavenumber

distances = np.linspace(4e-6, 25e-6, 22)

# print(f"Distances: {distances*1e6:.2f} um")

np.save("dimol2/distances.npy", distances)

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


for idx, dist in enumerate(distances):
    print(f"Distance: {dist*1e6:.2f} um")
    dn_array = np.zeros(X.shape)

    dn_array += dn1 * dn_func(X+(dist/2), Y, wx1, wy1)
    dn_array += dn1 * dn_func(X-(dist/2), Y, wx1, wy1)

    # Create the sparse matrix for the Helmholtz problem
    n = n0 + dn_array
    n = diags(n.reshape(Nx*Ny),(0))

    
    H = T + (k0**2 * n**2)

    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = linalg.eigsh(H, k=n_eigen, which='LM', sigma=(k0*(n0+dn1))**2)
    print((np.sqrt(eigenvalues[-1])/k0) - n0)
    print((np.sqrt(eigenvalues[-2])/k0) - n0)
    print(np.abs(np.sqrt(eigenvalues[-1]) - np.sqrt(eigenvalues[-2]))/2*1e-2)
    print("\n")
    np.save(f"dimol2/eigenvalues_{idx}.npy", eigenvalues)
    np.save(f"dimol2/eigenvectors_{idx}.npy", eigenvectors)