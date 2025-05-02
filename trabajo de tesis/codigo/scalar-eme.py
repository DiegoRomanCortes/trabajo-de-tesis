# Scalar Eigenmode Solver

# Copyright (C) 2025  Diego Roman-Cortes

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# contact: diego.roman.c@ug.uchile.cl

import numpy as np
from scipy.sparse import diags, linalg, spdiags, kronsum

# Define the parameters
Nx = 240
Ny = 240
Lx = 80E-6
Ly = 80E-6

x, dx = np.linspace(-Lx/2, Lx/2, Nx, retstep=True)
y, dy = np.linspace(-Ly/2, Ly/2, Ny, retstep=True)

X, Y = np.meshgrid(x, y, indexing="xy")


n0 = 1.48  # Refractive index of the background
dn1 = 4.00E-3  # Amplitude of the refractive index modulation
wavelength = 730E-9  # Wavelength of the light
k0 = 2 * np.pi / wavelength  # Wavenumber

n_eigen = 4

wx1 = 0.9E-6  # Width of the refractive index modulation
wy1 = 3.3E-6

a1 = 25E-6

##create matrix
if dx == dy:
    # Create the 2D Laplacian matrix for dx = dy using the 5-point stencil
    #  0  1  0
    #  1 -4  1
    #  0  1  0


    diag0 = -4 * np.ones(Nx * Ny)
    diag1 = np.ones(Nx * Ny - 1)
    diag1[np.arange(1, Nx * Ny) % Nx == 0] = 0  # Remove wrap-around connections
    diagNx = np.ones(Nx * Ny - Nx)

    diags_data = [diag0, diag1, diag1, diagNx, diagNx]
    diags_offsets = [0, -1, 1, -Nx, Nx]

    T = spdiags(diags_data, diags_offsets, Nx * Ny, Nx * Ny) / (dx**2)
else:
    # Just the Kronecker sum of the 1D Laplacian matrices
    diagx = np.ones([Nx])
    diagy = np.ones([Ny])

    diagsx = np.array([diagx, -2 * diagx, diagx])
    diagsy = np.array([diagy, -2 * diagy, diagy])
    Dx = spdiags(diagsx, np.array([-1, 0, 1]), Nx, Nx) / (dx**2)
    Dy = spdiags(diagsy, np.array([-1, 0, 1]), Ny, Ny) / (dy**2)

    T = kronsum(Dx, Dy)

def dn_func(X, Y, wx, wy):
    output = np.tanh(33.0*np.exp(-(X/wx)**2- ((Y)/wy)**2))
    return output


# Create the refractive index modulation
dn_array = np.zeros(X.shape)
dn_array += dn1 * dn_func(X, Y, wx1, wy1)

# Create the sparse matrix for the Helmholtz problem
n = n0 + dn_array
n = diags(n.reshape(Nx*Ny),(0))

H = T + (k0*n)**2

# Solve the eigenvalue problem
eigenvalues, eigenvectors = linalg.eigsh(H, k=n_eigen, which='LA')

np.save(f"eigenvalues.npy", eigenvalues)
np.save(f"eigenvectors.npy", eigenvectors)