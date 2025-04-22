from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from cupyx.scipy.sparse import diags, linalg, spdiags, kronsum
import scienceplots
from cmcrameri import cm

# Define the parameters
Nx = 120
Ny = 120
Lx = 60E-6
Ly = 60E-6


dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

x = cp.linspace(-Lx/2, Lx/2, Nx)
y = cp.linspace(-Ly/2, Ly/2, Ny)

X, Y = cp.meshgrid(x, y, indexing="xy")


n0 = 1.48  # Refractive index of the background
dn1 = 8.0E-4  # 4.4 Amplitude of the refractive index modulation
wavelength = 730E-9  # Wavelength of the light
k0 = 2 * cp.pi / wavelength  # Wavenumber

angles = cp.linspace(0, np.pi/2, 10)


n_eigen = 6 
eigenvalues_tot = cp.zeros((len(angles), n_eigen))
eigenvectors_tot = cp.zeros((len(angles), Nx*Ny, n_eigen))

wx1 = 1.4E-6  # Width of the refractive index modulation
wy1 = 3.1E-6

a1 = 18E-6
a2 = 18E-6

##create matrix
diagx = cp.ones([Nx])
diagy = cp.ones([Ny])

diagsx = cp.array([diagx, -2*diagx, diagx])
diagsy = cp.array([diagy, -2*diagy, diagy])
Dx = spdiags(diagsx, np.array([-1, 0, 1]), Nx, Nx)/(dx**2)
Dy = spdiags(diagsy, np.array([-1, 0, 1]), Ny, Ny)/(dy**2)

def dn_func(X, Y, wx, wy):
    output = cp.tanh(33.0*cp.exp(-(X/wx)**2- ((Y)/wy)**2))
    return output

zmin = 5E-3
zmax = 25E-3

zarr = cp.linspace(zmin, zmax, 1000)

propagation_dynamics = cp.zeros((len(zarr), len(angles), Nx), dtype=cp.complex128)

for idx, angle in enumerate(angles):
    print(f"Angle: {angle}")
    dn_array = cp.zeros(X.shape)

    dn_array += dn1 * dn_func(X+a1*cp.cos(angle)/2, Y+a1*cp.sin(angle)/2, wx1, wy1)
    dn_array += dn1 * dn_func(X-a1*cp.cos(angle)/2, Y-a1*cp.sin(angle)/2, wx1, wy1)

    # Create the sparse matrix for the Helmholtz problem
    n = n0 + dn_array
    n = diags(n.reshape(Nx*Ny),(0))

    T = kronsum(Dx,Dy)
    H = T + (k0*n)**2

    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = linalg.eigsh(H, k=n_eigen, which='LA')
    eigenvalues_tot[idx, :] = eigenvalues
    eigenvectors_tot[idx, :, :] = eigenvectors

    cp.save(f"eigenvalues_{idx}.npy", eigenvalues)