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
dn1 = 1.530E-3  # 4.4 Amplitude of the refractive index modulation
wavelength = 730E-9  # Wavelength of the light
k0 = 2 * np.pi / wavelength  # Wavenumber

angles = np.linspace(0, np.pi/2, 50)

np.save("angles.npy", angles)

n_eigen = 4
eigenvalues_tot = np.zeros((len(angles), n_eigen))
eigenvectors_tot = np.zeros((len(angles), Nx*Ny, n_eigen))

wx1 = 0.9E-6  # Width of the refractive index modulation
wy1 = 3.3E-6

a1 = 25E-6

##create matrix
diagx = np.ones([Nx])
diagy = np.ones([Ny])

diagsx = np.array([diagx, -2*diagx, diagx])
diagsy = np.array([diagy, -2*diagy, diagy])
Dx = spdiags(diagsx, np.array([-1, 0, 1]), Nx, Nx)/(dx**2)
Dy = spdiags(diagsy, np.array([-1, 0, 1]), Ny, Ny)/(dy**2)

def dn_func(X, Y, wx, wy):
    output = np.tanh(33.0*np.exp(-(X/wx)**2- ((Y)/wy)**2))
    return output

zmin = 5E-3
zmax = 25E-3

zarr = np.linspace(zmin, zmax, 1000)

propagation_dynamics = np.zeros((len(zarr), len(angles), Nx), dtype=np.complex128)

for idx, angle in enumerate(angles):
    print(f"Angle: {angle}")
    dn_array = np.zeros(X.shape)

    dn_array += dn1 * dn_func(X+a1*np.cos(angle)/2, Y+a1*np.sin(angle)/2, wx1, wy1)
    dn_array += dn1 * dn_func(X-a1*np.cos(angle)/2, Y-a1*np.sin(angle)/2, wx1, wy1)

    # Create the sparse matrix for the Helmholtz problem
    n = n0 + dn_array
    n = diags(n.reshape(Nx*Ny),(0))

    T = kronsum(Dx,Dy)
    H = T + (k0*n)**2

    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = linalg.eigsh(H, k=n_eigen, which='LA')
    eigenvalues_tot[idx, :] = eigenvalues
    eigenvectors_tot[idx, :, :] = eigenvectors
    print(np.sqrt(eigenvalues[0])/k0 - n0)
    print(np.sqrt(eigenvalues[1])/k0 - n0)
    print(np.abs(np.sqrt(eigenvalues[1]) - np.sqrt(eigenvalues[0]))/2*1e-2)
    print("\n")
    np.save(f"eigenvalues_{idx}.npy", eigenvalues)