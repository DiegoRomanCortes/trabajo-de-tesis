import numpy as np
from scipy.sparse import diags, linalg, spdiags, kronsum

# Define the parameters
Nx = 101
Ny = 101
Lx = 100E-6
Ly = 100E-6

x, dx = np.linspace(-Lx/2, Lx/2, Nx, retstep=True)
y, dy = np.linspace(-Ly/2, Ly/2, Ny, retstep=True)

print(f"dx: {dx*1e6:.2f} um")
print(f"dy: {dy*1e6:.2f} um")

X, Y = np.meshgrid(x, y, indexing="xy")


n0 = 1.48  # Refractive index of the background
dn1 = 4.4E-4  # 41.530E-3 Amplitude of the refractive index modulation
wavelength = 730E-9  # Wavelength of the light
k0 = 2 * np.pi / wavelength  # Wavenumber

distances = np.linspace(4e-6, 25e-6, 50)

np.save("dimol2/distances.npy", distances)

n_eigen = 2

wx1 = 1.0E-6  # Width of the refractive index modulation
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


for idx, dist in enumerate(distances):
    print(f"Distance: {dist*1e6:.2f} um")
    dn_array = np.zeros(X.shape)

    dn_array += dn1 * dn_func(X+(dist/2), Y, wx1, wy1)
    dn_array += dn1 * dn_func(X-(dist/2), Y, wx1, wy1)

    # Create the sparse matrix for the Helmholtz problem
    n = n0 + dn_array
    n = diags(n.reshape(Nx*Ny),(0))

    T = kronsum(Dx,Dy)
    H = T + ((k0*n)**2)

    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = linalg.eigsh(H, k=n_eigen, which='LM', sigma=(k0*(n0+dn1*3)**2))
    print((np.sqrt(eigenvalues[-1])/k0) - n0)
    print((np.sqrt(eigenvalues[-2])/k0) - n0)
    print(np.abs(np.sqrt(eigenvalues[-1]) - np.sqrt(eigenvalues[-2]))/2*1e-2)
    print("\n")
    np.save(f"dimol2/eigenvalues_{idx}.npy", eigenvalues)
    np.save(f"dimol2/eigenvectors_{idx}.npy", eigenvectors)