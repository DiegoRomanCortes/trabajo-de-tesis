import numpy as np
from scipy.sparse import diags, linalg, spdiags, kronsum

# Define the parameters
Nx = 400
Ny = 400
Lx = 200E-6
Ly = 200E-6

x, dx = np.linspace(-Lx/2, Lx/2, Nx, retstep=True)
y, dy = np.linspace(-Ly/2, Ly/2, Ny, retstep=True)

X, Y = np.meshgrid(x, y, indexing="xy")


n0 = 1.48  # Refractive index of the background
dn1 = 1.530E-3  # 4.4 Amplitude of the refractive index modulation
wavelength = 730E-9  # Wavelength of the light
k0 = 2 * np.pi / wavelength  # Wavenumber

angles = np.linspace(0.4, 1.1, 8+7*7)

np.save("angles.npy", angles)

n_eigen = 4

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

dn_array_kappa_sigma = np.zeros(X.shape)

dn_array_kappa_sigma += dn1 * dn_func(X+a1*np.cos(np.pi/2)/2, Y+a1*np.sin(np.pi/2)/2, wx1, wy1)
dn_array_kappa_sigma += dn1 * dn_func(X-a1*np.cos(np.pi/2)/2, Y-a1*np.sin(np.pi/2)/2, wx1, wy1)

# Create the sparse matrix for the Helmholtz problem
n_kappa_sigma = n0 + dn_array_kappa_sigma
n_kappa_sigma = diags(n_kappa_sigma.reshape(Nx*Ny),(0))

T = kronsum(Dx,Dy)
H_kappa_sigma = T + (k0*n_kappa_sigma)**2

# Solve the eigenvalue problem
eigenvalues_kappa_sigma = linalg.eigsh(H_kappa_sigma, k=n_eigen, which='LA', return_eigenvectors=False)

np.save(f"coupling_kappa_sigma.npy", np.abs(np.sqrt(eigenvalues_kappa_sigma[-3]) - np.sqrt(eigenvalues_kappa_sigma[-4]))/2*1e-2)


for idx, angle in enumerate(angles):
    print(f"Angle: {angle}")
    dn_array_kappa_theta = np.zeros(X.shape)
    dn_array_t_pi = np.zeros(X.shape)
    dn_array_t_theta_1 = np.zeros(X.shape)
    dn_array_t_theta_2 = np.zeros(X.shape)
    dn_array_t_theta_3 = np.zeros(X.shape)

    dn_array_kappa_theta += dn1 * dn_func(X+a1*np.cos(angle)/2, Y+a1*np.sin(angle)/2, wx1, wy1)
    dn_array_kappa_theta += dn1 * dn_func(X-a1*np.cos(angle)/2, Y-a1*np.sin(angle)/2, wx1, wy1)

    d_t_pi = 2*a1*np.cos(angle)
    dn_array_t_pi += dn1 * dn_func(X+d_t_pi*np.cos(0)/2, Y+d_t_pi*np.sin(0)/2, wx1, wy1)
    dn_array_t_pi += dn1 * dn_func(X-d_t_pi*np.cos(0)/2, Y-d_t_pi*np.sin(0)/2, wx1, wy1)

    d_t_theta_1 = a1 * np.sqrt(4*np.cos(angle)**2 + 1)
    angle_t_theta_1 = np.arctan(1/(2*np.cos(angle)))
    dn_array_t_theta_1 += dn1 * dn_func(X+d_t_theta_1*np.cos(angle_t_theta_1)/2, Y+d_t_theta_1*np.sin(angle_t_theta_1)/2, wx1, wy1)
    dn_array_t_theta_1 += dn1 * dn_func(X-d_t_theta_1*np.cos(angle_t_theta_1)/2, Y-d_t_theta_1*np.sin(angle_t_theta_1)/2, wx1, wy1)

    d_t_theta_2 = a1 * np.sqrt(2*(np.sin(angle)+1))
    angle_t_theta_2 = np.arctan((np.sin(angle)+1)/np.cos(angle))
    dn_array_t_theta_2 += dn1 * dn_func(X+d_t_theta_2*np.cos(angle_t_theta_2)/2, Y+d_t_theta_2*np.sin(angle_t_theta_2)/2, wx1, wy1)
    dn_array_t_theta_2 += dn1 * dn_func(X-d_t_theta_2*np.cos(angle_t_theta_2)/2, Y-d_t_theta_2*np.sin(angle_t_theta_2)/2, wx1, wy1)

    d_t_theta_3 = a1 * np.sqrt(2*np.sin(angle)+1)
    angle_t_theta_3 = np.pi/2
    dn_array_t_theta_3 += dn1 * dn_func(X+d_t_theta_3*np.cos(angle_t_theta_3)/2, Y+d_t_theta_3*np.sin(angle_t_theta_3)/2, wx1, wy1)
    dn_array_t_theta_3 += dn1 * dn_func(X-d_t_theta_3*np.cos(angle_t_theta_3)/2, Y-d_t_theta_3*np.sin(angle_t_theta_3)/2, wx1, wy1)

    print(a1)
    print(d_t_pi)
    print(d_t_theta_1)
    print(d_t_theta_2)
    print(d_t_theta_3)

    # Create the sparse matrix for the Helmholtz problem
    n_kappa_theta = n0 + dn_array_kappa_theta
    n_kappa_theta = diags(n_kappa_theta.reshape(Nx*Ny),(0))

    n_t_pi = n0 + dn_array_t_pi
    n_t_pi = diags(n_t_pi.reshape(Nx*Ny),(0))

    n_t_theta_1 = n0 + dn_array_t_theta_1
    n_t_theta_1 = diags(n_t_theta_1.reshape(Nx*Ny),(0))

    n_t_theta_2 = n0 + dn_array_t_theta_2
    n_t_theta_2 = diags(n_t_theta_2.reshape(Nx*Ny),(0))

    n_t_theta_3 = n0 + dn_array_t_theta_3
    n_t_theta_3 = diags(n_t_theta_3.reshape(Nx*Ny),(0))

    H_kappa_theta = T + (k0*n_kappa_theta)**2
    H_t_pi = T + (k0*n_t_pi)**2
    H_t_theta_1 = T + (k0*n_t_theta_1)**2
    H_t_theta_2 = T + (k0*n_t_theta_2)**2
    H_t_theta_3 = T + (k0*n_t_theta_3)**2

    # Solve the eigenvalue problem
    eigenvalues_kappa_theta = linalg.eigsh(H_kappa_theta, k=n_eigen, which='LA', return_eigenvectors=False)
    eigenvalues_t_pi = linalg.eigsh(H_t_pi, k=n_eigen, which='LA', return_eigenvectors=False)
    eigenvalues_t_theta_1 = linalg.eigsh(H_t_theta_1, k=n_eigen, which='LA', return_eigenvectors=False)
    eigenvalues_t_theta_2 = linalg.eigsh(H_t_theta_2, k=n_eigen, which='LA', return_eigenvectors=False)
    eigenvalues_t_theta_3 = linalg.eigsh(H_t_theta_3, k=n_eigen, which='LA', return_eigenvectors=False)

    np.save(f"coupling_kappa_theta_{idx}.npy", np.abs(np.sqrt(eigenvalues_kappa_theta[-3]) - np.sqrt(eigenvalues_kappa_theta[-4]))/2*1e-2)
    np.save(f"coupling_t_pi_{idx}.npy", np.abs(np.sqrt(eigenvalues_t_pi[-3]) - np.sqrt(eigenvalues_t_pi[-4]))/2*1e-2)
    np.save(f"coupling_t_theta_1_{idx}.npy", np.abs(np.sqrt(eigenvalues_t_theta_1[-3]) - np.sqrt(eigenvalues_t_theta_1[-4]))/2*1e-2)
    np.save(f"coupling_t_theta_2_{idx}.npy", np.abs(np.sqrt(eigenvalues_t_theta_2[-3]) - np.sqrt(eigenvalues_t_theta_2[-4]))/2*1e-2)
    np.save(f"coupling_t_theta_3_{idx}.npy", np.abs(np.sqrt(eigenvalues_t_theta_3[-3]) - np.sqrt(eigenvalues_t_theta_3[-4]))/2*1e-2)