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

n_eigen = 2

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

dn_array_kappa_sigma_1 = np.zeros(X.shape)
dn_array_kappa_sigma_2 = np.zeros(X.shape)

dn_array_kappa_sigma_1 += dn1 * dn_func(X+a1*np.cos(np.pi/2)/2, Y+a1*np.sin(np.pi/2)/2, wx1, wy1)
dn_array_kappa_sigma_2 += dn1 * dn_func(X-a1*np.cos(np.pi/2)/2, Y-a1*np.sin(np.pi/2)/2, wx1, wy1)

# Create the sparse matrix for the Helmholtz problem
n_kappa_sigma_1 = n0 + dn_array_kappa_sigma_1
n_kappa_sigma_1 = diags(n_kappa_sigma_1.reshape(Nx*Ny),(0))

n_kappa_sigma_2 = n0 + dn_array_kappa_sigma_2
n_kappa_sigma_2 = diags(n_kappa_sigma_2.reshape(Nx*Ny),(0))

T = kronsum(Dx,Dy)

H_kappa_sigma_1 = T + (k0*n_kappa_sigma_1)**2
H_kappa_sigma_2 = T + (k0*n_kappa_sigma_2)**2

# Solve the eigenvalue problem
eigenvalues_kappa_sigma_1, eigenvecs_kappa_sigma_1 = linalg.eigsh(H_kappa_sigma_1, k=n_eigen, which='LA', return_eigenvectors=True)
eigenvalues_kappa_sigma_2, eigenvecs_kappa_sigma_2 = linalg.eigsh(H_kappa_sigma_2, k=n_eigen, which='LA', return_eigenvectors=True)

overlap_kappa_sigma = np.dot(eigenvecs_kappa_sigma_1[:, -2], eigenvecs_kappa_sigma_2[:, -2]) * dx * dy
print(np.dot(eigenvecs_kappa_sigma_1[:, -2], eigenvecs_kappa_sigma_1[:, -2]) * dx * dy)
print(np.dot(eigenvecs_kappa_sigma_2[:, -2], eigenvecs_kappa_sigma_2[:, -2]) * dx * dy)

overlap_kappa_sigma /= np.dot(eigenvecs_kappa_sigma_1[:, -2], eigenvecs_kappa_sigma_1[:, -2]) * dx * dy

np.save(f"overlap_kappa_sigma.npy", overlap_kappa_sigma)


for idx, angle in enumerate(angles):
    print(f"Angle: {angle}")
    dn_array_kappa_theta_1 = np.zeros(X.shape)
    dn_array_kappa_theta_2 = np.zeros(X.shape)

    dn_array_t_pi_1 = np.zeros(X.shape)
    dn_array_t_pi_2 = np.zeros(X.shape)

    dn_array_t_theta_1_1 = np.zeros(X.shape)
    dn_array_t_theta_1_2 = np.zeros(X.shape)

    dn_array_t_theta_2_1 = np.zeros(X.shape)
    dn_array_t_theta_2_2 = np.zeros(X.shape)

    dn_array_t_theta_3_1 = np.zeros(X.shape)
    dn_array_t_theta_3_2 = np.zeros(X.shape)

    dn_array_kappa_theta_1 += dn1 * dn_func(X+a1*np.cos(angle)/2, Y+a1*np.sin(angle)/2, wx1, wy1)
    dn_array_kappa_theta_2 += dn1 * dn_func(X-a1*np.cos(angle)/2, Y-a1*np.sin(angle)/2, wx1, wy1)

    d_t_pi = 2*a1*np.cos(angle)
    dn_array_t_pi_1 += dn1 * dn_func(X+d_t_pi*np.cos(0)/2, Y+d_t_pi*np.sin(0)/2, wx1, wy1)
    dn_array_t_pi_2 += dn1 * dn_func(X-d_t_pi*np.cos(0)/2, Y-d_t_pi*np.sin(0)/2, wx1, wy1)

    d_t_theta_1 = a1 * np.sqrt(4*np.cos(angle)**2 + 1)
    angle_t_theta_1 = np.arctan(1/(2*np.cos(angle)))
    dn_array_t_theta_1_1 += dn1 * dn_func(X+d_t_theta_1*np.cos(angle_t_theta_1)/2, Y+d_t_theta_1*np.sin(angle_t_theta_1)/2, wx1, wy1)
    dn_array_t_theta_1_2 += dn1 * dn_func(X-d_t_theta_1*np.cos(angle_t_theta_1)/2, Y-d_t_theta_1*np.sin(angle_t_theta_1)/2, wx1, wy1)

    d_t_theta_2 = a1 * np.sqrt(2*(np.sin(angle)+1))
    angle_t_theta_2 = np.arctan((np.sin(angle)+1)/np.cos(angle))
    dn_array_t_theta_2_1 += dn1 * dn_func(X+d_t_theta_2*np.cos(angle_t_theta_2)/2, Y+d_t_theta_2*np.sin(angle_t_theta_2)/2, wx1, wy1)
    dn_array_t_theta_2_2 += dn1 * dn_func(X-d_t_theta_2*np.cos(angle_t_theta_2)/2, Y-d_t_theta_2*np.sin(angle_t_theta_2)/2, wx1, wy1)

    d_t_theta_3 = a1 * np.sqrt(2*np.sin(angle)+1)
    angle_t_theta_3 = np.pi/2
    dn_array_t_theta_3_1 += dn1 * dn_func(X+d_t_theta_3*np.cos(angle_t_theta_3)/2, Y+d_t_theta_3*np.sin(angle_t_theta_3)/2, wx1, wy1)
    dn_array_t_theta_3_2 += dn1 * dn_func(X-d_t_theta_3*np.cos(angle_t_theta_3)/2, Y-d_t_theta_3*np.sin(angle_t_theta_3)/2, wx1, wy1)

    # Create the sparse matrix for the Helmholtz problem
    n_kappa_theta_1 = n0 + dn_array_kappa_theta_1
    n_kappa_theta_2 = n0 + dn_array_kappa_theta_2
    
    n_kappa_theta_1 = diags(n_kappa_theta_1.reshape(Nx*Ny),(0))
    n_kappa_theta_2 = diags(n_kappa_theta_2.reshape(Nx*Ny),(0))

    n_t_pi_1 = n0 + dn_array_t_pi_1
    n_t_pi_2 = n0 + dn_array_t_pi_2

    n_t_pi_1 = diags(n_t_pi_1.reshape(Nx*Ny),(0))
    n_t_pi_2 = diags(n_t_pi_2.reshape(Nx*Ny),(0))

    n_t_theta_1_1 = n0 + dn_array_t_theta_1_1
    n_t_theta_1_2 = n0 + dn_array_t_theta_1_2

    n_t_theta_1_1 = diags(n_t_theta_1_1.reshape(Nx*Ny),(0))
    n_t_theta_1_2 = diags(n_t_theta_1_2.reshape(Nx*Ny),(0))

    n_t_theta_2_1 = n0 + dn_array_t_theta_2_1
    n_t_theta_2_2 = n0 + dn_array_t_theta_2_2

    n_t_theta_2_1 = diags(n_t_theta_2_1.reshape(Nx*Ny),(0))
    n_t_theta_2_2 = diags(n_t_theta_2_2.reshape(Nx*Ny),(0))

    n_t_theta_3_1 = n0 + dn_array_t_theta_3_1
    n_t_theta_3_2 = n0 + dn_array_t_theta_3_2

    n_t_theta_3_1 = diags(n_t_theta_3_1.reshape(Nx*Ny),(0))
    n_t_theta_3_2 = diags(n_t_theta_3_2.reshape(Nx*Ny),(0))

    H_kappa_theta_1 = T + (k0*n_kappa_theta_1)**2
    H_kappa_theta_2 = T + (k0*n_kappa_theta_2)**2

    H_t_pi_1 = T + (k0*n_t_pi_1)**2
    H_t_pi_2 = T + (k0*n_t_pi_2)**2

    H_t_theta_1_1 = T + (k0*n_t_theta_1_1)**2
    H_t_theta_1_2 = T + (k0*n_t_theta_1_2)**2

    H_t_theta_2_1 = T + (k0*n_t_theta_2_1)**2
    H_t_theta_2_2 = T + (k0*n_t_theta_2_2)**2

    H_t_theta_3_1 = T + (k0*n_t_theta_3_1)**2
    H_t_theta_3_2 = T + (k0*n_t_theta_3_2)**2

    # Solve the eigenvalue problem
    eigenvalues_kappa_theta_1, eigenvecs_kappa_theta_1 = linalg.eigsh(H_kappa_theta_1, k=n_eigen, which='LA', return_eigenvectors=True)
    eigenvalues_kappa_theta_2, eigenvecs_kappa_theta_2 = linalg.eigsh(H_kappa_theta_2, k=n_eigen, which='LA', return_eigenvectors=True)

    eigenvalues_t_pi_1, eigenvecs_t_pi_1 = linalg.eigsh(H_t_pi_1, k=n_eigen, which='LA', return_eigenvectors=True)
    eigenvalues_t_pi_2, eigenvecs_t_pi_2 = linalg.eigsh(H_t_pi_2, k=n_eigen, which='LA', return_eigenvectors=True)

    eigenvalues_t_theta_1_1, eigenvecs_t_theta_1_1 = linalg.eigsh(H_t_theta_1_1, k=n_eigen, which='LA', return_eigenvectors=True)
    eigenvalues_t_theta_1_2, eigenvecs_t_theta_1_2 = linalg.eigsh(H_t_theta_1_2, k=n_eigen, which='LA', return_eigenvectors=True)

    eigenvalues_t_theta_2_1, eigenvecs_t_theta_2_1 = linalg.eigsh(H_t_theta_2_1, k=n_eigen, which='LA', return_eigenvectors=True)
    eigenvalues_t_theta_2_2, eigenvecs_t_theta_2_2 = linalg.eigsh(H_t_theta_2_2, k=n_eigen, which='LA', return_eigenvectors=True)

    eigenvalues_t_theta_3_1, eigenvecs_t_theta_3_1 = linalg.eigsh(H_t_theta_3_1, k=n_eigen, which='LA', return_eigenvectors=True)
    eigenvalues_t_theta_3_2, eigenvecs_t_theta_3_2 = linalg.eigsh(H_t_theta_3_2, k=n_eigen, which='LA', return_eigenvectors=True)

    overlap_kappa_theta = np.dot(eigenvecs_kappa_theta_1[:, -2], eigenvecs_kappa_theta_2[:, -2]) * dx * dy
    print(np.dot(eigenvecs_kappa_theta_1[:, -2], eigenvecs_kappa_theta_1[:, -2]) * dx * dy)
    print(np.dot(eigenvecs_kappa_theta_2[:, -2], eigenvecs_kappa_theta_2[:, -2]) * dx * dy)
    overlap_kappa_theta /= np.dot(eigenvecs_kappa_theta_1[:, -2], eigenvecs_kappa_theta_1[:, -2]) * dx * dy
    print(overlap_kappa_theta)
    np.save(f"overlap_kappa_theta_{idx}.npy", overlap_kappa_theta)
    overlap_t_pi = np.dot(eigenvecs_t_pi_1[:, -2], eigenvecs_t_pi_2[:, -2]) * dx * dy
    print(np.dot(eigenvecs_t_pi_1[:, -2], eigenvecs_t_pi_1[:, -2]) * dx * dy)
    print(np.dot(eigenvecs_t_pi_2[:, -2], eigenvecs_t_pi_2[:, -2]) * dx * dy)
    overlap_t_pi /= np.dot(eigenvecs_t_pi_1[:, -2], eigenvecs_t_pi_1[:, -2]) * dx * dy
    print(overlap_t_pi)
    np.save(f"overlap_t_pi_{idx}.npy", overlap_t_pi)
    overlap_t_theta_1 = np.dot(eigenvecs_t_theta_1_1[:, -2], eigenvecs_t_theta_1_2[:, -2]) * dx * dy
    print(np.dot(eigenvecs_t_theta_1_1[:, -2], eigenvecs_t_theta_1_1[:, -2]) * dx * dy)
    print(np.dot(eigenvecs_t_theta_1_2[:, -2], eigenvecs_t_theta_1_2[:, -2]) * dx * dy)
    overlap_t_theta_1 /= np.dot(eigenvecs_t_theta_1_1[:, -2], eigenvecs_t_theta_1_1[:, -2]) * dx * dy
    print(overlap_t_theta_1)
    np.save(f"overlap_t_theta_1_{idx}.npy", overlap_t_theta_1)
    overlap_t_theta_2 = np.dot(eigenvecs_t_theta_2_1[:, -2], eigenvecs_t_theta_2_2[:, -2]) * dx * dy
    print(np.dot(eigenvecs_t_theta_2_1[:, -2], eigenvecs_t_theta_2_1[:, -2]) * dx * dy)
    print(np.dot(eigenvecs_t_theta_2_2[:, -2], eigenvecs_t_theta_2_2[:, -2]) * dx * dy)
    overlap_t_theta_2 /= np.dot(eigenvecs_t_theta_2_1[:, -2], eigenvecs_t_theta_2_1[:, -2]) * dx * dy
    print(overlap_t_theta_2)
    np.save(f"overlap_t_theta_2_{idx}.npy", overlap_t_theta_2)
    overlap_t_theta_3 = np.dot(eigenvecs_t_theta_3_1[:, -2], eigenvecs_t_theta_3_2[:, -2]) * dx * dy
    print(np.dot(eigenvecs_t_theta_3_1[:, -2], eigenvecs_t_theta_3_1[:, -2]) * dx * dy)
    print(np.dot(eigenvecs_t_theta_3_2[:, -2], eigenvecs_t_theta_3_2[:, -2]) * dx * dy)
    overlap_t_theta_3 /= np.dot(eigenvecs_t_theta_3_1[:, -2], eigenvecs_t_theta_3_1[:, -2]) * dx * dy
    print(overlap_t_theta_3)
    np.save(f"overlap_t_theta_3_{idx}.npy", overlap_t_theta_3)
