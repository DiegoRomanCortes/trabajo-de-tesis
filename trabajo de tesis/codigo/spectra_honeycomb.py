import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.linalg import eig, ishermitian
from cmcrameri import cm
angles = np.load("angles.npy")

overlap_kappa_sigma = -np.abs(np.load('overlap_kappa_sigma.npy'))

overlap_kappa_theta = [np.abs(np.load(f'overlap_kappa_theta_{i}.npy')) for i in range(len(angles))]
overlap_kappa_theta = np.array(overlap_kappa_theta)

idx_magic = np.argmin(np.abs(overlap_kappa_theta))+1
print(f"idx_magic: {idx_magic}")
overlap_kappa_theta[idx_magic:] *= -1

overlap_t_pi = [np.abs(np.load(f'overlap_t_pi_{i}.npy')) for i in range(len(angles))]
overlap_t_pi = np.array(overlap_t_pi)

overlap_t_theta_1 = [-np.abs(np.load(f'overlap_t_theta_1_{i}.npy')) for i in range(len(angles))]

overlap_t_theta_2 = [np.abs(np.load(f'overlap_t_theta_2_{i}.npy')) for i in range(len(angles))]
overlap_t_theta_2 = -np.array(overlap_t_theta_2)

overlap_t_theta_3 = [np.abs(np.load(f'overlap_t_theta_3_{i}.npy')) for i in range(len(angles))]
overlap_t_theta_3 = -np.array(overlap_t_theta_3)

coupling_kappa_sigma = -np.load('coupling_kappa_sigma.npy')

coupling_kappa_theta = [np.abs(np.load(f'coupling_kappa_theta_{i}.npy')) for i in range(len(angles))]
coupling_kappa_theta = np.array(coupling_kappa_theta)

idx_magic = np.argmin(np.abs(coupling_kappa_theta))+1
print(f"idx_magic: {idx_magic}")
coupling_kappa_theta[idx_magic:] *= -1

coupling_t_pi = [np.load(f'coupling_t_pi_{i}.npy') for i in range(len(angles))]
coupling_t_pi = np.array(coupling_t_pi)

coupling_t_theta_1 = [np.load(f'coupling_t_theta_1_{i}.npy') for i in range(len(angles))]
coupling_t_theta_1 = -np.array(coupling_t_theta_1)

coupling_t_theta_2 = [np.load(f'coupling_t_theta_2_{i}.npy') for i in range(len(angles))]
coupling_t_theta_2 = -np.array(coupling_t_theta_2)

coupling_t_theta_3 = [np.load(f'coupling_t_theta_3_{i}.npy') for i in range(len(angles))]
coupling_t_theta_3 = -np.array(coupling_t_theta_3)

# Generate the honeycomb lattice coordinates

Nx = 16#15
Ny = 15#16


guias = np.zeros((Nx*Ny, 2)) 

angle = 0.4

for i in range(Nx):
    for j in range(Ny):
        if i == 0:
            if j == 0:
                guias[0, :] = [0, 0]
            elif j % 4 == 1:
                guias[i*Ny+j, 0] = 0
                guias[i*Ny+j, 1] = guias[i*Ny+(j-1), 1] + 25e-6
            elif j % 4 == 2:
                guias[i*Ny+j, 0] = -10
                guias[i*Ny+j, 1] = -10
            elif j % 4 == 3:
                guias[i*Ny+j, 0] = -10
                guias[i*Ny+j, 1] = -10
            elif j % 4 == 0:
                guias[i*Ny+j, 0] = 0
                guias[i*Ny+j, 1] = guias[i*Ny+(j-3), 1] + (25e-6*(np.sin(angle)*2 + 1))
        elif i % 2 == 0 :
            if i != 0 and j == 0:
                guias[i*Ny+j, 0] = guias[(i-2)*Ny+j, 0] + (25e-6*np.cos(angle)*2)
                guias[i*Ny+j, 1] = guias[(i-2)*Ny+j, 1]
            elif j % 4 == 0 and j != 0:
                guias[i*Ny+j, 0] = guias[(i-2)*Ny+(j), 0] + (25e-6*np.cos(angle)*2)
                guias[i*Ny+j, 1] = guias[(i-2)*Ny+(j), 1]
            elif j % 4 == 1:
                guias[i*Ny+j, 0] = guias[i*Ny+(j-1), 0]
                guias[i*Ny+j, 1] = guias[i*Ny+(j-1), 1] + 25e-6
            elif j % 4 == 2:
                guias[i*Ny+j, 0] = -10
                guias[i*Ny+j, 1] = -10
            elif j % 4 == 3:
                guias[i*Ny+j, 0] = -10
                guias[i*Ny+j, 1] = -10

        elif i % 2 == 1:
            if j % 4 == 0:
                guias[i*Ny+j, 0] = -10
                guias[i*Ny+j, 1] = -10
            elif j % 4 == 1:
                guias[i*Ny+j, 0] = -10
                guias[i*Ny+j, 1] = -10
            elif j % 4 == 2:
                guias[i*Ny+j, 0] = guias[(i-1)*Ny+(j-1), 0] + (25e-6*np.cos(angle))
                guias[i*Ny+j, 1] = guias[(i-1)*Ny+(j-1), 1] + (25e-6*np.sin(angle))
            elif j % 4 == 3:
                guias[i*Ny+j, 0] = guias[i*Ny+(j-1), 0]
                guias[i*Ny+j, 1] = guias[i*Ny+(j-1), 1] + 25e-6 

for i in range(Nx):
    for j in range(Ny):
        if j//2 - (i) >= 1:
            guias[i*Ny+j, 0] = -10
            guias[i*Ny+j, 1] = -10
        if j//2 + (i) >= 15:
            guias[i*Ny+j, 0] = -10
            guias[i*Ny+j, 1] = -10



# Delete the negatives
guias_real = guias[guias[:, 0] >= 0]          

# Plot the honeycomb lattice
plt.style.use('science')
plt.figure(figsize=(6, 6), dpi=500)
plt.plot(guias_real[:, 0]*1e6, guias_real[:, 1]*1e6, '.')
plt.axis('equal')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Honeycomb Lattice')
plt.savefig('honeycomb_lattice.png')

# Generate the adjacency matrix
N = guias_real.shape[0]

lambdas = np.zeros((len(angles), N))
eigenmodes = np.zeros((len(angles), N, N))
for k in range(len(angles)):
    H = np.zeros((N, N))
    C = np.zeros((N, N))
    for i in range(N):
        # print(guias_real[i, 0]*1e6, guias_real[i, 1]*1e6)
        # print()
        for j in range(i, N):  # Only care about upper triangular part of the matrix since it is symmetric
            dx = abs(guias_real[i, 0] - guias_real[j, 0])
            dy = abs(guias_real[i, 1] - guias_real[j, 1])
            
            ds = np.sqrt(dx**2 + dy**2)
            # Fill matrix H with coupling values
            # print(f"dx: {dx}, dy: {dy}")
            if np.abs(dx) < 1e-6 and np.abs(dy) < 1e-6:
                C[i, j] = C[j, i] = 1
                # print(f"dx: {dx}, dy: {dy}, coupling_kappa_sigma: {coupling_kappa_sigma}")
            elif np.abs(dx) < 1e-6 and np.abs(dy - 25e-6) < 1e-6:
                H[i, j] = H[j, i] = coupling_kappa_sigma
                C[i, j] = C[j, i] = overlap_kappa_sigma
                # print(f"dx: {dx}, dy: {dy}, coupling_kappa_sigma: {coupling_kappa_sigma}")
            elif np.abs(dx - 25e-6*np.cos(angle)) < 1e-6 and np.abs(dy - 25e-6*np.sin(angle)) < 1e-6:
                H[i, j] = H[j, i] = coupling_kappa_theta[k]
                C[i, j] = C[j, i] = overlap_kappa_theta[k]
                # print(f"dx: {dx}, dy: {dy}, coupling_kappa_theta: {coupling_kappa_theta[k]}")
            elif np.abs(dx - 25e-6*np.cos(angle)*2) < 1e-6 and np.abs(dy) < 1e-6:
                H[i, j] = H[j, i] = coupling_t_pi[k]
                C[i, j] = C[j, i] = overlap_t_pi[k]
                # print(f"dx: {dx}, dy: {dy}, coupling_t_pi: {coupling_t_pi[k]}")
            elif np.abs(dx - 25e-6*np.cos(angle)*2) < 1e-6 and np.abs(dy -25e-6) < 1e-6:
                H[i, j] = H[j, i] = coupling_t_theta_1[k]
                C[i, j] = C[j, i] = overlap_t_theta_1[k]
                # print(f"dx: {dx}, dy: {dy}, coupling_t_theta_1: {coupling_t_theta_1[k]}")
            elif np.abs(dx - 25e-6*np.cos(angle)) < 1e-6 and np.abs(dy - 25e-6*(1+np.sin(angle))) < 1e-6:
                H[i, j] = H[j, i] = coupling_t_theta_2[k]
                C[i, j] = C[j, i] = overlap_t_theta_2[k]
                # print(f"dx: {dx}, dy: {dy}, coupling_t_theta_2: {coupling_t_theta_2[k]}")
            elif np.abs(dx) < 1e-6 and np.abs(dy - 25e-6*(1+2*np.sin(angle))) < 1e-6:
                H[i, j] = H[j, i] = coupling_t_theta_3[k]
                C[i, j] = C[j, i] = overlap_t_theta_3[k]
                # print(f"dx: {dx}, dy: {dy}, coupling_t_theta_3: {coupling_t_theta_3[k]}")

    # print(H)
    # Compute the eigenvalues and eigenvectors
    print("\r", f'Computing eigenvalues and eigenvectors for angle {angles[k]:.2f}...', end='')
    # V = np.linalg.inv(C) @ H #/ 2.512547031739796e-13
    
    print(ishermitian(C))
    T = np.linalg.cholesky(C)
    
    # V = np.linalg.inv(np.conj(T)) @ H @ np.linalg.inv(T)
    V = np.linalg.inv(C) @ H
    print(ishermitian(V))
    eigenvalues, eigenvectors = eig(V)
    lambdas[k, :] = eigenvalues
    eigenmodes[k, :, :] = eigenvectors
    print("Eigenvalues:", eigenvalues)
    print(f" {k/len(angles)*100:.0f} % complete")

    # for i in range(N):
    #     if np.abs(lambdas[k, i]) < 0.2:
    #         fig, ax = plt.subplots(1, 1, dpi=400)
    #         ax.set_facecolor('black')
    #         vmax = np.max(np.abs(eigenmodes[k, :, i]))
    #         sc = ax.scatter(guias_real[:, 0], guias_real[:, 1], c=(eigenmodes[k, :, i])/vmax, cmap=cm.berlin, s=10, vmin=-1, vmax=1)
    #         ax.set_title(r'Angle = '+f'{angles[k]:.2f}, Eigenvalue = {lambdas[k, i]:.2f}')
    #         fig.colorbar(sc, label='Intensity', aspect=5)
    #         if k < 9:
    #             if i < 9:
    #                 fig.savefig(f'eigenmode_0{k+1}_0{i+1}.png')
    #             else:
    #                 fig.savefig(f'eigenmode_0{k+1}_{i+1}.png')
    #         else:
    #             if i < 9:
    #                 fig.savefig(f'eigenmode_{k+1}_0{i+1}.png')
    #             else:
    #                 fig.savefig(f'eigenmode_{k+1}_{i+1}.png')
    #         plt.close('all')
    

# Plot the eigenvalues vs the coupling constant
plt.style.use('science')
fig, ax = plt.subplots(1, 1, dpi=500)
vmax = -np.inf
vmin = np.inf

mpl.rcParams['scatter.marker'] = '.'

for i in range(len(angles)):
    IPR = np.sum(np.abs(eigenmodes[i, :, :])**4, axis=0)/np.sum(np.abs(eigenmodes[i, :, :])**2, axis=0)**2
    if np.max(IPR) > vmax:
        vmax = np.max(IPR)
    if np.min(IPR) < vmin:
        vmin = np.min(IPR)
    # ax.set_ylim(-2, 2)

for i in range(len(angles)):
    IPR = np.sum(np.abs(eigenmodes[i, :, :])**4, axis=0)/np.sum(np.abs(eigenmodes[i, :, :])**2, axis=0)**2
    sc = ax.scatter(angles[i]*np.ones(N), np.abs(lambdas[i, :]), c=IPR, cmap=cm.batlow, s=1, marker=mpl.markers.MarkerStyle("."), vmin=vmin, vmax=vmax)

fig.colorbar(sc, label='IPR')
ax.set_xlabel(r'Angle $\theta$ (rad)')
# ax.set_xlabel('Angle (rad)')
ax.set_ylabel(r'Eigenvalues (cm$^{-1}$)')
# ax.set_title('Eigenvalues vs Coupling Constant,'+r' $C_V$ = '+f'{coupling_pp_vertical:.2f}')
ax.set_title('Eigenvalues vs Angle')
fig.savefig('honeycomb_eigenvalues_vs_angle.png')