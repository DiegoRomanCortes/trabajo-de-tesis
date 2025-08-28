import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# Define the parameters for the SSH model
n_cells = 100  # Number of sites
t_prime = 1.0  # Hopping parameter
d = 1.0  # Lattice constant

delta = 0.5  # Pairing parameter

# Define the crystal momentum
k = 2*np.pi*np.linspace(-n_cells/2, n_cells/2, n_cells+1) / (n_cells * d)

def d_x_func(k, delta):
    t = t_prime * delta
    return t + t_prime * np.cos(k*d)

def d_y_func(k, delta):
    t = t_prime * delta
    return t_prime * np.sin(k*d)

def phase_func(k, delta):
    return np.arctan2(d_y_func(k, delta), d_x_func(k, delta))

delta_values = [0.5, 1.0, 1.5]

# # Improved Plot
# plt.style.use('science')
# fig, axs = plt.subplots(2, 3, dpi=400, figsize=(12, 8), constrained_layout=False, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1, 1]})


# for i, delta in enumerate(delta_values):
#     d_x = d_x_func(k, delta)
#     d_y = d_y_func(k, delta)
#     phase = phase_func(k, delta)
#     t = t_prime * delta

#     # Top row: d_x vs d_y with phase as color
#     scatter = axs[0, i].scatter(d_x, d_y, c=phase, s=1, cmap='hsv', vmin=-np.pi, vmax=np.pi)
#     axs[0, i].quiver(0, 0, d_x[int(-n_cells/3)], d_y[int(-n_cells/3)], angles='xy', scale_units='xy', scale=1, color='black', label='d_x, d_y')
#     axs[0, i].scatter(0, 0, c='black', s=10)
#     axs[0, i].set_title(f'$\delta = {delta}$', fontsize=10)
#     axs[0, i].set_xlabel('$d_x$', fontsize=8)
#     axs[0, i].set_ylabel('$d_y$', fontsize=8)
#     axs[0, i].set_aspect('equal')

#     # Add individual colorbar for phase plot
#     cbar = fig.colorbar(scatter, ax=axs[0, i], orientation='vertical', fraction=0.05, pad=0.02)
#     cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
#     cbar.set_ticklabels(['$-\\pi$', '$-\\frac{\\pi}{2}$', '$0$', '$\\frac{\\pi}{2}$', '$\\pi$'])

#     # Bottom row: Energy bands
#     axs[1, i].plot(k * d, np.sqrt(t**2 + t_prime**2 + 2 * t * t_prime * np.cos(k * d)), label='Upper Band')
#     axs[1, i].plot(k * d, -np.sqrt(t**2 + t_prime**2 + 2 * t * t_prime * np.cos(k * d)), label='Lower Band')
#     axs[1, i].set_title(f'Energy Bands ($\delta = {delta}$)', fontsize=10)
#     axs[1, i].set_xlabel('$k d$', fontsize=8)
#     axs[1, i].legend(fontsize=6, loc='center')

#     # Set tick labels to multiples of pi
#     axs[1, i].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
#     axs[1, i].set_xticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'], fontsize=8)

# # Adjust layout to align bottom plots with respect to the colorbar space
# fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for colorbars
# plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing between subplots

# # Share y-axis for the bottom row
# for ax in axs[1, :]:
#     ax.set_ylabel('Energy', fontsize=8)
#     ax.set_ylim(-2.5, 2.5)  # Adjust limits to align all plots

# plt.show()

# Now solving for open boundary conditions
def solve_ssh_open(n_cells, t_prime, delta):
    # Construct the Hamiltonian matrix for open boundary conditions
    H = np.zeros((2*n_cells, 2*n_cells))

    # Fill the Hamiltonian matrix
    for i in range(1, n_cells-1):
        H[2*i, 2*i+1] = t_prime * delta
        H[2*i+1, 2*i+2] = t_prime

        H[2*i+1, 2*i] = t_prime * delta
        H[2*i+2, 2*i+1] = t_prime

    H[0, 1] = t_prime * delta
    H[1, 0] = t_prime * delta

    H[1, 2] = t_prime
    H[2, 1] = t_prime

    H[2*n_cells-2, 2*n_cells-1] = t_prime * delta
    H[2*n_cells-1, 2*n_cells-2] = t_prime * delta

    eigenvalues, eigenvectors = np.linalg.eigh(H)
    # Extract the eigenvalues
    return eigenvalues

delta_values = np.linspace(0, 2, num=100)
plt.style.use('science')
fig, ax = plt.subplots(1, 1, dpi=300)

for delta in delta_values:
    ax.plot(delta*np.ones(2*n_cells), solve_ssh_open(n_cells, t_prime, delta), "r.", markersize=0.1)

ax.set_title('Energy Spectrum for SSH Model \n with Open Boundary Conditions', fontsize=12)
ax.set_xlabel('$\delta \equiv \\frac{t}{t\'}$', fontsize=12)
ax.set_ylabel('Energy', fontsize=12)
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Adjust top space for the title
plt.show()