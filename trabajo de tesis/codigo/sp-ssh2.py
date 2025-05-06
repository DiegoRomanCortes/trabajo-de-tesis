import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Brillouin zone: k from -pi to pi
k_vals = np.linspace(-np.pi, np.pi, 500)

# --- Define your d-vector components here ---
# Example placeholders — replace with your own functions!
# Set model parameters
delta_beta = 0.0
kappa_sp1 = 1.0
kappa_sp2 = 1.0

# Define components of d(k)
def d_x1(k): return 2 * kappa_sp1 * np.cos(k) + 2 * kappa_sp2
def d_y1(k): return 2 * kappa_sp1 * np.sin(k)
def d_z1(k): return delta_beta * np.ones_like(k)

# Define components of d(k)
def d_x2(k): return 2 * kappa_sp1 * np.cos(k) - 2 * kappa_sp2
def d_y2(k): return -2 * kappa_sp1 * np.sin(k)
def d_z2(k): return delta_beta * np.ones_like(k)

# Evaluate d-vector
dx1 = d_x1(k_vals)
dy1 = d_y1(k_vals)
dz1 = d_z1(k_vals)

# Evaluate d-vector for the second set of parameters
dx2 = d_x2(k_vals)
dy2 = d_y2(k_vals)
dz2 = d_z2(k_vals)



# Plot 3D trajectory of d(k)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(dx1, dy1, dz1, c=np.arctan2(dy1, dx1), s=10, cmap='hsv', vmin=-np.pi, vmax=np.pi)
ax.scatter(dx2, dy2, dz2, c=np.arctan2(dy2, dx2), s=10, cmap='hsv', vmin=-np.pi, vmax=np.pi)

# Optional: mark start and end points
ax.scatter(dx1[0], dy1[0], dz1[0], color='green', label='k = -π')
ax.scatter(dx1[len(k_vals)//2], dy1[len(k_vals)//2], dz1[len(k_vals)//2], color='blue', label='k = 0')
ax.scatter(dx1[-1], dy1[-1], dz1[-1], color='red', label='k = π')

# Optional: mark the origin

ax.scatter(0, 0, 0, color='black', label='origin')

# Optional: mark the end points of the second trajectory
ax.scatter(dx2[0], dy2[0], dz2[0], color='orange', label='k = -π (2nd)')
ax.scatter(dx2[len(k_vals)//2], dy2[len(k_vals)//2], dz2[len(k_vals)//2], color='purple', label='k = 0 (2nd)')
ax.scatter(dx2[-1], dy2[-1], dz2[-1], color='brown', label='k = π (2nd)')


# Labels and aesthetics
ax.set_xlabel(r'$d_x$')
ax.set_ylabel(r'$d_y$')
ax.set_zlabel(r'$d_z$')
ax.set_title('Trajectory of $\hat{d}(k)$ over the Brillouin zone')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
