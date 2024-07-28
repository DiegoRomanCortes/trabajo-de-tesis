import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scienceplots
from numpy.ma import masked_array as marr
from scipy.constants import epsilon_0, mu_0, c

n0 = 1.48
dn1 = 5e-3

n1 = n0 + dn1

a = 3e-6
wavelength = 730e-9
k0 = 2 * np.pi / wavelength
omega = k0*c
kz = k0*(n0+n1)/2

alpha = np.sqrt(k0**2*n1**2 - kz**2)
beta = np.sqrt(kz**2 - k0**2*n0**2)


def eq11(x, y):
    return np.zeros_like(x)
def eq12(x, y):
    return -omega*mu_0*np.sin(alpha*x)/alpha
def eq21(x, y):
    return np.zeros_like(x)
def eq22(x, y):
    return -omega*mu_0*np.abs(x)/x*np.exp(beta*a)*np.sin(alpha*a)*np.exp(-beta*np.abs(x))/beta
# Grid of x, y points
nx, ny = 100, 100
x = np.linspace(-3*a, 3*a, nx)
y = np.linspace(-3*a, 3*a, ny)
X, Y = np.meshgrid(x, y)

# Create a multipole with nq charges of alternating sign, equally spaced
# on the unit circle.
# nq = int(sys.argv[1])

# Electric field vector, E=(Ex, Ey), as separate xcomponents
# Ex, Ey = np.zeros((ny, nx)), np.zeros((ny, nx))



mask = ((-a<=X) * (X<=a))

res11 = marr(eq11(X, Y), mask)
res12 = marr(eq12(X, Y), mask)
res21 = marr(eq21(X, Y), ~mask)
res22 = marr(eq22(X, Y), ~mask)

plt.style.use(['science'])
fig = plt.figure(dpi=400)
ax = fig.add_subplot(111)

# Plot the streamlines with an appropriate colormap and arrow style
color1 = 2 * np.log(np.hypot(res11, res12))
ax.streamplot(x*1e6, y*1e6, res11, res12, color=color1, linewidth=1, cmap=plt.cm.inferno,
              density=1.5, arrowstyle='->', arrowsize=1)

color2 = 2 * np.log(np.hypot(res21, res22))
ax.streamplot(x*1e6, y*1e6, res21, res22, color=color2, linewidth=1, cmap=plt.cm.inferno,
              density=2, arrowstyle='->', arrowsize=1.5)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
# ax.set_xlim(-2,2)
# ax.set_ylim(-2,2)
# ax.set_aspect('equal')
plt.show()