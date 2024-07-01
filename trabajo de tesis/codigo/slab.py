import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# Constants
n0 = 1.48
dn1 = 1e-3
n1 = n0 + dn1

dn2 = 15e-3
n2 = n0 + dn2

dn3 = 30e-3
n3 = n0 + dn3

a = 3e-6
wavelength = 730e-9
k0 = 2 * np.pi / wavelength

R1 = k0*a*np.sqrt(n1**2-n0**2)
R2 = k0*a*np.sqrt(n2**2-n0**2)
R3 = k0*a*np.sqrt(n3**2-n0**2)

x = np.linspace(0, 6*np.pi/2, 800)

y1 = x * np.tan(x)
y2 = - x / np.tan(x)
y1[:-1][np.diff(y1) < 0] = np.nan
y2[:-1][np.diff(y2) < 0] = np.nan

y3 = x * np.tan(x) * (n0/n1)**2
y4 = - x / np.tan(x) * (n0/n1)**2
y3[:-1][np.diff(y1) < 0] = np.nan
y4[:-1][np.diff(y2) < 0] = np.nan



sol1x = []
sol1y = []

sol2x = []
sol2y = []

sol3x = []
sol3y = []

for i in range(len(x)):
    for j in range(len(x)):
        if (np.abs(y1[i]-np.sqrt(R1**2-x[j]**2)) < 9e-3) and np.abs(x[i]-x[j]) < (x[1]-x[0])*0.9:
            sol1x.append(x[j])
            sol1y.append(y1[i])
        if (np.abs(y2[i]-np.sqrt(R1**2-x[j]**2)) < 2e-2) and np.abs(x[i]-x[j]) < (x[1]-x[0])*1.0:
            sol1x.append(x[j])
            sol1y.append(y2[i])
        if (np.abs(y1[i]-np.sqrt(R2**2-x[j]**2)) < 5e-2) and np.abs(x[i]-x[j]) < (x[1]-x[0])*1.0:
            sol2x.append(x[j])
            sol2y.append(y1[i])
        if (np.abs(y2[i]-np.sqrt(R2**2-x[j]**2)) < 5e-2) and np.abs(x[i]-x[j]) < (x[1]-x[0])*1.0:
            sol2x.append(x[j])
            sol2y.append(y2[i])
        if (np.abs(y1[i]-np.sqrt(R3**2-x[j]**2)) < 15e-2) and np.abs(x[i]-x[j]) < (x[1]-x[0])*1.0:
            sol3x.append(x[j])
            sol3y.append(y1[i])
        if (np.abs(y2[i]-np.sqrt(R3**2-x[j]**2)) < 10e-2) and np.abs(x[i]-x[j]) < (x[1]-x[0])*1.0:
            sol3x.append(x[j])
            sol3y.append(y2[i])

sol2xaux = []
sol2yaux = []
sol2xaux2 = []
sol2yaux2 = []
for i in range(len(sol2x)):
    if i < len(sol2x)-1:
        if np.abs(sol2x[i]-sol2x[i+1]) < 1e-1:
            sol2xaux.append(sol2x[i]) 
            sol2yaux.append(sol2y[i])
        else:
            sol2xaux.append(sol2x[i]) 
            sol2yaux.append(sol2y[i])
            sol2xaux2.append(np.mean(np.array(sol2xaux)))
            sol2yaux2.append(np.mean(np.array(sol2yaux)))
            sol2xaux = []
            sol2yaux = []
    else:
        sol2xaux2.append(np.mean(np.array(sol2xaux)))
        sol2yaux2.append(np.mean(np.array(sol2yaux)))

sol3xaux = []
sol3yaux = []
sol3xaux2 = []
sol3yaux2 = []
for i in range(len(sol3x)):
    if i < len(sol3x)-1:
        if np.abs(sol3x[i]-sol3x[i+1]) < 1e-1:
            sol3xaux.append(sol3x[i]) 
            sol3yaux.append(sol3y[i])
        else:
            sol3xaux.append(sol3x[i]) 
            sol3yaux.append(sol3y[i])
            sol3xaux2.append(np.mean(np.array(sol3xaux)))
            sol3yaux2.append(np.mean(np.array(sol3yaux)))
            sol3xaux = []
            sol3yaux = []
    else:
        sol3xaux2.append(np.mean(np.array(sol3xaux)))
        sol3yaux2.append(np.mean(np.array(sol3yaux)))

plt.style.use(['science'])
fig, ax = plt.subplots(dpi=300)

colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']

ax.plot(x, y1, label='pares', color=colors[0])
ax.plot(x, y2, label='impares', color=colors[1])

ax.plot(x, np.sqrt(R1**2-x**2), label=r'$V_1$', color=colors[2])
ax.plot(x, np.sqrt(R2**2-x**2), label=r'$V_2$', color=colors[3])
ax.plot(x, np.sqrt(R3**2-x**2), label=r'$V_3$', color=colors[4])

# ax.plot(x, np.sqrt(R1**2-(x*(n1/n0))**2), "--", label=r'$V_1$', color=colors[2])
# ax.plot(x, np.sqrt(R2**2-(x*(n2/n0))**2), "--", label=r'$V_2$', color=colors[3])
# ax.plot(x, np.sqrt(R3**2-(x*(n3/n0))**2), "--", label=r'$V_3$', color=colors[4])


ax.plot(sol1x, sol1y, 'o', color=colors[2])
ax.plot(sol2xaux2, sol2yaux2, 'o', color=colors[3])
ax.plot(sol3xaux2, sol3yaux2, 'o', color=colors[4])


ax.set_ylim(0.0, 1.1*R3)
# ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
# ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

# ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
# ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

# ax.set_xlabel(r'$\alpha a$')
# ax.set_ylabel(r'$\beta a$')

fig.legend()
fig.savefig('../media/slabgraphical.png')

plt.close('all')