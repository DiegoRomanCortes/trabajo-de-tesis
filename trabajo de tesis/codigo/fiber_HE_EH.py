import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.special import jv, kn, jvp, kvp

# Constants
n0 = 1.48
dn1 = 5e-3
n1 = n0 + dn1

dn2 = 15e-3
n2 = n0 + dn2

dn3 = 40e-3
n3 = n0 + dn3

n1 = n3


a = 3e-6
wavelength = 730e-9
k0 = 2 * np.pi / wavelength

# kz = np.linspace(k0*n0, k0*n1, 1000)
l = 1


V = k0*a*np.sqrt(n1**2-n0**2)

alpha = np.linspace(0.0, V, 1000)/a
beta = np.sqrt(V**2 - (alpha*a)**2)/a

kz = np.sqrt(k0**2*n1**2 - alpha**2)

plt.style.use(['science'])
fig,ax = plt.subplots(dpi=300)


rhs_HE =  -(n1**2+n0**2)/(2*n1**2)*kvp(l, beta*a) /( kn(l, beta*a)*beta*a)
rhs_EH =  (n1**2+n0**2)/(2*n1**2)*kvp(l, beta*a) /( kn(l, beta*a)*beta*a)

disc =  (((n1**2-n0**2)/(2*n1**2)*kvp(l, beta*a) /( kn(l, beta*a)*beta*a))**2 + (kz*l/(k0*n1))**2 * ((1/(alpha*a))**2 + (1/(beta*a))**2))**2
l_part = l/(alpha*a)**2 - np.sqrt(disc)

rhs_HE += l_part
rhs_EH += l_part

# rhs_HE *= (alpha*a)
# rhs_EH *= (alpha*a)

lhs_HE =  jv(l-1, alpha*a) / (jv(l, alpha*a)*(alpha*a)) 
lhs_EH = jv(l+1, alpha*a) / (jv(l, alpha*a)*(alpha*a)) 


# kz /= k0

lhs_HE[:-1][np.diff(lhs_HE) > 0] = np.nan
# rhs_HE[:-1][np.diff(rhs_HE) > 0] = np.nan

lhs_EH[:-1][np.diff(lhs_EH) < 0] = np.nan
# rhs_EH[:-1][np.diff(rhs_EH) < 0] = np.nan

colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']


ax.plot(alpha*a, lhs_HE, label=r'$\frac{J_{\ell-1}(\alpha a)}{\alpha a J_\ell(\alpha a)}$', color=colors[0])
ax.plot(alpha*a, rhs_HE, "--", label=r'HE', color=colors[0])

# ax.plot(alpha*a, np.sqrt(V**2 - (alpha*a)**2))
ax.plot(alpha*a, -lhs_EH, label=r'$-\frac{J_{\ell+1}(\alpha a)}{\alpha a J_\ell(\alpha a)}$', color=colors[1])
ax.plot(alpha*a, -rhs_EH, "--", label=r'EH',color=colors[1])
# ax.plot(kz, rhs_TM, "--", label='TM', color=colors[2])
# ax.set_xscale('log')
ax.set_ylim(0, 2)
ax.set_xlabel(r'$\alpha a$')
ax.set_title(r"$\ell = {}$".format(l))
fig.legend(loc='lower left', fontsize='xx-small', bbox_to_anchor=(0.25, 0.59))
fig.savefig('../media/fibergraphicalHE_EH.pdf')

plt.close('all')