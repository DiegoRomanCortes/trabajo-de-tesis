 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy.special import eval_genlaguerre
from scipy import signal

my_dpi = 120 #120
plt.style.use('dark_background')

WIDTH = 1920
HEIGHT = 1080

a = 6 # rango de parametrizacion
b = 0.3
sigma = HEIGHT/4#HEIGHT/15#2.2
sigmax = HEIGHT/2.2/4*2.2

sigmay = HEIGHT/5.0*1.1#*2.5*0

sigma = HEIGHT/8
n = 0
l = 2
num = 1

x = np.linspace(-WIDTH/2, WIDTH/2, num=WIDTH)
y = np.linspace(-HEIGHT/2, HEIGHT/2, num=HEIGHT)

Xn, Yn = np.meshgrid(x, y, indexing='xy')

angle = -0.05
X = Xn*np.cos(angle) - Yn*np.sin(angle)
Y = Yn*np.cos(angle) + Xn*np.sin(angle)

angle = 10*np.pi/45

R = np.sqrt(X**2 + Y**2)
PHI = np.arctan2(Y, X)
Z = np.zeros(X.shape, dtype=complex)

A = 1.0
sigmax *= 0.75
sigma /= 1.5

sigmay = sigmax*1.9
sigmax = sigma*1.9
sigma *= 1.3
dy = -sigma*1.5*0
dx1 = sigma*0.1
dx2 = -sigma*0.4*0

per = 100
kick = np.pi/2*0
alpha = 0.3



Z += (np.exp(-((X)/sigma/1.0)**2)*np.exp(-((Y)/sigma)**2))

PHI = np.arctan2(Y, X)

sigma *= 3
Z = (R*np.sqrt(2)/sigma)**np.abs(l) * np.exp(-(R/sigma)**2) * eval_genlaguerre(n, np.abs(l), 2*R/sigma) * np.exp(-1j*l*PHI)

Z /= np.sqrt(np.sum(np.abs(Z)**2))

phase = (np.angle(Z)+np.pi) * 255.0 / (2*np.pi)

plt.imsave('fase.png', phase, cmap="gray", vmin=0, vmax=255)
plt.close("all")

fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)

blaze = (signal.sawtooth(Xn*2.0*np.pi/5.0) + 1)/2.0*255

plt.imsave('blaze.png', blaze, cmap="gray", vmin=0, vmax=255)
plt.close("all")

fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)

amplitude = (np.abs(Z)**2/np.max(np.abs(Z)**2)) * blaze/255
amplitude /= amplitude.max()
amplitude *= 255


plt.imsave('amplitude.png', amplitude, cmap="gray", vmin=0, vmax=255)
plt.close("all")


im1 = np.array(Image.open('amplitude.png').convert('L'), dtype="uint16")
im2 = np.array(Image.open('fase.png').convert('L'), dtype="uint16")

imf = (((amplitude + phase) % 255))
im = (imf).astype(np.uint8)

plt.imsave('vortex2.png', im, cmap="gray", vmin=0, vmax=255)
