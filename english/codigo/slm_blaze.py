#    Copyright (C) 2024  Diego Roman-Cortes
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#    e-mail: diego.roman.c@ug.uchile.cl

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy.special import eval_genlaguerre
from scipy import signal

my_dpi = 120 #120
plt.style.use('dark_background')

WIDTH = 1920
HEIGHT = 1080

sigma = HEIGHT/4

sigmax = HEIGHT/2.2/4*2.2
sigmay = HEIGHT/5.0*1.1

x = np.linspace(-WIDTH/2, WIDTH/2, num=WIDTH)
y = np.linspace(-HEIGHT/2, HEIGHT/2, num=HEIGHT)

Xn, Yn = np.meshgrid(x, y, indexing='xy')

angle = -0.05
X = Xn*np.cos(angle) - Yn*np.sin(angle)
Y = Yn*np.cos(angle) + Xn*np.sin(angle)

Z = np.zeros(X.shape, dtype=complex)
Z += (np.exp(-((X)/sigma/1.0)**2)*np.exp(-((Y)/sigma)**2)) # any function of x and y
Z /= np.sqrt(np.sum(np.abs(Z)**2))

phase = (np.angle(Z)+np.pi) * 255.0 / (2*np.pi)

fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)

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
