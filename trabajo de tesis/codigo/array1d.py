import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import os

# Set the style for the plots
plt.style.use('science')

os.chdir("1darray")

for file in os.listdir():
    fig, ax = plt.subplots(figsize=(8, 6))
    data = np.loadtxt(file).reshape(700, 700)
    ax.imshow(data.T, cmap='hot', aspect='equal')
    fig.savefig(file.replace('.txt', '.png'), dpi=300)
    plt.close(fig)