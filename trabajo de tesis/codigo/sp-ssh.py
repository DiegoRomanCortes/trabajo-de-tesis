import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Initialize all parameters
init_Delta = 0.5
init_kappa_sp1 = 1.0
init_kappa_sp2 = 1.0
init_kappa_ss1 = 0.5
init_kappa_ss2 = 1.0
init_kappa_pp1 = 2.0
init_kappa_pp2 = 2.0

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(bottom=0.35)  # Adjust for additional sliders

# Generate k values
k = np.linspace(-np.pi, np.pi, 200)

# Calculate d-vector and energy (now including all kappa terms)
def calculate_d_and_energy(k, Delta, kappa_sp1, kappa_sp2, kappa_ss1, kappa_ss2, kappa_pp1, kappa_pp2):
    # d_x(k) = Δ (constant)
    k_mat_1_minus = np.array([[kappa_ss1, -kappa_sp1], [kappa_sp1, -kappa_pp1]])
    k_mat_2_plus = np.array([[kappa_ss2, kappa_sp2], [-kappa_sp2, -kappa_pp2]])
    h_mat = np.exp(-1j * k[:, np.newaxis, np.newaxis]) * k_mat_1_minus + k_mat_2_plus
    det_h = np.linalg.det(h_mat)
    dx = np.real(det_h)
    
    # d_z(k) = -κ_{sp,1} e^{-ik} + κ_{sp,2} (real part)
    dz = np.imag(det_h)
    
    # Energy = ±sqrt(dx² + dz²) (ignoring d0 for simplicity)
    E = np.sqrt(dx.real**2 + dz.imag**2)
    return dx, dz, E

# Initial calculation
dx, dz, E = calculate_d_and_energy(k, init_Delta, init_kappa_sp1, init_kappa_sp2, 
                                    init_kappa_ss1, init_kappa_ss2, init_kappa_pp1, init_kappa_pp2)

# Plot initial d-vector trajectory
scatter = ax1.scatter(dx, dz, c=np.arctan2(dz, dx), s=10, cmap='hsv', vmin=-np.pi, vmax=np.pi)
ax1.scatter(0, 0, c='black', s=30, zorder=3)
ax1.set_xlabel('Re[Det[$\hat{h}$]]', fontsize=12)
ax1.set_ylabel('Im[Det[$\hat{h}$]]', fontsize=12)
ax1.set_title('$\mathbf{d}(k)$ trajectory', fontsize=14)
ax1.grid(True)
# # ax1.set_aspect('equal')
# ax1.set_xlim(-2, 2)
# ax1.set_ylim(-2, 2)

# Plot initial energy bands
line_upper, = ax2.plot(k, E, 'r-', lw=2, label='$E_+(k)$')
line_lower, = ax2.plot(k, -E, 'b-', lw=2, label='$E_-(k)$')
ax2.set_xlabel('$k$', fontsize=12)
ax2.set_ylabel('Energy', fontsize=12)
ax2.set_title('Energy bands', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True)
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels(['$-\pi$', '0', '$\pi$'])

# Add sliders for all parameters
slider_y_pos = 0.25
slider_height = 0.03
slider_spacing = 0.04

ax_Delta = plt.axes([0.25, slider_y_pos, 0.65, slider_height])
Delta_slider = Slider(ax=ax_Delta, label='$\Delta$', valmin=0, valmax=6, valinit=init_Delta, valstep=0.1)

ax_kappa_sp1 = plt.axes([0.25, slider_y_pos - slider_spacing, 0.65, slider_height])
kappa_sp1_slider = Slider(ax=ax_kappa_sp1, label='$\kappa_{sp,1}$', valmin=0, valmax=3, valinit=init_kappa_sp1, valstep=0.1)

ax_kappa_sp2 = plt.axes([0.25, slider_y_pos - 2*slider_spacing, 0.65, slider_height])
kappa_sp2_slider = Slider(ax=ax_kappa_sp2, label='$\kappa_{sp,2}$', valmin=0, valmax=3, valinit=init_kappa_sp2, valstep=0.1)

ax_kappa_ss1 = plt.axes([0.25, slider_y_pos - 3*slider_spacing, 0.65, slider_height])
kappa_ss1_slider = Slider(ax=ax_kappa_ss1, label='$\kappa_{ss,1}$', valmin=0, valmax=3, valinit=init_kappa_ss1, valstep=0.1)

ax_kappa_ss2 = plt.axes([0.25, slider_y_pos - 5*slider_spacing, 0.65, slider_height])
kappa_ss2_slider = Slider(ax=ax_kappa_ss2, label='$\kappa_{ss,2}$', valmin=0, valmax=3, valinit=init_kappa_ss2, valstep=0.1)

ax_kappa_pp1 = plt.axes([0.25, slider_y_pos - 4*slider_spacing, 0.65, slider_height])
kappa_pp1_slider = Slider(ax=ax_kappa_pp1, label='$\kappa_{pp,1}$', valmin=0, valmax=3, valinit=init_kappa_pp1, valstep=0.1)

ax_kappa_pp2 = plt.axes([0.25, slider_y_pos - 6*slider_spacing, 0.65, slider_height])
kappa_pp2_slider = Slider(ax=ax_kappa_pp2, label='$\kappa_{pp,2}$', valmin=0, valmax=3, valinit=init_kappa_pp2, valstep=0.1)

# Update function (now includes all parameters)
def update(val):
    dx, dz, E = calculate_d_and_energy(
        k, 
        Delta_slider.val, 
        kappa_sp1_slider.val, 
        kappa_sp2_slider.val,
        kappa_ss1_slider.val,
        kappa_ss2_slider.val,
        kappa_pp1_slider.val,
        kappa_pp2_slider.val
    )
    
    # Update scatter plot
    scatter.set_offsets(np.column_stack((dx, dz)))
    scatter.set_array(np.arctan2(dz, dx))
    
    # Update energy bands
    line_upper.set_ydata(E)
    line_lower.set_ydata(-E)
    
    fig.canvas.draw_idle()

# Register update for all sliders
sliders = [Delta_slider, kappa_sp1_slider, kappa_sp2_slider, 
           kappa_ss1_slider, kappa_pp1_slider, kappa_ss2_slider, kappa_pp2_slider]
for slider in sliders:
    slider.on_changed(update)

# Add reset button
resetax = plt.axes([0.8, 0.9, 0.1, 0.04])
button = Button(resetax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

def reset(event):
    for slider in sliders:
        slider.reset()
button.on_clicked(reset)

plt.show()