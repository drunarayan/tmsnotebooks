import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*NumPy version.*")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import make_interp_spline
from IPython.display import HTML

plt.rcParams['animation.embed_limit'] = 50.0

def simulate_journey(time_pts, speed_pts, title="Journey Simulation", 
                     time_unit="s", dist_unit="m", speed_unit="m/s", 
                     runner_type="person"):
    """
    Unified Calculus Engine for The Meenakshi School.
    Automatically handles unit conversions, spline interpolation, area integration,
    and 3-panel synchronized animation.
    """
    total_time = time_pts[-1]
    time_dense = np.linspace(0, total_time, 100)
    
    # Smooth interpolation for speed curve
    k_order = 2 if len(time_pts) <= 6 else 3
    spline = make_interp_spline(time_pts, speed_pts, k=min(k_order, len(time_pts)-1))
    speed_dense = np.maximum(spline(time_dense), 0)
    
    # Calculate dt factor for unit consistency (minutes -> hours if needed)
    dt_raw = time_dense[1] - time_dense[0]
    dt_conv = dt_raw / 60.0 if (time_unit == "min" and speed_unit == "km/h") else dt_raw
    
    # Area integration -> Accumulated Distance
    distance_dense = np.cumsum(speed_dense) * dt_conv
    total_distance = distance_dense[-1]
    
    # Tick markers every 10% of total time
    tick_times = [int(t) for t in np.linspace(0, total_time, 7)]
    tick_indices = [int(t * (len(time_dense) - 1) / total_time) for t in tick_times]
    tick_distances = [distance_dense[idx] for idx in tick_indices]

    # Figure Layout
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_anim = fig.add_subplot(gs[1, :])

    # 1. Speed Plot Base
    ax1.plot(time_dense, speed_dense, color='tab:blue', linewidth=1.5, linestyle=':', alpha=0.4)
    ax1.scatter(time_pts, speed_pts, color='navy', zorder=5)
    speed_line, = ax1.plot([], [], color='tab:blue', linewidth=2.5, label='Speed $v(t)$')
    speed_point, = ax1.plot([], [], 'o', color='tab:blue', markersize=7)
    ax1.set_xlim(0, total_time)
    ax1.set_ylim(0, max(max(speed_dense) * 1.25, 0.1))
    ax1.set_title(f"{title}: Speed vs Time", fontsize=11, weight='bold')
    ax1.set_xlabel(f"Time ({time_unit})")
    ax1.set_ylabel(f"Speed ({speed_unit})")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 2. Distance Plot Base
    ax2.plot(time_dense, distance_dense, color='tab:orange', linewidth=1.5, linestyle=':', alpha=0.4)
    dist_line, = ax2.plot([], [], color='tab:orange', linewidth=2.5, label='Distance $d(t)$')
    dist_point, = ax2.plot([], [], 'o', color='tab:orange', markersize=7)
    ax2.set_xlim(0, total_time)
    ax2.set_ylim(0, max(total_distance * 1.15, 0.1))
    ax2.set_title(f"{title}: Distance vs Time", fontsize=11, weight='bold')
    ax2.set_xlabel(f"Time ({time_unit})")
    ax2.set_ylabel(f"Distance ({dist_unit})")
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 3. Track Graphics (Person vs Car)
    ax_anim.set_xlim(-2, max(total_distance + 5, 10))
    ax_anim.set_ylim(-3, 4)
    ax_anim.set_aspect('equal')
    ax_anim.axis('off')
    ax_anim.set_title("Movement Track (Area Under Speed Curve = Distance Traveled)", fontsize=11, weight='bold')
    ax_anim.plot([-5, total_distance + 50], [0, 0], color='black', linewidth=2.5)

    if runner_type == "car":
        body, = ax_anim.plot([], [], color='tab:red', linewidth=6)
        roof, = ax_anim.plot([], [], color='tab:red', linewidth=4)
        w1 = plt.Circle((0, 0.2), 0.3, color='black')
        w2 = plt.Circle((0, 0.2), 0.3, color='black')
        ax_anim.add_patch(w1)
        ax_anim.add_patch(w2)
    else:
        head = plt.Circle((0, 3.2), 0.4, color='black')
        torso, = ax_anim.plot([], [], color='black', linewidth=3)
        leg_l, = ax_anim.plot([], [], color='black', linewidth=2.5)
        leg_r, = ax_anim.plot([], [], color='black', linewidth=2.5)
        ax_anim.add_patch(head)

    area_text = ax1.text(0.03, 0.85, '', transform=ax1.transAxes, fontsize=10, weight='bold', color='tab:blue')
    status_text = ax_anim.text(0.02, 0.85, '', transform=ax_anim.transAxes, fontsize=11, weight='bold')

    fill_container = [None]
    tick_lines = []
    tick_texts = []

    plt.tight_layout()

    def animate(i):
        t_curr = time_dense[:i+1]
        v_curr = speed_dense[:i+1]
        d_curr = distance_dense[:i+1]
        
        x = distance_dense[i]
        t = time_dense[i]

        speed_line.set_data(t_curr, v_curr)
        speed_point.set_data([t], [speed_dense[i]])
        
        if fill_container[0] is not None:
            fill_container[0].remove()
        if i > 0:
            fill_container[0] = ax1.fill_between(t_curr, v_curr, color='tab:blue', alpha=0.35)
        
        area_text.set_text(f"Area = {distance_dense[i]:.2f} {dist_unit}")

        dist_line.set_data(t_curr, d_curr)
        dist_point.set_data([t], [x])

        if runner_type == "car":
            body.set_data([x - 1.2, x + 1.2], [0.6, 0.6])
            roof.set_data([x - 0.5, x + 0.5], [1.2, 1.2])
            w1.center = (x - 0.8, 0.3)
            w2.center = (x + 0.8, 0.3)
        else:
            head.center = (x, 3.2)
            torso.set_data([x, x], [1.8, 2.8])
            stride = np.sin(2 * np.pi * (i / len(time_dense)) * 10) * 0.8
            leg_l.set_data([x, x + stride], [1.8, 0])
            leg_r.set_data([x, x - stride], [1.8, 0])

        # Drop ground tick marks
        sec_reached = int(np.floor(t))
        if sec_reached in tick_times and len(tick_lines) <= tick_times.index(sec_reached):
            tick_idx = tick_times.index(sec_reached)
            tick_x = tick_distances[tick_idx]
            line, = ax_anim.plot([tick_x, tick_x], [0, -0.8], color='tab:red', linewidth=2, zorder=3)
            txt_t = ax_anim.text(tick_x, -1.4, f"{sec_reached}{time_unit}", color='tab:red', fontsize=9, ha='center', weight='bold')
            txt_d = ax_anim.text(tick_x, -2.2, f"{tick_x:.1f}{dist_unit}", color='navy', fontsize=8, ha='center')
            
            tick_lines.append(line)
            tick_texts.extend([txt_t, txt_d])

        status_text.set_text(f"Time: {t:.1f} {time_unit} | Distance: {x:.2f} {dist_unit} | Speed: {speed_dense[i]:.1f} {speed_unit}")
        
        if runner_type == "car":
            return [speed_line, speed_point, dist_line, dist_point, body, roof, w1, w2, area_text, status_text] + tick_lines + tick_texts
        else:
            return [speed_line, speed_point, dist_line, dist_point, head, torso, leg_l, leg_r, area_text, status_text] + tick_lines + tick_texts

    anim = FuncAnimation(fig, animate, frames=len(time_dense), interval=60, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())