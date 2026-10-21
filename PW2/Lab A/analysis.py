import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# 1. Read data
data = np.loadtxt('freefall.csv', delimiter=',', skiprows=1)

t = data[:, 0]
y = data[:, 1]

# 2. Calculate Velocity and Acceleration (providing precise time steps is crucial)
v = np.gradient(y, t)
a = np.gradient(v, t)

mean_a = np.mean(a)
std_a = np.std(a)

print(f"Mean acceleration: {mean_a:.2f} m/s^2")
print(f"Standard deviation (Noise): {std_a:.2f}")

# 3. Integration (from acceleration to velocity, and velocity to position)

v_rec = cumulative_trapezoid(a, t, initial=0) + v[0]

y_rec = cumulative_trapezoid(v_rec, t, initial=0) + y[0]

max_diff = np.max(np.abs(y - y_rec))
print(f"Maximum difference (Original vs Recovered y): {max_diff:.4f} m")

# 4. Plot
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Position
axs[0].plot(t, y, label='Original noisy y', color='blue')
axs[0].plot(t, y_rec, label='Recovered y (from integration)', color='red', linestyle='--')
axs[0].set_ylabel('Position y (m)')
axs[0].set_title('Position, Velocity, and Acceleration Analysis')
axs[0].legend()
axs[0].grid(True)

# Velocity
axs[1].plot(t, v, label='Calculated Velocity (v)', color='green')
axs[1].set_ylabel('Velocity v (m/s)')
axs[1].legend()
axs[1].grid(True)

# Acceleration
axs[2].plot(t, a, label='Calculated Acceleration (a)', color='orange', alpha=0.7)
axs[2].axhline(y=-9.81, color='black', linestyle='--', label='Theoretical -9.81 m/s²')
axs[2].set_xlabel('Time t (s)')
axs[2].set_ylabel('Acceleration a (m/s²)')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig('motion.png', dpi=300)
print("Plot saved as 'motion.png'.")