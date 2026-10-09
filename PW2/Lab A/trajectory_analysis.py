import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('trajectory.csv', delimiter=',', skiprows=1)

t = data[:, 0]
x = data[:, 1]
y = data[:, 2]

vx = np.gradient(x, t)
vy = np.gradient(y, t)

speed = np.sqrt(vx**2 + vy**2)

fig, axs = plt.subplots(1, 2 , figsize=(12, 5))

axs[0].plot(x, y, label='2D Trajectory', color='purple')
axs[0].set_xlabel('Position x (m)')
axs[0].set_ylabel('Position y (m)')
axs[0].set_title('Tracked Path (x vs y)')
axs[0].grid(True)
axs[0].legend()

# Panel 2: Speed over time
axs[1].plot(t, speed, label=r'Speed $\sqrt{v_x^2 + v_y^2}$', color='crimson')
axs[1].set_xlabel('Time t (s)')
axs[1].set_ylabel('Speed (m/s)')
axs[1].set_title('Speed vs Time')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()

# we save graph
plt.savefig('trajectory_plot.png', dpi=300)
print("Bonus qrafik 'trajectory_plot.png' adı ilə saxlanıldı.")

plt.show()