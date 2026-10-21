import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read titration.csv
data = pd.read_csv("titration.csv")
V = data["volume_base"].values
pH = data["pH"].values

# Compute derivative dpH/dV
slope = np.gradient(pH, V)

# Max slope index
eq_idx = np.argmax(slope)
eq_volume = V[eq_idx]

print(f"Equivalence point is at V ≈ {eq_volume:.2f} mL")

# Two side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(V, pH, 'b.-', label="pH Curve")
ax1.axvline(eq_volume, color='r', linestyle='--', label=f"Equivalence ({eq_volume} mL)")
ax1.set_xlabel("Volume Base (mL)")
ax1.set_ylabel("pH")
ax1.set_title("Titration Curve")
ax1.legend()
ax1.grid(True)

ax2.plot(V, slope, 'g.-', label="dpH / dV")
ax2.axvline(eq_volume, color='r', linestyle='--', label=f"Peak ({eq_volume} mL)")
ax2.set_xlabel("Volume Base (mL)")
ax2.set_ylabel("Slope (dpH/dV)")
ax2.set_title("First Derivative (Slope)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("titration.png")
plt.close()
print("Saved titration.png")