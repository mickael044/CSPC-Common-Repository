import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = pd.read_csv("kinetics.csv")
t = data["time"].values
C_measured = data["concentration"].values
C0 = C_measured[0]

# 2. Define total error function
def total_error(k):
    C_model = C0 * np.exp(-k * t)
    return np.sum((C_measured - C_model)**2)

# 3. Minimize with SLSQP
res = minimize(total_error, x0=0.5, method="SLSQP", bounds=[(0, 5)])
fitted_k = res.x[0]

print(f"Fitted rate constant k ≈ {fitted_k:.4f}")

# 4. Plot data and fitted curve
plt.figure(figsize=(8, 5))
plt.scatter(t, C_measured, color="red", label="Measured Data")
plt.plot(t, C0 * np.exp(-fitted_k * t), label=f"Fitted Curve (k={fitted_k:.4f})", color="blue")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Chemical Kinetics: Reaction Rate Fitting")
plt.legend()
plt.grid(True)
plt.savefig("kinetics.png")
plt.close()
print("Saved kinetics.png")