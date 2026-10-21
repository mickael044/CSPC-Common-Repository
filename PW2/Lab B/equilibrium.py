import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 50.0

# Define the equilibrium imbalance function: (2x)^2 / ((1-x)(1-x)) - K = 0
def k_imbalance(x):
    return ((2 * x)**2) / ((1 - x)**2) - K

# Derivative of k_imbalance for Newton's method
def dk_imbalance(x):
    return 8 * x / ((1 - x)**3)

# 1. Newton's method with a closer initial guess (x0=0.6) to avoid divergence
try:
    x_newton = newton(k_imbalance, x0=0.6, fprime=dk_imbalance)
except RuntimeError:
    # Secant fallback if derivative stepping exceeds tolerance
    x_newton = newton(k_imbalance, x0=0.6, x1=0.7)

# 2. SLSQP minimizer minimizing the squared imbalance
res = minimize(lambda x: k_imbalance(x[0])**2, x0=[0.5], method="SLSQP", bounds=[(0.01, 0.99)])
x_slsqp = res.x[0]

print(f"Equilibrium extent x (Newton): {x_newton:.4f}")
print(f"Equilibrium extent x (SLSQP) : {x_slsqp:.4f}")

# Calculate equilibrium mole amounts
nH2 = 1 - x_slsqp
nI2 = 1 - x_slsqp
nHI = 2 * x_slsqp
print(f"Equilibrium amounts -> H2: {nH2:.2f} mol, I2: {nI2:.2f} mol, HI: {nHI:.2f} mol")

# Plot mole amounts vs reaction extent x
x_vals = np.linspace(0, 0.9, 100)
plt.figure(figsize=(8, 5))
plt.plot(x_vals, 1 - x_vals, label="H2 / I2", color="blue")
plt.plot(x_vals, 2 * x_vals, label="HI", color="green")
plt.axvline(x=x_slsqp, color="red", linestyle="--", label=f"Equilibrium (x={x_slsqp:.2f})")
plt.xlabel("Reaction Extent (x)")
plt.ylabel("Moles")
plt.title("Chemical Equilibrium Composition")
plt.legend()
plt.grid(True)
plt.savefig("equilibrium.png")
plt.close()
print("Saved equilibrium.png")