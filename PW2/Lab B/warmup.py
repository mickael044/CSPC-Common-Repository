import numpy as np
from scipy.optimize import minimize, newton

def f(x):
    return (x - 3)**2 + 1
def df(x):
    return 2 * (x - 3)
def d2f(x):
    return 2

x0 = 0.0

# 1. Gradient descent (manual)
x_gd = x0
lr = 0.1
for _ in range(100):
    x_gd = x_gd - lr * df(x_gd)

x_newton = newton(df, x0, fprime=d2f)

# 3. SLSQP
res_slsqp = minimize(f, x0, method="SLSQP")

print("=== Part 2A Results ===")
print(f"Gradient Descent: x ≈ {x_gd:.4f}")
print(f"Newton's Method:  x ≈ {x_newton:.4f}")
print(f"SLSQP:            x ≈ {res_slsqp.x[0]:.4f}")

# --- Part 2B: A harder landscape ---
def g(x):
    return x**4 - 3 * x**2 + x + 5

def dg(x):
    return 4 * x**3 - 6 * x + 1

def d2g(x):
    return 12 * x**2 - 6

print("\n=== Part 2B Results ===")
for start_x in [0.0, 2.0]:
    # Newton's method on g'(x) = 0
    x_newt = newton(dg, start_x, fprime=d2g)
    second_deriv = d2g(x_newt)
    point_type = "minimum" if second_deriv > 0 else ("maximum" if second_deriv < 0 else "inversion")
    
    # SLSQP
    res_g = minimize(g, start_x, method="SLSQP")
    
    print(f"\nStart x0 = {start_x}:")
    print(f"  Newton found stationary point x ≈ {x_newt:.4f} (g'' = {second_deriv:.2f} -> {point_type})")
    print(f"  SLSQP found minimum at x ≈ {res_g.x[0]:.4f} with g(x) = {res_g.fun:.4f}")