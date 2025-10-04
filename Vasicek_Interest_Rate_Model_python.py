"""
Vasicek Interest Rate Model
"""

import pandas as pd
import urllib.request

CSV_URL = "https://home.treasury.gov/system/files/276/yield-curve-rates-1990-2024.csv"
LOCAL_FILE = "treasury_yield_curve.csv"

with urllib.request.urlopen(CSV_URL) as resp:
    data = resp.read()
    
with open(LOCAL_FILE, "wb") as f:
    f.write(data)



data = pd.read_csv('treasury_yield_curve.csv')

time_horizon = '3 Mo'
n_days = 252
n_years = 1
selected_data = pd.DataFrame(data[['Date',time_horizon]])
selected_data.set_index('Date', inplace = True)


actual_rates = selected_data / 100
actual_rates = actual_rates[:n_years * n_days]



# === MLE ===

import numpy as np
from scipy.optimize import minimize

r = actual_rates[time_horizon].dropna().to_numpy(float)
r_t, r_next = r[:-1], r[1:]
dt = 1.0 / n_days

def nll(theta):
    a, b, sigma = theta
    if a <= 0.0 or sigma <= 0.0:
        return 1e50
    exp_adt = np.exp(-a * dt)
    mean = r_t * exp_adt + b * (1.0 - exp_adt)
    var = (sigma**2) * (1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a)
    if np.any(var <= 0.0) or not np.isfinite(var).all():
        return 1e50
    resid = r_next - mean
    return 0.5 * np.sum(np.log(2.0 * np.pi * var) + (resid**2) / var)

# inizialization
rng = np.random.default_rng(42)
a0 = rng.uniform(0.05, 1.5)
b0 = float(np.median(r))
sigma0 = max(np.std(np.diff(r), ddof=1) / np.sqrt(dt), 1e-4)
x0 = np.array([a0, b0, sigma0])

bounds = [(1e-6, 10.0), (-0.10, 0.20), (1e-6, 1.0)]
res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)

a_hat, b_hat, sigma_hat = res.x
r0 = float(r[0])

print("\n=== Output MLE ===")
print(f"a (mean reversion): {a_hat:.6f}")
print(f"b (long-run mean) : {b_hat*100:.4f}%")
print(f"sigma (vol)       : {sigma_hat*100:.4f}%")
print(f"r0 (iniziale)     : {r0*100:.4f}%")


