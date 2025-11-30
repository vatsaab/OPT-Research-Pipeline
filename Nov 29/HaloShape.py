import numpy as np
import matplotlib.pyplot as plt
from CenterOfMass2 import CenterOfMass

def compute_inertia_tensor(positions, masses):
    r_squared = np.sum(positions**2, axis=1)
    I = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            delta = 1.0 if i == j else 0.0
            I[i, j] = np.sum(masses * (r_squared * delta - positions[:, i] * positions[:, j]))
    return I

def compute_axis_ratios(inertia_tensor):
    eigvals, _ = np.linalg.eigh(inertia_tensor)
    eigvals = np.sort(eigvals)[::-1]
    return np.sqrt(eigvals[1] / eigvals[0]), np.sqrt(eigvals[2] / eigvals[0]), eigvals

def radial_shell_axis_ratios(positions, masses, r_bins):
    num_shells = len(r_bins) - 1
    ba_array = np.full(num_shells, np.nan)
    ca_array = np.full(num_shells, np.nan)
    radii = np.linalg.norm(positions, axis=1)

    for i in range(num_shells):
        r1, r2 = r_bins[i], r_bins[i + 1]
        mask = (radii >= r1) & (radii < r2)
        if np.sum(mask) >= 20:
            I_shell = compute_inertia_tensor(positions[mask], masses[mask])
            ba, ca, _ = compute_axis_ratios(I_shell)
            ba_array[i] = ba
            ca_array[i] = ca
    return ba_array, ca_array

MW = CenterOfMass("MW_445.txt", 1)
M31 = CenterOfMass("M31_445.txt", 1)

x = np.concatenate((MW.x, M31.x))
y = np.concatenate((MW.y, M31.y))
z = np.concatenate((MW.z, M31.z))
m = np.concatenate((MW.m, M31.m))
positions = np.vstack((x, y, z)).T

x_com, y_com, z_com = MW.COMdefine(x, y, z, m)
positions -= np.array([x_com, y_com, z_com])

I_tensor = compute_inertia_tensor(positions, m)
b_to_a, c_to_a, eigvals = compute_axis_ratios(I_tensor)

print("Snapshot 445 — Inertia Tensor:\n", I_tensor)
print(f"Global Axis Ratios: b/a = {b_to_a:.3f}, c/a = {c_to_a:.3f}")

if np.isclose(b_to_a, 1.0, atol=0.05) and np.isclose(c_to_a, 1.0, atol=0.05):
    shape = "spherical"
elif np.isclose(b_to_a, c_to_a, atol=0.05):
    shape = "prolate" if c_to_a < 1.0 else "oblate"
else:
    shape = "triaxial"

print(f"→ Global halo shape: {shape.upper()}")
