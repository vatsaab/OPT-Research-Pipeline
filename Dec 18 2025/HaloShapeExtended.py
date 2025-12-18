"""
MW–M31 Halo Shape Evolution
Standard Inertia Tensor + Convex Hull Geometry

This script computes:
1. Mass-weighted inertia-tensor axis ratios (standard practice)
2. Purely geometric convex-hull axis ratios
3. Convex-hull volume
4. Ellipsoidal volume from inertia axes
5. Volume inflation factor (geometric irregularity indicator)

Designed for merger snapshots of the MW–M31 system.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from CenterOfMass2 import CenterOfMass


# ------------------------------------------------------------
# Inertia tensor–based shape (standard practice)
# ------------------------------------------------------------

def compute_inertia_tensor(positions, masses):
    """
    Compute the unreduced inertia tensor:
        I_ij = sum_k m_k (r_k^2 δ_ij - x_{k,i} x_{k,j})
    """
    r2 = np.sum(positions**2, axis=1)
    I = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            delta = 1.0 if i == j else 0.0
            I[i, j] = np.sum(masses * (r2 * delta - positions[:, i] * positions[:, j]))
    return I


def inertia_axis_ratios(I):
    """
    Return (b/a, c/a) from inertia tensor eigenvalues.
    """
    eigvals = np.sort(np.linalg.eigvalsh(I))[::-1]
    b_over_a = np.sqrt(eigvals[1] / eigvals[0])
    c_over_a = np.sqrt(eigvals[2] / eigvals[0])
    return b_over_a, c_over_a, eigvals


# ------------------------------------------------------------
# Convex hull–based geometric shape (novel method)
# ------------------------------------------------------------

def convex_hull_properties(positions):
    """
    Compute convex hull volume and principal axis ratios.

    Axis ratios are computed from the covariance of hull vertices,
    not from particle masses.
    """
    hull = ConvexHull(positions)
    verts = positions[hull.vertices]

    # Covariance of hull geometry
    cov = np.cov(verts.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]

    b_over_a = np.sqrt(eigvals[1] / eigvals[0])
    c_over_a = np.sqrt(eigvals[2] / eigvals[0])

    return hull.volume, b_over_a, c_over_a


def ellipsoid_volume_from_inertia(eigvals):
    """
    Construct an equivalent ellipsoid volume from inertia eigenvalues.
    Used only for comparison, not as a physical halo model.
    """
    a = np.sqrt(eigvals[0])
    b = np.sqrt(eigvals[1])
    c = np.sqrt(eigvals[2])
    return (4.0 / 3.0) * np.pi * a * b * c


# ------------------------------------------------------------
# Snapshot loop
# ------------------------------------------------------------

snaps = np.arange(1, 802)  # 801 snapshots

# Inertia-based time series
ba_inertia = np.zeros(len(snaps))
ca_inertia = np.zeros(len(snaps))

# Convex-hull–based time series
ba_hull = np.zeros(len(snaps))
ca_hull = np.zeros(len(snaps))
hull_volume = np.zeros(len(snaps))
volume_inflation = np.zeros(len(snaps))


for i, s in enumerate(snaps):

    mw_file = f"MW_{s}.txt"
    m31_file = f"M31_{s}.txt"

    MW = CenterOfMass(mw_file, 1)
    M31 = CenterOfMass(m31_file, 1)

    # Combine particle data
    x = np.concatenate((MW.x, M31.x))
    y = np.concatenate((MW.y, M31.y))
    z = np.concatenate((MW.z, M31.z))
    m = np.concatenate((MW.m, M31.m))
    pos = np.vstack((x, y, z)).T

    # Center on MW center of mass
    xcom, ycom, zcom = MW.COMdefine(x, y, z, m)
    pos -= np.array([xcom, ycom, zcom])

    # --- Inertia tensor shape ---
    I = compute_inertia_tensor(pos, m)
    bI, cI, eigvals = inertia_axis_ratios(I)

    ba_inertia[i] = bI
    ca_inertia[i] = cI

    # --- Convex hull shape ---
    Vh, bH, cH = convex_hull_properties(pos)
    ba_hull[i] = bH
    ca_hull[i] = cH
    hull_volume[i] = Vh

    # --- Volume comparison ---
    Vell = ellipsoid_volume_from_inertia(eigvals)
    volume_inflation[i] = Vh / Vell

    print(f"Snapshot {s:3d} | Inertia b/a={bI:.3f}, c/a={cI:.3f} | "
          f"Hull b/a={bH:.3f}, c/a={cH:.3f}")


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------

# Axis ratios
plt.figure(figsize=(7, 5))
plt.plot(snaps, ba_inertia, label="Inertia b/a")
plt.plot(snaps, ca_inertia, label="Inertia c/a")
plt.plot(snaps, ba_hull, "--", label="Hull b/a")
plt.plot(snaps, ca_hull, "--", label="Hull c/a")
plt.xlabel("Snapshot")
plt.ylabel("Axis Ratio")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("halo_axis_ratio_comparison.png", dpi=300)

# Hull volume
plt.figure(figsize=(7, 5))
plt.plot(snaps, hull_volume)
plt.xlabel("Snapshot")
plt.ylabel("Convex Hull Volume")
plt.grid()
plt.tight_layout()
plt.savefig("halo_hull_volume.png", dpi=300)

# Volume inflation factor
plt.figure(figsize=(7, 5))
plt.plot(snaps, volume_inflation)
plt.xlabel("Snapshot")
plt.ylabel("Hull Volume / Ellipsoid Volume")
plt.grid()
plt.tight_layout()
plt.savefig("halo_volume_inflation.png", dpi=300)

plt.show()
