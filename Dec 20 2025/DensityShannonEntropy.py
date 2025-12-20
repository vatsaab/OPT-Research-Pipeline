import numpy as np
import matplotlib.pyplot as plt
from CenterOfMass2 import CenterOfMass
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

# Snapshot range
start_snap = 0
end_snap = 600
snaps = np.arange(start_snap, end_snap + 1)
nsnaps = len(snaps)

# Radial binning (logarithmic to resolve inner halo and outskirts)
r_bins = np.logspace(-1, 2.6, 40)
r_mid = np.sqrt(r_bins[:-1] * r_bins[1:])
nbins = len(r_mid)

# Angular discretization
# Chosen to balance angular resolution and particle noise
n_theta = 12   # polar bins
n_phi = 24     # azimuthal bins
n_ang_bins = n_theta * n_phi

# Storage for angular entropy as a function of time and radius
angular_entropy = np.full((nsnaps, nbins), np.nan)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def spherical_angles(pos):
    """
    Convert Cartesian positions into spherical coordinates.

    Parameters
    ----------
    pos : (N,3) array
        Particle positions in a centered reference frame.

    Returns
    -------
    r : array
        Radial distances.
    theta : array
        Polar angles in [0, pi].
    phi : array
        Azimuthal angles in [0, 2pi).

    Notes
    -----
    Only angular information is used in the entropy calculation;
    the radial coordinate is used exclusively for shell selection.
    """
    r = np.linalg.norm(pos, axis=1)
    theta = np.arccos(np.clip(pos[:,2] / r, -1, 1))
    phi = np.mod(np.arctan2(pos[:,1], pos[:,0]), 2*np.pi)
    return r, theta, phi


def angular_entropy_shell(theta, phi, masses):
    """
    Compute the normalized angular information entropy
    of the mass distribution on a spherical shell.

    Parameters
    ----------
    theta, phi : arrays
        Angular coordinates of particles in the shell.
    masses : array
        Particle masses.

    Returns
    -------
    S_norm : float
        Shannon entropy normalized to the maximum possible value.

    Physical Meaning
    ----------------
    - S_norm ~ 1 : isotropic angular distribution
    - S_norm < 1 : anisotropy, substructure, or tidal features

    This statistic is independent of radius and total mass
    and does not assume ellipsoidal symmetry.
    """
    theta_bins = np.linspace(0, np.pi, n_theta + 1)
    phi_bins = np.linspace(0, 2*np.pi, n_phi + 1)

    # Mass-weighted angular histogram
    hist = np.zeros((n_theta, n_phi))

    theta_idx = np.digitize(theta, theta_bins) - 1
    phi_idx = np.digitize(phi, phi_bins) - 1

    valid = (
        (theta_idx >= 0) & (theta_idx < n_theta) &
        (phi_idx >= 0) & (phi_idx < n_phi)
    )

    for t, p, m in zip(theta_idx[valid], phi_idx[valid], masses[valid]):
        hist[t, p] += m

    total_mass = np.sum(hist)
    if total_mass == 0:
        return np.nan

    # Probability distribution over angular bins
    prob = hist.flatten() / total_mass
    prob = prob[prob > 0]

    # Shannon entropy
    S = -np.sum(prob * np.log(prob))
    S_max = np.log(n_ang_bins)

    return S / S_max


# ============================================================
# MAIN ANALYSIS LOOP
# ============================================================

processed = 0

for i, s in enumerate(snaps):
    mw_file = f"MW_{s:03d}.txt"
    m31_file = f"M31_{s:03d}.txt"

    # Load particle data
    try:
        MW = CenterOfMass(mw_file, 1)
        M31 = CenterOfMass(m31_file, 1)
        processed += 1
        print(f"Processed {processed}/{nsnaps} snapshots (s={s})")
    except Exception:
        continue

    # Combine MW and M31 particles into a single system
    x = np.concatenate((MW.x, M31.x))
    y = np.concatenate((MW.y, M31.y))
    z = np.concatenate((MW.z, M31.z))
    m = np.concatenate((MW.m, M31.m))
    pos = np.vstack((x, y, z)).T

    # Recenter using the MW center of mass
    # This removes bulk motion while preserving merger asymmetries
    xcom, ycom, zcom = MW.COMdefine(x, y, z, m)
    pos -= np.array([xcom, ycom, zcom])

    # Convert to spherical coordinates
    r, theta, phi = spherical_angles(pos)

    # Compute angular entropy in each radial shell
    for b in range(nbins):
        shell = (r >= r_bins[b]) & (r < r_bins[b+1])

        # Require sufficient sampling to suppress shot noise
        if np.sum(shell) < 50:
            continue

        angular_entropy[i, b] = angular_entropy_shell(
            theta[shell], phi[shell], m[shell]
        )

# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

im = plt.imshow(
    angular_entropy.T,
    origin="lower",
    aspect="auto",
    extent=[snaps[0], snaps[-1], r_mid[0], r_mid[-1]],
    cmap="magma"
)

plt.colorbar(im, label="Normalized Angular Entropy")
plt.ylabel("Radius (kpc)")
plt.xlabel("Snapshot Number")
plt.title("Angular Information Entropy Evolution of the MW–M31 Halo")

plt.tight_layout()
plt.savefig("angular_entropy_halo_evolution.png", dpi=300)
plt.show()
