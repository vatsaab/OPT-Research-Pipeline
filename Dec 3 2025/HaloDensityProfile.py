# Halo Density Profile for the merger snapshot 
import numpy as np
import matplotlib.pyplot as plt
from ReadFile import Read
from CenterOfMass2 import CenterOfMass

def density_profile(positions, masses, r_bins):
    r = np.linalg.norm(positions, axis=1)
    density = np.zeros(len(r_bins)-1)
    for i in range(len(r_bins)-1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        shell_mass = np.sum(masses[mask])
        r1, r2 = r_bins[i], r_bins[i+1]
        vol = (4/3) * np.pi * (r2**3 - r1**3)
        density[i] = shell_mass / vol
    return density

# Load one snapshot (change number if desired)
snap = 445
mw_file = f"MW_{snap}.txt"
m31_file = f"M31_{snap}.txt"

time_mw, count_mw, data_mw = Read(mw_file)
time_m31, count_m31, data_m31 = Read(m31_file)

MW = CenterOfMass(mw_file, 1)
M31 = CenterOfMass(m31_file, 1)

x = np.concatenate((MW.x, M31.x))
y = np.concatenate((MW.y, M31.y))
z = np.concatenate((MW.z, M31.z))
m = np.concatenate((MW.m, M31.m))
pos = np.vstack((x, y, z)).T

xcom, ycom, zcom = MW.COMdefine(x, y, z, m)
pos -= np.array([xcom, ycom, zcom])

r_bins = np.logspace(-1, 2.6, 40)
dens = density_profile(pos, m, r_bins)
r_mid = np.sqrt(r_bins[:-1] * r_bins[1:])

# Hernquist
Mtot = np.sum(m)
a = 30
hern = (Mtot / (2*np.pi)) * (a / (r_mid * (r_mid + a)**3))

# NFW
rho0 = 0.04
rs = 25
nfw = rho0 / ((r_mid/rs)*(1+r_mid/rs)**2)

# Isothermal
sigma2 = 100
iso = sigma2 / (2*np.pi * r_mid**2)

plt.figure(figsize=(8,6))
plt.loglog(r_mid, dens, label="Simulated MW+M31 Density", marker='o')
plt.loglog(r_mid, hern, label="Hernquist", linestyle='--')
plt.loglog(r_mid, nfw, label="NFW", linestyle='--')
plt.loglog(r_mid, iso, label="Isothermal", linestyle='--')

plt.xlabel("Radius (kpc)")
plt.ylabel("Density")
plt.title("Density Profile of MW+M31 Halo (Snapshot {})".format(snap))
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig("density_profile_comparison.png", dpi=300)
plt.show()
