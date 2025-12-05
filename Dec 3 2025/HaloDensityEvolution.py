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

snapshots = np.arange(100, 801)
r_bins = np.logspace(-1, 2.6, 40)
dens_matrix = np.zeros((len(snapshots), len(r_bins)-1))

for idx, snap in enumerate(snapshots):
    mw_file = f"MW_{snap}.txt"
    m31_file = f"M31_{snap}.txt"
    
    MW = CenterOfMass(mw_file, 1)
    M31 = CenterOfMass(m31_file, 1)
    
    x = np.concatenate((MW.x, M31.x))
    y = np.concatenate((MW.y, M31.y))
    z = np.concatenate((MW.z, M31.z))
    m = np.concatenate((MW.m, M31.m))
    pos = np.vstack((x, y, z)).T
    
    xcom, ycom, zcom = MW.COMdefine(x, y, z, m)
    pos -= np.array([xcom, ycom, zcom])
    
    dens_matrix[idx] = density_profile(pos, m, r_bins)
    print(f"Processed snapshot {snap}")

r_mid = np.sqrt(r_bins[:-1]*r_bins[1:])

plt.figure(figsize=(10,6))
plt.imshow(np.log10(dens_matrix.T), aspect='auto', origin='lower',
           extent=[snapshots[0], snapshots[-1], r_mid[0], r_mid[-1]],
           cmap='viridis')
plt.colorbar(label='log10(Density)')
plt.yscale('log')
plt.xlabel("Snapshot Number")
plt.ylabel("Radius (kpc)")
plt.title("Time Evolution of MW+M31 Halo Density Profile")
plt.tight_layout()
plt.savefig("density_evolution_heatmap.png", dpi=300)
plt.show()
