# Differential Density Profile via Enclosed Mass Method
import numpy as np
from ReadFile import Read
from CenterOfMass2 import CenterOfMass
import matplotlib.pyplot as plt

snap = 445
mw_file = f"MW_{snap}.txt"
m31_file = f"M31_{snap}.txt"

MW = CenterOfMass(mw_file, 1)
M31 = CenterOfMass(m31_file, 1)

x = np.concatenate((MW.x, M31.x))
y = np.concatenate((MW.y, M31.y))
z = np.concatenate((MW.z, M31.z))
m = np.concatenate((MW.m, M31.m))
positions = np.vstack((x, y, z)).T

xcom, ycom, zcom = MW.COMdefine(x, y, z, m)
positions -= np.array([xcom, ycom, zcom])

radii = np.linalg.norm(positions, axis=1)
sort_idx = np.argsort(radii)
r_sorted = radii[sort_idx]
m_sorted = m[sort_idx]

M_enc = np.cumsum(m_sorted)
dr = np.diff(r_sorted)
dM = np.diff(M_enc)
rho_enc = dM / (4 * np.pi * r_sorted[1:]**2 * dr)

plt.figure()
plt.loglog(r_sorted[1:], rho_enc, marker='o')
plt.xlabel("Radius [kpc]")
plt.ylabel("Differential Density [1e10 Msun/kpc³]")
plt.title("Enclosed Mass / Differential Density Profile")
plt.grid(True, which='both')
plt.show()
