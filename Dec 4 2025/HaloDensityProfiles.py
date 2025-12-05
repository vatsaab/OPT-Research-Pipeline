# Code for calculating and plotting various density profiles for the halo 
import numpy as np
import matplotlib.pyplot as plt
from ReadFile import Read
from CenterOfMass2 import CenterOfMass
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree, Delaunay

snap = 445
mw_file = f"MW_{snap}.txt"
m31_file = f"M31_{snap}.txt"

time_mw, _, _ = Read(mw_file)
time_m31, _, _ = Read(m31_file)

MW = CenterOfMass(mw_file, 1)
M31 = CenterOfMass(m31_file, 1)

x = np.concatenate((MW.x, M31.x))
y = np.concatenate((MW.y, M31.y))
z = np.concatenate((MW.z, M31.z))
m = np.concatenate((MW.m, M31.m))
pos = np.vstack((x, y, z)).T

xcom, ycom, zcom = MW.COMdefine(x, y, z, m)
pos -= np.array([xcom, ycom, zcom])

r_all = np.linalg.norm(pos, axis=1)
r_bins = np.logspace(-1, 2.6, 60)
r_mid = np.sqrt(r_bins[:-1] * r_bins[1:])

def shell_mass_density(positions, masses, r_bins):
    r = np.linalg.norm(positions, axis=1)
    dens = np.zeros(len(r_bins)-1)
    for i in range(len(r_bins)-1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        shell_mass = np.sum(masses[mask])
        vol = (4/3) * np.pi * (r_bins[i+1]**3 - r_bins[i]**3)
        dens[i] = shell_mass / vol
    return r_mid, dens

def kde_spherical(positions, masses, r_grid, h=None):
    r = np.linalg.norm(positions, axis=1)
    if h is None:
        h = np.median(np.diff(np.sort(r))) * 5
    kernel = lambda d: np.exp(-(d/h)**2) / (np.sqrt(np.pi) * h)
    dens = np.zeros_like(r_grid)
    for i, R in enumerate(r_grid):
        d = np.abs(R - r)
        dens[i] = np.sum(masses * kernel(d)) / (4*np.pi*R**2 + 1e-30)
    return dens

def nearest_neighbor_density(positions, masses, k=32):
    tree = cKDTree(positions)
    distances, _ = tree.query(positions, k=k)
    rN = distances[:, -1]
    vol = (4/3) * np.pi * rN**3
    dens_particles = (k * np.mean(masses)) / vol
    return np.linalg.norm(positions, axis=1), dens_particles

def octree_density(positions, masses, max_particles=200, max_depth=10):
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    cells = [(mins, maxs, np.arange(len(positions)), 0)]
    cell_centers = []
    cell_densities = []
    while cells:
        mn, mx, idxs, depth = cells.pop()
        pts_idx = idxs
        n = pts_idx.size
        vol = np.prod(mx - mn)
        if n <= max_particles or depth >= max_depth:
            mass = np.sum(masses[pts_idx])
            if vol > 0:
                cell_centers.append((mn + mx) / 2.0)
                cell_densities.append(mass / vol)
            continue
        mid = (mn + mx) / 2.0
        for ix in range(2):
            for iy in range(2):
                for iz in range(2):
                    low = mn.copy()
                    high = mx.copy()
                    low[0] = mn[0] if ix == 0 else mid[0]
                    high[0] = mid[0] if ix == 0 else mx[0]
                    low[1] = mn[1] if iy == 0 else mid[1]
                    high[1] = mid[1] if iy == 0 else mx[1]
                    low[2] = mn[2] if iz == 0 else mid[2]
                    high[2] = mid[2] if iz == 0 else mx[2]
                    mask = np.all((positions[pts_idx] >= low) & (positions[pts_idx] < high), axis=1)
                    child_idxs = pts_idx[mask]
                    if child_idxs.size > 0:
                        cells.append((low, high, child_idxs, depth+1))
    centers = np.array(cell_centers)
    dens = np.array(cell_densities)
    radii = np.linalg.norm(centers, axis=1)
    order = np.argsort(radii)
    return radii[order], dens[order]

def delaunay_density(positions, masses):
    tri = Delaunay(positions)
    simplices = tri.simplices
    pts = positions
    tetra_vols = np.abs(np.linalg.det(pts[simplices][:,1:] - pts[simplices][:,0,None])) / 6.0
    tetra_masses = masses[simplices].sum(axis=1)
    dens = tetra_masses / tetra_vols
    centroids = pts[simplices].mean(axis=1)
    radii = np.linalg.norm(centroids, axis=1)
    order = np.argsort(radii)
    return radii[order], dens[order]

def enclosed_differential(positions, masses, r_bins):
    r = np.linalg.norm(positions, axis=1)
    cumM = np.array([np.sum(masses[r <= R]) for R in r_bins])
    dens = np.diff(cumM) / ((4/3)*np.pi*(r_bins[1:]**3 - r_bins[:-1]**3))
    rmid = np.sqrt(r_bins[:-1] * r_bins[1:])
    return rmid, dens

r_shell, dens_shell = shell_mass_density(pos, m, r_bins)
r_kde = np.logspace(-1, 2.6, 200)
dens_kde = kde_spherical(pos, m, r_kde, h=None)
r_nn, dens_nn_particles = nearest_neighbor_density(pos, m, k=32)
r_oct, dens_oct = octree_density(pos, m, max_particles=150, max_depth=8)
r_del, dens_del = delaunay_density(pos, m)
r_enc, dens_enc = enclosed_differential(pos, m, r_bins)

plt.figure(figsize=(8,6))
plt.loglog(r_shell, dens_shell, marker='o', label='Shell Mass')
plt.loglog(r_kde, dens_kde, label='KDE (spherical kernel)')
plt.loglog(r_enc, dens_enc, linestyle='--', label='Enclosed Diff')
plt.loglog(r_nn, dens_nn_particles, marker='.', linestyle='None', label='Nearest-Neighbor (particles)')
plt.loglog(r_oct, dens_oct, marker='x', linestyle='None', label='Octree cells')
plt.loglog(r_del, dens_del, marker='.', linestyle='None', label='Delaunay tetrahedra')
plt.xlabel('Radius (kpc)')
plt.ylabel('Density (mass units / kpc^3)')
plt.title(f'Density Estimates (Snapshot {snap})')
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig('density_all5_methods.png', dpi=300)
plt.show()
