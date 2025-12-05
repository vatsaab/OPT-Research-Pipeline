import numpy as np
import matplotlib.pyplot as plt
from ReadFile import Read
from CenterOfMass2 import CenterOfMass
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree, Delaunay
import warnings

warnings.filterwarnings("ignore")

start_snap = 0
end_snap = 600
snaps = np.arange(start_snap, end_snap + 1)
nsnaps = len(snaps)

r_bins = np.logspace(-1, 2.6, 40)
r_mid = np.sqrt(r_bins[:-1] * r_bins[1:])
nbins = len(r_mid)

dens_kde_all = np.full((nsnaps, nbins), np.nan)
dens_enc_all = np.full((nsnaps, nbins), np.nan)
dens_nn_all = np.full((nsnaps, nbins), np.nan)
dens_oct_all = np.full((nsnaps, nbins), np.nan)
dens_del_all = np.full((nsnaps, nbins), np.nan)

for i, s in enumerate(snaps):
    mw_file = f"MW_{s:03d}.txt"
    m31_file = f"M31_{s:03d}.txt"
    try:
        MW = CenterOfMass(mw_file, 1)
        M31 = CenterOfMass(m31_file, 1)
    except Exception:
        continue

    x = np.concatenate((MW.x, M31.x))
    y = np.concatenate((MW.y, M31.y))
    z = np.concatenate((MW.z, M31.z))
    m = np.concatenate((MW.m, M31.m))
    pos = np.vstack((x, y, z)).T

    xcom, ycom, zcom = MW.COMdefine(x, y, z, m)
    pos -= np.array([xcom, ycom, zcom])

    r = np.linalg.norm(pos, axis=1)

    # Method: Enclosed mass / differential density
    try:
        cumM = np.array([np.sum(m[r <= R]) for R in r_bins])
        dens_enc = np.diff(cumM) / ((4.0/3.0) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3))
        dens_enc_all[i, :] = dens_enc
    except Exception:
        pass

    # Method: KDE on radial distances (spherical KDE)
    try:
        weights = m / np.sum(m)
        kde = gaussian_kde(r, weights=weights)
        dens_kde = kde(r_mid)
        dens_kde_all[i, :] = dens_kde
    except Exception:
        pass

    # Method: Nearest-neighbor per-particle then bin to r_bins
    try:
        tree = cKDTree(pos)
        k = 32
        distances, _ = tree.query(pos, k=k)
        rN = distances[:, -1]
        vol = (4.0/3.0) * np.pi * rN**3
        # approximate local particle density (mass per volume) using particle mass proportionality
        # use each particle's mass scaled by average particle mass ratio if masses vary
        dens_particles = (k * np.mean(m)) / vol
        # bin particle densities by radius
        inds = np.digitize(r, bins=r_bins) - 1
        for b in range(nbins):
            mask = inds == b
            if np.sum(mask) > 0:
                dens_nn_all[i, b] = np.nanmean(dens_particles[mask])
    except Exception:
        pass

    # Method: Octree (recursive subdivision) -> collect cell centers/densities then bin by radius
    try:
        def octree_cells(positions, masses, max_particles=200, max_depth=8):
            mins = positions.min(axis=0)
            maxs = positions.max(axis=0)
            cells = [(mins, maxs, np.arange(len(positions)), 0)]
            centers = []
            dens = []
            while cells:
                mn, mx, idxs, depth = cells.pop()
                n = idxs.size
                vol = np.prod(mx - mn)
                if n <= max_particles or depth >= max_depth or vol == 0:
                    mass = np.sum(masses[idxs]) if n > 0 else 0.0
                    if vol > 0 and mass > 0:
                        centers.append((mn + mx) / 2.0)
                        dens.append(mass / vol)
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
                            pts = positions[idxs]
                            if pts.size == 0:
                                continue
                            mask = np.all((pts >= low) & (pts < high), axis=1)
                            child_idxs = idxs[mask]
                            if child_idxs.size > 0:
                                cells.append((low, high, child_idxs, depth+1))
            if len(centers) == 0:
                return np.array([]), np.array([])
            centers = np.array(centers)
            dens = np.array(dens)
            return centers, dens

        centers, cell_dens = octree_cells(pos, m, max_particles=150, max_depth=7)
        if centers.size:
            radii_cells = np.linalg.norm(centers, axis=1)
            inds = np.digitize(radii_cells, bins=r_bins) - 1
            for b in range(nbins):
                mask = inds == b
                if np.sum(mask) > 0:
                    dens_oct_all[i, b] = np.nanmean(cell_dens[mask])
    except Exception:
        pass

    # Method: Delaunay tessellation -> tetrahedral cell densities then bin by centroid radius
    try:
        tri = Delaunay(pos)
        simplices = tri.simplices
        pts = pos
        tetra = pts[simplices]
        vols = np.abs(np.linalg.det(tetra[:,1:] - tetra[:,0,None])) / 6.0
        tetra_m = m[simplices].sum(axis=1)
        tetra_dens = tetra_m / vols
        centroids = tetra.mean(axis=1)
        radii_cent = np.linalg.norm(centroids, axis=1)
        inds = np.digitize(radii_cent, bins=r_bins) - 1
        for b in range(nbins):
            mask = inds == b
            if np.sum(mask) > 0:
                dens_del_all[i, b] = np.nanmean(tetra_dens[mask])
    except Exception:
        pass

# plotting heatmaps for each method
methods = [
    ("KDE", dens_kde_all),
    ("EnclosedDiff", dens_enc_all),
    ("NearestNeighbor", dens_nn_all),
    ("Octree", dens_oct_all),
    ("Delaunay", dens_del_all),
]

fig, axes = plt.subplots(len(methods), 1, figsize=(10, 3*len(methods)), sharex=True)
for ax, (name, data) in zip(axes, methods):
    im = ax.imshow(np.log10(np.where(data > 0, data, np.nan)).T,
                   origin="lower", aspect="auto",
                   extent=[snaps[0], snaps[-1], r_mid[0], r_mid[-1]],
                   cmap="viridis")
    ax.set_ylabel("Radius (kpc)")
    ax.set_title(name)
    fig.colorbar(im, ax=ax, label='log10(density)')
axes[-1].set_xlabel("Snapshot number")
plt.tight_layout()
plt.savefig("density_methods_evolution_heatmaps.png", dpi=300)
plt.show()
