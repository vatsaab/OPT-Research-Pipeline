import os
import tarfile
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from ReadFile import Read
from CenterOfMass2 import CenterOfMass

start_snap = 0
end_snap = 800
ptype = 1
r_bins = np.logspace(-1, 2.6, 20)  # kpc
snapshots = np.arange(start_snap, end_snap + 1)

tmpdir = tempfile.mkdtemp(prefix="snapshots_")
for fn in os.listdir("."):
    if fn.endswith(".tar"):
        with tarfile.open(fn, "r") as tar:
            for member in tar.getmembers():
                if member.isfile() and ("MW_" in member.name or "M31_" in member.name):
                    tar.extract(member, path=tmpdir)

G_kpc_kms2_Msun = 4.30091e-6  # (kpc (km/s)^2) / Msun
mass_unit = 1e10  # snapshot masses are in 1e10 Msun

def compute_profiles_for_snapshot(mw_path, m31_path):
    MW = CenterOfMass(mw_path, ptype)
    M31 = CenterOfMass(m31_path, ptype)

    x = np.concatenate((MW.x, M31.x))
    y = np.concatenate((MW.y, M31.y))
    z = np.concatenate((MW.z, M31.z))
    vx = np.concatenate((MW.vx, M31.vx))
    vy = np.concatenate((MW.vy, M31.vy))
    vz = np.concatenate((MW.vz, M31.vz))
    m = np.concatenate((MW.m, M31.m)) * mass_unit  # convert to Msun

    pos = np.vstack((x, y, z)).T
    vel = np.vstack((vx, vy, vz)).T

    xcom, ycom, zcom = MW.COMdefine(x, y, z, np.concatenate((MW.m, M31.m)))
    vcom, = (None,)  # placeholder for shape
    try:
        vCOM = MW.COM_V(xcom * MW.x.unit, ycom * MW.y.unit, zcom * MW.z.unit)[0]
    except Exception:
        # fallback: mass-weighted average velocity of inner 15 kpc
        r = np.linalg.norm(pos - np.array([xcom, ycom, zcom]), axis=1)
        mask = r < 15.0
        vCOM = np.array([np.sum(vel[mask, i] * m[mask]) / np.sum(m[mask]) for i in range(3)])

    vel_rel = vel - vCOM  # km/s

    r_vec = pos - np.array([xcom, ycom, zcom])
    r_mag = np.linalg.norm(r_vec, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_hat = (r_vec.T / r_mag).T
    v_radial = np.einsum("ij,ij->i", vel_rel, r_hat)
    v_tang2 = np.sum(vel_rel**2, axis=1) - v_radial**2
    v_tang2[v_tang2 < 0] = 0.0
    v_tang = np.sqrt(v_tang2)

    # specific angular momentum vector per particle (kpc * km/s)
    j_vec = np.cross(r_vec, vel_rel)  # units: (kpc * km/s)
    j_mag = np.linalg.norm(j_vec, axis=1)

    # cumulative mass for enclosed M(<r)
    sort_idx = np.argsort(r_mag)
    r_sorted = r_mag[sort_idx]
    m_sorted = m[sort_idx]
    M_enc = np.cumsum(m_sorted)

    # radial-bin profiles
    nb = len(r_bins) - 1
    sigma_r = np.full(nb, np.nan)
    sigma_t = np.full(nb, np.nan)
    v_rot = np.full(nb, np.nan)
    j_spec = np.full(nb, np.nan)
    M_enclosed_bin = np.full(nb, np.nan)
    v_esc = np.full(nb, np.nan)
    beta = np.full(nb, np.nan)

    inds = np.digitize(r_mag, r_bins) - 1
    for b in range(nb):
        mask = inds == b
        if np.sum(mask) < 10:
            continue
        w = m[mask]
        vr = v_radial[mask]
        vt = v_tang[mask]
        # mass-weighted velocity dispersions
        vr_mean = np.sum(w * vr) / np.sum(w)
        vt_mean = np.sum(w * vt) / np.sum(w)
        sigma_r[b] = np.sqrt(np.sum(w * (vr - vr_mean)**2) / np.sum(w))
        sigma_t[b] = np.sqrt(np.sum(w * (vt - vt_mean)**2) / np.sum(w))
        # mean rotation about z: v_phi ~ <(-x v_y + y v_x)/R>
        R = np.sqrt(r_vec[mask,0]**2 + r_vec[mask,1]**2)
        nonzero = R > 0
        if np.any(nonzero):
            vphi = np.mean(( - r_vec[mask,0][nonzero]*vel_rel[mask,1][nonzero] +
                             r_vec[mask,1][nonzero]*vel_rel[mask,0][nonzero]) / R[nonzero])
            v_rot[b] = vphi
        j_spec[b] = np.sum(w * j_mag[mask]) / np.sum(w)
        # enclosed mass at outer edge of bin
        r_outer = r_bins[b+1]
        M_encl = np.sum(m[r_mag <= r_outer])
        M_enclosed_bin[b] = M_encl
        if r_outer > 0 and M_encl > 0:
            v_esc[b] = np.sqrt(2 * G_kpc_kms2_Msun * (M_encl) / (r_outer))
        # anisotropy
        if sigma_r[b] > 0:
            beta[b] = 1.0 - (sigma_t[b]**2) / (2.0 * sigma_r[b]**2)

    # global quantities
    total_j = np.sum(m * j_mag) / np.sum(m)
    mean_vrad = np.sum(m * v_radial) / np.sum(m)
    mean_vt = np.sum(m * v_tang) / np.sum(m)
    sigma_r_global = np.sqrt(np.sum(m * (v_radial - mean_vrad)**2) / np.sum(m))
    sigma_t_global = np.sqrt(np.sum(m * (v_tang - mean_vt)**2) / np.sum(m))

    results = {
        "r_mid": 0.5*(r_bins[:-1]+r_bins[1:]),
        "sigma_r": sigma_r,
        "sigma_t": sigma_t,
        "v_rot": v_rot,
        "j_spec": j_spec,
        "M_enclosed_bin": M_enclosed_bin,
        "v_esc": v_esc,
        "beta": beta,
        "total_j": total_j,
        "sigma_r_global": sigma_r_global,
        "sigma_t_global": sigma_t_global,
        "mean_vrad": mean_vrad,
        "mean_vt": mean_vt,
        "time": MW.time.value if hasattr(MW, "time") else None
    }
    return results

nb = len(r_bins)-1
ns = len(snapshots)
sigma_r_ts = np.full((ns, nb), np.nan)
sigma_t_ts = np.full((ns, nb), np.nan)
vrot_ts = np.full((ns, nb), np.nan)
j_ts = np.full((ns, nb), np.nan)
beta_ts = np.full((ns, nb), np.nan)
time_list = np.full(ns, np.nan)

for i, s in enumerate(snapshots):
    mw_file = os.path.join(tmpdir, f"MW_{s:03d}.txt")
    m31_file = os.path.join(tmpdir, f"M31_{s:03d}.txt")
    if not (os.path.exists(mw_file) and os.path.exists(m31_file)):
        continue
    out = compute_profiles_for_snapshot(mw_file, m31_file)
    time_list[i] = out["time"] if out["time"] is not None else s
    sigma_r_ts[i, :] = out["sigma_r"]
    sigma_t_ts[i, :] = out["sigma_t"]
    vrot_ts[i, :] = out["v_rot"]
    j_ts[i, :] = out["j_spec"]
    beta_ts[i, :] = out["beta"]
    print(f"Processed snap {s}")

# plots: global evolution (averaged over inner 30 kpc) and heatmaps
inner_mask = (0.5*(r_bins[:-1]+r_bins[1:]) <= 30.0)
sigma_r_inner = np.nanmean(sigma_r_ts[:, inner_mask], axis=1)
sigma_t_inner = np.nanmean(sigma_t_ts[:, inner_mask], axis=1)
vrot_inner = np.nanmean(vrot_ts[:, inner_mask], axis=1)
j_inner = np.nanmean(j_ts[:, inner_mask], axis=1)

plt.figure(figsize=(8,5))
plt.plot(time_list, sigma_r_inner, label='sigma_r (inner)')
plt.plot(time_list, sigma_t_inner, label='sigma_t (inner)')
plt.plot(time_list, vrot_inner, label='v_rot (inner)')
plt.xlabel('Snapshot / Time')
plt.ylabel('km/s')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("kinematics_inner_evolution.png", dpi=300)

fig, ax = plt.subplots(2,1, figsize=(10,8), sharex=True)
im1 = ax[0].imshow(np.where(sigma_r_ts>0,np.log10(sigma_r_ts), np.nan).T, aspect='auto',
                   origin='lower', extent=[snapshots[0], snapshots[-1], r_bins[0], r_bins[-1]])
ax[0].set_ylabel('Radius (kpc)')
ax[0].set_title('log10(sigma_r) evolution')
fig.colorbar(im1, ax=ax[0])
im2 = ax[1].imshow(np.where(beta_ts!=np.nan,np.nan_to_num(beta_ts, nan=np.nan)).T, aspect='auto',
                   origin='lower', extent=[snapshots[0], snapshots[-1], r_bins[0], r_bins[-1]], cmap='bwr', vmin=-1, vmax=1)
ax[1].set_ylabel('Radius (kpc)')
ax[1].set_xlabel('Snapshot')
ax[1].set_title('anisotropy beta evolution')
fig.colorbar(im2, ax=ax[1])
plt.tight_layout()
plt.savefig("kinematics_heatmaps.png", dpi=300)

shutil.rmtree(tmpdir)
