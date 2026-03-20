# visualize_nbody.py — reads plain-text "state"/"energy" and makes PNGs
import matplotlib
matplotlib.use("Agg")  # save-only backend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- paths ----------
BASE = Path.cwd()  # run from the folder with 'state' and 'energy'

def find(stem: str) -> Path:
    for p in (BASE/stem, BASE/f"{stem}.txt", BASE/f"{stem}.csv"):
        if p.exists():
            return p
    raise FileNotFoundError(f"Couldn't find {stem}(.txt/.csv) in {BASE}")

state_path  = find("state")
energy_path = find("energy")

# ---------- read text (space-delimited) ----------
state  = pd.read_csv(state_path,  sep=r"\s+", engine="python")
energy = pd.read_csv(energy_path, sep=r"\s+", engine="python")

# ---------- constants (match your Fortran) ----------
AU   = 1.495978707e11
day  = 86400.0
year = 365.25 * day

# Auto-detect N from columns: t + 6*N
cols = list(state.columns)
if cols[0] != "t[s]":
    raise RuntimeError("First column must be 't[s]' (from your Fortran header).")
N = (len(cols) - 1) // 6
if 1 + 6*N != len(cols):
    raise RuntimeError("Column count doesn't match 1 + 6*N. Check the state header.")

# ---------- naming & masses ----------
default_names  = ["Sun","Mercury","Venus","Earth","Moon","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto"]
default_masses = np.array([
    1.98847e30, 3.3011e23, 4.8675e24, 5.9722e24, 7.342e22, 6.4171e23,
    1.89813e27, 5.6834e26, 8.6810e25, 1.02413e26, 1.303e22
], dtype=float)

if N <= len(default_names):
    names  = default_names[:N]
    masses = default_masses[:N]
else:
    names  = default_names + [f"Body{i}" for i in range(len(default_names)+1, N+1)]
    # extend masses with small placeholders for extra bodies
    masses = np.concatenate([default_masses, np.full(N-len(default_masses), 1.0)])

print(f"Detected N={N} bodies:")
print("  " + ", ".join(names))

# convenience
def col(sym, i):
    return state[f"{sym}{i}"].to_numpy()

t_years = state["t[s]"].to_numpy()/year

# ---------- 1) Orbits in the xy-plane ----------
plt.figure(figsize=(9,7))
for k, label in enumerate(names, start=1):
    plt.plot(col("x",k)/AU, col("y",k)/AU, label=label, linewidth=0.9)
plt.axis("equal")
plt.xlabel("x [AU]"); plt.ylabel("y [AU]")
plt.title("Orbits (barycentric COM frame)")
# compact legend for many bodies
ncol = 2 if N <= 10 else 3
plt.legend(fontsize=8, ncol=ncol, frameon=False)
plt.tight_layout()
plt.savefig(BASE/"orbits_xy.png", dpi=200); plt.close()

# ---------- 2) Heliocentric distances (skip Moon; plot separately) ----------
plt.figure(figsize=(9,7))
sun_idx = 1  # your Fortran writes Sun as body 1
for k, label in enumerate(names, start=1):
    if k == sun_idx:           # skip Sun
        continue
    if label.lower() == "moon":  # handle Moon separately below
        continue
    dx = col("x",k) - col("x",sun_idx)
    dy = col("y",k) - col("y",sun_idx)
    dz = col("z",k) - col("z",sun_idx)
    r  = np.sqrt(dx*dx + dy*dy + dz*dz)/AU
    plt.plot(t_years, r, label=label, linewidth=1.0)
plt.xlabel("Time [years]"); plt.ylabel("Distance to Sun [AU]")
plt.title("Heliocentric distances")
plt.legend(fontsize=8, ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(BASE/"heliocentric_distances.png", dpi=200); plt.close()

# ---------- 3) Earth–Moon distance (if Moon present) ----------
if "Moon" in names:
    iE = names.index("Earth")+1
    iM = names.index("Moon")+1
    dx = col("x",iM) - col("x",iE)
    dy = col("y",iM) - col("y",iE)
    dz = col("z",iM) - col("z",iE)
    r_em = np.sqrt(dx*dx + dy*dy + dz*dz)
    plt.figure(figsize=(9,6))
    plt.plot(t_years, r_em/1.0e3)  # km
    plt.xlabel("Time [years]"); plt.ylabel("Earth–Moon distance [km]")
    plt.title("Earth–Moon distance")
    plt.tight_layout()
    plt.savefig(BASE/"earth_moon_distance.png", dpi=200); plt.close()

# ---------- 4) Energy error ----------
plt.figure(figsize=(9,7))
if "dE/E0" in energy.columns:
    plt.plot(energy["t[s]"]/year, np.abs(energy["dE/E0"]))
else:
    # fallback: compute relative to first energy
    e = energy["E[J]"].to_numpy()
    plt.plot(energy["t[s]"]/year, np.abs((e - e[0])/e[0]))
plt.yscale("log")
plt.xlabel("Time [years]"); plt.ylabel("|ΔE/E0|")
plt.title("Relative Energy Error")
plt.grid(True, which="both", linestyle=":")
plt.tight_layout()
plt.savefig(BASE/"energy_error.png", dpi=200); plt.close()

# ---------- 5) COM drift ----------
Mtot = masses.sum()
xcom = np.zeros(len(state)); ycom = np.zeros(len(state)); zcom = np.zeros(len(state))
for k in range(1, N+1):
    xcom += masses[k-1]*col("x",k)
    ycom += masses[k-1]*col("y",k)
    zcom += masses[k-1]*col("z",k)
xcom /= Mtot; ycom /= Mtot; zcom /= Mtot
com_mag = np.sqrt(xcom*xcom + ycom*ycom + zcom*zcom)/AU

plt.figure(figsize=(9,7))
plt.plot(t_years, com_mag)
plt.yscale("log")
plt.xlabel("Time [years]"); plt.ylabel("|r_com| [AU]")
plt.title("COM Drift")
plt.grid(True, which="both", linestyle=":")
plt.tight_layout()
plt.savefig(BASE/"com_drift.png", dpi=200); plt.close()

# ---------- console summary ----------
de_col = "dE/E0" if "dE/E0" in energy.columns else None
if de_col:
    de = np.abs(energy[de_col].to_numpy())
    de_max = float(np.nanmax(de))
else:
    e = energy["E[J]"].to_numpy()
    de = np.abs((e - e[0])/e[0]); de_max = float(np.nanmax(de))

print(f"Read: {state_path.name} and {energy_path.name}")
print(f"Bodies (N={N}): " + ", ".join(names))
print(f"Saved: orbits_xy.png, heliocentric_distances.png, energy_error.png, com_drift.png"
      + (", earth_moon_distance.png" if "Moon" in names else ""))
print(f"Max |ΔE/E0| ≈ {de_max:.3e}")
print(f"Max COM offset ≈ {np.nanmax(com_mag):.3e} AU")
