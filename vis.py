# vis.py — read plain-text state/energy from Hermite Fortran run and save PNGs
# Outputs:
#   COM_drift.png
#   energy_err.png
#   helilocentric_distances.png
#   orbits_xy.png
#   orbits_xy_innerplanetsincludingjuipter.png
#   orbits_xy_innerplanets_to_mars.png

import matplotlib
matplotlib.use("Agg")  # save-only backend (no GUI)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

BASE = Path.cwd()

# ---------- locate newest state_*/energy_* files ----------
'''
def find_latest(prefix: str) -> Path:
    # Accept timestamped names from Fortran, with or without .txt/.csv
    pats = [f"{prefix}*.txt", f"{prefix}*.csv", f"{prefix}*"]
    cands = [Path(p) for pat in pats for p in glob(str(BASE / pat))]
    if not cands:
        # Fallback to plain 'state'/'energy' names if user saved fixed names
        p = BASE / prefix.rstrip("_")
        if p.exists():
            return p
        ptxt = p.with_suffix(".txt")
        if ptxt.exists():
            return ptxt
        raise FileNotFoundError(f"No files matching {prefix}* in {BASE}")
    return max(cands, key=lambda p: p.stat().st_mtime)
'''

def find_latest(prefix: str) -> Path:
    # Accept timestamped names from Fortran, with .txt/.csv
    pats = [f"{prefix}*.txt", f"{prefix}*.csv"]
    cands = [Path(p) for pat in pats for p in glob(str(BASE / pat))]
    if not cands:
        # Fallback to plain 'state'/'energy' if user saved fixed names
        p = BASE / prefix.rstrip("_")
        if p.exists():
            return p
        ptxt = p.with_suffix(".txt")
        if ptxt.exists():
            return ptxt
        raise FileNotFoundError(f"No files matching {prefix}*.txt or {prefix}*.csv in {BASE}")
    return max(cands, key=lambda p: p.stat().st_mtime)


state_path  = find_latest("state_")
energy_path = find_latest("energy_")

# ---------- read files ----------
state  = pd.read_csv(state_path,  sep=r"\s+", engine="python")
energy = pd.read_csv(energy_path, sep=r"\s+", engine="python")

# ---------- constants (must match Fortran, up to scaling) ----------
AU   = 1.495978707e11
day  = 86400.0
year = 365.25 * day

# ---------- infer number of bodies ----------
cols = list(state.columns)
if cols[0] != "t[s]":
    raise RuntimeError("First column must be 't[s]'. Check the state header.")
N = (len(cols) - 1) // 6
if 1 + 6 * N != len(cols):
    raise RuntimeError(f"Column count mismatch: got {len(cols)} but expected 1 + 6*N.")

# ---------- names & masses ----------
# Order must match your Fortran: Sun, Mercury, Venus, Earth, Moon, Mars, ...
default_names  = ["Sun","Mercury","Venus","Earth","Moon","Mars",
                  "Jupiter","Saturn","Uranus","Neptune","Pluto"]

# Match your updated Fortran: Mercury and Moon are massless (0.0)
default_masses = np.array([
    1.98847e30,   # Sun
    0.0,          # Mercury (massless in this experiment)
    4.8675e24,    # Venus
    5.9722e24,    # Earth
    0.0,          # Moon (massless in this experiment)
    6.4171e23,    # Mars
    1.89813e27,   # Jupiter
    5.6834e26,    # Saturn
    8.6810e25,    # Uranus
    1.02413e26,   # Neptune
    1.303e22      # Pluto
], dtype=float)

if N <= len(default_names):
    names  = default_names[:N]
    masses = default_masses[:N]
else:
    names  = default_names + [f"Body{i}" for i in range(len(default_names)+1, N+1)]
    pad    = np.full(N - len(default_masses), 1.0, dtype=float)
    masses = np.concatenate([default_masses, pad])

# Bodies we never want to plot (massless test particles)
SKIP_BODIES = {"Mercury", "Moon"}

# ---------- convenience column accessor ----------
def col(sym: str, i: int) -> np.ndarray:
    return state[f"{sym}{i}"].to_numpy()

t_years = state["t[s]"].to_numpy() / year

# ================================================================
#  PLOTS
# ================================================================

# ---------- ORBITS: all bodies (x–y) ----------
plt.figure(figsize=(9, 7))
for k, label in enumerate(names, start=1):
    if label in SKIP_BODIES:
        continue
    plt.plot(col("x", k)/AU, col("y", k)/AU, label=label, linewidth=0.9)
plt.axis("equal")
plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.title("Orbits (barycentric COM frame)")
plt.legend(fontsize=8, ncol=2 if N <= 10 else 3, frameon=False)
plt.tight_layout()
plt.savefig(BASE / "orbits_xy.png", dpi=200)
plt.close()

# ---------- ORBITS: inner planets up to Jupiter ----------
plt.figure(figsize=(8, 6))
for k, label in enumerate(names, start=1):
    if label in SKIP_BODIES:
        continue
    # keep only Venus, Earth, Mars, Jupiter
    if label not in {"Venus", "Earth", "Mars", "Jupiter"}:
        continue
    plt.plot(col("x", k)/AU, col("y", k)/AU, label=label, linewidth=1.0)
plt.axis("equal")
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.title("Inner orbits (including Jupiter)")
plt.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(BASE / "orbits_xy_innerplanetsincludingjuipter.png", dpi=200)
plt.close()

# ---------- ORBITS: inner planets up to Mars (no Jupiter) ----------
plt.figure(figsize=(8, 6))
for k, label in enumerate(names, start=1):
    if label in SKIP_BODIES:
        continue
    # keep only Venus, Earth, Mars
    if label not in {"Venus", "Earth", "Mars"}:
        continue
    plt.plot(col("x", k)/AU, col("y", k)/AU, label=label, linewidth=1.0)
plt.axis("equal")
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.title("Inner orbits (up to Mars)")
plt.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(BASE / "orbits_xy_innerplanets_to_mars.png", dpi=200)
plt.close()

# ---------- HELIOCENTRIC distances (skip Mercury & Moon) ----------
'''
sun_idx = 1  # Sun is body 1 in your output

def heliodist(i: int) -> np.ndarray:
    dx = col("x", i) - col("x", sun_idx)
    dy = col("y", i) - col("y", sun_idx)
    dz = col("z", i) - col("z", sun_idx)
    return np.sqrt(dx*dx + dy*dy + dz*dz) / AU

# linear-scale version (old one)
plt.figure(figsize=(9, 7))
for k, label in enumerate(names, start=1):
    if k == sun_idx:
        continue
    if label in SKIP_BODIES:
        continue
    if label.lower() == "moon":
        continue
    plt.plot(t_years, heliodist(k), label=label, linewidth=1.0)
plt.xlabel("Time [years]")
plt.ylabel("Distance to Sun [AU]")
plt.title("Heliocentric distances")
plt.legend(fontsize=8, ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(BASE / "helilocentric_distances.png", dpi=200)
plt.close()
'''
# log-scale version
'''
plt.figure(figsize=(9, 7))
for k, label in enumerate(names, start=1):
    if k == sun_idx:
        continue
    if label in SKIP_BODIES:
        continue
    if label.lower() == "moon":
        continue
    plt.plot(t_years, heliodist(k), label=label, linewidth=1.0)
plt.xlabel("Time [years]")
plt.ylabel("Distance to Sun [AU]")
plt.title("Heliocentric distances (log scale)")
plt.yscale("log")
plt.grid(True, which="both", linestyle=":")
plt.legend(fontsize=8, ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(BASE / "helilocentric_distances_log.png", dpi=200)
plt.close()

'''
'''
plt.figure(figsize=(9, 7))
for k, label in enumerate(names, start=1):
    if k == sun_idx:
        continue
    if label in SKIP_BODIES:
        continue
    if label.lower() == "moon":
        continue
    r = heliodist(k)
    frac = (r - r.mean()) / r.mean()   # fractional deviation
    plt.plot(t_years, frac * 100.0, label=label, linewidth=1.0)

plt.xlabel("Time [years]")
plt.ylabel("Δr / ⟨r⟩ [%]")
plt.title("Fractional heliocentric distance variation")
plt.grid(True, linestyle=":")
plt.legend(fontsize=8, ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(BASE / "helilocentric_distances_frac.png", dpi=200)
plt.close()
'''

# ---------- HELIOCENTRIC distances (skip Mercury & Moon) ----------
sun_idx = 1  # Sun is body 1 in your output

def heliodist(i: int) -> np.ndarray:
    dx = col("x", i) - col("x", sun_idx)
    dy = col("y", i) - col("y", sun_idx)
    dz = col("z", i) - col("z", sun_idx)
    return np.sqrt(dx*dx + dy*dy + dz*dz) / AU

# --- 1) linear-scale distances (all planets except massless ones) ---
plt.figure(figsize=(9, 7))
for k, label in enumerate(names, start=1):
    if k == sun_idx:
        continue
    if label in SKIP_BODIES:
        continue
    if label.lower() == "moon":
        continue
    plt.plot(t_years, heliodist(k), label=label, linewidth=1.0)

plt.xlabel("Time [years]")
plt.ylabel("Distance to Sun [AU]")
plt.title("Heliocentric distances")
plt.legend(fontsize=8, ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(BASE / "helilocentric_distances.png", dpi=200)
plt.close()

# --- 2) log-scale distances (all planets) ---
plt.figure(figsize=(9, 7))
for k, label in enumerate(names, start=1):
    if k == sun_idx:
        continue
    if label in SKIP_BODIES:
        continue
    if label.lower() == "moon":
        continue
    plt.plot(t_years, heliodist(k), label=label, linewidth=1.0)

plt.xlabel("Time [years]")
plt.ylabel("Distance to Sun [AU]")
plt.title("Heliocentric distances (log scale)")
plt.yscale("log")
plt.grid(True, which="both", linestyle=":")
plt.legend(fontsize=8, ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(BASE / "helilocentric_distances_log.png", dpi=200)
plt.close()

# --- 3a) fractional variation: inner (Venus–Earth–Mars–Jupiter) ---
inner_frac_bodies = {"Venus", "Earth", "Mars", "Jupiter"}

plt.figure(figsize=(9, 7))
for k, label in enumerate(names, start=1):
    if label not in inner_frac_bodies:
        continue
    if label in SKIP_BODIES:
        continue
    r = heliodist(k)
    frac = (r - r.mean()) / r.mean()   # fractional deviation from mean
    plt.plot(t_years, frac * 100.0, label=label, linewidth=1.0)

plt.xlabel("Time [years]")
plt.ylabel("Δr / ⟨r⟩ [%]")
plt.title("Fractional heliocentric distance variation (inner planets)")
plt.grid(True, linestyle=":")
plt.legend(fontsize=8, ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(BASE / "helilocentric_distances_frac_inner.png", dpi=200)
plt.close()

# --- 3b) fractional variation: outer (Saturn–Uranus–Neptune–Pluto) ---
outer_frac_bodies = {"Saturn", "Uranus", "Neptune", "Pluto"}

plt.figure(figsize=(9, 7))
for k, label in enumerate(names, start=1):
    if label not in outer_frac_bodies:
        continue
    if label in SKIP_BODIES:
        continue
    r = heliodist(k)
    frac = (r - r.mean()) / r.mean()
    plt.plot(t_years, frac * 100.0, label=label, linewidth=1.0)

plt.xlabel("Time [years]")
plt.ylabel("Δr / ⟨r⟩ [%]")
plt.title("Fractional heliocentric distance variation (outer planets)")
plt.grid(True, linestyle=":")
plt.legend(fontsize=8, ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(BASE / "helilocentric_distances_frac_outer.png", dpi=200)
plt.close()


# ---------- ENERGY error (log scale) ----------
plt.figure(figsize=(9, 7))
if "dE/E0" in energy.columns:
    plt.plot(energy["t[s]"] / year, np.abs(energy["dE/E0"]))
else:
    e = energy["E[J]"].to_numpy()
    plt.plot(energy["t[s]"] / year, np.abs((e - e[0]) / e[0]))
plt.yscale("log")
plt.xlabel("Time [years]")
plt.ylabel("|ΔE/E0|")
plt.title("Relative Energy Error")
plt.grid(True, which="both", linestyle=":")
plt.tight_layout()
plt.savefig(BASE / "energy_err.png", dpi=200)
plt.close()

# ---------- COM drift (barycentric) ----------
Mtot = masses.sum()
xcom = np.zeros(len(state))
ycom = np.zeros(len(state))
zcom = np.zeros(len(state))
for k in range(1, N + 1):
    # if our names array is shorter than masses (for weird N), guard index
    if k-1 >= len(masses):
        break
    xcom += masses[k-1] * col("x", k)
    ycom += masses[k-1] * col("y", k)
    zcom += masses[k-1] * col("z", k)
xcom /= Mtot
ycom /= Mtot
zcom /= Mtot
com_mag = np.sqrt(xcom*xcom + ycom*ycom + zcom*zcom) / AU

plt.figure(figsize=(9, 7))
plt.plot(t_years, com_mag)
plt.yscale("log")
plt.xlabel("Time [years]")
plt.ylabel("|r_com| [AU]")
plt.title("COM Drift")
plt.grid(True, which="both", linestyle=":")
plt.tight_layout()
plt.savefig(BASE / "COM_drift.png", dpi=200)
plt.close()

# ---------- console summary ----------
if "dE/E0" in energy.columns:
    de = np.abs(energy["dE/E0"].to_numpy())
else:
    e = energy["E[J]"].to_numpy()
    de = np.abs((e - e[0]) / e[0])
de_max = float(np.nanmax(de))

print(f"Read files:\n  {state_path.name}\n  {energy_path.name}")
print("Saved images:\n"
      "  COM_drift.png\n"
      "  energy_err.png\n"
      "  helilocentric_distances.png\n"
      "  helilocentric_distances_log.png\n"
      "  helilocentric_distances_frac_inner.png\n"
      "  helilocentric_distances_frac_outer.png\n"
      "  orbits_xy.png\n"
      "  orbits_xy_innerplanetsincludingjuipter.png\n"
      "  orbits_xy_innerplanets_to_mars.png")
print(f"Detected N={N} bodies: {', '.join(names)}")
print(f"Max |ΔE/E0| ≈ {de_max:.3e}")
print(f"Max COM offset ≈ {np.nanmax(com_mag):.3e} AU")
