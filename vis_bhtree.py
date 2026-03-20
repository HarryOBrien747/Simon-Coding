import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

pc = 3.0856775814913673e16
Myr = 1.0e6 * 365.25 * 86400.0

def newest(pattern: str):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]

def read_energy(energy_file: str):
    data = np.loadtxt(energy_file, skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    t = data[:, 0]
    E = data[:, 1]
    dE = data[:, 2]
    return t, E, dE

def read_masses_optional(mass_file, N_expected=None):
    if mass_file is None:
        return None
    if not os.path.exists(mass_file):
        return None
    m = np.loadtxt(mass_file)
    if m.ndim != 1:
        return None
    if N_expected is not None and m.size != N_expected:
        print(f"[warn] mass file size mismatch: expected {N_expected}, got {m.size} -> ignoring masses")
        return None
    return m

def iter_snaps_bin(snaps_file: str):
    """
    Record format (repeated):
      int32 N
      float64 t
      float32 x[N], y[N], z[N]   in pc
    """
    with open(snaps_file, "rb") as f:
        while True:
            nbytes = f.read(4)
            if not nbytes:
                break
            N = np.frombuffer(nbytes, dtype=np.int32)[0]
            t_bytes = f.read(8)
            if len(t_bytes) < 8:
                break
            t = np.frombuffer(t_bytes, dtype=np.float64)[0]

            # read x,y,z arrays
            x = np.fromfile(f, dtype=np.float32, count=N)
            if x.size != N: break
            y = np.fromfile(f, dtype=np.float32, count=N)
            if y.size != N: break
            z = np.fromfile(f, dtype=np.float32, count=N)
            if z.size != N: break

            yield int(N), float(t), x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)

def quantiles_partition(r, qs=(0.1, 0.5, 0.9)):
    """
    Fast-ish quantiles using np.partition (no full sort).
    Returns radii at given quantiles (by number).
    """
    N = r.size
    out = []
    for q in qs:
        k = int(q * (N - 1))
        rk = np.partition(r, k)[k]
        out.append(rk)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", default=None, help="energy_*.txt (default: newest)")
    ap.add_argument("--snaps", default=None, help="snaps_*.bin (default: newest)")
    ap.add_argument("--masses", default=None, help="optional masses txt (one mass per line, kg)")
    ap.add_argument("--traj_k", type=int, default=10, help="how many trajectories to plot")
    ap.add_argument("--outdir", default="plots", help="folder for figures")
    ap.add_argument("--density_bins", type=int, default=300, help="bins for XY density map")
    args = ap.parse_args()

    energy_file = args.energy or newest("energy_*.txt")
    snaps_file  = args.snaps  or newest("snaps_*.bin")

    if energy_file is None:
        raise FileNotFoundError("No energy_*.txt found in this folder.")
    if snaps_file is None:
        raise FileNotFoundError("No snaps_*.bin found in this folder.")

    os.makedirs(args.outdir, exist_ok=True)

    # --- Energy plots ---
    tE, E, dE = read_energy(energy_file)
    tE_myr = tE / Myr

    plt.figure()
    plt.plot(tE_myr, dE)
    plt.xlabel("Time [Myr]")
    plt.ylabel("dE / E0")
    plt.title("Energy Error")
    plt.grid(True)
    plt.savefig(os.path.join(args.outdir, "energy_error.png"), dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.semilogy(tE_myr, np.abs(dE) + 1e-30)
    plt.xlabel("Time [Myr]")
    plt.ylabel("|dE / E0|")
    plt.title("Absolute Energy Error (log scale)")
    plt.grid(True, which="both")
    plt.savefig(os.path.join(args.outdir, "energy_error_abs_log.png"), dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(tE_myr, E)
    plt.xlabel("Time [Myr]")
    plt.ylabel("Total Energy [J]")
    plt.title("Total Energy")
    plt.grid(True)
    plt.savefig(os.path.join(args.outdir, "total_energy.png"), dpi=220, bbox_inches="tight")
    plt.close()

    # --- Snapshots processing ---
    # We don’t know N until we read first record
    snaps = list(iter_snaps_bin(snaps_file))
    if len(snaps) == 0:
        raise RuntimeError("snaps_*.bin exists but contains no readable snapshots.")

    N0 = snaps[0][0]
    m = read_masses_optional(args.masses, N_expected=N0)

    nsnap = len(snaps)
    mid_idx = nsnap // 2
    end_idx = nsnap - 1

    times = np.zeros(nsnap)
    r10 = np.zeros(nsnap)
    r50 = np.zeros(nsnap)
    r90 = np.zeros(nsnap)
    rmean = np.zeros(nsnap)
    rrms = np.zeros(nsnap)

    # sample trajectories (use fixed IDs, consistent ordering across snapshots)
    traj_k = min(args.traj_k, N0)
    ids = np.arange(traj_k, dtype=int)
    traj_x = {i: [] for i in ids}
    traj_y = {i: [] for i in ids}

    # store radii arrays for start/mid/end hist plots
    hist_r = {}

    # store final positions for XY density map
    final_xy = None

    for s, (N, t, x, y, z) in enumerate(snaps):
        if N != N0:
            raise ValueError(f"Snapshot {s} has N={N} but first snapshot had N={N0}")

        # centre on COM each snapshot (mean if masses missing, mass-weighted if present)
        if m is None:
            x0 = x - np.mean(x)
            y0 = y - np.mean(y)
            z0 = z - np.mean(z)
        else:
            mtot = np.sum(m)
            x0 = x - np.sum(m * x) / mtot
            y0 = y - np.sum(m * y) / mtot
            z0 = z - np.sum(m * z) / mtot

        r = np.sqrt(x0*x0 + y0*y0 + z0*z0)  # in pc

        times[s] = t
        r10[s], r50[s], r90[s] = quantiles_partition(r, qs=(0.1, 0.5, 0.9))
        rmean[s] = np.mean(r)
        rrms[s] = np.sqrt(np.mean(r*r))

        # trajectories (XY)
        for i in ids:
            traj_x[i].append(x0[i])
            traj_y[i].append(y0[i])

        if s in (0, mid_idx, end_idx):
            hist_r[s] = r.copy()

        if s == end_idx:
            final_xy = (x0.copy(), y0.copy())

    t_myr = times / Myr

    # Lagrange radii by number
    plt.figure()
    plt.plot(t_myr, r10, label="r10 (number)")
    plt.plot(t_myr, r50, label="r50 (number)")
    plt.plot(t_myr, r90, label="r90 (number)")
    plt.xlabel("Time [Myr]")
    plt.ylabel("Radius [pc]")
    plt.title("Lagrange Radii (by number)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.outdir, "lagrange_radii_number.png"), dpi=220, bbox_inches="tight")
    plt.close()

    # Mean and RMS radius
    plt.figure()
    plt.plot(t_myr, rmean, label="mean r")
    plt.plot(t_myr, rrms, label="RMS r")
    plt.xlabel("Time [Myr]")
    plt.ylabel("Radius [pc]")
    plt.title("Mean and RMS Radius")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.outdir, "mean_r_and_rms.png"), dpi=220, bbox_inches="tight")
    plt.close()

    # Expansion factor
    plt.figure()
    plt.plot(t_myr, r50 / max(r50[0], 1e-30))
    plt.xlabel("Time [Myr]")
    plt.ylabel("r50(t) / r50(0)")
    plt.title("Expansion Factor (half-number radius)")
    plt.grid(True)
    plt.savefig(os.path.join(args.outdir, "expansion_factor_r50.png"), dpi=220, bbox_inches="tight")
    plt.close()

    # Radial histograms start/mid/end
    plt.figure()
    for idx, label in [(0, "start"), (mid_idx, "mid"), (end_idx, "end")]:
        if idx in hist_r:
            plt.hist(hist_r[idx], bins=60, histtype="step", density=True, label=label)
    plt.xlabel("r [pc]")
    plt.ylabel("Probability density")
    plt.title("Radial Distribution (start / mid / end)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.outdir, "radial_hist_start_mid_end.png"), dpi=220, bbox_inches="tight")
    plt.close()

    # Sample XY trajectories
    plt.figure()
    for i in ids:
        plt.plot(traj_x[i], traj_y[i], linewidth=1)
    plt.xlabel("x [pc]")
    plt.ylabel("y [pc]")
    plt.title(f"Sample XY Trajectories (k={len(ids)})")
    plt.grid(True)
    plt.savefig(os.path.join(args.outdir, "sample_xy_trajectories.png"), dpi=220, bbox_inches="tight")
    plt.close()

    # XY surface density map at final snapshot
    if final_xy is not None:
        x0, y0 = final_xy
        H, xedges, yedges = np.histogram2d(x0, y0, bins=args.density_bins)
        plt.figure()
        plt.imshow(
            H.T,
            origin="lower",
            aspect="equal",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
        )
        plt.xlabel("x [pc]")
        plt.ylabel("y [pc]")
        plt.title("XY Surface Density (final snapshot)")
        plt.colorbar(label="counts per bin")
        plt.savefig(os.path.join(args.outdir, "xy_density_final.png"), dpi=220, bbox_inches="tight")
        plt.close()

    print(f"Done. Saved plots to: {args.outdir}")
    print(f"Used energy: {energy_file}")
    print(f"Used snaps : {snaps_file}")
    if m is not None:
        print("Mass file loaded -> (not currently used for extra plots unless you ask).")
    else:
        print("No masses loaded (fine).")

if __name__ == "__main__":
    main()
