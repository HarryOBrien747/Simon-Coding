import argparse
import glob
import os
import struct
import numpy as np
import matplotlib.pyplot as plt

pc = 3.085677581e16
Myr = 1.0e6 * 365.25 * 86400.0

def newest(glob_pattern: str):
    files = glob.glob(glob_pattern)
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

def snap_reader_bin(path):
    """
    Record format repeated:
      int32 N
      float64 t
      float32 x[N], y[N], z[N]    (in pc units)
    """
    with open(path, "rb") as f:
        while True:
            n_bytes = f.read(4)
            if not n_bytes:
                return
            N = struct.unpack("<i", n_bytes)[0]
            t = struct.unpack("<d", f.read(8))[0]
            raw = f.read(3 * N * 4)
            if len(raw) != 3 * N * 4:
                return
            xyz = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
            x = xyz[0:N]
            y = xyz[N:2*N]
            z = xyz[2*N:3*N]
            yield N, t, x, y, z

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def percentile_radii(r, ps=(10, 50, 90, 99)):
    return [np.percentile(r, p) for p in ps]

def density_profile(r, nbins=40):
    # log-spaced bins in r (avoid 0)
    r = np.maximum(r, 1e-12)
    rmin = np.percentile(r, 1)
    rmax = np.percentile(r, 99.9)
    edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    counts, _ = np.histogram(r, bins=edges)
    # shell volumes in pc^3 (since r is in pc here)
    vol = (4.0/3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    nden = counts / np.maximum(vol, 1e-30)
    centers = np.sqrt(edges[1:] * edges[:-1])  # geometric mean
    return centers, nden

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", default=None, help="energy_*.txt (default: newest)")
    ap.add_argument("--snaps", default=None, help="snaps_*.bin (default: newest)")
    ap.add_argument("--outdir", default="plots", help="output folder")
    ap.add_argument("--ktraj", type=int, default=10, help="number of trajectories")
    ap.add_argument("--esc_r_pc", type=float, default=10.0, help="escaper radius threshold [pc] (position-only proxy)")
    ap.add_argument("--density_bins", type=int, default=50, help="radial density profile bins")
    ap.add_argument("--seed", type=int, default=1, help="random seed for trajectory selection")
    args = ap.parse_args()

    energy_file = args.energy or newest("energy_*.txt")
    snaps_file  = args.snaps  or newest("snaps_*.bin")

    if energy_file is None:
        raise FileNotFoundError("Could not find energy_*.txt in this folder.")
    if snaps_file is None:
        raise FileNotFoundError("Could not find snaps_*.bin in this folder.")

    ensure_dir(args.outdir)

    # --- energy plots ---
    tE, E, dE = read_energy(energy_file)
    tE_myr = tE / Myr

    plt.figure()
    plt.plot(tE_myr, dE)
    plt.xlabel("Time [Myr]")
    plt.ylabel("dE / E0")
    plt.title("Energy Error")
    plt.grid(True)
    plt.savefig(os.path.join(args.outdir, "energy_error.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(tE_myr, np.abs(dE) + 1e-30)
    plt.yscale("log")
    plt.xlabel("Time [Myr]")
    plt.ylabel("|dE / E0|")
    plt.title("Absolute Energy Error (log scale)")
    plt.grid(True)
    plt.savefig(os.path.join(args.outdir, "energy_error_abs_log.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(tE_myr, E)
    plt.xlabel("Time [Myr]")
    plt.ylabel("Total Energy [J]")
    plt.title("Total Energy")
    plt.grid(True)
    plt.savefig(os.path.join(args.outdir, "total_energy.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- stream snapshots, compute time series ---
    rng = np.random.default_rng(args.seed)

    times = []
    r_mean = []
    r_rms = []
    r_med = []
    r10 = []
    r50 = []
    r90 = []
    r99 = []
    esc_count = []

    # For trajectories
    traj_ids = None
    traj_x = None
    traj_y = None

    # For hist/density start/mid/end
    stored_r = {}   # index -> r array
    stored_xyz = {} # final snapshot for density maps

    snaps = list(snap_reader_bin(snaps_file))
    if len(snaps) == 0:
        raise RuntimeError("No snapshots found in snaps file (file empty or format mismatch).")

    # sanity
    N0 = snaps[0][0]
    nsnap = len(snaps)
    mid_idx = nsnap // 2
    end_idx = nsnap - 1

    # choose traj ids from first snapshot
    if args.ktraj > 0:
        k = min(args.ktraj, N0)
        traj_ids = rng.choice(N0, size=k, replace=False)
        traj_x = {i: [] for i in traj_ids}
        traj_y = {i: [] for i in traj_ids}

    for s, (N, t, x, y, z) in enumerate(snaps):
        if N != N0:
            raise RuntimeError(f"Snapshot {s} has N={N}, expected N={N0}.")
        r = np.sqrt(x*x + y*y + z*z)  # in pc

        times.append(t)
        r_mean.append(np.mean(r))
        r_rms.append(np.sqrt(np.mean(r*r)))
        r_med.append(np.median(r))
        p10, p50, p90, p99 = percentile_radii(r, ps=(10, 50, 90, 99))
        r10.append(p10); r50.append(p50); r90.append(p90); r99.append(p99)

        esc_count.append(int(np.sum(r > args.esc_r_pc)))

        if traj_ids is not None:
            for i in traj_ids:
                traj_x[i].append(x[i])
                traj_y[i].append(y[i])

        if s in (0, mid_idx, end_idx):
            stored_r[s] = r.copy()

        if s == end_idx:
            stored_xyz["final"] = (x.copy(), y.copy(), z.copy())

    times = np.array(times)
    t_myr = times / Myr
    r_mean = np.array(r_mean)
    r_rms  = np.array(r_rms)
    r_med  = np.array(r_med)
    r10 = np.array(r10); r50 = np.array(r50); r90 = np.array(r90); r99 = np.array(r99)
    esc_count = np.array(esc_count)

    # --- radius summaries ---
    plt.figure()
    plt.plot(t_myr, r_mean, label="mean r")
    plt.plot(t_myr, r_rms,  label="RMS r")
    plt.plot(t_myr, r_med,  label="median r")
    plt.xlabel("Time [Myr]")
    plt.ylabel("Radius [pc]")
    plt.title("Mean / RMS / Median Radius")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.outdir, "mean_r_and_rms.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(t_myr, r10, label="r10")
    plt.plot(t_myr, r50, label="r50")
    plt.plot(t_myr, r90, label="r90")
    plt.plot(t_myr, r99, label="r99")
    plt.xlabel("Time [Myr]")
    plt.ylabel("Radius [pc]")
    plt.title("Lagrange Radii (by number)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.outdir, "lagrange_radii_number.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(t_myr, r50 / r50[0])
    plt.xlabel("Time [Myr]")
    plt.ylabel("r50(t) / r50(0)")
    plt.title("Expansion Factor (half-number radius)")
    plt.grid(True)
    plt.savefig(os.path.join(args.outdir, "expansion_factor_r50.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- escaper proxy ---
    plt.figure()
    plt.plot(t_myr, esc_count)
    plt.xlabel("Time [Myr]")
    plt.ylabel(f"N(r > {args.esc_r_pc} pc)")
    plt.title("Escaper Proxy (position-only)")
    plt.grid(True)
    plt.savefig(os.path.join(args.outdir, "escapers_position_proxy.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- radial distributions start/mid/end ---
    plt.figure()
    for idx, label in [(0, "start"), (mid_idx, "mid"), (end_idx, "end")]:
        if idx in stored_r:
            plt.hist(stored_r[idx], bins=60, histtype="step", density=True, label=label)
    plt.xlabel("r [pc]")
    plt.ylabel("Probability density")
    plt.title("Radial Distribution (start / mid / end)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.outdir, "radial_hist_start_mid_end.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- radial number density profiles start/mid/end (log-log) ---
    plt.figure()
    for idx, label in [(0, "start"), (mid_idx, "mid"), (end_idx, "end")]:
        if idx in stored_r:
            rr, nden = density_profile(stored_r[idx], nbins=args.density_bins)
            plt.plot(rr, nden, label=label)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("r [pc]")
    plt.ylabel("number density [1/pc^3]")
    plt.title("Radial Number Density Profile (start / mid / end)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.outdir, "radial_number_density_profiles.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- sample XY trajectories ---
    if traj_ids is not None:
        plt.figure()
        for i in traj_ids:
            plt.plot(traj_x[i], traj_y[i], linewidth=1)
        plt.xlabel("x [pc]")
        plt.ylabel("y [pc]")
        plt.title(f"Sample XY Trajectories (k={len(traj_ids)})")
        plt.grid(True)
        plt.savefig(os.path.join(args.outdir, "sample_xy_trajectories.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # --- final snapshot density maps ---
    if "final" in stored_xyz:
        x, y, z = stored_xyz["final"]

        # auto range: show out to r99
        r = np.sqrt(x*x + y*y + z*z)
        R = max(5.0, float(np.percentile(r, 99)))
        bins = 400

        H, xe, ye = np.histogram2d(x, y, bins=bins, range=[[-R, R], [-R, R]])
        plt.figure()
        plt.imshow(H.T, origin="lower",
                   extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   aspect="equal")
        plt.colorbar(label="counts per bin")
        plt.xlabel("x [pc]")
        plt.ylabel("y [pc]")
        plt.title("XY Surface Density (final snapshot)")
        plt.savefig(os.path.join(args.outdir, "xy_density_final.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # zoomed-in core view (use r50 scale)
        Rz = max(2.0, float(np.percentile(r, 90)))
        H, xe, ye = np.histogram2d(x, y, bins=bins, range=[[-Rz, Rz], [-Rz, Rz]])
        plt.figure()
        plt.imshow(H.T, origin="lower",
                   extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   aspect="equal")
        plt.colorbar(label="counts per bin")
        plt.xlabel("x [pc]")
        plt.ylabel("y [pc]")
        plt.title("XY Surface Density (final, zoom)")
        plt.savefig(os.path.join(args.outdir, "xy_density_final_zoom.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # XZ map too
        H, xe, ze = np.histogram2d(x, z, bins=bins, range=[[-R, R], [-R, R]])
        plt.figure()
        plt.imshow(H.T, origin="lower",
                   extent=[xe[0], xe[-1], ze[0], ze[-1]],
                   aspect="equal")
        plt.colorbar(label="counts per bin")
        plt.xlabel("x [pc]")
        plt.ylabel("z [pc]")
        plt.title("XZ Surface Density (final snapshot)")
        plt.savefig(os.path.join(args.outdir, "xz_density_final.png"), dpi=200, bbox_inches="tight")
        plt.close()

    print("Done. Saved plots to:", args.outdir)
    print("Energy file:", energy_file)
    print("Snaps file :", snaps_file)
    print(f"N = {N0}, snapshots = {nsnap}, t_end = {t_myr[-1]:.6f} Myr")

if __name__ == "__main__":
    main()
