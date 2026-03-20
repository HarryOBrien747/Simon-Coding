import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

Myr = 1.0e6 * 365.25 * 86400.0  # seconds in Myr


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


def read_masses(mass_file: str, N_expected: int):
    if mass_file is None or (not os.path.exists(mass_file)):
        return None
    m = np.loadtxt(mass_file)
    if m.ndim != 1 or m.size != N_expected:
        print(f"[warn] masses file size mismatch: expected {N_expected}, got {m.size}. Ignoring masses.")
        return None
    return m  # Msun


def file_snapshot_count(snaps_file: str, N: int):
    # record size (bytes): int32 + float64 + 3*N*float32
    rec_bytes = 4 + 8 + 3 * N * 4
    size = os.path.getsize(snaps_file)
    if size % rec_bytes != 0:
        print(f"[warn] snaps file size not multiple of record bytes. size={size}, rec={rec_bytes}.")
    return size // rec_bytes


def read_one_snapshot(f):
    nbuf = f.read(4)
    if not nbuf:
        return None
    N = np.frombuffer(nbuf, dtype=np.int32, count=1)[0]
    t = np.fromfile(f, dtype=np.float64, count=1)[0]
    x = np.fromfile(f, dtype=np.float32, count=N)
    y = np.fromfile(f, dtype=np.float32, count=N)
    z = np.fromfile(f, dtype=np.float32, count=N)
    return int(N), float(t), x, y, z


def percentiles_number(r, fracs=(0.1, 0.5, 0.9)):
    # r in pc
    rs = np.sort(r)
    out = []
    n = len(rs)
    for f in fracs:
        j = int(f * (n - 1))
        out.append(rs[j])
    return out


def lagrange_radii_mass(r, m, fracs=(0.1, 0.5, 0.9)):
    # r in pc, m in Msun (any units OK)
    idx = np.argsort(r)
    r_sorted = r[idx]
    m_sorted = m[idx]
    cum = np.cumsum(m_sorted)
    tot = cum[-1]
    out = []
    for f in fracs:
        target = f * tot
        j = np.searchsorted(cum, target)
        j = min(max(j, 0), len(r_sorted) - 1)
        out.append(r_sorted[j])
    return out


def radial_density_profile(r, nbins=40):
    # 3D density profile rho(r) ~ counts / shell volume
    r = np.asarray(r)
    rmax = np.max(r)
    if rmax <= 0:
        return None
    bins = np.logspace(np.log10(max(rmin_positive(r), 1e-6)), np.log10(rmax), nbins + 1)
    counts, edges = np.histogram(r, bins=bins)
    r_in = edges[:-1]
    r_out = edges[1:]
    shell_vol = (4.0 / 3.0) * np.pi * (r_out**3 - r_in**3)
    rho = counts / shell_vol
    r_mid = np.sqrt(r_in * r_out)
    return r_mid, rho


def rmin_positive(r):
    rp = r[r > 0]
    return float(np.min(rp)) if rp.size else 1e-6


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def savefig(path, dpi=250):
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", default=None, help="snaps_*.bin (default: newest)")
    ap.add_argument("--energy", default=None, help="energy_*.txt (default: newest)")
    ap.add_argument("--masses", default=None, help="masses_*.txt (default: newest if present)")
    ap.add_argument("--masshist", default=None, help="masshist_*.txt (default: newest if present)")
    ap.add_argument("--outdir", default="plots", help="output folder for figures")
    ap.add_argument("--traj_k", type=int, default=10, help="number of tracked star trajectories")
    ap.add_argument("--traj_mode", choices=["topmass", "random", "first"], default="topmass",
                    help="which stars to track (topmass needs masses)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth snapshot (1=all)")
    ap.add_argument("--xy_bins", type=int, default=300, help="bins for XY surface density plot")
    args = ap.parse_args()

    snaps_file = args.snaps or newest("snaps_*.bin")
    energy_file = args.energy or newest("energy_*.txt")
    mass_file = args.masses or newest("masses_*.txt")
    masshist_file = args.masshist or newest("masshist_*.txt")

    if snaps_file is None:
        raise FileNotFoundError("Could not find snaps_*.bin in this folder.")
    if energy_file is None:
        print("[warn] No energy_*.txt found. Energy plots will be skipped.")

    ensure_outdir(args.outdir)

    # ---- read first snapshot to get N and set up ----
    with open(snaps_file, "rb") as f:
        first = read_one_snapshot(f)
        if first is None:
            raise RuntimeError("snaps file appears empty.")
        N, t0, x0, y0, z0 = first

    nsnap = file_snapshot_count(snaps_file, N)
    if nsnap <= 0:
        raise RuntimeError("Could not infer snapshot count.")

    idx_start = 0
    idx_mid = (nsnap - 1) // 2
    idx_end = nsnap - 1

    print(f"Using snaps:  {snaps_file}")
    print(f"N = {N}, snapshots = {nsnap}, mid index = {idx_mid}")
    print(f"First snapshot time = {t0/Myr:.6f} Myr")

    # ---- optional masses ----
    m = read_masses(mass_file, N)
    if m is not None:
        print(f"Using masses: {mass_file}")
        print(f"Mass range [Msun]: min={m.min():.4f}, max={m.max():.4f}")
    else:
        print("No valid masses file -> mass-weighted plots skipped.")

    # ---- choose trajectories to track ----
    rng = np.random.default_rng(0)
    if args.traj_mode == "topmass" and m is not None:
        ids = np.argsort(m)[-args.traj_k:]
    elif args.traj_mode == "random":
        ids = rng.choice(N, size=min(args.traj_k, N), replace=False)
    else:
        ids = np.arange(min(args.traj_k, N))

    traj_x = {int(i): [] for i in ids}
    traj_y = {int(i): [] for i in ids}

    # ---- time series arrays (processed snapshots) ----
    nproc = (nsnap + args.stride - 1) // args.stride
    t_series = np.zeros(nproc)

    mean_r = np.zeros(nproc)
    rms_r = np.zeros(nproc)
    r10_num = np.zeros(nproc)
    r50_num = np.zeros(nproc)
    r90_num = np.zeros(nproc)

    # mass-based
    r10_m = r50_m = r90_m = None
    seg_top = seg_bot = None
    if m is not None:
        r10_m = np.zeros(nproc)
        r50_m = np.zeros(nproc)
        r90_m = np.zeros(nproc)
        seg_top = np.zeros(nproc)
        seg_bot = np.zeros(nproc)

    # keep radii for selected snapshots
    hist_r = {}

    # keep final snapshot positions for XY density
    final_xyz = None

    # ---- stream through snapshots ----
    p = 0
    with open(snaps_file, "rb") as f:
        s = 0
        while True:
            snap = read_one_snapshot(f)
            if snap is None:
                break
            N_read, t, x, y, z = snap
            if N_read != N:
                raise RuntimeError(f"N changed inside file: {N_read} vs {N}")

            if (s % args.stride) != 0:
                s += 1
                continue

            r = np.sqrt(x.astype(np.float64)**2 + y.astype(np.float64)**2 + z.astype(np.float64)**2)  # pc

            t_series[p] = t
            mean_r[p] = np.mean(r)
            rms_r[p] = np.sqrt(np.mean(r*r))

            r10_num[p], r50_num[p], r90_num[p] = percentiles_number(r, (0.1, 0.5, 0.9))

            # trajectories
            for i in ids:
                ii = int(i)
                traj_x[ii].append(float(x[ii]))
                traj_y[ii].append(float(y[ii]))

            # store start/mid/end hist radii (based on raw snapshot index s)
            if s in (idx_start, idx_mid, idx_end):
                hist_r[s] = r.copy()

            # store final snapshot xyz
            if s == idx_end:
                final_xyz = (x.copy(), y.copy(), z.copy())

            # mass-based metrics
            if m is not None:
                r10_m[p], r50_m[p], r90_m[p] = lagrange_radii_mass(r, m, (0.1, 0.5, 0.9))

                idx_sorted = np.argsort(m)
                bot = idx_sorted[: max(1, N // 2)]
                top = idx_sorted[int(0.9 * N):]
                seg_bot[p] = np.mean(r[bot])
                seg_top[p] = np.mean(r[top])

            p += 1
            s += 1

    # trim in case of any mismatch
    t_series = t_series[:p]
    mean_r = mean_r[:p]
    rms_r = rms_r[:p]
    r10_num = r10_num[:p]
    r50_num = r50_num[:p]
    r90_num = r90_num[:p]
    if m is not None:
        r10_m = r10_m[:p]
        r50_m = r50_m[:p]
        r90_m = r90_m[:p]
        seg_top = seg_top[:p]
        seg_bot = seg_bot[:p]

    t_myr = t_series / Myr
    print(f"Last processed time = {t_myr[-1]:.6f} Myr (stride={args.stride})")

    # =========================
    # ENERGY PLOTS
    # =========================
    if energy_file is not None and os.path.exists(energy_file):
        tE, E, dE = read_energy(energy_file)
        tE_myr = tE / Myr

        plt.figure()
        plt.plot(tE_myr, E)
        plt.xlabel("Time [Myr]")
        plt.ylabel("Total Energy [J]")
        plt.title("Total Energy")
        plt.grid(True)
        savefig(os.path.join(args.outdir, "total_energy.png"))

        plt.figure()
        plt.plot(tE_myr, dE)
        plt.xlabel("Time [Myr]")
        plt.ylabel("dE / E0")
        plt.title("Energy Error")
        plt.grid(True)
        savefig(os.path.join(args.outdir, "energy_error.png"))

        plt.figure()
        plt.semilogy(tE_myr, np.abs(dE) + 1e-30)
        plt.xlabel("Time [Myr]")
        plt.ylabel("|dE / E0|")
        plt.title("Absolute Energy Error (log scale)")
        plt.grid(True)
        savefig(os.path.join(args.outdir, "energy_error_abs_log.png"))

    # =========================
    # RADIUS / STRUCTURE PLOTS
    # =========================
    plt.figure()
    plt.plot(t_myr, mean_r, label="mean r")
    plt.plot(t_myr, rms_r, label="RMS r")
    plt.xlabel("Time [Myr]")
    plt.ylabel("Radius [pc]")
    plt.title("Mean and RMS Radius")
    plt.grid(True)
    plt.legend()
    savefig(os.path.join(args.outdir, "mean_r_and_rms.png"))

    plt.figure()
    plt.plot(t_myr, r10_num, label="r10 (number)")
    plt.plot(t_myr, r50_num, label="r50 (number)")
    plt.plot(t_myr, r90_num, label="r90 (number)")
    plt.xlabel("Time [Myr]")
    plt.ylabel("Radius [pc]")
    plt.title("Lagrange Radii (by number)")
    plt.grid(True)
    plt.legend()
    savefig(os.path.join(args.outdir, "lagrange_radii_number.png"))

    plt.figure()
    plt.plot(t_myr, r50_num / max(r50_num[0], 1e-30))
    plt.xlabel("Time [Myr]")
    plt.ylabel("r50(t) / r50(0)")
    plt.title("Expansion Factor (half-number radius)")
    plt.grid(True)
    savefig(os.path.join(args.outdir, "expansion_factor_r50.png"))

    # cumulative radial distribution start vs end
    if idx_start in hist_r and idx_end in hist_r:
        r_start = np.sort(hist_r[idx_start])
        r_end = np.sort(hist_r[idx_end])
        f_start = np.linspace(0, 1, len(r_start))
        f_end = np.linspace(0, 1, len(r_end))

        plt.figure()
        plt.plot(r_start, f_start, label="start")
        plt.plot(r_end, f_end, label="end")
        plt.xlabel("r [pc]")
        plt.ylabel("Cumulative fraction")
        plt.title("Cumulative Radial Distribution")
        plt.grid(True)
        plt.legend()
        savefig(os.path.join(args.outdir, "cumulative_radial_distribution.png"))

    # radial hist start/mid/end
    plt.figure()
    for idx, label in [(idx_start, "start"), (idx_mid, "mid"), (idx_end, "end")]:
        if idx in hist_r:
            plt.hist(hist_r[idx], bins=60, histtype="step", density=True, label=label)
    plt.xlabel("r [pc]")
    plt.ylabel("Probability density")
    plt.title("Radial Distribution (start / mid / end)")
    plt.grid(True)
    plt.legend()
    savefig(os.path.join(args.outdir, "radial_hist_start_mid_end.png"))

    # 3D radial density profile (final)
    if idx_end in hist_r:
        prof = radial_density_profile(hist_r[idx_end], nbins=45)
        if prof is not None:
            rmid, rho = prof
            plt.figure()
            plt.loglog(rmid, rho + 1e-30)
            plt.xlabel("r [pc]")
            plt.ylabel("Number density [1/pc^3] (arb)")
            plt.title("Radial Number Density Profile (final)")
            plt.grid(True, which="both")
            savefig(os.path.join(args.outdir, "radial_density_profile_final.png"))

    # sample XY trajectories
    plt.figure()
    for ii in ids:
        ii = int(ii)
        plt.plot(traj_x[ii], traj_y[ii], linewidth=1)
    plt.xlabel("x [pc]")
    plt.ylabel("y [pc]")
    plt.title(f"Sample XY Trajectories (k={len(ids)})")
    plt.grid(True)
    savefig(os.path.join(args.outdir, "sample_xy_trajectories.png"))

    # XY surface density (final snapshot)
    if final_xyz is not None:
        xf, yf, zf = final_xyz
        H, xedges, yedges = np.histogram2d(
            xf.astype(np.float64), yf.astype(np.float64),
            bins=args.xy_bins
        )
        plt.figure()
        plt.imshow(
            H.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="equal"
        )
        plt.colorbar(label="counts per bin")
        plt.xlabel("x [pc]")
        plt.ylabel("y [pc]")
        plt.title("XY Surface Density (final snapshot)")
        savefig(os.path.join(args.outdir, "xy_density_final.png"))

    # =========================
    # MASS PLOTS (if masses exist)
    # =========================
    if m is not None:
        # mass histogram
        plt.figure()
        plt.hist(m, bins=60, histtype="step")
        plt.xlabel("Mass [Msun]")
        plt.ylabel("Count")
        plt.title("IMF Sample (masses_*.txt)")
        plt.grid(True)
        savefig(os.path.join(args.outdir, "mass_histogram.png"))

        # mass-lagrange radii
        plt.figure()
        plt.plot(t_myr, r10_m, label="r10 (mass)")
        plt.plot(t_myr, r50_m, label="r50 (mass)")
        plt.plot(t_myr, r90_m, label="r90 (mass)")
        plt.xlabel("Time [Myr]")
        plt.ylabel("Radius [pc]")
        plt.title("Lagrange Radii (by mass)")
        plt.grid(True)
        plt.legend()
        savefig(os.path.join(args.outdir, "lagrange_radii_mass.png"))

        # mass segregation proxy
        plt.figure()
        plt.plot(t_myr, seg_top, label="mean r (top 10% mass)")
        plt.plot(t_myr, seg_bot, label="mean r (bottom 50% mass)")
        plt.xlabel("Time [Myr]")
        plt.ylabel("Mean radius [pc]")
        plt.title("Mass Segregation Proxy")
        plt.grid(True)
        plt.legend()
        savefig(os.path.join(args.outdir, "mass_segregation_proxy.png"))

        # print key percentages for your bins of interest
        def pct_in(lo, hi):
            return 100.0 * np.mean((m >= lo) & (m < hi))

        #print("\nMass percentages (from masses_*.txt):")
        #print(f"  3.8–4.0 Msun : {pct_in(3.8, 4.0):.4f}%")
        #print(f"  3.6–3.8 Msun : {pct_in(3.6, 3.8):.4f}%")
        #print(f"  0.2–0.4 Msun : {pct_in(0.2, 0.4):.4f}%")

    # =========================
    # MASSHIST TEXT FILE (if present)
    # =========================
    if masshist_file is not None and os.path.exists(masshist_file):
        try:
            mh = np.loadtxt(masshist_file, skiprows=1)

            # If the file has only one row, loadtxt returns 1D; force 2D
            if mh.ndim == 1:
                mh = mh[None, :]

            # columns: m_lo, m_hi, count, percent
            m_lo = mh[:, 0]
            m_hi = mh[:, 1]
            cnt  = mh[:, 2].astype(int)
            pct  = mh[:, 3]

            centers = 0.5 * (m_lo + m_hi)
            width   = (m_hi - m_lo)

            plt.figure()
            plt.bar(centers, pct, width=width * 0.95)
            plt.xlabel("Mass bin center [Msun]")
            plt.ylabel("Percent of stars [%]")
            plt.title("Mass Bin Percentages (masshist_*.txt)")
            plt.grid(True)
            savefig(os.path.join(args.outdir, "masshist_percent_bar.png"))

            print(f"\nRead masshist file: {masshist_file}")
            print("\nMass percentages (from masshist_*.txt):")
            for lo, hi, c, p in zip(m_lo, m_hi, cnt, pct):
                print(f"{lo:6.3f}–{hi:6.3f} Msun: {p:9.4f}%  ({c} stars)")

            print(f"Sum of percentages = {pct.sum():.4f}%")

        except Exception as e:
            print(f"[warn] Could not parse masshist file: {masshist_file} ({e})")



    print(f"\nDone. Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()
