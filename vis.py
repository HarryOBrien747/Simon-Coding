import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

G = 6.67430e-11
pc = 3.085677581e16
Myr = 1.0e6 * 365.25 * 86400.0

def newest(glob_pattern: str):
    files = glob.glob(glob_pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]

def infer_N_from_header(state_file: str) -> int:
    with open(state_file, "r") as f:
        header = f.readline().strip().split()
    # header: t[s] then 6N columns
    ncols = len(header) - 1
    if ncols % 6 != 0:
        raise ValueError(f"State header columns not divisible by 6: got {ncols}")
    return ncols // 6

def count_snapshots(state_file: str) -> int:
    n = 0
    with open(state_file, "r") as f:
        _ = f.readline()  # header
        for line in f:
            if line.strip():
                n += 1
    return n

def read_energy(energy_file: str):
    data = np.loadtxt(energy_file, skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    t = data[:, 0]
    E = data[:, 1]
    dE = data[:, 2]
    return t, E, dE

def try_read_masses(mass_file: str, N: int):
    if mass_file is None:
        return None
    if not os.path.exists(mass_file):
        return None
    m = np.loadtxt(mass_file)
    if m.ndim != 1 or m.size != N:
        print(f"[warn] mass file exists but size mismatch: expected {N}, got {m.size}")
        return None
    return m

def r50_r90_number(radii):
    rs = np.sort(radii)
    r50 = rs[int(0.50 * (len(rs) - 1))]
    r90 = rs[int(0.90 * (len(rs) - 1))]
    return r50, r90

def lagrange_radii_mass(radii, masses, fracs=(0.1, 0.5, 0.9)):
    idx = np.argsort(radii)
    r_sorted = radii[idx]
    m_sorted = masses[idx]
    cum = np.cumsum(m_sorted)
    tot = cum[-1]
    out = []
    for f in fracs:
        target = f * tot
        j = np.searchsorted(cum, target)
        j = min(max(j, 0), len(r_sorted) - 1)
        out.append(r_sorted[j])
    return out  # same order as fracs

def velocity_dispersion(v):
    # 1D dispersion (unweighted): std over all components
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    sigx = np.std(vx)
    sigy = np.std(vy)
    sigz = np.std(vz)
    sigma1d = (sigx + sigy + sigz) / 3.0
    vrms = np.sqrt(np.mean(vx*vx + vy*vy + vz*vz))
    return sigma1d, vrms

def approximate_escapers(radii, speeds, Mtot):
    # crude: v > sqrt(2GM/r)
    # avoid r=0:
    r = np.maximum(radii, 1e-30)
    vesc = np.sqrt(2.0 * G * Mtot / r)
    return np.sum(speeds > vesc)

def select_ids(N, m=None, k=10):
    if m is not None:
        # top-mass stars
        return np.argsort(m)[-k:]
    # otherwise first k
    return np.arange(min(k, N))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=None, help="cluster_state_*.txt (default: newest)")
    ap.add_argument("--energy", default=None, help="cluster_energy_*.txt (default: newest)")
    ap.add_argument("--masses", default=None, help="optional cluster_masses_*.txt (default: try newest)")
    ap.add_argument("--traj_k", type=int, default=10, help="how many sample trajectories to plot")
    ap.add_argument("--outdir", default="plots", help="output folder for figures")
    ap.add_argument("--virial_stride", type=int, default=0,
                    help="if >0 and masses provided, compute virial ratio every Nth snapshot (can be slow)")
    args = ap.parse_args()

    state_file = args.state or newest("cluster_state_*.txt")
    energy_file = args.energy or newest("cluster_energy_*.txt")
    if state_file is None or energy_file is None:
        raise FileNotFoundError("Could not find cluster_state_*.txt and/or cluster_energy_*.txt in this folder.")

    N = infer_N_from_header(state_file)
    nsnap = count_snapshots(state_file)
    mid_idx = nsnap // 2
    end_idx = nsnap - 1

    mass_file = args.masses
    if mass_file is None:
        mass_file = newest("cluster_masses_*.txt")  # optional
    m = try_read_masses(mass_file, N)
    Mtot = np.sum(m) if m is not None else None

    os.makedirs(args.outdir, exist_ok=True)

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
    plt.plot(tE_myr, E)
    plt.xlabel("Time [Myr]")
    plt.ylabel("Total Energy [J]")
    plt.title("Total Energy")
    plt.grid(True)
    plt.savefig(os.path.join(args.outdir, "total_energy.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- prepare time series containers from state ---
    times = np.zeros(nsnap)
    r50_num = np.zeros(nsnap)
    r90_num = np.zeros(nsnap)
    sigma1d = np.zeros(nsnap)
    vrms = np.zeros(nsnap)
    escapers = np.zeros(nsnap, dtype=int)

    # optional mass-based
    r10_m = r50_m = r90_m = None
    seg_top = seg_bot = None
    Q_vir = None
    Q_t = None

    if m is not None:
        r10_m = np.zeros(nsnap)
        r50_m = np.zeros(nsnap)
        r90_m = np.zeros(nsnap)
        seg_top = np.zeros(nsnap)  # mean radius of top 10% mass
        seg_bot = np.zeros(nsnap)  # mean radius of bottom 50% mass

        if args.virial_stride > 0:
            Q_vir = []
            Q_t = []

    # sample trajectories
    ids = select_ids(N, m=m, k=args.traj_k)
    traj_x = {i: [] for i in ids}
    traj_y = {i: [] for i in ids}

    # radial hist snapshots
    hist_data = {}

    # read state snapshots streaming (fast-ish)
    with open(state_file, "r") as f:
        _ = f.readline()  # header
        for s, line in enumerate(f):
            if not line.strip():
                continue
            arr = np.fromstring(line, sep=" ")
            t = arr[0]
            data = arr[1:].reshape(N, 6)
            pos = data[:, 0:3]
            vel = data[:, 3:6]

            rmag = np.sqrt(np.sum(pos*pos, axis=1))     # [m]
            speed = np.sqrt(np.sum(vel*vel, axis=1))    # [m/s]

            times[s] = t
            r50_num[s], r90_num[s] = r50_r90_number(rmag)

            sig, v_rms = velocity_dispersion(vel)
            sigma1d[s] = sig
            vrms[s] = v_rms

            # approximate escapers needs total mass; if masses missing, we can’t do v_esc meaningfully
            if Mtot is not None:
                escapers[s] = approximate_escapers(rmag, speed, Mtot)
            else:
                escapers[s] = 0

            # trajectories (XY)
            for i in ids:
                traj_x[i].append(pos[i, 0] / pc)
                traj_y[i].append(pos[i, 1] / pc)

            # radial hists at start/mid/end
            if s in (0, mid_idx, end_idx):
                hist_data[s] = rmag / pc

            # mass-based metrics (if available)
            if m is not None:
                r10, r50, r90 = lagrange_radii_mass(rmag, m, fracs=(0.1, 0.5, 0.9))
                r10_m[s], r50_m[s], r90_m[s] = r10, r50, r90

                # mass segregation proxy: mean radius top 10% mass vs bottom 50%
                idx_sorted = np.argsort(m)
                bot = idx_sorted[: max(1, N//2)]
                top = idx_sorted[int(0.9*N):]
                seg_bot[s] = np.mean(rmag[bot]) / pc
                seg_top[s] = np.mean(rmag[top]) / pc

                # optional virial ratio (expensive if done every snapshot)
                if args.virial_stride > 0 and (s % args.virial_stride == 0):
                    # Kinetic energy
                    T = 0.5 * np.sum(m * np.sum(vel*vel, axis=1))
                    # Potential energy exact O(N^2) with a loop (N=1000 ok if not too frequent)
                    U = 0.0
                    for i in range(N-1):
                        rij = pos[i+1:] - pos[i]
                        dist = np.sqrt(np.sum(rij*rij, axis=1))
                        U -= G * m[i] * np.sum(m[i+1:] / np.maximum(dist, 1e-30))
                    Q = 2.0 * T / abs(U)
                    Q_vir.append(Q)
                    Q_t.append(t / Myr)

    t_myr = times / Myr

    # --- plots from state series ---
    plt.figure()
    plt.plot(t_myr, r50_num/pc, label="R50 (by number)")
    plt.plot(t_myr, r90_num/pc, label="R90 (by number)")
    plt.xlabel("Time [Myr]")
    plt.ylabel("Radius [pc]")
    plt.title("Cluster Size (Number Lagrange Radii)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.outdir, "r50_r90_number.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(t_myr, sigma1d/1e3, label="sigma_1D")
    plt.plot(t_myr, vrms/1e3, label="v_rms")
    plt.xlabel("Time [Myr]")
    plt.ylabel("Velocity [km/s]")
    plt.title("Velocity Dispersion")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.outdir, "velocity_dispersion.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(t_myr, escapers)
    plt.xlabel("Time [Myr]")
    plt.ylabel("N(escapers) [approx]")
    plt.title("Approx Escapers vs Time")
    plt.grid(True)
    plt.savefig(os.path.join(args.outdir, "escapers_approx.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # radial histograms start/mid/end
    plt.figure()
    for idx, label in [(0, "start"), (mid_idx, "mid"), (end_idx, "end")]:
        if idx in hist_data:
            plt.hist(hist_data[idx], bins=40, histtype="step", label=label, density=True)
    plt.xlabel("r [pc]")
    plt.ylabel("Probability density")
    plt.title("Radial Distribution (start / mid / end)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.outdir, "radial_histograms.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # sample XY trajectories
    plt.figure()
    for i in ids:
        plt.plot(traj_x[i], traj_y[i], linewidth=1)
    plt.xlabel("x [pc]")
    plt.ylabel("y [pc]")
    plt.title(f"Sample XY Trajectories (N={len(ids)})")
    plt.grid(True)
    plt.savefig(os.path.join(args.outdir, "sample_xy_trajectories.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # mass-based plots if masses exist
    if m is not None:
        plt.figure()
        plt.plot(t_myr, r10_m, label="R10 (mass)")
        plt.plot(t_myr, r50_m, label="R50 (mass)")
        plt.plot(t_myr, r90_m, label="R90 (mass)")
        plt.xlabel("Time [Myr]")
        plt.ylabel("Radius [pc]")
        plt.title("Mass Lagrange Radii")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(args.outdir, "lagrange_radii_mass.png"), dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(t_myr, seg_top, label="Mean r (top 10% mass)")
        plt.plot(t_myr, seg_bot, label="Mean r (bottom 50% mass)")
        plt.xlabel("Time [Myr]")
        plt.ylabel("Mean radius [pc]")
        plt.title("Mass Segregation Proxy")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(args.outdir, "mass_segregation_proxy.png"), dpi=200, bbox_inches="tight")
        plt.close()

        if Q_vir is not None and len(Q_vir) > 0:
            plt.figure()
            plt.plot(Q_t, Q_vir)
            plt.xlabel("Time [Myr]")
            plt.ylabel("Q = 2T/|U|")
            plt.title(f"Virial Ratio (every {args.virial_stride} snapshots)")
            plt.grid(True)
            plt.savefig(os.path.join(args.outdir, "virial_ratio.png"), dpi=200, bbox_inches="tight")
            plt.close()

    print(f"Done. Saved plots to: {args.outdir}")
    print(f"Used state:  {state_file}")
    print(f"Used energy: {energy_file}")
    if m is not None:
        print(f"Used masses: {mass_file}")
    else:
        print("No mass file found -> mass-weighted plots + virial ratio skipped.")

if __name__ == "__main__":
    main()
