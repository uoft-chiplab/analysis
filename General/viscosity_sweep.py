"""
Shear-viscosity sensitivity sweep for the unitary pipeline.

Runs one unitary analysis per shot, but instead of using the per-shot 2nd-moment
estimate of the shear viscosity (alpha_visc), it FORCES a grid of alpha values and
records how T, N, TF and T/TF respond. The expensive upstream work (load .mat, OD,
Gaussian fit, 2nd moment) is done once per shot; only the alpha-dependent tail
(expansionParams -> Abel -> pressure -> pres1dFit) is repeated per alpha.

Usage:
    python viscosity_sweep.py                 # sweeps the default run dir below
    python viscosity_sweep.py "<run folder>"  # any run folder containing imgs/

Outputs (written into the run folder):
    viscosity_sweep.csv   long-form table: one row per (shot, alpha)
    viscosity_sweep.png   T/TF, T[nK] and N vs alpha, one line per shot

Edit ALPHAS / MAX_SHOTS below to change the grid or how many shots are swept.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
PYTHERMOMETRY = r"D:\LocalCode\pythermometry"
sys.path.insert(0, PYTHERMOMETRY)   # so `import settings`, etc. resolve
os.chdir(PYTHERMOMETRY)             # FermiFitter/tables/* are loaded relative to cwd
import helpers.constants as c
from settings import settings
from scanFormatter import scanInitialize
from chipImaging import calculateOD, createBlanks
from analysis_runner import prepare_fermifit_od
from fermifitter import fermifitMonitor

# ---------------------------------------------------------------- configuration
DEFAULT_RUN = r"F:\Data\2026\08 August2026\11August2026\I_odt1_2_0p7_2p3_UShots"
ALPHAS = np.round(np.arange(1.0, 8.01, 0.5), 3)   # shear-viscosity grid to sweep
MAX_SHOTS = 12                                     # cap number of shots (speed)
READ_COLS = ["T", "TF", "N", "ToTF", "q", "bx", "by", "alpha_visc"]


def build_settings(run_dir):
    s = settings(run_dir)
    # force the unitary path regardless of how the run's JSON is flagged
    s.analysis_cfg["unitary"] = True
    s.analysis_cfg["isSSI"] = False
    s.analysis_cfg["offchip"] = False
    s.analysis_cfg["bz_boxes"] = False
    s.analysis_cfg["atomson"] = True
    # calculateOD reads these display toggles (normally set by run_analysis); the
    # sweep never wants per-shot image dumps
    s.output_cfg.update({"save_OD": False, "save_imgs": False, "average": False})
    return s


def make_expv(s, sp):
    expv = c.fermifit_expv(np.array(s.fermifit3D_cfg["trap_freqs"]))
    expv.update_atom_num(s.fermifit3D_cfg["N_B"])
    expv.TOF = s.fermifit3D_cfg["TOF"]
    expv.m = sp["m"]
    return expv


def fit_once(df, i, ff_od, s, expv, alpha):
    """Run the unitary fit for shot i at a fixed alpha; return a dict of outputs
    (or None if the fit failed). alpha=None uses the normal per-shot estimate."""
    ok = fermifitMonitor(df, i, ff_od, s, expv, talk=0, alpha_override=alpha)
    if not ok:
        return None
    row = df.loc[i, READ_COLS]
    return {k: float(row[k]) for k in READ_COLS}


def main():
    run_dir = sys.argv[1] if (len(sys.argv) > 1 and os.path.isdir(sys.argv[1])) else DEFAULT_RUN
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run folder not found: {run_dir}")
    print(f"run folder : {run_dir}")
    print(f"alpha grid : {list(ALPHAS)}")

    s = build_settings(run_dir)
    cam = c.camera(s.analysis_cfg["pixis"])

    # ref-region border mask needed by the unitary OD (chipImaging), replicated
    # from analysis_runner.run_analysis
    ROI = s.img_eval_cfg["ROI"]
    ref_roi_mask = np.zeros(cam.size)
    ref_roi_mask[max(ROI[0] - 6, 0):min(ROI[1] + 5, cam.size[0]),
                 max(ROI[2] - 6, 0):min(ROI[3] + 5, cam.size[1])] = 1
    ref_roi_mask[ROI[0] - 1:ROI[1], ROI[2] - 1:ROI[3]] = 0
    s.ref_roi_mask = ref_roi_mask

    atom_blanks, ref_blanks = createBlanks(s, cam.size)
    sp = c.SPECIES.get(s.analysis_cfg.get("species", "K"), c.SPECIES["K"])
    expv = make_expv(s, sp)

    df = scanInitialize(s)

    # pick shots: those with an image file, capped at MAX_SHOTS
    have_img = [i for i in df.index if isinstance(df.loc[i, "imgfiles"], str)]
    shots = have_img[:MAX_SHOTS]
    print(f"sweeping {len(shots)} shots x {len(ALPHAS)} alphas "
          f"= {len(shots) * (len(ALPHAS) + 1)} fits\n")

    records, baselines = [], []
    for n, i in enumerate(shots):
        try:
            atom, bg, ref, od, failed = calculateOD(df, i, s, atom_blanks, ref_blanks)
        except Exception as e:
            print(f"  shot {i}: OD failed ({e}); skipping")
            continue
        if failed:
            print(f"  shot {i}: OD returned no image; skipping")
            continue
        # simple shutter-failure autoskip (as in run_analysis)
        if abs(df.loc[i, "roi.at_med"] - df.loc[i, "roi.ref_med"]) > 1000:
            print(f"  shot {i}: autoskip (shutter failure)")
            continue

        ff_od = prepare_fermifit_od(df, i, s, bg, ref, od, expv, sp)
        freq = df.loc[i, "freq"] if "freq" in df.columns else i

        # baseline: the pipeline's own per-shot estimate (alpha_override=None)
        base = fit_once(df, i, ff_od, s, expv, None)
        if base is not None:
            baselines.append({"cycle": i, "freq": freq,
                              "auto_alpha": base["alpha_visc"],
                              "T_nK": base["T"] * 1e9, "N": base["N"],
                              "ToTF": base["ToTF"]})

        for a in ALPHAS:
            out = fit_once(df, i, ff_od, s, expv, float(a))
            if out is None:
                continue
            records.append({"cycle": i, "freq": freq, "alpha": float(a),
                            "T_nK": out["T"] * 1e9, "TF_nK": out["TF"] * 1e9,
                            "N": out["N"], "ToTF": out["ToTF"],
                            "q": out["q"], "bx": out["bx"], "by": out["by"]})
        if base:
            print(f"  shot {i:3d} (freq {freq}): auto alpha "
                  f"{base['alpha_visc']:.2f} -> ToTF {base['ToTF']:.3f}")
        else:
            print(f"  shot {i:3d} (freq {freq}): baseline fit failed")

    if not records:
        raise SystemExit("no successful fits; nothing to report")

    sweep = pd.DataFrame(records)
    base_df = pd.DataFrame(baselines)
    csv_path = os.path.join(run_dir, "viscosity_sweep.csv")
    sweep.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}  ({len(sweep)} rows)")

    # ---- plots: T/TF, T[nK], N vs alpha, one line per shot ------------------
    metrics = [("ToTF", "T/T_F"), ("T_nK", "T [nK]"), ("N", "N [atoms]")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (col, label) in zip(axes, metrics):
        for cyc, g in sweep.groupby("cycle"):
            g = g.sort_values("alpha")
            line, = ax.plot(g["alpha"], g[col], "-", lw=1, alpha=0.8,
                            label=f"{cyc}")
            # mark each shot's auto-estimated alpha
            br = base_df[base_df["cycle"] == cyc]
            if len(br):
                bcol = {"ToTF": "ToTF", "T_nK": "T_nK", "N": "N"}[col]
                ax.plot(br["auto_alpha"], br[bcol], "o", ms=5,
                        color=line.get_color())
        ax.set_xlabel("shear viscosity alpha")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs alpha")
    axes[0].legend(title="cycle", fontsize=7, ncol=2, loc="best")
    fig.suptitle("Viscosity sensitivity (dots = per-shot auto estimate)")
    fig.tight_layout()
    png_path = os.path.join(run_dir, "viscosity_sweep.png")
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    print(f"wrote {png_path}")

    # ---- text summary: sensitivity of ToTF to alpha per shot ----------------
    print("\nToTF sensitivity (per shot):")
    print(f"{'cycle':>5} {'freq':>7} {'autoA':>6} {'ToTF@lo':>8} "
          f"{'ToTF@auto':>9} {'ToTF@hi':>8} {'dToTF/dA':>9}")
    for cyc, g in sweep.groupby("cycle"):
        g = g.sort_values("alpha")
        lo, hi = g.iloc[0], g.iloc[-1]
        slope = (hi["ToTF"] - lo["ToTF"]) / (hi["alpha"] - lo["alpha"])
        br = base_df[base_df["cycle"] == cyc]
        aA = br["auto_alpha"].iloc[0] if len(br) else float("nan")
        tA = br["ToTF"].iloc[0] if len(br) else float("nan")
        print(f"{cyc:5d} {g['freq'].iloc[0]:7} {aA:6.2f} {lo['ToTF']:8.3f} "
              f"{tA:9.3f} {hi['ToTF']:8.3f} {slope:9.4f}")


if __name__ == "__main__":
    main()
