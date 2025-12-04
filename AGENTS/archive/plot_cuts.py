#!/usr/bin/env python3
import argparse, pathlib, csv, math
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def load(path):
    return pd.read_csv(path, dtype={"logN":str, "num_cores":str})

def ensure_fig_dir(out_dir):
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d

def plot_cut_curves(df, figs_dir):
    # For each (model, logN, num_cores)
    for (model, logN, cores), g in df.groupby(["model","logN","num_cores"]):
        g = g.sort_values("cut_position")
        plt.figure()
        plt.plot(g.cut_position, g.forward_time_total, marker="o", label="Forward")
        plt.plot(g.cut_position, g.backprop_time_total, marker="s", label="Backprop+Update")
        plt.xlabel("Cut Position")
        plt.ylabel("Time (s)")
        plt.title(f"{model} logN={logN} cores={cores}")
        plt.legend()
        out = figs_dir / f"{model}_logN{logN}_cores{cores}_cut_curve.png"
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close()

def plot_logn_comparison(df, figs_dir):
    # Fix model, cores; vary logN
    for (model, cores), g in df.groupby(["model","num_cores"]):
        # Need multiple logN
        if g.logN.nunique() < 2: continue
        plt.figure()
        for logN, g2 in g.groupby("logN"):
            g2 = g2.sort_values("cut_position")
            plt.plot(g2.cut_position, g2.forward_time_total, marker="o", label=f"logN {logN} (F)")
        plt.xlabel("Cut Position")
        plt.ylabel("Forward Time (s)")
        plt.title(f"{model} cores={cores} – Forward vs cut by logN")
        plt.legend()
        out = figs_dir / f"{model}_cores{cores}_forward_logn_compare.png"
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close()

def plot_core_scaling(df, figs_dir):
    # Fix model, logN; vary cores
    for (model, logN), g in df.groupby(["model","logN"]):
        if g.num_cores.nunique() < 2: continue
        plt.figure()
        for cores, g2 in g.groupby("num_cores"):
            g2 = g2.sort_values("cut_position")
            plt.plot(g2.cut_position, g2.forward_time_total, marker="o", label=f"{cores} cores (F)")
        plt.xlabel("Cut Position")
        plt.ylabel("Forward Time (s)")
        plt.title(f"{model} logN={logN} – Forward vs cut by cores")
        plt.legend()
        out = figs_dir / f"{model}_logN{logN}_forward_core_compare.png"
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="AGENTS/results/cut_aggregates.csv")
    ap.add_argument("--outdir", default="AGENTS/results/figs")
    args = ap.parse_args()

    df = load(args.agg)
    figs = ensure_fig_dir(args.outdir)

    plot_cut_curves(df, figs)
    plot_logn_comparison(df, figs)
    plot_core_scaling(df, figs)

    print(f"Figures written to {figs}")

if __name__ == "__main__":
    main() 