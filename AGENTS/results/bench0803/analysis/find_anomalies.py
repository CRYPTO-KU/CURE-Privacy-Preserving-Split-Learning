#!/usr/bin/env python3
import csv, glob, os, re, math, argparse, collections

def load_csvs(root):
    paths = sorted(glob.glob(os.path.join(root, "bench_results_cores*_logn*.csv")))
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
    return rows

def to_float(s):
    try:
        return float(s)
    except Exception:
        return 0.0

def to_int(s):
    try:
        return int(s)
    except Exception:
        return 0

def total_time(row):
    return to_float(row["forward_time"]) + to_float(row["backward_time"])  # ignore update in total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(__file__)))
    ap.add_argument("--out", default="anomalies.csv")
    ap.add_argument("--eps", type=float, default=0.02, help="tolerance for monotonic comparisons")
    args = ap.parse_args()

    rows = load_csvs(args.root)
    # Filter to HE rows only for speedup analysis
    rows_he = [r for r in rows if r.get("mode") == "HE"]

    # Group by (model, layer, logN, cores)
    key = lambda r: (r["model"], r["layer"], r["logN"], int(r["num_cores"]))
    grouped = collections.defaultdict(list)
    for r in rows_he:
        grouped[key(r)].append(r)

    # Aggregate mean total per group to smooth noise
    agg = {}
    for k, samples in grouped.items():
        times = [total_time(s) for s in samples]
        agg[k] = sum(times) / len(times)

    # Build index by (model, layer, logN) -> {cores: total}
    idx = collections.defaultdict(dict)
    for (model, layer, logN, cores), val in agg.items():
        idx[(model, layer, logN)][cores] = val

    anomalies = []
    rerun_plan = set()

    for (model, layer, logN), core_map in idx.items():
        cores_present = sorted(core_map.keys())
        if 1 not in core_map:
            continue  # need baseline
        base = core_map[1]
        for c in cores_present:
            if c == 1:
                continue
            speedup = base / core_map[c] if core_map[c] > 0 else math.inf
            if speedup > c * (1.0 + args.eps):
                anomalies.append({
                    "type": "superlinear",
                    "model": model,
                    "layer": layer,
                    "logN": logN,
                    "cores": c,
                    "speedup": f"{speedup:.2f}",
                    "baseline_total": f"{base:.6f}",
                    "current_total": f"{core_map[c]:.6f}",
                })
                # Rerun highest cores for this model/logN
                rerun_plan.add((model, logN, c))

        # 32 better than 40 anomaly
        if 32 in core_map and 40 in core_map:
            if core_map[32] < core_map[40] * (1.0 - args.eps):
                anomalies.append({
                    "type": "32_better_than_40",
                    "model": model,
                    "layer": layer,
                    "logN": logN,
                    "cores": 40,
                    "total_32": f"{core_map[32]:.6f}",
                    "total_40": f"{core_map[40]:.6f}",
                })
                rerun_plan.add((model, logN, 32))
                rerun_plan.add((model, logN, 40))

    # Write anomalies CSV
    with open(args.out, "w", newline="") as f:
        if anomalies:
            fieldnames = sorted({k for a in anomalies for k in a.keys()})
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for a in anomalies:
                w.writerow(a)
        else:
            f.write("type,model,layer,logN,cores,detail\n")

    # Print a minimal rerun command suggestion
    # Group reruns by (model, logN)
    plan_by_model = collections.defaultdict(set)
    for model, logN, c in rerun_plan:
        plan_by_model[(model, logN)].add(c)

    print("Suggested reruns:")
    for (model, logN), cores in sorted(plan_by_model.items()):
        cores_csv = ",".join(str(c) for c in sorted(cores))
        print(f"  ./bench_rerun_linux_amd64 --models {model} --logNs {logN} --cores {cores_csv} --out bench_results_rerun_{model}_logn{logN}.csv")

if __name__ == "__main__":
    main()


