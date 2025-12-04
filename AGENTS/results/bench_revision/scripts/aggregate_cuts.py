#!/usr/bin/env python3
import argparse, csv, pathlib, re, collections

# Regular expressions to help parse layer names if needed later
LIN_RE = re.compile(r"Linear_(\d+)_(\d+)")
ACT_RE = re.compile(r"Activation_.*")

def discover_csvs(root_dir: pathlib.Path):
    return list(root_dir.rglob("bench_results_*.csv"))

def load_micro_rows(paths):
    rows = []
    for p in paths:
        with p.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                # Normalize logN: sometimes '-' for Plain; fill from filename if needed
                if r["logN"] in ("", "-", None):
                    # try to extract from parent folder name e.g. cores2_logn15
                    parent = p.parent.name
                    m = re.search(r"logn(\d+)", parent)
                    r["logN"] = m.group(1) if m else ""
                rows.append(r)
    return rows

def build_layer_order(rows_per_model):
    """
    We reconstruct canonical forward order by the first HE occurrence ordering
    (HE rows will have same sequence each file). Activation / Linear appear multiple times (repeats in repeated runs).
    Strategy: for a model, take the earliest occurrence index of each unique layer name
    in a HE or Plain context, keeping relative order.
    """
    seen = {}
    order = []
    for idx, r in enumerate(rows_per_model):
        layer = r["layer"]
        if layer not in seen:
            seen[layer] = idx
            order.append(layer)
    return order

def aggregate(rows):
    # Group by (model, logN, num_cores, mode, layer)
    grouped = collections.defaultdict(list)
    for r in rows:
        key = (r["model"], r["layer"], r["mode"], r["logN"], r["num_cores"])
        grouped[key].append(r)

    # For stability compute mean times if multiple samples per tuple
    stats = {}
    for key, samples in grouped.items():
        def mean(field):
            vals = [float(s[field]) for s in samples]
            return sum(vals)/len(vals)
        stats[key] = {
            "forward": mean("forward_time"),
            "backward": mean("backward_time"),
            "update": mean("update_time")
        }

    # Re-group by (model, logN, num_cores) to create cut aggregates
    result_rows = []
    by_model_param = collections.defaultdict(list)
    for (model, layer, mode, logN, cores), v in stats.items():
        by_model_param[(model, logN, cores)].append((layer, mode, v))

    for (model, logN, cores), layer_entries in by_model_param.items():
        # Determine canonical layer order
        # Sort by first appearance heuristic: order by min forward+backward time rank
        # Better: reconstruct using insertion order of original rows filtered by model/logN/cores
        # We'll just sort by an index derived from appearance in original list
        layer_order = build_layer_order([r for r in rows if r["model"]==model])
        # Filter only layers we have stats for
        layer_order = [L for L in layer_order if any(e[0]==L for e in layer_entries)]

        # Build per-layer dict: {layer: {mode: times}}
        layer_dict = collections.defaultdict(dict)
        for layer, mode, times in layer_entries:
            layer_dict[layer][mode] = times

        n_layers = len(layer_order)
        for cut in range(n_layers + 1):
            f_total = 0.0
            b_total = 0.0
            # Forward
            for i, layer in enumerate(layer_order):
                if i < cut: # server HE
                    times = layer_dict[layer].get("HE")
                else:       # client Plain
                    times = layer_dict[layer].get("Plain")
                if not times:
                    # If missing a mode, skip (or raise). For robustness we skip.
                    continue
                f_total += times["forward"]
            # Backprop (includes update)
            # Backprop order is reverse but summation is commutative, we just choose correct mode
            for i, layer in enumerate(layer_order):
                if i >= cut: # client side Plain backprop
                    times = layer_dict[layer].get("Plain")
                else:        # server side HE backprop
                    times = layer_dict[layer].get("HE")
                if not times:
                    continue
                b_total += times["backward"] + times["update"]

            result_rows.append({
                "model": model,
                "logN": logN,
                "num_cores": cores,
                "cut_position": cut,
                "forward_time_total": f_total,
                "backprop_time_total": b_total
            })
    return result_rows

def write_output(rows, out_path: pathlib.Path):
    fieldnames = ["model","logN","num_cores","cut_position","forward_time_total","backprop_time_total"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        def safe_int(val):
            try:
                return int(val) if val else 0
            except (ValueError, TypeError):
                return 0
        for r in sorted(rows, key=lambda x:(x["model"], safe_int(x["logN"]), int(x["num_cores"]), x["cut_position"])):
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Root directory containing microbenchmark CSVs")
    ap.add_argument("--out", default="cut_aggregates.csv")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    csvs = discover_csvs(root)
    if not csvs:
        raise SystemExit(f"No bench_results_*.csv found under {root}")
    rows = load_micro_rows(csvs)
    agg = aggregate(rows)
    write_output(agg, pathlib.Path(args.out))
    print(f"Wrote {len(agg)} rows to {args.out}")

if __name__ == "__main__":
    main()
