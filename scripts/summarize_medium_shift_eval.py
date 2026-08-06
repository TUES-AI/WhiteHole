import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize source, shifted, and adapted medium-shift evals."
    )
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--medium-json", required=True)
    parser.add_argument("--delta-json", required=True)
    parser.add_argument("--diagonal-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-rmse", type=float, default=15.0)
    parser.add_argument("--max-rollout-ratio", type=float, default=1000.0)
    return parser.parse_args()


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def maybe_mean(values):
    if values is None:
        return None
    return mean(float(value) for value in values)


def baseline_row(name: str, data: dict, shift: str):
    rollout = data.get("latent_rollout_mse_by_horizon")
    return {
        "condition": name,
        "appearance_shift": shift,
        "adapter": "none",
        "probe_rmse_px": data.get("linear_probe_rmse_pixels"),
        "probe_vs_mean_ratio": data.get("linear_probe_vs_mean_rmse_ratio"),
        "rollout_vs_persistence_ratio": data.get(
            "latent_rollout_vs_persistence_mse_ratio"
        ),
        "rollout_mse_mean": maybe_mean(rollout),
        "rollout_mse_h1": rollout[0] if rollout else None,
        "rollout_mse_last": rollout[-1] if rollout else None,
        "paired_latent_mse_after": None,
        "rollout_to_source_mse_mean": None,
        "delta_l2": None,
        "scale_mean": None,
        "scale_std": None,
    }


def adapter_row(name: str, data: dict, adapter: str):
    rollout = data.get("adapted_rollout_mse_by_horizon")
    rollout_to_source = data.get("adapted_rollout_to_source_mse_by_horizon")
    return {
        "condition": name,
        "appearance_shift": data.get("appearance_shift", "medium"),
        "adapter": adapter,
        "probe_rmse_px": data.get("adapted_linear_probe_rmse_pixels"),
        "probe_vs_mean_ratio": data.get("adapted_linear_probe_vs_mean_rmse_ratio"),
        "rollout_vs_persistence_ratio": data.get(
            "adapted_rollout_vs_persistence_mse_ratio"
        ),
        "rollout_mse_mean": maybe_mean(rollout),
        "rollout_mse_h1": rollout[0] if rollout else None,
        "rollout_mse_last": rollout[-1] if rollout else None,
        "paired_latent_mse_after": data.get("paired_latent_mse_after_adapter"),
        "rollout_to_source_mse_mean": maybe_mean(rollout_to_source),
        "delta_l2": data.get("delta_l2"),
        "scale_mean": data.get("scale_mean"),
        "scale_std": data.get("scale_std"),
    }


def main():
    args = parse_args()
    source = load_json(args.source_json)
    medium = load_json(args.medium_json)
    delta = load_json(args.delta_json)
    diagonal = load_json(args.diagonal_json)

    rows = [
        baseline_row("source_baseline", source, "source"),
        baseline_row("medium_unadapted", medium, "medium"),
        adapter_row("medium_delta_pairw_1d0", delta, "constant_delta"),
        adapter_row("medium_diagonal_affine_pairw_0d3", diagonal, "diagonal_affine"),
    ]

    for row in rows:
        rmse = row["probe_rmse_px"]
        rollout = row["rollout_vs_persistence_ratio"]
        row["passes_probe_threshold"] = rmse is not None and rmse <= args.max_rmse
        row["passes_rollout_threshold"] = (
            rollout is not None and rollout < args.max_rollout_ratio
        )
        row["passes_both_thresholds"] = (
            row["passes_probe_threshold"] and row["passes_rollout_threshold"]
        )

    best_probe = min(
        rows,
        key=lambda row: (
            float("inf") if row["probe_rmse_px"] is None else row["probe_rmse_px"]
        ),
    )
    best_rollout = min(
        rows,
        key=lambda row: (
            float("inf")
            if row["rollout_vs_persistence_ratio"] is None
            else row["rollout_vs_persistence_ratio"]
        ),
    )
    passing = [row for row in rows if row["passes_both_thresholds"]]

    summary = {
        "thresholds": {
            "max_rmse": args.max_rmse,
            "max_rollout_ratio": args.max_rollout_ratio,
        },
        "best_probe_condition": best_probe["condition"],
        "best_rollout_condition": best_rollout["condition"],
        "passing_conditions": [row["condition"] for row in passing],
        "rows": rows,
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n")

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
