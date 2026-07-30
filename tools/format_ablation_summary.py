#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def metric(mapping, *keys):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def text(value):
    return "-" if value is None else "%.6f" % float(value)


def main():
    parser = argparse.ArgumentParser(description="Format hybrid ablation metrics.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    names = list(summary.get("estimates", {}).keys())
    rows = []
    for name in names:
        components = summary.get("component_metrics", {}).get(name, {})
        rows.append({
            "mode": name,
            "ape_rmse_m": metric(summary, "metrics", name, "ape", "rmse"),
            "rpe_1m_rmse_m": metric(summary, "metrics", name, "rpe", "rmse"),
            "yaw_rmse_deg": components.get("rmse_yaw_deg"),
            "z_rmse_m": components.get("rmse_z"),
            "z_mean_abs_m": components.get("mean_abs_z"),
            "associated_poses": components.get("n"),
        })

    print("| mode | APE RMSE (m) | RPE 1m RMSE (m) | yaw RMSE (deg) | Z RMSE (m) | Z MAE (m) | poses |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print("| {mode} | {ape} | {rpe} | {yaw} | {z} | {z_mae} | {poses} |".format(
            mode=row["mode"], ape=text(row["ape_rmse_m"]),
            rpe=text(row["rpe_1m_rmse_m"]), yaw=text(row["yaw_rmse_deg"]),
            z=text(row["z_rmse_m"]), z_mae=text(row["z_mean_abs_m"]),
            poses=row["associated_poses"] if row["associated_poses"] is not None else "-"))

    csv_path = args.csv or args.summary.parent / "ablation_metrics.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else ["mode"])
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote comparison CSV: %s" % csv_path)


if __name__ == "__main__":
    main()
