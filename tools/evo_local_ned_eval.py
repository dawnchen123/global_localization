#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import re
from collections import Counter
from pathlib import Path


def finite(values):
    return all(math.isfinite(v) for v in values)


def normalize_quat(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12 or not math.isfinite(n):
        return 0.0, 0.0, 0.0, 1.0
    return qx / n, qy / n, qz / n, qw / n


def normalize_coordinate_mode(mode):
    normalized = (mode or "as_is").strip().lower()
    aliases = {
        "as_is": "as_is",
        "ros_enu": "as_is",
        "enu": "as_is",
        "enu_to_ned": "enu_to_ned",
        "ros_enu_to_ned": "enu_to_ned",
        "enu_flu_to_ned_frd": "enu_flu_to_ned_frd",
        "ros_enu_flu_to_ned_frd": "enu_flu_to_ned_frd",
        "ned": "enu_flu_to_ned_frd",
        "local_ned": "enu_flu_to_ned_frd",
    }
    if normalized not in aliases:
        raise RuntimeError("unsupported coordinate_mode=%s" % mode)
    return aliases[normalized]


def quat_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def convert_coordinate(row, coordinate_mode):
    if coordinate_mode == "as_is":
        return row
    # R_ned_enu maps ROS ENU coordinates into i2Nav local NED:
    # [[0, 1, 0], [1, 0, 0], [0, 0, -1]].  It must left-multiply the
    # world-from-body quaternion just like the position transform.
    q_ned_enu = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    output_quaternion = quat_multiply(q_ned_enu, row[4:8])
    if coordinate_mode == "enu_flu_to_ned_frd":
        # The state publisher uses ROS base_link (FLU), whereas i2Nav ground
        # truth uses the INS body frame (FRD).  Change the body basis too so
        # the exported quaternion and position describe the same local-NED pose.
        output_quaternion = quat_multiply(output_quaternion, (1.0, 0.0, 0.0, 0.0))
    qx, qy, qz, qw = output_quaternion
    qx, qy, qz, qw = normalize_quat(qx, qy, qz, qw)
    return (row[0], row[2], row[1], -row[3], qx, qy, qz, qw)


def read_tum8(path):
    rows = []
    bad = 0
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 8:
                bad += 1
                continue
            try:
                vals = [float(x) for x in parts[:8]]
            except ValueError:
                bad += 1
                continue
            if not finite(vals):
                bad += 1
                continue
            qx, qy, qz, qw = normalize_quat(vals[4], vals[5], vals[6], vals[7])
            rows.append((vals[0], vals[1], vals[2], vals[3], qx, qy, qz, qw))
    if not rows:
        raise RuntimeError("no valid 8-column trajectory rows in %s" % path)
    return rows, bad


def filter_cluster(rows, mode, width):
    if mode in ("all", "none", ""):
        return rows, {"mode": "all", "kept": len(rows), "total": len(rows)}
    keys = [int(math.floor(r[0] / width)) for r in rows]
    counts = Counter(keys)
    if mode == "auto":
        key, count = counts.most_common(1)[0]
    else:
        key = int(mode)
        count = counts.get(key, 0)
    filtered = [r for r, k in zip(rows, keys) if k == key]
    if not filtered:
        raise RuntimeError("time cluster %s kept zero rows" % mode)
    return filtered, {
        "mode": mode,
        "cluster_key": key,
        "cluster_width": width,
        "kept": len(filtered),
        "total": len(rows),
        "all_clusters": counts.most_common(),
    }


def sort_dedupe(rows):
    rows = sorted(rows, key=lambda r: r[0])
    out = []
    last_t = None
    dropped = 0
    for r in rows:
        if last_t is not None and abs(r[0] - last_t) < 1e-9:
            dropped += 1
            continue
        out.append(r)
        last_t = r[0]
    return out, dropped


def normalize_time(rows, mode, time_shift, time_reference=None):
    if mode == "absolute":
        return [(r[0] + time_shift,) + r[1:] for r in rows], rows[0][0]
    if mode != "relative":
        raise RuntimeError("unsupported time_mode=%s" % mode)
    t0 = rows[0][0] if time_reference is None else time_reference
    return [(r[0] - t0 + time_shift,) + r[1:] for r in rows], t0


def crop_time(rows, t_start, t_end):
    out = rows
    if t_start is not None:
        out = [r for r in out if r[0] >= t_start]
    if t_end is not None:
        out = [r for r in out if r[0] <= t_end]
    if not out:
        raise RuntimeError("time crop kept zero rows")
    return out


def write_tum(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write("%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % r)


def convert_file(src, dst, time_mode, cluster_mode, cluster_width, time_shift, t_start, t_end,
                 coordinate_mode="as_is", time_reference=None):
    rows, bad = read_tum8(src)
    coordinate_mode = normalize_coordinate_mode(coordinate_mode)
    rows = [convert_coordinate(row, coordinate_mode) for row in rows]
    rows, cluster_info = filter_cluster(rows, cluster_mode, cluster_width)
    rows, duplicates = sort_dedupe(rows)
    rows, time_origin = normalize_time(rows, time_mode, time_shift, time_reference)
    rows = crop_time(rows, t_start, t_end)
    write_tum(dst, rows)
    return {
        "src": str(src),
        "dst": str(dst),
        "valid_rows": len(rows),
        "bad_rows": bad,
        "dropped_duplicate_timestamps": duplicates,
        "time_origin": time_origin,
        "time_reference": time_reference,
        "time_shift": time_shift,
        "coordinate_mode": coordinate_mode,
        "time_min": rows[0][0],
        "time_max": rows[-1][0],
        "time_span": rows[-1][0] - rows[0][0],
        "cluster": cluster_info,
    }


def run_command(cmd, log_path, env):
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout)
    return proc.returncode, proc.stdout


def parse_evo_statistics(output):
    statistics = {}
    pattern = re.compile(
        r"^\s*(max|mean|median|min|rmse|sse|std)\s+"
        r"([-+0-9.eE]+)\s*$"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            statistics[match.group(1)] = float(match.group(2))
    return statistics


def quat_to_yaw(qx, qy, qz, qw):
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def quat_to_rotation(qx, qy, qz, qw):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for component metrics") from exc
    qx, qy, qz, qw = normalize_quat(qx, qy, qz, qw)
    return np.asarray([
        [1.0 - 2.0 * (qy * qy + qz * qz),
         2.0 * (qx * qy - qz * qw),
         2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw),
         1.0 - 2.0 * (qx * qx + qz * qz),
         2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw),
         2.0 * (qy * qz + qx * qw),
         1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=float)


def yaw_of_rotation(rotation):
    return math.atan2(rotation[1, 0], rotation[0, 0])


def wrap_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def associate_rows(gt_rows, est_rows, max_dt):
    pairs = []
    i = 0
    for e in est_rows:
        while i + 1 < len(gt_rows) and gt_rows[i + 1][0] <= e[0]:
            i += 1
        best = None
        for k in (i, i + 1):
            if 0 <= k < len(gt_rows):
                dt = abs(gt_rows[k][0] - e[0])
                if best is None or dt < best[0]:
                    best = (dt, gt_rows[k])
        if best is not None and best[0] <= max_dt:
            pairs.append((best[1], e, best[0]))
    return pairs


def rigid_alignment(gt_xyz, est_xyz, align_mode):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for component metrics") from exc

    gt = np.asarray(gt_xyz, dtype=float)
    est = np.asarray(est_xyz, dtype=float)
    R = np.eye(3)
    t = np.zeros(3)
    if len(gt) == 0:
        return R, t
    if align_mode == "origin":
        t = gt[0] - est[0]
        return R, t
    if align_mode == "none":
        return R, t
    if align_mode != "se3":
        raise RuntimeError("unsupported align=%s" % align_mode)

    gt_mean = gt.mean(axis=0)
    est_mean = est.mean(axis=0)
    X = est - est_mean
    Y = gt - gt_mean
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = gt_mean - R @ est_mean
    return R, t


def write_component_metrics(gt_tum, est_tum, args, name):
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy is required for component metrics"}

    gt_rows, _ = read_tum8(gt_tum)
    est_rows, _ = read_tum8(est_tum)
    pairs = associate_rows(gt_rows, est_rows, args.max_time_diff)
    if len(pairs) < 3:
        return {"error": "too few associated poses", "n": len(pairs)}

    gt_xyz = [(g[1], g[2], g[3]) for g, _, _ in pairs]
    est_xyz = [(e[1], e[2], e[3]) for _, e, _ in pairs]
    R, t = rigid_alignment(gt_xyz, est_xyz, args.align)
    rows = []
    err_x = []
    err_y = []
    err_z = []
    err_xy = []
    err_3d = []
    yaw_err_deg = []
    rotation_err_deg = []
    for g, e, dt in pairs:
        est_aligned = R @ np.asarray([e[1], e[2], e[3]], dtype=float) + t
        dx = float(est_aligned[0] - g[1])
        dy = float(est_aligned[1] - g[2])
        dz = float(est_aligned[2] - g[3])
        exy = math.hypot(dx, dy)
        e3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        gt_rotation = quat_to_rotation(g[4], g[5], g[6], g[7])
        aligned_est_rotation = R @ quat_to_rotation(e[4], e[5], e[6], e[7])
        gyaw = yaw_of_rotation(gt_rotation)
        eyaw = yaw_of_rotation(aligned_est_rotation)
        dyaw_deg = wrap_angle(eyaw - gyaw) * 180.0 / math.pi
        rotation_delta = gt_rotation.T @ aligned_est_rotation
        rotation_cosine = max(-1.0, min(1.0, (float(np.trace(rotation_delta)) - 1.0) * 0.5))
        rotation_error_deg = math.acos(rotation_cosine) * 180.0 / math.pi
        err_x.append(dx)
        err_y.append(dy)
        err_z.append(dz)
        err_xy.append(exy)
        err_3d.append(e3d)
        yaw_err_deg.append(dyaw_deg)
        rotation_err_deg.append(rotation_error_deg)
        rows.append((
            g[0], g[1], g[2], g[3], e[1], e[2], e[3],
            float(est_aligned[0]), float(est_aligned[1]), float(est_aligned[2]),
            dx, dy, dz, exy, e3d, dyaw_deg, rotation_error_deg, dt,
        ))

    csv_path = args.out_dir / ("components_%s.csv" % name)
    with open(csv_path, "w") as f:
        f.write(
            "stamp,gt_x,gt_y,gt_z,est_x,est_y,est_z,aligned_x,aligned_y,aligned_z,"
            "err_x,err_y,err_z,err_xy,err_3d,yaw_err_deg,rotation_err_deg,dt\n"
        )
        for r in rows:
            f.write(",".join("%.9f" % v for v in r) + "\n")

    def rmse(values):
        return math.sqrt(sum(v * v for v in values) / max(1, len(values)))

    return {
        "csv": str(csv_path),
        "n": len(rows),
        "align": args.align,
        "rmse_x": rmse(err_x),
        "rmse_y": rmse(err_y),
        "rmse_z": rmse(err_z),
        "rmse_xy": rmse(err_xy),
        "rmse_3d": rmse(err_3d),
        "mean_abs_z": sum(abs(v) for v in err_z) / max(1, len(err_z)),
        "max_abs_z": max(abs(v) for v in err_z),
        "rmse_yaw_deg": rmse(yaw_err_deg),
        "rmse_rotation_deg": rmse(rotation_err_deg),
    }


def configure_evo(evo_bin, evo_home):
    evo_home.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HOME"] = str(evo_home)
    env["MPLBACKEND"] = "Agg"
    # Keep evo's NumPy/BLAS subprocesses deterministic and avoid native
    # thread-pool crashes when plots and alignment run in constrained hosts.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    cmd = [str(evo_bin / "evo_config"), "set", "plot_backend", "Agg"]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    return env


def build_metric_cmd(evo_bin, metric, gt_tum, est_tum, args, name):
    exe = evo_bin / ("evo_" + metric)
    if metric == "ape":
        cmd = [
            str(exe),
            "tum",
            str(gt_tum),
            str(est_tum),
            "-r",
            args.pose_relation,
            "--t_max_diff",
            str(args.max_time_diff),
            "--plot_mode",
            args.plot_mode,
            "--save_results",
            str(args.out_dir / ("ape_%s.zip" % name)),
            "--save_plot",
            str(args.out_dir / ("ape_%s.pdf" % name)),
            "--no_warnings",
        ]
    else:
        cmd = [
            str(exe),
            "tum",
            str(gt_tum),
            str(est_tum),
            "-r",
            args.pose_relation,
            "--delta",
            str(args.rpe_delta),
            "--delta_unit",
            args.rpe_delta_unit,
            "--t_max_diff",
            str(args.max_time_diff),
            "--plot_mode",
            args.plot_mode,
            "--save_results",
            str(args.out_dir / ("rpe_%s.zip" % name)),
            "--save_plot",
            str(args.out_dir / ("rpe_%s.pdf" % name)),
            "--no_warnings",
        ]
    if args.align == "se3":
        cmd.append("--align")
    elif args.align == "origin":
        cmd.append("--align_origin")
    elif args.align != "none":
        raise RuntimeError("unsupported align=%s" % args.align)
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate local NED 8-column trajectories with evo.")
    parser.add_argument("--dataset_dir", type=Path,
                        default=Path("/home/dawn/document/phd_exp/street02"))
    parser.add_argument("--gt", type=Path, default=None,
                        help="GroundTruth file. Default: dataset_dir/street02_trajectory.csv")
    parser.add_argument("--est", type=Path, nargs="+", default=None,
                        help="Estimated trajectory files. Default: fast_livo2_local_ned.csv fused_local_ned.csv")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Names for estimated trajectories.")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--evo_bin", type=Path,
                        default=Path("/home/dawn/document/phd_exp/evo_env/bin"))
    parser.add_argument("--evo_home", type=Path,
                        default=Path("/home/dawn/document/phd_exp/evo_home"))
    parser.add_argument("--time_mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--gt_time_shift", type=float, default=0.0,
                        help="Seconds added to converted GroundTruth timestamps after time normalization.")
    parser.add_argument("--est_time_shift", type=float, default=0.0,
                        help="Seconds added to converted estimate timestamps after time normalization.")
    parser.add_argument("--est_time_shifts", type=float, nargs="+", default=None,
                        help="Per-estimate timestamp shifts. Overrides --est_time_shift.")
    parser.add_argument("--est_time_reference", type=float, default=None,
                        help="Shared absolute estimate timestamp used as the relative-time origin. "
                             "Use this when trajectories start after different initialization delays.")
    parser.add_argument("--gt_time_cluster", default="all",
                        help="all, auto, or integer floor(timestamp / cluster_width)")
    parser.add_argument("--est_time_cluster", default="auto",
                        help="all, auto, or integer floor(timestamp / cluster_width)")
    parser.add_argument("--gt_coordinate_mode", default="as_is",
                        help="as_is, enu_to_ned, or enu_flu_to_ned_frd.")
    parser.add_argument("--est_coordinate_modes", nargs="+", default=None,
                        help="Per-estimate coordinate modes: as_is, enu_to_ned, or enu_flu_to_ned_frd.")
    parser.add_argument("--cluster_width", type=float, default=1000000.0)
    parser.add_argument("--t_start", type=float, default=None)
    parser.add_argument("--t_end", type=float, default=None)
    parser.add_argument("--max_time_diff", type=float, default=0.05)
    parser.add_argument("--align", choices=["se3", "origin", "none"], default="se3")
    parser.add_argument("--pose_relation", default="trans_part",
                        choices=["full", "trans_part", "rot_part", "angle_deg", "angle_rad", "point_distance"])
    parser.add_argument("--plot_mode", default="xy",
                        choices=["xy", "xz", "yx", "yz", "zx", "zy", "xyz"])
    parser.add_argument("--run_rpe", action="store_true")
    parser.add_argument("--rpe_delta", type=float, default=1.0)
    parser.add_argument("--rpe_delta_unit", default="f", choices=["f", "d", "r", "m"])
    parser.add_argument("--skip_component_metrics", action="store_true",
                        help="Do not write aligned XY/Z/yaw component error CSV files.")
    args = parser.parse_args()

    args.dataset_dir = args.dataset_dir.resolve()
    if args.gt is None:
        args.gt = args.dataset_dir / "street02_trajectory.csv"
    if args.est is None:
        args.est = [
            args.dataset_dir / "fast_livo2_local_ned.csv",
            args.dataset_dir / "fused_local_ned.csv",
        ]
    if args.names is None:
        args.names = [p.stem for p in args.est]
    if len(args.names) != len(args.est):
        raise RuntimeError("--names length must match --est length")
    if args.est_time_shifts is None:
        args.est_time_shifts = [args.est_time_shift] * len(args.est)
    elif len(args.est_time_shifts) == 1 and len(args.est) > 1:
        args.est_time_shifts *= len(args.est)
    elif len(args.est_time_shifts) != len(args.est):
        raise RuntimeError("--est_time_shifts length must be one or match --est")
    if args.est_coordinate_modes is None:
        args.est_coordinate_modes = ["as_is"] * len(args.est)
    elif len(args.est_coordinate_modes) == 1 and len(args.est) > 1:
        args.est_coordinate_modes *= len(args.est)
    elif len(args.est_coordinate_modes) != len(args.est):
        raise RuntimeError("--est_coordinate_modes length must be one or match --est")
    args.gt_coordinate_mode = normalize_coordinate_mode(args.gt_coordinate_mode)
    args.est_coordinate_modes = [normalize_coordinate_mode(mode)
                                 for mode in args.est_coordinate_modes]
    if args.out_dir is None:
        args.out_dir = args.dataset_dir / "evo_eval"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    converted_dir = args.out_dir / "tum"
    gt_tum = converted_dir / "groundtruth.tum"
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "time_mode": args.time_mode,
        "gt_time_shift": args.gt_time_shift,
        "est_time_shift": args.est_time_shift,
        "est_time_shifts": args.est_time_shifts,
        "est_time_reference": args.est_time_reference,
        "align": args.align,
        "max_time_diff": args.max_time_diff,
        "rpe_delta": args.rpe_delta,
        "rpe_delta_unit": args.rpe_delta_unit,
        "gt": convert_file(
        args.gt, gt_tum, args.time_mode, args.gt_time_cluster,
            args.cluster_width, args.gt_time_shift, args.t_start, args.t_end,
            args.gt_coordinate_mode),
        "estimates": {},
        "commands": {},
        "component_metrics": {},
        "metrics": {},
    }

    env = configure_evo(args.evo_bin, args.evo_home)

    for name, est, time_shift, coordinate_mode in zip(
            args.names, args.est, args.est_time_shifts, args.est_coordinate_modes):
        est_tum = converted_dir / ("%s.tum" % name)
        summary["estimates"][name] = convert_file(
            est, est_tum, args.time_mode, args.est_time_cluster,
            args.cluster_width, time_shift, args.t_start, args.t_end,
            coordinate_mode, args.est_time_reference)

        if not args.skip_component_metrics:
            summary["component_metrics"][name] = write_component_metrics(
                gt_tum, est_tum, args, name)

        ape_cmd = build_metric_cmd(args.evo_bin, "ape", gt_tum, est_tum, args, name)
        code, out = run_command(ape_cmd, args.out_dir / ("ape_%s.log" % name), env)
        summary["commands"]["ape_" + name] = {
            "cmd": ape_cmd,
            "returncode": code,
            "log": str(args.out_dir / ("ape_%s.log" % name)),
        }
        summary["metrics"].setdefault(name, {})["ape"] = parse_evo_statistics(out)
        print("\n[APE %s] returncode=%d" % (name, code))
        print(out)

        if args.run_rpe:
            rpe_cmd = build_metric_cmd(args.evo_bin, "rpe", gt_tum, est_tum, args, name)
            code, out = run_command(rpe_cmd, args.out_dir / ("rpe_%s.log" % name), env)
            summary["commands"]["rpe_" + name] = {
                "cmd": rpe_cmd,
                "returncode": code,
                "log": str(args.out_dir / ("rpe_%s.log" % name)),
            }
            summary["metrics"].setdefault(name, {})["rpe"] = parse_evo_statistics(out)
            print("\n[RPE %s] returncode=%d" % (name, code))
            print(out)

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nWrote summary: %s" % summary_path)
    print("Converted TUM files: %s" % converted_dir)
    print("Result directory: %s" % args.out_dir)


if __name__ == "__main__":
    main()
