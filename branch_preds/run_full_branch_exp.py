#!/usr/bin/env python3
"""
run_full_branch_experiments.py

Automates gem5 runs for branch predictor experiments:
- Runs multiple predictors on multiple workloads
- Optional warmup (checkpoint) support or simple ROI mode
- Repeats each configuration 'repeats' times, aggregates mean/std
- Extracts many gem5 stats (robust to different stat-key naming)
- Produces CSV summary and plots (matplotlib)

USAGE (example):
python3 run_full_branch_experiments.py \
    --gem5 /home/user/gem5/build/X86/gem5.opt \
    --config /home/user/gem5/configs/example/se.py \
    --outdir results \
    --repeats 3 \
    --mode checkpoint \
    --warmup-insts 2000000 \
    --roi-insts 50000000

Before running: edit WORKLOADS dict below to point to your MiBench qsort/basicmath executables.
"""

import argparse
import math
import os
import re
import subprocess
import time
from statistics import (
    mean,
    stdev,
)

import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# =========== CONFIG ==========
# -----------------------------
# Edit WORKLOADS to your mibench builds (absolute paths recommended)
WORKLOADS = {
    "qsort": {
        "cmd": "/mnt/c/Users/HP/COL718/Assignment1/gem5/mibench/automotive/qsort/qsort_small",
        "options": "/mnt/c/Users/HP/COL718/Assignment1/gem5/mibench/automotive/qsort/input_small.dat",
    },
    "basicmath": {
        "cmd": "/mnt/c/Users/HP/COL718/Assignment1/gem5/mibench/automotive/basicmath/basicmath_small",
        "options": "",
    },
}

# Default predictors and flags (modify if your gem5 accepts different flags)
PREDICTORS = {
    "BiModeBP": "--bp-type=BiModeBP",
    "GShareBP": "--bp-type=GShareBP",
    "LocalBP": "--bp-type=LocalBP",
    "TournamentBP": "--bp-type=TournamentBP",
    # Add Perceptron flags if available
}

# Stat name candidate lists (robustness for different gem5 versions)
STAT_CANDIDATES = {
    "sim_seconds": ["sim_seconds", "sim_sec", "sim_time"],
    "committed_instructions": [
        "system.cpu.committed_instructions",
        "system.cpu.committedInsts",
        "system.cpu.committed_instructions::total",
        "system.cpu.instrs_committed",
        "system.cpu.committed_instructions::total",
    ],
    "ipc": ["system.cpu.ipc", "ipc"],
    "branchPredicted": [
        "system.cpu.branchPredicted",
        "system.cpu.branchPredictions",
        "branchPredicted",
    ],
    "branchMispredicted": [
        "system.cpu.branchMispredicted",
        "system.cpu.branchMispredictions",
        "branchMispredicted",
    ],
    "pipeline_stalls": [
        "system.cpu.pipeline_stalls",
        "system.cpu.stallCycles",
        "system.cpu.stall_cycles",
    ],
    "fetch_bubbles": ["system.cpu.fetch_bubbles", "fetchBubbles"],
    "squash_count": ["system.cpu.squash_count", "system.cpu.squashes"],
    "l1d_accesses": [
        "system.cpu.dcache.overall_accesses::total",
        "system.cpu.dcache.overall_accesses",
        "system.l1d.overall_accesses::total",
        "system.cpu.dcache.accesses",
    ],
    "l1i_accesses": [
        "system.cpu.icache.overall_accesses::total",
        "system.cpu.icache.overall_accesses",
        "system.l1i.overall_accesses::total",
        "system.cpu.icache.accesses",
    ],
    "ROBFullEvents": [
        "system.cpu.ROBFullEvents",
        "system.cpu.rob_full_events",
    ],
    "IQFullEvents": ["system.cpu.IQFullEvents", "system.cpu.iq_full_events"],
    "MispredRecoveryCycles": [
        "system.cpu.branchPred.MispredRecoveryCycles",
        "branchPred.MispredRecoveryCycles",
    ],
    "numCycles": ["system.cpu.numCycles", "numCycles"],
    "execute_cycles": ["system.cpu.execute_cycles", "execCycles"],
}


# -----------------------------
# ========== Helpers ==========
# -----------------------------
def run_process(cmd, stdout_path=None, stderr_path=None):
    """Run subprocess, streaming output to optional files. Returns returncode."""
    stdout_f = open(stdout_path, "w") if stdout_path else subprocess.DEVNULL
    stderr_f = open(stderr_path, "w") if stderr_path else subprocess.DEVNULL
    try:
        return subprocess.run(cmd, stdout=stdout_f, stderr=stderr_f).returncode
    finally:
        if stdout_path:
            stdout_f.close()
        if stderr_path:
            stderr_f.close()


def parse_stats_file(stats_path):
    """Parse stats.txt robustly and return dict of stat_name -> float."""
    stats = {}
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"stats file not found: {stats_path}")
    num_re = re.compile(r"^\s*([^\s]+)\s+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")
    with open(stats_path) as f:
        for line in f:
            m = num_re.match(line)
            if m:
                key = m.group(1).strip()
                val = float(m.group(2))
                stats[key] = val
    return stats


def find_stat(stats_map, candidates):
    """Return first matching stat value from stats_map given candidates list."""
    for c in candidates:
        if c in stats_map:
            return stats_map[c]
    # try fuzzy match by prefix
    for key in stats_map:
        for c in candidates:
            if key.endswith(c) or c.endswith(key) or c in key:
                return stats_map[key]
    return None


def safe_div(a, b):
    try:
        return a / b if b and not math.isclose(b, 0.0) else None
    except Exception:
        return None


# -----------------------------
# ========== Main ============
# -----------------------------
def run_experiment(args):
    os.makedirs(args.outdir, exist_ok=True)
    per_run_records = []

    for workload_name, workload in WORKLOADS.items():
        for predictor_name, predictor_flags in PREDICTORS.items():
            print(
                f"\n=== Workload: {workload_name} | Predictor: {predictor_name} ==="
            )
            run_results = []
            for runno in range(1, args.repeats + 1):
                print(f" Run {runno}/{args.repeats}...")
                run_id = f"{workload_name}_{predictor_name}_run{runno}_{int(time.time())}"
                out_dir = os.path.join(args.outdir, run_id)
                os.makedirs(out_dir, exist_ok=True)

                # Construct common gem5 command parts
                base_cmd = [
                    args.gem5,
                    args.config,
                    "--cpu-type=DerivO3CPU",
                    "--caches",
                    "--l2cache",
                    # f"--rob-size={args.rob_size}",
                    # f"--iq-size={args.iq_size}",
                    # f"--fetch-width={args.fetch_width}",
                    # f"--issue-width={args.issue_width}"
                ]
                # add predictor-specific flags
                base_cmd += predictor_flags.split()

                # set workload cmd/options
                base_cmd += ["--cmd", workload["cmd"]]
                if workload["options"]:
                    base_cmd += ["--options", workload["options"]]

                # Two modes: 'simple' -> just run ROI (no warmup)
                # 'checkpoint' -> run warmup to create checkpoint, then run ROI by restoring
                if args.mode == "simple":
                    # Run single gem5 instance for ROI_insts
                    cmd = base_cmd + [
                        "--maxinsts",
                        str(args.roi_insts),
                        "--outdir",
                        out_dir,
                    ]
                    stdout_log = os.path.join(out_dir, "gem5_stdout.log")
                    stderr_log = os.path.join(out_dir, "gem5_stderr.log")
                    rc = run_process(cmd, stdout_log, stderr_log)
                    if rc != 0:
                        print(
                            f"Warning: gem5 returned code {rc} (see {stderr_log})"
                        )
                    stats_path = os.path.join(out_dir, "stats.txt")
                    if not os.path.exists(stats_path):
                        print(
                            "stats.txt missing - gem5 might have used different outdir. Searching..."
                        )
                        # try look for stats in default m5out
                        alt = os.path.join("m5out", "stats.txt")
                        if os.path.exists(alt):
                            stats_path = alt

                elif args.mode == "checkpoint":
                    # Step 1: warmup run that creates a checkpoint
                    warm_out = os.path.join(out_dir, "warm")
                    os.makedirs(warm_out, exist_ok=True)
                    warm_cmd = base_cmd + [
                        "--maxinsts",
                        str(args.warmup_insts),
                        "--outdir",
                        warm_out,
                        "--take-checkpoint",
                    ]
                    # NOTE: --take-checkpoint is the flag used by many gem5 versions; if your gem5 uses a different flag,
                    # update args.checkpoint_flag accordingly.
                    stdout_log = os.path.join(warm_out, "gem5_warm_stdout.log")
                    stderr_log = os.path.join(warm_out, "gem5_warm_stderr.log")
                    rc1 = run_process(warm_cmd, stdout_log, stderr_log)
                    if rc1 != 0:
                        print(
                            f"Warning: warmup gem5 returned code {rc1}. Check {stderr_log}"
                        )
                    # Step 2: restore checkpoint and run ROI
                    # Common flags in many gem5s: --restore-from-checkpoint <n> and --checkpoint-dir <dir>
                    # We'll try a few common possibilities; user may need to adapt to their gem5 version.
                    restored = False
                    # Attempt 1: restore-from-checkpoint + checkpoint-dir
                    restore_cmd = base_cmd + [
                        "--restore-from-checkpoint",
                        "0",
                        "--checkpoint-dir",
                        warm_out,
                        "--maxinsts",
                        str(args.roi_insts),
                        "--outdir",
                        out_dir,
                    ]
                    stdout_log = os.path.join(
                        out_dir, "gem5_restore_stdout.log"
                    )
                    stderr_log = os.path.join(
                        out_dir, "gem5_restore_stderr.log"
                    )
                    rc2 = run_process(restore_cmd, stdout_log, stderr_log)
                    if rc2 == 0:
                        restored = True
                        stats_path = os.path.join(out_dir, "stats.txt")
                    else:
                        # Attempt 2: --restore-checkpoint flag
                        restore_cmd2 = base_cmd + [
                            "--restore-checkpoint",
                            "0",
                            "--checkpoint-dir",
                            warm_out,
                            "--maxinsts",
                            str(args.roi_insts),
                            "--outdir",
                            out_dir,
                        ]
                        rc3 = run_process(restore_cmd2, stdout_log, stderr_log)
                        if rc3 == 0:
                            restored = True
                            stats_path = os.path.join(out_dir, "stats.txt")
                        else:
                            print(
                                "Warning: Couldn't restore checkpoint automatically. You may need to adapt flags for your gem5 version."
                            )
                            # fall back to running roi directly (not warmed)
                            cmd = base_cmd + [
                                "--maxinsts",
                                str(args.roi_insts),
                                "--outdir",
                                out_dir,
                            ]
                            rc_f = run_process(cmd, stdout_log, stderr_log)
                            stats_path = os.path.join(out_dir, "stats.txt")
                else:
                    raise ValueError(
                        "Unknown mode. Use 'simple' or 'checkpoint'."
                    )

                # Small wait to ensure files written
                time.sleep(0.5)
                if not os.path.exists(stats_path):
                    raise FileNotFoundError(
                        f"stats.txt not found for run: {stats_path}"
                    )

                stats_map = parse_stats_file(stats_path)

                # extract metrics robustly
                sim_seconds = find_stat(
                    stats_map, STAT_CANDIDATES["sim_seconds"]
                )
                committed = find_stat(
                    stats_map, STAT_CANDIDATES["committed_instructions"]
                )
                ipc = find_stat(stats_map, STAT_CANDIDATES["ipc"])
                if ipc is None and committed and sim_seconds:
                    ipc = safe_div(committed, sim_seconds)
                branch_pred = find_stat(
                    stats_map, STAT_CANDIDATES["branchPredicted"]
                )
                branch_misp = find_stat(
                    stats_map, STAT_CANDIDATES["branchMispredicted"]
                )
                mispred_rate = (
                    safe_div(branch_misp, branch_pred) if branch_pred else None
                )
                pipeline_stalls = find_stat(
                    stats_map, STAT_CANDIDATES["pipeline_stalls"]
                )
                fetch_bubbles = find_stat(
                    stats_map, STAT_CANDIDATES["fetch_bubbles"]
                )
                squash_count = find_stat(
                    stats_map, STAT_CANDIDATES["squash_count"]
                )
                l1d = find_stat(stats_map, STAT_CANDIDATES["l1d_accesses"])
                l1i = find_stat(stats_map, STAT_CANDIDATES["l1i_accesses"])
                rob_events = find_stat(
                    stats_map, STAT_CANDIDATES["ROBFullEvents"]
                )
                iq_events = find_stat(
                    stats_map, STAT_CANDIDATES["IQFullEvents"]
                )
                mrecovery = find_stat(
                    stats_map, STAT_CANDIDATES["MispredRecoveryCycles"]
                )
                numCycles = find_stat(stats_map, STAT_CANDIDATES["numCycles"])
                exec_cycles = find_stat(
                    stats_map, STAT_CANDIDATES["execute_cycles"]
                )

                rec = {
                    "workload": workload_name,
                    "predictor": predictor_name,
                    "run": runno,
                    "sim_seconds": sim_seconds,
                    "committed_instructions": committed,
                    "ipc": ipc,
                    "branchPredicted": branch_pred,
                    "branchMispredicted": branch_misp,
                    "misprediction_rate": mispred_rate,
                    "pipeline_stalls": pipeline_stalls,
                    "fetch_bubbles": fetch_bubbles,
                    "squash_count": squash_count,
                    "l1d_accesses": l1d,
                    "l1i_accesses": l1i,
                    "ROBFullEvents": rob_events,
                    "IQFullEvents": iq_events,
                    "MispredRecoveryCycles": mrecovery,
                    "numCycles": numCycles,
                    "execute_cycles": exec_cycles,
                    "stats_path": stats_path,
                    "out_dir": out_dir,
                }
                run_results.append(rec)
                per_run_records.append(rec)
                print(
                    f"  Collected: IPC={ipc}, mispred_rate={mispred_rate}, committed={committed}"
                )

            # compute aggregates for this workload/predictor
            fields_to_agg = [
                "ipc",
                "misprediction_rate",
                "committed_instructions",
                "pipeline_stalls",
                "l1d_accesses",
                "l1i_accesses",
            ]
            agg = {
                "workload": workload_name,
                "predictor": predictor_name,
                "runs": args.repeats,
            }
            for f in fields_to_agg:
                vals = [r[f] for r in run_results if r[f] is not None]
                agg[f"_mean_{f}"] = mean(vals) if vals else None
                agg[f"_std_{f}"] = stdev(vals) if len(vals) > 1 else None
            # also copy some representative fields
            agg["sim_seconds_mean"] = (
                mean(
                    [
                        r["sim_seconds"]
                        for r in run_results
                        if r["sim_seconds"] is not None
                    ]
                )
                if any(r["sim_seconds"] for r in run_results)
                else None
            )
            # write aggregated summary
            summary_csv = os.path.join(
                args.outdir, f"summary_{workload_name}_{predictor_name}.csv"
            )
            pd.DataFrame(run_results).to_csv(summary_csv, index=False)

            print(f" Wrote per-run summary to {summary_csv}")

    # Write global per-run CSV and aggregated CSV
    per_run_df = pd.DataFrame(per_run_records)
    per_run_csv = os.path.join(args.outdir, "per_run_results.csv")
    per_run_df.to_csv(per_run_csv, index=False)
    print(f"\nWrote per-run results to {per_run_csv}")

    # Create aggregated view (group by workload/predictor)
    agg_rows = []
    grouped = per_run_df.groupby(["workload", "predictor"])
    for (wk, pr), group in grouped:
        row = {"workload": wk, "predictor": pr, "runs": len(group)}
        for col in [
            "ipc",
            "misprediction_rate",
            "committed_instructions",
            "pipeline_stalls",
            "l1d_accesses",
            "l1i_accesses",
        ]:
            vals = group[col].dropna().tolist()
            row[f"{col}_mean"] = mean(vals) if vals else None
            row[f"{col}_std"] = stdev(vals) if len(vals) > 1 else None
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(args.outdir, "aggregated_results.csv")
    agg_df.to_csv(agg_csv, index=False)
    print(f"Wrote aggregated results to {agg_csv}")

    # Generate plots per workload
    metrics_to_plot = [
        "ipc_mean",
        "misprediction_rate_mean",
        "pipeline_stalls_mean",
        "l1d_accesses_mean",
        "l1i_accesses_mean",
    ]
    for workload in agg_df["workload"].unique():
        subset = agg_df[agg_df["workload"] == workload]
        predictors = subset["predictor"].tolist()
        for metric in metrics_to_plot:
            vals = subset[metric].tolist()
            plt.figure(figsize=(8, 5))
            # Replace None values with 0 for safe plotting
            vals = [0 if v is None else v for v in vals]
            plt.bar(predictors, vals)
            plt.title(
                f"{metric.replace('_mean','').replace('_',' ').title()} for {workload}"
            )
            plt.xlabel("Predictor")
            plt.ylabel(metric.replace("_mean", ""))
            plt.xticks(rotation=20)
            plt.tight_layout()
            plot_path = os.path.join(args.outdir, f"{workload}_{metric}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot {plot_path}")

    print("\nALL DONE. Results and plots are in:", args.outdir)


# -----------------------------
# ========== CLI =============
# -----------------------------
def cli():
    parser = argparse.ArgumentParser(
        description="Run gem5 experiments for branch predictors and collect stats."
    )
    parser.add_argument(
        "--gem5", required=True, help="Path to gem5.opt binary"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to gem5 config script (e.g., configs/example/se.py)",
    )
    parser.add_argument(
        "--outdir", default="gem5_results", help="Directory to store outputs"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of runs per configuration",
    )
    parser.add_argument(
        "--mode",
        choices=["simple", "checkpoint"],
        default="simple",
        help="Warmup mode: simple(no warmup) or checkpoint (create warmup checkpoint then restore)",
    )
    parser.add_argument(
        "--warmup-insts",
        type=int,
        default=2000000,
        help="Warmup instruction count (used in checkpoint mode)",
    )
    parser.add_argument(
        "--roi-insts", type=int, default=50000000, help="ROI instruction count"
    )
    parser.add_argument(
        "--rob-size", type=int, default=128, help="ROB entries"
    )
    parser.add_argument("--iq-size", type=int, default=64, help="IQ entries")
    parser.add_argument(
        "--fetch-width", type=int, default=4, help="Fetch width"
    )
    parser.add_argument(
        "--issue-width", type=int, default=4, help="Issue width"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    # sanity-check workloads
    for wname, w in WORKLOADS.items():
        if not os.path.exists(w["cmd"]):
            print(f"ERROR: workload executable not found: {w['cmd']}")
            print(
                "Edit WORKLOADS in this script to point to your mibench builds."
            )
            raise SystemExit(1)
    if not os.path.exists(args.gem5):
        print("ERROR: gem5 binary not found:", args.gem5)
        raise SystemExit(1)
    if not os.path.exists(args.config):
        print("ERROR: gem5 config script not found:", args.config)
        raise SystemExit(1)

    run_experiment(args)
