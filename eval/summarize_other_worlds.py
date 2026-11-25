"""Aggregate OtherWorlds navigation benchmark JSONs into CSV, summary JSON, and LNEE figures."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def setup_lnee_style() -> None:
    """Configure matplotlib parameters for Springer Nature LNEE vector standards."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#222222",
        "grid.color": "#d0d0d0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def summarize_other_worlds(
    results_dir: str = "results/other_worlds",
    fig_dir: str = "paper_figs",
    dpi: int = 600,
) -> Dict[str, Any]:
    pattern = os.path.join(results_dir, "*_*.json")
    files = sorted(
        f for f in glob.glob(pattern)
        if not os.path.basename(f).startswith("benchmark_summary")
    )
    rows: List[Dict[str, Any]] = []
    for fpath in files:
        base = os.path.splitext(os.path.basename(fpath))[0]
        m = re.match(r"^(.+)_(m1_full|m2_static|b1_lod0|b2_lod2|b3_mae)_(rl1|rl2)$", base)
        if not m:
            continue
        world, model_type, task = m.group(1), m.group(2), m.group(3)
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
        rows.append({
            "world": world,
            "model_type": model_type,
            "task": task,
            "success_rate_pct": float(d.get("success_rate_pct", 0.0)),
            "spl": float(d.get("spl", 0.0)),
            "chns": float(d.get("chns", 0.0)),
            "waypoint_completion_pct": float(d.get("waypoint_completion_pct", 0.0)),
            "mean_targets_reached": float(d.get("mean_targets_reached", 0.0)),
            "hazard_collision_rate_pct": float(d.get("hazard_collision_rate_pct", 0.0)),
            "mean_episode_length": float(d.get("mean_episode_length", 0.0)),
            "total_episodes": int(d.get("total_episodes", 0)),
        })

    os.makedirs(results_dir, exist_ok=True)
    summary_json = os.path.join(results_dir, "benchmark_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"runs": rows}, f, indent=2)

    summary_csv = os.path.join(results_dir, "benchmark_summary.csv")
    if rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Generate Springer LNEE grouped bar chart if results exist
    os.makedirs(fig_dir, exist_ok=True)
    pdf_path = os.path.join(fig_dir, "other_worlds_benchmark.pdf")
    png_path = os.path.join(fig_dir, "other_worlds_benchmark.png")

    if rows:
        setup_lnee_style()
        worlds = sorted({r["world"] for r in rows})
        series_keys = sorted({f"{r['model_type']} ({r['task'].upper()})" for r in rows})
        colors = ["#004D40", "#1E88E5", "#D81B60", "#E65100"]
        hatches = ["", "///", "\\\\", "..."]

        fig, (ax_sr, ax_spl, ax_chns) = plt.subplots(1, 3, figsize=(9.6, 3.0), dpi=300)
        x = np.arange(len(worlds))
        w = 0.75 / max(1, len(series_keys))

        for idx, sk in enumerate(series_keys):
            sr_vals, spl_vals, chns_vals = [], [], []
            for w_name in worlds:
                match = [
                    r for r in rows
                    if r["world"] == w_name and f"{r['model_type']} ({r['task'].upper()})" == sk
                ]
                sr_vals.append(match[0]["success_rate_pct"] if match else 0.0)
                spl_vals.append(match[0]["spl"] if match else 0.0)
                chns_vals.append(match[0]["chns"] if match else 0.0)

            offset = (idx - (len(series_keys) - 1) / 2.0) * w
            c = colors[idx % len(colors)]
            h = hatches[idx % len(hatches)]
            ax_sr.bar(x + offset, sr_vals, w, label=sk, color=c, edgecolor="#222222", hatch=h, linewidth=0.7)
            ax_spl.bar(x + offset, spl_vals, w, label=sk, color=c, edgecolor="#222222", hatch=h, linewidth=0.7)
            ax_chns.bar(x + offset, chns_vals, w, label=sk, color=c, edgecolor="#222222", hatch=h, linewidth=0.7)

        ax_sr.set_title("(a) Zero-Shot Success Rate (%)", fontweight="bold")
        ax_sr.set_ylabel("Success Rate (%)")
        ax_sr.set_ylim(0, 105)
        ax_sr.set_xticks(x)
        ax_sr.set_xticklabels([w.replace("_", "\n") for w in worlds], fontsize=7.5)
        ax_sr.grid(axis="y")

        ax_spl.set_title("(b) Path Efficiency (SPL)", fontweight="bold")
        ax_spl.set_ylabel("SPL")
        ax_spl.set_ylim(0, 1.05)
        ax_spl.set_xticks(x)
        ax_spl.set_xticklabels([w.replace("_", "\n") for w in worlds], fontsize=7.5)
        ax_spl.grid(axis="y")

        ax_chns.set_title("(c) CHNS Score (0-100)", fontweight="bold")
        ax_chns.set_ylabel("CHNS")
        ax_chns.set_ylim(0, 105)
        ax_chns.set_xticks(x)
        ax_chns.set_xticklabels([w.replace("_", "\n") for w in worlds], fontsize=7.5)
        ax_chns.grid(axis="y")

        handles, labels = ax_sr.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=len(series_keys), frameon=True)
        plt.tight_layout(rect=[0.0, 0.12, 1.0, 1.0])
        plt.savefig(pdf_path, format="pdf")
        plt.savefig(png_path, format="png", dpi=dpi)
        plt.close(fig)

    return {"runs": len(rows), "json": summary_json, "csv": summary_csv, "pdf": pdf_path, "png": png_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize OtherWorlds benchmark results and generate plots.")
    parser.add_argument("--results_dir", type=str, default="results/other_worlds")
    parser.add_argument("--fig_dir", type=str, default="paper_figs")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    out = summarize_other_worlds(args.results_dir, args.fig_dir, args.dpi)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
