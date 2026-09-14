#!/usr/bin/env python3
"""Select and plot abrupt/calm wind samples from Sdwpfh1.

The sample IDs follow the rolling-forecast test split used by
``config/rolling_forecast_config.json``.  With the defaults in this script,
sample 0 starts at ``int(0.8 * len(data))`` and subsequent samples advance by
one hour.

Examples
--------
First rank abrupt and calm candidates::

    python scripts/w_future/TempXer/plot_wind_samples.py select

Then draw the complete history/future windows for chosen sample IDs::

    python scripts/w_future/TempXer/plot_wind_samples.py plot \
        --sample-ids 120 2048
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ts_benchmark.data.utils import read_data


DEFAULT_DATA = REPO_ROOT / "dataset/forecasting/Sdwpfh1.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "result/Sdwpfh1/TempXer/sample_visualization"

# Variable roles follow Appendix A.1 of the GCGNet paper:
# https://arxiv.org/html/2603.08032#A1.SS1
# Definitions and units follow the original SDWPF data descriptor:
# https://doi.org/10.1038/s41597-024-03427-5
# RelH is stored as a fraction in this local file (observed range 0--1).
VARIABLE_INFO = {
    "T2m": ("2 m air temperature from ERA5", "°C"),
    "Sp": ("Surface pressure from ERA5", "Pa"),
    "RelH": ("Relative humidity derived from ERA5", "fraction"),
    "Wspd_w": ("10 m wind speed from ERA5", "m/s"),
    "Wdir_w": ("10 m wind direction from ERA5", "°"),
    "Tp": ("Total precipitation from ERA5", "m"),
    "Patv": ("Active power produced by the wind turbine", "kW"),
}


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seq-len", type=int, default=720)
    parser.add_argument("--horizon", type=int, default=360)
    parser.add_argument(
        "--tv-ratio",
        type=float,
        default=0.8,
        help="Train/validation fraction used by rolling_forecast_config.json.",
    )
    parser.add_argument(
        "--rolling-stride",
        type=int,
        default=1,
        help="Distance between consecutive rolling samples, in data rows.",
    )
    parser.add_argument("--wind-column", default="Wspd_w")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank and visualize wind-change samples using benchmark sample IDs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser(
        "select", help="Rank abrupt and calm samples and draw a wind-speed gallery."
    )
    add_common_arguments(select_parser)
    select_parser.add_argument("--top-k", type=int, default=3)
    select_parser.add_argument(
        "--min-separation",
        type=int,
        default=None,
        help="Minimum separation between selected forecast starts; default=horizon.",
    )

    plot_parser = subparsers.add_parser(
        "plot", help="Plot all variables for one or more benchmark sample IDs."
    )
    add_common_arguments(plot_parser)
    plot_parser.add_argument("--sample-ids", type=int, nargs="+", required=True)
    plot_parser.add_argument(
        "--format", choices=("png", "pdf", "both"), default="both"
    )

    return parser.parse_args()


def validate_inputs(args: argparse.Namespace, data: pd.DataFrame) -> None:
    if not 0 < args.tv_ratio < 1:
        raise ValueError("--tv-ratio must be between 0 and 1.")
    if args.seq_len <= 0 or args.horizon <= 1 or args.rolling_stride <= 0:
        raise ValueError("seq-len, horizon and rolling-stride must be positive.")
    if args.wind_column not in data.columns:
        raise KeyError(
            f"Wind column {args.wind_column!r} is absent; columns={list(data.columns)}"
        )
    train_length = int(args.tv_ratio * len(data))
    if train_length < args.seq_len:
        raise ValueError("The first test sample has insufficient history.")
    if len(data) - train_length < args.horizon:
        raise ValueError("The test split is shorter than the forecast horizon.")


def rolling_starts(
    data_len: int, tv_ratio: float, horizon: int, stride: int
) -> np.ndarray:
    """Reproduce RollingForecast._get_index for the test split."""
    train_length = int(tv_ratio * data_len)
    starts = list(range(train_length, data_len - horizon + 1, stride))
    final_start = data_len - horizon
    if (data_len - train_length - horizon) % stride != 0:
        starts.append(final_start)
    return np.asarray(starts, dtype=np.int64)


def score_samples(
    data: pd.DataFrame,
    wind_column: str,
    starts: np.ndarray,
    horizon: int,
) -> pd.DataFrame:
    """Score wind changes inside the future window consumed by SPOT."""
    wind = data[wind_column].to_numpy(dtype=np.float64)
    windows = sliding_window_view(wind, horizon)[starts]
    abs_steps = np.abs(np.diff(windows, axis=1))
    jump_offsets = np.argmax(abs_steps, axis=1) + 1
    jump_rows = starts + jump_offsets

    scores = pd.DataFrame(
        {
            "sample_id": np.arange(len(starts), dtype=np.int64),
            "forecast_start_row": starts,
            "forecast_end_row": starts + horizon - 1,
            "forecast_start_time": data.index[starts].astype(str),
            "forecast_end_time": data.index[starts + horizon - 1].astype(str),
            "largest_jump_time": data.index[jump_rows].astype(str),
            "max_abs_wind_step": np.max(abs_steps, axis=1),
            "p95_abs_wind_step": np.percentile(abs_steps, 95, axis=1),
            "mean_abs_wind_step": np.mean(abs_steps, axis=1),
            "wind_std": np.std(windows, axis=1),
            "wind_range": np.ptp(windows, axis=1),
        }
    )
    return scores


def select_separated(
    ranked: pd.DataFrame, top_k: int, min_separation: int
) -> pd.DataFrame:
    selected_rows = []
    selected_starts: list[int] = []
    for _, row in ranked.iterrows():
        start = int(row["forecast_start_row"])
        if all(abs(start - previous) >= min_separation for previous in selected_starts):
            selected_rows.append(row)
            selected_starts.append(start)
            if len(selected_rows) == top_k:
                break
    if not selected_rows:
        return ranked.iloc[:0].copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def choose_candidates(
    scores: pd.DataFrame, top_k: int, min_separation: int
) -> pd.DataFrame:
    abrupt_ranked = scores.sort_values(
        ["max_abs_wind_step", "p95_abs_wind_step", "wind_std"],
        ascending=[False, False, False],
    )
    calm_ranked = scores.sort_values(
        ["max_abs_wind_step", "wind_std", "p95_abs_wind_step"],
        ascending=[True, True, True],
    )
    abrupt = select_separated(abrupt_ranked, top_k, min_separation)
    # Do not present two heavily overlapping windows as opposite regimes.
    abrupt_starts = abrupt["forecast_start_row"].to_numpy(dtype=np.int64)
    calm_ranked = calm_ranked[
        calm_ranked["forecast_start_row"].map(
            lambda start: np.all(np.abs(abrupt_starts - int(start)) >= min_separation)
        )
    ]
    calm = select_separated(calm_ranked, top_k, min_separation)
    abrupt.insert(0, "category", "abrupt")
    calm.insert(0, "category", "calm")
    abrupt.insert(1, "category_rank", np.arange(1, len(abrupt) + 1))
    calm.insert(1, "category_rank", np.arange(1, len(calm) + 1))
    return pd.concat([abrupt, calm], ignore_index=True)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def plot_candidate_gallery(
    data: pd.DataFrame,
    candidates: pd.DataFrame,
    wind_column: str,
    horizon: int,
    output_path: Path,
) -> None:
    abrupt = candidates[candidates["category"] == "abrupt"].reset_index(drop=True)
    calm = candidates[candidates["category"] == "calm"].reset_index(drop=True)
    rows = max(len(abrupt), len(calm))
    fig, axes = plt.subplots(rows, 2, figsize=(12, max(3.0, 2.25 * rows)), squeeze=False)
    groups = (
        ("Abrupt candidates", abrupt, "#D55E00"),
        ("Relatively calm candidates", calm, "#0072B2"),
    )

    for column_index, (group_title, group, color) in enumerate(groups):
        for row_index in range(rows):
            ax = axes[row_index, column_index]
            if row_index >= len(group):
                ax.set_visible(False)
                continue
            item = group.iloc[row_index]
            start = int(item["forecast_start_row"])
            end = start + horizon
            ax.plot(data.index[start:end], data[wind_column].iloc[start:end], color=color, linewidth=1.1)
            jump_time = pd.Timestamp(item["largest_jump_time"])
            ax.axvline(jump_time, color="#000000", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.set_title(
                f"ID {int(item['sample_id'])} | start {item['forecast_start_time']} | "
                rf"max $|\Delta w|$={item['max_abs_wind_step']:.3f} m/s",
                loc="left",
            )
            ax.set_ylabel("Wind speed (m/s)")
            ax.grid(axis="y", color="0.9", linewidth=0.6)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            if row_index == 0:
                ax.text(0.5, 1.30, group_title, transform=ax.transAxes, ha="center", fontweight="bold")
    fig.suptitle(
        f"Rolling-test sample candidates ({wind_column}: {VARIABLE_INFO[wind_column][0]})\n"
        "Dashed line: largest one-step wind-speed change in the future window",
        y=1.01,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def variable_label(name: str) -> str:
    meaning, unit = VARIABLE_INFO.get(name, ("Dataset variable", "unit not specified"))
    return f"{name} ({unit})\n{meaning}"


def plot_full_sample(
    data: pd.DataFrame,
    sample_id: int,
    start: int,
    seq_len: int,
    horizon: int,
    wind_column: str,
    output_dir: Path,
    formats: Iterable[str],
) -> None:
    history_start = start - seq_len
    future_end = start + horizon
    window = data.iloc[history_start:future_end]
    columns = list(data.columns)
    ncols = 2
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.0 * nrows), sharex=True, squeeze=False)
    axes_flat = axes.ravel()

    wind_future = data[wind_column].iloc[start:future_end].to_numpy(dtype=float)
    jump_offset = int(np.argmax(np.abs(np.diff(wind_future)))) + 1
    jump_row = start + jump_offset
    max_jump = float(abs(wind_future[jump_offset] - wind_future[jump_offset - 1]))

    for ax, column in zip(axes_flat, columns):
        color = "#D55E00" if column == wind_column else "#0072B2"
        ax.plot(window.index, window[column], color=color, linewidth=1.0)
        ax.axvspan(data.index[start], data.index[future_end - 1], color="#56B4E9", alpha=0.12)
        ax.axvline(data.index[start], color="#000000", linestyle="--", linewidth=1.0)
        if column == wind_column:
            ax.axvline(data.index[jump_row], color="#D55E00", linestyle=":", linewidth=1.2)
            ax.scatter(data.index[jump_row], data[column].iloc[jump_row], color="#D55E00", s=22, zorder=3)
        ax.set_ylabel(variable_label(column))
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    for ax in axes_flat[len(columns) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"Sample ID {sample_id} | history rows [{history_start}, {start - 1}] | "
        f"future rows [{start}, {future_end - 1}]\n"
        f"forecast start: {data.index[start]} | largest future {wind_column} step: "
        f"{max_jump:.3f} m/s at {data.index[jump_row]}\n"
        "Blue shading = future/forecast window; black dashed line = forecast start; "
        "orange dotted line = largest wind-speed step",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    stem = output_dir / f"sample_{sample_id:05d}"
    for extension in formats:
        fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def run_select(args: argparse.Namespace, data: pd.DataFrame) -> None:
    starts = rolling_starts(len(data), args.tv_ratio, args.horizon, args.rolling_stride)
    scores = score_samples(data, args.wind_column, starts, args.horizon)
    min_separation = args.horizon if args.min_separation is None else args.min_separation
    candidates = choose_candidates(scores, args.top_k, min_separation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "sample_scores.csv"
    candidates_path = args.output_dir / "sample_candidates.csv"
    variables_path = args.output_dir / "variable_dictionary.csv"
    gallery_path = args.output_dir / "sample_candidates.png"
    scores.to_csv(scores_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    variable_rows = []
    for name in data.columns:
        meaning, unit = VARIABLE_INFO.get(name, ("Dataset variable", "unit not specified"))
        variable_rows.append(
            {
                "variable": name,
                "role": "endogenous target" if name == "Patv" else "exogenous variable",
                "meaning": meaning,
                "unit": unit,
            }
        )
    pd.DataFrame(variable_rows).to_csv(variables_path, index=False)
    plot_candidate_gallery(data, candidates, args.wind_column, args.horizon, gallery_path)

    print(f"Total rolling-test samples: {len(scores)} (IDs 0..{len(scores) - 1})")
    print(f"All scores: {scores_path}")
    print(f"Candidates: {candidates_path}")
    print(f"Variable dictionary: {variables_path}")
    print(f"Gallery: {gallery_path}")
    print()
    print(
        candidates[
            [
                "category",
                "category_rank",
                "sample_id",
                "forecast_start_time",
                "max_abs_wind_step",
                "wind_std",
            ]
        ].to_string(index=False)
    )


def run_plot(args: argparse.Namespace, data: pd.DataFrame) -> None:
    starts = rolling_starts(len(data), args.tv_ratio, args.horizon, args.rolling_stride)
    invalid_ids = [sample_id for sample_id in args.sample_ids if sample_id < 0 or sample_id >= len(starts)]
    if invalid_ids:
        raise IndexError(
            f"Invalid sample IDs {invalid_ids}; valid range is 0..{len(starts) - 1}."
        )
    formats = ("png", "pdf") if args.format == "both" else (args.format,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in args.sample_ids:
        plot_full_sample(
            data=data,
            sample_id=sample_id,
            start=int(starts[sample_id]),
            seq_len=args.seq_len,
            horizon=args.horizon,
            wind_column=args.wind_column,
            output_dir=args.output_dir,
            formats=formats,
        )
        print(f"Plotted sample ID {sample_id}: {args.output_dir / f'sample_{sample_id:05d}'}")


def main() -> None:
    args = parse_args()
    configure_plot_style()
    data = read_data(str(args.data))
    validate_inputs(args, data)
    if args.command == "select":
        run_select(args, data)
    else:
        run_plot(args, data)


if __name__ == "__main__":
    main()
