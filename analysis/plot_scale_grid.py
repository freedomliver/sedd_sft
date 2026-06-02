#!/usr/bin/env python3
"""Plot SEDD SFT scale-grid and hyperparameter search summaries.

The data here is intentionally a compact, verified snapshot from
CURRENT_PROJECT_RECORD.md.  Add new rows when a remote run finishes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass(frozen=True)
class EvalRow:
    group: str
    run: str
    config: str
    train_size: int
    valid_size: int
    step: int
    train_eval_n: int
    valid_eval_n: int
    train_exact: int
    valid_exact: int
    train_no_eos: int
    valid_no_eos: int
    train_loss: float | None = None
    valid_loss: float | None = None


ROWS: list[EvalRow] = [
    # Scale curve, lr=3e-4, r32/alpha64, dropout0, wd0.
    EvalRow("scale", "scale128_seed10_lr3e4_ga2_dwdse_only", "128/lr3e-4", 128, 128, 400, 128, 128, 70, 3, 1, 0, 1.6210, 3.4638),
    EvalRow("scale", "scale128_seed10_lr3e4_ga2_dwdse_only", "128/lr3e-4", 128, 128, 800, 128, 128, 101, 5, 1, 1, 0.4348, 5.0422),
    EvalRow("scale", "scale128_seed10_lr3e4_ga2_dwdse_only", "128/lr3e-4", 128, 128, 1200, 128, 128, 112, 4, 0, 0, 0.2161, 3.7589),
    EvalRow("scale", "scale128_seed10_lr3e4_ga2_dwdse_only", "128/lr3e-4", 128, 128, 1600, 128, 128, 111, 10, 1, 0, 0.2271, 3.8031),
    EvalRow("scale", "scale256t176v40_seed10_lr3e4_ga2_dwdse_only", "256 total/lr3e-4", 176, 40, 400, 64, 40, 33, 1, 6, 1, 0.9409, 3.4174),
    EvalRow("scale", "scale256t176v40_seed10_lr3e4_ga2_dwdse_only", "256 total/lr3e-4", 176, 40, 800, 64, 40, 49, 3, 0, 0, 0.7534, 2.7869),
    EvalRow("scale", "scale256t176v40_seed10_lr3e4_ga2_dwdse_only", "256 total/lr3e-4", 176, 40, 1200, 64, 40, 44, 2, 1, 0, 0.3598, 2.4850),
    EvalRow("scale", "scale256t176v40_seed10_lr3e4_ga2_dwdse_only", "256 total/lr3e-4", 176, 40, 1600, 64, 40, 55, 2, 0, 0, 0.3322, 3.3795),
    EvalRow("scale", "scale586t458v64_seed10_lr3e4_ga2_dwdse_only", "full le32/lr3e-4", 458, 64, 400, 64, 64, 10, 2, 34, 36, 2.5688, 3.5289),
    EvalRow("scale", "scale586t458v64_seed10_lr3e4_ga2_dwdse_only", "full le32/lr3e-4", 458, 64, 800, 64, 64, 41, 4, 2, 0, 1.5233, 3.0571),
    EvalRow("scale", "scale586t458v64_seed10_lr3e4_ga2_dwdse_only", "full le32/lr3e-4", 458, 64, 1200, 64, 64, 40, 4, 0, 0, 1.0642, 3.0529),
    EvalRow("scale", "scale586t458v64_seed10_lr3e4_ga2_dwdse_only", "full le32/lr3e-4", 458, 64, 1600, 64, 64, 48, 6, 1, 0, 0.6943, 3.3356),
    # LR scan on 128.
    EvalRow("lr", "scale128_seed10_lr2e4_ga2_save100_dwdse_only", "128/lr2e-4", 128, 128, 200, 128, 128, 22, 1, 0, 0, 1.6189, 2.5769),
    EvalRow("lr", "scale128_seed10_lr2e4_ga2_save100_dwdse_only", "128/lr2e-4", 128, 128, 400, 128, 128, 78, 4, 0, 1, 1.3814, 3.3852),
    EvalRow("lr", "scale128_seed10_lr2e4_ga2_save100_dwdse_only", "128/lr2e-4", 128, 128, 800, 128, 128, 100, 6, 0, 0, 0.5546, 5.0887),
    EvalRow("lr", "scale128_seed10_lr2e4_ga2_save100_dwdse_only", "128/lr2e-4", 128, 128, 1200, 128, 128, 104, 6, 0, 0, 0.3419, 3.7041),
    EvalRow("lr", "scale128_seed10_lr2e4_ga2_save100_dwdse_only", "128/lr2e-4", 128, 128, 1600, 128, 128, 104, 6, 2, 2, 0.3776, 3.7420),
    EvalRow("lr", "scale128_seed10_lr5e4_ga2_dwdse_only", "128/lr5e-4", 128, 128, 400, 128, 128, 74, 5, 0, 0, 1.1116, 3.3864),
    EvalRow("lr", "scale128_seed10_lr5e4_ga2_dwdse_only", "128/lr5e-4", 128, 128, 800, 128, 128, 109, 6, 0, 0, 0.4162, 4.2605),
    EvalRow("lr", "scale128_seed10_lr5e4_ga2_dwdse_only", "128/lr5e-4", 128, 128, 1200, 128, 128, 113, 4, 0, 0, 0.1760, 3.8704),
    EvalRow("lr", "scale128_seed10_lr5e4_ga2_dwdse_only", "128/lr5e-4", 128, 128, 1600, 128, 128, 115, 4, 0, 0, 0.1607, 3.8167),
    # Rank / alpha.
    EvalRow("rank_alpha", "scale128_seed10_lr3e4_r16a32_ga2_dwdse_only", "r16/a32", 128, 128, 400, 128, 128, 74, 4, 0, 0, 0.8353, 3.6871),
    EvalRow("rank_alpha", "scale128_seed10_lr3e4_r16a32_ga2_dwdse_only", "r16/a32", 128, 128, 800, 128, 128, 97, 3, 0, 0, 0.5740, 4.6599),
    EvalRow("rank_alpha", "scale128_seed10_lr3e4_r16a32_ga2_dwdse_only", "r16/a32", 128, 128, 1200, 128, 128, 106, 4, 1, 0, 0.4157, 3.6686),
    EvalRow("rank_alpha", "scale128_seed10_lr3e4_r16a32_ga2_dwdse_only", "r16/a32", 128, 128, 1600, 128, 128, 111, 5, 1, 0, 0.2587, 3.7060),
    EvalRow("rank_alpha", "scale128_seed10_lr3e4_r16a16_ga2_dwdse_only", "r16/a16", 128, 128, 400, 128, 128, 74, 2, 0, 1, 0.9862, 3.6853),
    EvalRow("rank_alpha", "scale128_seed10_lr3e4_r16a16_ga2_dwdse_only", "r16/a16", 128, 128, 800, 128, 128, 89, 4, 0, 0, 0.9910, 5.2482),
    EvalRow("rank_alpha", "scale128_seed10_lr3e4_r16a16_ga2_dwdse_only", "r16/a16", 128, 128, 1200, 128, 128, 96, 4, 1, 2, 0.5074, 3.6698),
    EvalRow("rank_alpha", "scale128_seed10_lr3e4_r16a16_ga2_dwdse_only", "r16/a16", 128, 128, 1600, 128, 128, 106, 4, 0, 2, 0.4294, 3.8219),
    # Dropout.
    EvalRow("dropout", "scale128_seed10_lr3e4_r32a64_do005_ga2_dwdse_only", "dropout0.05", 128, 128, 400, 128, 128, 77, 3, 4, 5, 0.8096, 4.5248),
    EvalRow("dropout", "scale128_seed10_lr3e4_r32a64_do005_ga2_dwdse_only", "dropout0.05", 128, 128, 800, 128, 128, 82, 6, 4, 4, 0.4378, 4.2163),
    EvalRow("dropout", "scale128_seed10_lr3e4_r32a64_do005_ga2_dwdse_only", "dropout0.05", 128, 128, 1200, 128, 128, 100, 4, 4, 0, 0.3876, 4.6673),
    EvalRow("dropout", "scale128_seed10_lr3e4_r32a64_do005_ga2_dwdse_only", "dropout0.05", 128, 128, 1600, 128, 128, 102, 5, 2, 0, 0.2530, 5.1617),
    # Weight decay.
    EvalRow("weight_decay", "scale128_seed10_lr3e4_r32a64_wd001_ga2_dwdse_only", "wd0.001", 128, 128, 400, 128, 128, 76, 5, 0, 1, 1.2665, 3.4917),
    EvalRow("weight_decay", "scale128_seed10_lr3e4_r32a64_wd001_ga2_dwdse_only", "wd0.001", 128, 128, 800, 128, 128, 103, 6, 2, 1, 0.5590, 4.7377),
    EvalRow("weight_decay", "scale128_seed10_lr3e4_r32a64_wd001_ga2_dwdse_only", "wd0.001", 128, 128, 1200, 128, 128, 111, 4, 0, 0, 0.2567, 3.5672),
    EvalRow("weight_decay", "scale128_seed10_lr3e4_r32a64_wd001_ga2_dwdse_only", "wd0.001", 128, 128, 1600, 128, 128, 112, 5, 1, 0, 0.2166, 3.8148),
    EvalRow("weight_decay", "scale128_seed10_lr3e4_r32a64_wd01_ga2_dwdse_only", "wd0.01", 128, 128, 400, 128, 128, 77, 5, 9, 6, 0.9793, 3.4908),
    EvalRow("weight_decay", "scale128_seed10_lr3e4_r32a64_wd01_ga2_dwdse_only", "wd0.01", 128, 128, 800, 128, 128, 96, 4, 2, 1, 0.5121, 4.8669),
    EvalRow("weight_decay", "scale128_seed10_lr3e4_r32a64_wd01_ga2_dwdse_only", "wd0.01", 128, 128, 1200, 128, 128, 106, 5, 0, 0, 0.2525, 3.7499),
    EvalRow("weight_decay", "scale128_seed10_lr3e4_r32a64_wd01_ga2_dwdse_only", "wd0.01", 128, 128, 1600, 128, 128, 114, 7, 1, 0, 0.2345, 3.8281),
    # Max steps / early stopping.
    EvalRow("max_steps", "scale128_seed10_lr3e4_ga2_dwdse_only_resume1600_to3200", "128 1600->3200", 128, 128, 2000, 128, 128, 116, 7, 1, 0, 0.2090, 3.9452),
    EvalRow("max_steps", "scale128_seed10_lr3e4_ga2_dwdse_only_resume1600_to3200", "128 1600->3200", 128, 128, 2400, 128, 128, 116, 5, 0, 0, 0.1463, 5.1491),
    EvalRow("max_steps", "scale128_seed10_lr3e4_ga2_dwdse_only_resume1600_to3200", "128 1600->3200", 128, 128, 2800, 128, 128, 119, 4, 0, 0, 0.0725, 4.0788),
    EvalRow("max_steps", "scale128_seed10_lr3e4_ga2_dwdse_only_resume1600_to3200", "128 1600->3200", 128, 128, 3200, 128, 128, 120, 5, 0, 0, 0.0942, 4.1557),
    EvalRow("max_steps", "scale586t458v64_seed10_lr3e4_ga2_dwdse_only_resume1600_to3200", "full 1600->3200", 458, 64, 2000, 64, 64, 48, 6, 1, 0, 0.5133, 4.0583),
    EvalRow("max_steps", "scale586t458v64_seed10_lr3e4_ga2_dwdse_only_resume1600_to3200", "full 1600->3200", 458, 64, 2400, 64, 64, 51, 6, 0, 0, 0.4339, 2.9297),
    EvalRow("max_steps", "scale586t458v64_seed10_lr3e4_ga2_dwdse_only_resume1600_to3200", "full 1600->3200", 458, 64, 2800, 64, 64, 50, 5, 1, 0, 0.5847, 3.2654),
    EvalRow("max_steps", "scale586t458v64_seed10_lr3e4_ga2_dwdse_only_resume1600_to3200", "full 1600->3200", 458, 64, 3200, 64, 64, 55, 6, 1, 0, 0.4093, 3.5352),
]


def make_frame() -> pd.DataFrame:
    df = pd.DataFrame(asdict(row) for row in ROWS)
    df["train_exact_rate"] = df["train_exact"] / df["train_eval_n"]
    df["valid_exact_rate"] = df["valid_exact"] / df["valid_eval_n"]
    df["train_no_eos_rate"] = df["train_no_eos"] / df["train_eval_n"]
    df["valid_no_eos_rate"] = df["valid_no_eos"] / df["valid_eval_n"]
    df["overfit_gap"] = df["train_exact_rate"] - df["valid_exact_rate"]
    return df


def best_by_run(df: pd.DataFrame) -> pd.DataFrame:
    order = df.sort_values(["valid_exact_rate", "valid_exact", "step"], ascending=[False, False, True])
    best = order.groupby("run", as_index=False).head(1).copy()
    columns = [
        "group",
        "config",
        "run",
        "train_size",
        "valid_size",
        "step",
        "train_exact",
        "train_eval_n",
        "train_exact_rate",
        "valid_exact",
        "valid_eval_n",
        "valid_exact_rate",
        "train_no_eos",
        "valid_no_eos",
        "overfit_gap",
    ]
    return best[columns].sort_values(["group", "valid_exact_rate"], ascending=[True, False])


def plot_scale_curve(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    scale = df[df["group"] == "scale"]
    for config, part in scale.groupby("config", sort=False):
        ax.plot(part["step"], part["valid_exact_rate"] * 100, marker="o", linewidth=2, label=config)
    ax.set_title("Valid Exact by Sample Scale")
    ax.set_xlabel("training step")
    ax.set_ylabel("valid exact (%)")
    ax.set_xticks([400, 800, 1200, 1600])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scale_valid_curve.png")
    plt.close(fig)


def plot_best_bars(best: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    data = best.copy()
    data["label"] = data["group"] + ": " + data["config"]
    data = data.sort_values("valid_exact_rate")
    ax.barh(data["label"], data["valid_exact_rate"] * 100, color="#4C78A8")
    for idx, row in enumerate(data.itertuples(index=False)):
        ax.text(
            row.valid_exact_rate * 100 + 0.15,
            idx,
            f"{row.valid_exact}/{row.valid_eval_n} @ {row.step}",
            va="center",
            fontsize=8,
        )
    ax.set_title("Best Valid Exact by Run")
    ax.set_xlabel("best valid exact (%)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "best_valid_by_run.png")
    plt.close(fig)


def plot_overfit(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
    latest = df.sort_values("step").groupby("run", as_index=False).tail(1)
    colors = {
        "scale": "#4C78A8",
        "lr": "#F58518",
        "rank_alpha": "#54A24B",
        "dropout": "#B279A2",
        "weight_decay": "#E45756",
    }
    for group, part in latest.groupby("group"):
        ax.scatter(
            part["train_exact_rate"] * 100,
            part["valid_exact_rate"] * 100,
            label=group,
            s=52,
            color=colors.get(group),
        )
        for row in part.itertuples(index=False):
            ax.annotate(row.config, (row.train_exact_rate * 100, row.valid_exact_rate * 100), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_title("Latest Checkpoint: Train vs Valid Exact")
    ax.set_xlabel("train exact (%)")
    ax.set_ylabel("valid exact (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "latest_train_vs_valid.png")
    plt.close(fig)


def _plot_heatmap(
    data: pd.DataFrame,
    *,
    title: str,
    output_name: str,
    config_order: list[str],
    step_order: list[int],
) -> None:
    value = data.pivot_table(index="config", columns="step", values="valid_exact_rate", aggfunc="max")
    counts = data.pivot_table(index="config", columns="step", values="valid_exact", aggfunc="max")
    ns = data.pivot_table(index="config", columns="step", values="valid_eval_n", aggfunc="max")
    value = value.reindex(index=config_order, columns=step_order)
    counts = counts.reindex(index=config_order, columns=step_order)
    ns = ns.reindex(index=config_order, columns=step_order)

    fig_height = max(3.0, 0.42 * len(value.index) + 1.6)
    fig, ax = plt.subplots(figsize=(7.2, fig_height), dpi=160)
    image = ax.imshow(value.to_numpy() * 100, cmap="YlGnBu", aspect="auto", vmin=0)
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.set_ylabel("config")
    ax.set_xticks(range(len(value.columns)), [str(step) for step in value.columns])
    ax.set_yticks(range(len(value.index)), value.index)

    for row_idx, config in enumerate(value.index):
        for col_idx, step in enumerate(value.columns):
            pct = value.loc[config, step]
            if pd.isna(pct):
                label = "-"
                color = "#111827"
            else:
                label = f"{int(counts.loc[config, step])}/{int(ns.loc[config, step])}\n{pct * 100:.1f}%"
                color = "white" if pct * 100 >= 6.5 else "#111827"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("valid exact (%)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / output_name)
    plt.close(fig)


def plot_heatmaps(df: pd.DataFrame) -> None:
    scale = df[df["group"] == "scale"]
    _plot_heatmap(
        scale,
        title="Scale Runs: Valid Exact Heatmap",
        output_name="scale_valid_heatmap.png",
        config_order=["128/lr3e-4", "256 total/lr3e-4", "full le32/lr3e-4"],
        step_order=[400, 800, 1200, 1600],
    )

    hyper = df[
        (df["valid_eval_n"] == 128)
        & (
            (df["group"] != "scale")
            | (df["run"] == "scale128_seed10_lr3e4_ga2_dwdse_only")
        )
    ].copy()
    hyper.loc[hyper["run"] == "scale128_seed10_lr3e4_ga2_dwdse_only", "config"] = "baseline lr3e-4"
    _plot_heatmap(
        hyper,
        title="128-Sample Hyperparameter Runs: Valid Exact Heatmap",
        output_name="hyperparam_valid_heatmap.png",
        config_order=[
            "baseline lr3e-4",
            "128 1600->3200",
            "128/lr2e-4",
            "128/lr5e-4",
            "r16/a32",
            "r16/a16",
            "dropout0.05",
            "wd0.001",
            "wd0.01",
        ],
        step_order=[200, 400, 800, 1200, 1600],
    )


def write_markdown(df: pd.DataFrame, best: pd.DataFrame) -> None:
    scale_best = best[best["group"] == "scale"].sort_values("train_size")
    baseline = best[best["run"] == "scale128_seed10_lr3e4_ga2_dwdse_only"].iloc[0]
    lines = [
        "# Scale Grid Summary",
        "",
        "Generated from `analysis/plot_scale_grid.py`.",
        "",
        "## Scale Curve",
        "",
        scale_best[["config", "train_size", "valid_size", "step", "valid_exact", "valid_eval_n", "valid_exact_rate"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Hyperparameter Runs",
        "",
        best[best["group"] != "scale"][["group", "config", "step", "valid_exact", "valid_eval_n", "valid_exact_rate", "train_exact", "train_eval_n"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Current Reading",
        "",
        f"- Current 128 baseline best: {baseline.valid_exact}/{baseline.valid_eval_n} = {baseline.valid_exact_rate:.1%} at step {baseline.step}.",
        "- Scaling from 128 -> 256 -> full le32 gives only a small valid lift at full le32, not a clear generalization jump.",
        "- LR/rank/dropout/weight-decay/max-steps scans did not beat the no-WD r32/a64 lr3e-4 baseline on the 128 split.",
        "- Extending 128 from 1600 to 3200 raises train exact to 120/128 but drops valid below the step1600 baseline.",
        "- Extending full le32 from 1600 to 3200 stays around 6/64 valid exact, so longer training does not create a generalization jump.",
        "",
        "## Figures",
        "",
        "- `scale_valid_curve.png`: scale curve over saved checkpoints.",
        "- `scale_valid_heatmap.png`: scale-by-step valid exact map.",
        "- `hyperparam_valid_heatmap.png`: 128-sample hyperparameter valid exact map.",
        "- `best_valid_by_run.png`: best checkpoint per run.",
        "- `latest_train_vs_valid.png`: latest checkpoint train/valid exact scatter.",
        "",
    ]
    (OUT_DIR / "scale_grid_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = make_frame()
    best = best_by_run(df)
    df.to_csv(OUT_DIR / "scale_grid_results.csv", index=False)
    best.to_csv(OUT_DIR / "scale_grid_best_by_run.csv", index=False)
    plot_scale_curve(df)
    plot_best_bars(best)
    plot_overfit(df)
    plot_heatmaps(df)
    write_markdown(df, best)
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
