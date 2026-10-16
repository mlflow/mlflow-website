"""Regenerate judge-comparison.png from the scored Jev QA benchmark results.

The original chart was made with plot_comparison.py in the Jev pilot artifacts.
This copy keeps its typography, palette, and three-panel layout while using the
corrected 29-case synthetic QA reference set and adding the OpenAI Decisions
run. QA-018 has an unclear human label and is excluded from every agreement
denominator.

Run: python3 plot_judge_comparison.py
"""

from __future__ import annotations

import os
from pathlib import Path

# Keep Matplotlib's cache in a writable location on machines with a read-only home.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/jev-judge-chart-mpl")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

try:
    from fontTools.ttLib import TTCollection
except ImportError:
    TTCollection = None


OUTPUT = Path(__file__).with_name("judge-comparison.png")

# September 20 pilot: BASELINE_COMPARISON.md / baseline-comparison.json in the
# frozen jev-mlflow-qa-pilot artifacts. OpenAI Decisions: October 8 scored run
# 20261008T212531.060423Z-openai-decisions-part1 (29 completed, 0 failed;
# QA-002 was a false accept). Latencies are milliseconds; costs are estimated
# USD per 1,000 scored calls. p95 is retained for the accompanying table;
# this image follows the original three-panel chart and plots only the median.
MODELS = [
    {"label": "Jev\n1.13.0", "correct": 29, "median": 368.7, "p95": 411.1, "cost": 0.0247},
    {
        "label": "OpenAI\nDecisions\n(gpt-6-luna)",
        "correct": 28,
        "median": 171.942,
        "p95": 554.178,
        "cost": 0.0459,
    },
    {"label": "GPT-5.6\nTerra", "correct": 29, "median": 1091.0, "p95": 1924.0, "cost": 0.8960},
    {"label": "GPT-5.6\nLuna", "correct": 29, "median": 946.7, "p95": 1435.2, "cost": 0.0896},
    {"label": "Claude\nSonnet 4.6", "correct": 26, "median": 1609.5, "p95": 3545.6, "cost": 1.6721},
    {"label": "Claude\nOpus 4.8", "correct": 28, "median": 1965.9, "p95": 9279.1, "cost": 3.7750},
    {"label": "DeepSeek\nV4.1-Flash", "correct": 28, "median": 910.5, "p95": 1187.7, "cost": 0.0624},
]
SCORED_CASES = 29

INK = "#162B45"
MUTED = "#5C6E83"
BLUE = "#008FD5"
TEAL = "#168B91"
SLATE = "#637E9E"
GRID = "#E6ECF3"
BACKGROUND = "#FFFFFF"


def configure_fonts() -> None:
    font_path = Path("/System/Library/Fonts/Avenir Next.ttc")
    if font_path.exists() and TTCollection is not None:
        cache = Path(matplotlib.get_cachedir()) / "jev-fonts"
        cache.mkdir(exist_ok=True)
        with TTCollection(font_path) as collection:
            for font in collection.fonts:
                weight = font["OS/2"].usWeightClass
                style = font["name"].getDebugName(2)
                if weight in (400, 700) and style in ("Regular", "Bold"):
                    path = cache / f"avenir-next-{weight}.ttf"
                    font.save(path)
                    font_manager.fontManager.addfont(path)
        family = "Avenir Next"
    else:
        family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 12,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": INK,
        }
    )


def panel(
    ax,
    title: str,
    values: list[float],
    labels: list[str],
    maximum: float,
    ticks: list[float],
    *,
    currency: bool = False,
    agreement: bool = False,
) -> None:
    ax.set_title(title, fontsize=16.5, weight="bold", pad=18)
    colors = [BLUE, TEAL] + [SLATE] * (len(MODELS) - 2)
    ax.bar(range(len(MODELS)), values, width=0.61, color=colors, zorder=3)
    ax.set_xticks(range(len(MODELS)), [model["label"] for model in MODELS])
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(0, maximum)
    ax.set_yticks(ticks)
    if currency:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    elif agreement:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))

    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(axis="both", length=0)
    ax.tick_params(axis="x", labelsize=10.5, pad=10, labelcolor=INK)
    ax.tick_params(axis="y", labelsize=11.5, pad=7)
    for index, tick in enumerate(ax.get_xticklabels()):
        if index in (0, 1):
            tick.set_color(colors[index])
            tick.set_weight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)

    for index, (value, label) in enumerate(zip(values, labels, strict=True)):
        ax.annotate(
            label,
            (index, value),
            xytext=(0, 8),
            textcoords="offset points",
            va="bottom",
            ha="center",
            fontsize=11.5 if agreement else 12,
            linespacing=1.0,
            color=colors[index] if index < 2 else INK,
            weight="bold" if index < 2 else "normal",
        )


def main() -> None:
    assert len(MODELS) == 7
    assert all(0 <= model["correct"] <= SCORED_CASES for model in MODELS)
    configure_fonts()
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.7), facecolor=BACKGROUND)
    fig.subplots_adjust(left=0.035, right=0.99, bottom=0.19, top=0.89, wspace=0.22)

    agreement_values = [100 * model["correct"] / SCORED_CASES for model in MODELS]
    agreement_labels = []
    for model, value in zip(MODELS, agreement_values, strict=True):
        percentage = "100%" if model["correct"] == SCORED_CASES else f"{value:.1f}%"
        agreement_labels.append(f"{model['correct']}/{SCORED_CASES}\n{percentage}")
    median_values = [model["median"] for model in MODELS]
    cost_values = [model["cost"] for model in MODELS]

    panel(
        axes[0],
        "Alignment with human labels",
        agreement_values,
        agreement_labels,
        120,
        [0, 25, 50, 75, 100],
        agreement=True,
    )
    panel(
        axes[1],
        "Median latency (ms)",
        median_values,
        [f"{value:,.0f}" for value in median_values],
        2450,
        [0, 500, 1000, 1500, 2000],
    )
    panel(
        axes[2],
        "Estimated cost ($ / 1,000 judgments)",
        cost_values,
        [f"${value:.4f}" for value in cost_values],
        4.5,
        [0, 1, 2, 3, 4],
        currency=True,
    )

    fig.savefig(OUTPUT, dpi=200, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.13)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
