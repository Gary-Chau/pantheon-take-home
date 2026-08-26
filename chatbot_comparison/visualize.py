"""Generates comparison charts (response length, generation speed, latency)
from the aggregated results, and saves them as PNGs under result/charts/.

Usage:
    python visualize.py
"""
from __future__ import annotations

import os

import hydra
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig

from aggregate_to_excel import load_records


def plot_metric_by_model(df: pd.DataFrame, column: str, ylabel: str, title: str, out_path: str) -> None:
    means = df.groupby("model")[column].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_by_model_and_task(df: pd.DataFrame, column: str, ylabel: str, title: str, out_path: str) -> None:
    pivot = df.pivot_table(index="task_id", columns="model", values=column, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlabel("Task")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.charts_dir, exist_ok=True)
    df = load_records(cfg.raw_result_dir)
    df = df[df["error"].isna()].copy()  # skip failed calls in the charts

    plot_metric_by_model(
        df, "response_word_count", "Avg. words per response",
        "Average Response Length by Model", os.path.join(cfg.charts_dir, "avg_response_length.png"),
    )
    plot_metric_by_model(
        df, "tokens_per_second", "Tokens / second",
        "Average Generation Speed by Model (CPU)", os.path.join(cfg.charts_dir, "avg_tokens_per_second.png"),
    )
    plot_metric_by_model(
        df, "total_duration_sec", "Seconds",
        "Average Total Response Time by Model", os.path.join(cfg.charts_dir, "avg_total_duration.png"),
    )
    plot_metric_by_model_and_task(
        df, "response_word_count", "Words",
        "Response Length by Model and Task", os.path.join(cfg.charts_dir, "response_length_by_task.png"),
    )

    print(f"Charts saved to {cfg.charts_dir}/")


if __name__ == "__main__":
    main()
