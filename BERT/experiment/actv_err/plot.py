#!/usr/bin/env python3
"""
python plot.py --logdir <run_dir>
  * デフォルト: ./logs/actv-err_pp_rev_fp32_cifar-10_9999/ai-h200-brc
  * その配下にある actv_err_<stage>.csv をまとめ、
    mse_boxplot.png / cos_boxplot.png を生成します。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_LOGDIR = (
    Path("./logs")
    / "actv-err_pp_rev_bf16_cifar-10_9999"
    / "ai-h200-brc"
)


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Activation-error box-plot maker")
    ap.add_argument("--logdir", type=str, default=str(DEFAULT_LOGDIR),
                    help="Root dir that contains actv_err_*.csv")
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)

    root = Path(args.logdir).expanduser().resolve()
    if not root.exists():
        sys.exit(f"[ERROR] logdir does not exist: {root}")

    csv_paths = sorted(root.rglob("actv_err_*.csv"))
    if not csv_paths:
        sys.exit(f"[ERROR] no actv_err_*.csv found under {root}")

    dfs: list[pd.DataFrame] = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception as exc:
            print(f"[WARN] failed to read {p}: {exc}")
            continue

        # 何も有効な列が無い場合はスキップ（FutureWarning 回避）
        if df.empty or set(df.columns) <= {"Unnamed: 0"}:
            print(f"[WARN] {p} is empty – skipped")
            continue

        # actv_err_<stage>.csv から番号を取得
        try:
            stage_idx = int(p.stem.split("_")[-1])
        except ValueError:
            stage_idx = -1  # 不明なら末尾へ
        df["stage"] = stage_idx
        dfs.append(df)

    if not dfs:
        sys.exit("[ERROR] no valid CSV files could be parsed")

    data = pd.concat(dfs, ignore_index=True)
    data.sort_values("stage", inplace=True)

    def _save(col: str, ylabel: str, dest: Path):
        plt.figure(figsize=(8, 4.5))
        data.boxplot(column=col, by="stage")
        plt.title("")
        plt.suptitle("")
        plt.xlabel("Stage")
        plt.ylabel(ylabel)
        plt.tight_layout()
        dest.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(dest, dpi=150)
        print(f"[INFO] saved {dest}")
        plt.close()

    mse_png = root.with_name(root.name + "_mse.png")
    cos_png = root.with_name(root.name + "_cos.png")

    _save("mse", "MSE", mse_png)
    _save("cos", "Cosine Similarity", cos_png)


if __name__ == "__main__":
    main()
