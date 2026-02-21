#!/usr/bin/env python3
"""
plot_legends_only.py
====================================================
GPU メモリ・プロファイラ用 **凡例だけ** をファイルに書き出すスクリプト。

* **メモリカテゴリー凡例**
* **タイムラインイベント凡例**

両方とも **PDF + PNG** の 2 形式で保存できます。図表本体の描画では
`add_legends=False` で凡例を省き、本スクリプトの出力を組み合わせて
論文やスライドに貼り付ける想定です。

----------------------------------------------------
Usage::

    # 既定 (figs/ に 4 つのファイルを書き出し)
    $ python plot_legends_only.py

    # 出力ディレクトリを指定
    $ python plot_legends_only.py --outdir fig_output

    # ファイル名を変える
    $ python plot_legends_only.py \
        --mem-name mem_legend --evt-name evt_legend

    # 出力するイベント凡例を絞る
    $ python plot_legends_only.py --evt-labels forward_one_chunk optimize


Requirements
------------
このスクリプトは `experiment.profiling.constants` から
`COLORS` と `EVENT_SPECS` をインポートします。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from experiment.profiling.constants import COLORS, EVENT_SPECS, DEFAULT_ORDER
except ModuleNotFoundError as e:  # スクリプト単体実行時のフォールバック
    raise SystemExit("このスクリプトは experiment パッケージ内で実行してください") from e

# ---------------------------------------------------------------------------
# 内部ヘルパ
# ---------------------------------------------------------------------------

def _save_legend(handles: Sequence[Patch], *, stem: Path, ncol: int) -> None:
    """`stem` (拡張子なし Path) をベースに PDF / PNG を保存"""
    for ext in (".pdf", ".png"):
        outfile = stem.with_suffix(ext)
        fig = plt.figure(figsize=(8, 0.8))
        fig.legend(
            handles=handles,
            loc="center",
            ncol=ncol,
            frameon=False,
            fontsize=7,
        )
        fig.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"[saved] {outfile}")


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export legends (mem & events) as PDF/PNG.")
    parser.add_argument("--outdir", type=Path, default=Path("figs"), help="出力ディレクトリ (default: figs/)")
    parser.add_argument("--mem-name", default="legend_memory", help="メモリ凡例ファイル名 (拡張子除く)")
    parser.add_argument("--evt-name", default="legend_events", help="イベント凡例ファイル名 (拡張子除く)")
    parser.add_argument("--evt-labels", nargs="*", metavar="LABEL", help="含めるイベント名を列挙 (省略時は全種)")

    args = parser.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Memory category legend ------------------------------------------
    mem_handles = [
        Patch(facecolor=COLORS[c], edgecolor=COLORS[c], label=c) for c in DEFAULT_ORDER
    ]
    _save_legend(mem_handles, stem=outdir / args.mem_name, ncol=len(mem_handles))

    # ---- Event legend -----------------------------------------------------
    # evt_labels = set(args.evt_labels) if args.evt_labels else set(EVENT_SPECS.keys())
    evt_labels = (
        args.evt_labels
        if args.evt_labels is not None
        else list(EVENT_SPECS.keys())
    )
    if not evt_labels:
        print("[skip] no event labels specified / found")
    else:
        evt_handles = [
            Patch(facecolor=EVENT_SPECS[l]["color"], edgecolor="none", alpha=0.5, label=l)
            for l in evt_labels if l in EVENT_SPECS
        ]
        if not evt_handles:
            print("[skip] no matching EVENT_SPECS for given labels")
        else:
            _save_legend(evt_handles, stem=outdir / args.evt_name, ncol=len(evt_handles))


if __name__ == "__main__":
    main()
