from __future__ import annotations

"""plot_memory
----------------------------------
GPU メモリ使用量の *Actual / Event* を可視化し、
★ピーク時のメモリ内訳を記録する★ ユーティリティ。
"""

import json
from pathlib import Path
from typing import Sequence

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 定数は環境に合わせてimportしてください
from .constants import BASE_CATS, DEFAULT_ORDER, COLORS, EVENT_SPECS
from .events import EventSpan

# ======================================================================
# メイン関数
# ======================================================================

def plot_memory(
    json_path: Path,
    *,
    save_to: Path,
    peak_alloc: int,
    event_spans: Sequence[EventSpan] | None = None,
    cats_order: Sequence[str] | None = None,
    profiler_warmup: int = 0,
) -> None:
    """GPU メモリの実測値を描画し、ピーク時の内訳をJSONに保存する。"""

    # ------------------------------------------------------------------
    # 0. 実測データ読み込み
    # ------------------------------------------------------------------
    with json_path.open() as f:
        timestamps, mem_sizes = json.load(f)

    t0_us = timestamps[0]
    ts_ms = [(t - t0_us) / 1e3 for t in timestamps]
    # 単位を GiB に変換
    mem_gib = [[b / 1024 ** 3 for b in cat] for cat in mem_sizes]

    df_act = pd.DataFrame(mem_gib, columns=BASE_CATS)
    df_act["Timestamp"] = ts_ms

    order = list(cats_order) if cats_order else list(DEFAULT_ORDER)

    # ------------------------------------------------------------------
    # 1. Warmup スキップ処理
    # ------------------------------------------------------------------
    df_act, _, event_spans = _apply_skip(
        df_act, None, list(event_spans or []), profiler_warmup
    )

    # ------------------------------------------------------------------
    # 2. ★ここが追加機能★ ピーク時の内訳を特定・保存
    # ------------------------------------------------------------------
    # 各時刻の合計メモリ量を計算
    total_series = df_act[order].sum(axis=1)
    
    # プロファイラ上の最大値とそのインデックスを取得
    act_sum_max = total_series.max()
    peak_idx = total_series.idxmax()
    
    # ピーク時の行データを取得
    peak_row = df_act.iloc[peak_idx]
    peak_time_ms = peak_row["Timestamp"]

    # 保存用辞書を作成
    peak_stats = {
        "peak_total_gib": act_sum_max,
        "timestamp_ms": peak_time_ms,
        "profiler_peak_breakdown_gib": {
            cat: peak_row[cat] for cat in order if cat in peak_row and peak_row[cat] > 0
        },
        # 参考: torch.cudaが報告するハードウェア的な絶対最大値
        "cuda_max_alloc_gib": peak_alloc / 1024**3
    }

    # JSONとして保存 (画像の隣に _peak_stats.json として保存)
    stats_path = save_to.with_name(save_to.stem + "_peak_stats.json")
    with open(stats_path, "w") as f:
        json.dump(peak_stats, f, indent=4)
    
    print(f"  [Info] Peak memory breakdown saved to: {stats_path}")
    print(f"  [Info] Peak detected at {peak_time_ms:.2f} ms: {act_sum_max:.4f} GiB")

    # ------------------------------------------------------------------
    # 3. y-limit とピーク値 (描画用)
    # ------------------------------------------------------------------
    alloc_gib = peak_alloc / 1024 ** 3
    global_ymax = max(alloc_gib, act_sum_max) * 1.15

    # ------------------------------------------------------------------
    # 4. 描画処理 (以下、変更なし)
    # ------------------------------------------------------------------
    fig, (ax_act, ax_evt) = plt.subplots(
        2, 1, figsize=(8, 4.0), sharex=True,
        gridspec_kw=dict(height_ratios=[8, 1]),
    )

    # ---- Actual Stack ----
    _stack_fill(ax_act, df_act, order)

    _draw_peak_line(
        ax_act, act_sum_max, global_ymax,
        f"act_max:  {act_sum_max:.2f}", "purple", x_frac=0.25
    )
    _draw_alloc_line(ax_act, alloc_gib, global_ymax)

    ax_act.set_ylim(0, global_ymax)
    ax_act.set_ylabel("Memory [GiB]")
    ax_act.text(0.0, -0.1, "[GiB]", transform=ax_act.transAxes, ha="right", va="bottom", fontsize=8)

    # ---- Events ----
    _draw_events(ax_evt, event_spans or [])
    ax_evt.set_xlabel("Time [ms]")

    # ---- Legends & Save ----
    _add_legends(fig, order, event_spans or [])
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(save_to, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ======================================================================
# 補助関数 (変更なし)
# ======================================================================

def _apply_skip(df_act, df_pred, spans, profiler_warmup):
    if profiler_warmup == 0:
        return df_act, df_pred, spans
    
    end_ts = [ev.t1_ms for ev in spans if ev.label == "optimize"]
    if len(end_ts) < profiler_warmup:
        return df_act, df_pred, spans
        
    t_skip = end_ts[profiler_warmup - 1]
    
    # Actual 切り詰め
    last_before = df_act[df_act["Timestamp"] < t_skip].tail(1)
    df_act = df_act.loc[df_act["Timestamp"] >= t_skip].copy()
    df_act["Timestamp"] -= t_skip
    if not last_before.empty:
        snap0 = last_before.iloc[0].copy()
        snap0["Timestamp"] = 0.0
        df_act = pd.concat([snap0.to_frame().T, df_act], ignore_index=True)

    # Event 切り詰め
    new_spans = []
    for ev in spans:
        if ev.t1_ms < t_skip: continue
        new_spans.append(EventSpan(ev.label, ev.cat, ev.t0_ms - t_skip, ev.t1_ms - t_skip))

    # 末尾切り詰め
    if new_spans:
        t_end = new_spans[-1].t1_ms
        df_act = df_act.loc[df_act["Timestamp"] <= t_end].copy()
        if not df_act.empty and df_act["Timestamp"].iloc[-1] < t_end:
            snap_end = df_act.iloc[-1].copy()
            snap_end["Timestamp"] = t_end
            df_act = pd.concat([df_act, snap_end.to_frame().T], ignore_index=True)

    return df_act, None, new_spans

def _stack_fill(ax, df, order):
    bottom = pd.Series(0.0, index=df.index)
    for cat in order:
        if cat not in df.columns: continue
        ax.fill_between(
            df["Timestamp"], bottom, bottom + df[cat],
            color=COLORS.get(cat, "gray"), edgecolor=COLORS.get(cat, "gray"),
            linewidth=0.4, alpha=0.8,
        )
        bottom += df[cat]

def _draw_peak_line(ax, y, y_max, label, color, x_frac):
    x_pos = ax.get_xlim()[1] * x_frac
    delta = y_max / 100
    ax.axhline(y, ls="--", lw=1, color=color)
    ax.text(x_pos, y + delta, label, ha="center", color=color, fontsize=9)

def _draw_alloc_line(ax, alloc_gib, y_max):
    _draw_peak_line(ax, alloc_gib, y_max, f"max_alloc:    {alloc_gib:.2f}", "red", x_frac=0.75)

def _draw_events(ax, spans):
    for ev in spans:
        ax.broken_barh([(ev.t0_ms, ev.duration_ms)], (0, 0.8),
            facecolors=EVENT_SPECS.get(ev.label, {}).get("color", "lightgreen"), alpha=0.8)
    ax.set_ylim(-0.2, 1)
    ax.set_yticks([])
    ax.grid(False)

def _add_legends(fig, order, spans):
    mem_handles = [Patch(facecolor=COLORS.get(c, "gray"), edgecolor=COLORS.get(c, "gray"), label=c) for c in order]
    fig.legend(handles=mem_handles, loc="upper center", bbox_to_anchor=(0.53, 0.9), ncol=len(mem_handles), frameon=False, fontsize=8)
    evt_labels = {ev.label for ev in spans}
    if evt_labels:
        evt_handles = [Patch(facecolor=EVENT_SPECS.get(l, {}).get("color", "gray"), edgecolor="none", alpha=0.5, label=l) for l in evt_labels]
        fig.legend(handles=evt_handles, loc="lower center", bbox_to_anchor=(0.53, 0.05), ncol=len(evt_handles), frameon=False, fontsize=8)