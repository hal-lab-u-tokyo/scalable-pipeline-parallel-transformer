from __future__ import annotations

"""plot_memory_compare
----------------------------------
GPU メモリ使用量の *Actual / Predicted / Event* を比較するグラフを描画するユーティリティ。

改修点（2025-06-24）:
----------------------------------
* **メモリ要素の凡例 (memory categories)**
    - 図表タイトルの直下に横並びで配置。
* **イベント凡例 (timeline events)**
    - 図表の最下部、時間軸のさらに下に横並びで配置。
* 右側の凡例領域を撤廃し、可視領域全体をプロットへ使用。

追加改修（2025-06-27）:
----------------------------------
* `show_pred` 引数を廃止し、代わりに `show_cat` 引数を導入。
* `show_cat=False` の場合は Predicted スタックを描画せず、
  Actual + Event の 2 段構成とする。
* その際、**予測カテゴリ全体の総和** を Actual 段に折れ線グラフとして重ね描画。

追加改修（2025-07-10）:
----------------------------------
* **予測・実測ピークの追加表示**  
  - *予測カテゴリ総和の最大値* (`pred_sum_max`)  
  - *実測カテゴリ総和の最大値* (`act_sum_max`)  
* ラベルは **左：予測 → 中：実測 → 右:alloc** の順で横並び。
"""

import json
from pathlib import Path
from typing import Sequence, List, Tuple, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from datetime import datetime
from zoneinfo import ZoneInfo

from .constants import BASE_CATS, DEFAULT_ORDER, COLORS, EVENT_SPECS
from .events import EventSpan

# ======================================================================
# メイン
# ======================================================================


def plot_memory_compare(
    json_path: Path,
    *,
    save_to: Path,
    peak_alloc: int,                        # ← max_memory_allocated
    peak_reserved: int,                     # 使わないが呼び出し互換のため残す
    event_spans: Sequence[EventSpan] | None = None,
    cats_order: Sequence[str] | None = None,
    pred_timeline: Tuple[List[float], Dict[str, List[float]]] | None = None,
    pred_max: float = 0.0,
    profiler_warmup: int = 0,
    show_cat: bool = True,
) -> None:
    """GPU メモリの Actual / Predicted / Event を比較するグラフを描画する。

    3 段 (Actual / Predicted / Event) もしくは 2 段 (Actual+Event) の構成で描画する。

    Parameters
    ----------
    json_path : Path
        torch profiler のメモリ使用量 JSON ファイルへのパス。
    save_to : Path
        書き出し先 PNG パス。
    peak_alloc : int
        実測の max_memory_allocated (bytes)。
    peak_reserved : int
        互換性維持のため残しているが未使用。
    event_spans : Sequence[EventSpan] | None, default None
        イベント区間。
    cats_order : Sequence[str] | None, default None
        スタック順序。None の場合は DEFAULT_ORDER。
    pred_timeline : (List[float], Dict[str, List[float]]) | None, default None
        予測結果 (timestamp list, trace dict)。
    profiler_warmup : int, default 0
        Active #n までの部分をスキップして描画。
    show_cat : bool, default True
        * **True**  の場合: Predicted をスタック面グラフで描画し、3 段構成。
        * **False** の場合: Predicted を描画せず、Actual + Event の 2 段構成とし、
            予測カテゴリ総和の折れ線 (---) を Actual 段に重ねる。
    """

    # ------------------------------------------------------------------
    # 0. 実測データ読み込み
    # ------------------------------------------------------------------
    with json_path.open() as f:
        timestamps, mem_sizes = json.load(f)

    t0_us = timestamps[0]
    ts_ms = [(t - t0_us) / 1e3 for t in timestamps]
    mem_gib = [[b / 1024 ** 3 for b in cat] for cat in mem_sizes]

    df_act = pd.DataFrame(mem_gib, columns=BASE_CATS)
    df_act["Timestamp"] = ts_ms

    order = list(cats_order) if cats_order else list(DEFAULT_ORDER)

    # ------------------------------------------------------------------
    # 1. 予測データ
    # ------------------------------------------------------------------
    if pred_timeline is not None:
        ts_pred, traces_pred = pred_timeline
        df_pred = pd.DataFrame(traces_pred)
        df_pred["Timestamp"] = ts_pred
        for cat in order:
            if cat not in df_pred.columns:
                df_pred[cat] = 0.0
    else:
        df_pred = None

    df_act, df_pred, event_spans = _apply_skip(
        df_act, df_pred, list(event_spans or []), profiler_warmup
    )

    # ------------------------------------------------------------------
    # 2. y-limit とピーク値
    # ------------------------------------------------------------------
    alloc_gib     = peak_alloc / 1024 ** 3
    act_sum_max   = df_act[order].sum(axis=1).max()
    pred_sum_max  = df_pred[order].sum(axis=1).max() if df_pred is not None else 0.0

    global_ymax = max(alloc_gib, act_sum_max, pred_sum_max) * 1.15
    # ↓ レイアウト固定用 (必要ならコメントアウト可)
    # global_ymax = 0.9

    # ------------------------------------------------------------------
    # 3. Figure / Axes
    # ------------------------------------------------------------------
    draw_pred_stack = show_cat and (df_pred is not None)

    if draw_pred_stack:
        fig, (ax_act, ax_pred, ax_evt) = plt.subplots(
            3, 1, figsize=(8, 2 + 2 + 1), sharex=True,
            gridspec_kw=dict(height_ratios=[4, 4, 1]),
        )
    else:
        fig, (ax_act, ax_evt) = plt.subplots(
            2, 1, figsize=(8, 2.0), sharex=True,
            gridspec_kw=dict(height_ratios=[8, 1]),
        )
        ax_pred: Optional[plt.Axes] = None  # type: ignore

    # ----------------------- Actual ------------------------
    _stack_fill(ax_act, df_act, order)

    # ①予測ピーク → ②実測ピーク → ③alloc の順で左→中→右へ
    _draw_peak_line(
        ax_act, pred_sum_max, global_ymax,
        f"pred_max_prev: {pred_sum_max:.2f}", "black", x_frac=0.25
    )
    _draw_peak_line(
        ax_act, pred_max, global_ymax,
        f"pred_max: {pred_max:.2f}", "blue", x_frac=0.25
    )
    _draw_peak_line(
        ax_act, act_sum_max, global_ymax,
        f"act_max:  {act_sum_max:.2f}", "purple", x_frac=0.50
    )
    _draw_alloc_line(ax_act, alloc_gib, global_ymax)  # red, 右端寄せ

    # show_cat=False の場合は予測総和折れ線を Actual に重ね描画
    # 一旦コメントアウト
    # if (not show_cat) and (df_pred is not None):
    #     _draw_pred_total_line(ax_act, df_pred, order)

    ax_act.set_ylim(0, global_ymax)
    ax_act.text(
        0.0, -0.25, "[GiB]",
        transform=ax_act.transAxes,
        ha="right", va="bottom", fontsize=8
    )

    # ----------------------- Predicted ---------------------
    if draw_pred_stack and ax_pred is not None:
        _stack_fill(ax_pred, df_pred, order)          # type: ignore[arg-type]
        _draw_peak_line(
            ax_pred, pred_sum_max, global_ymax,
            f"pred_max: {pred_sum_max:.2f}", "blue", x_frac=0.25
        )
        ax_pred.set_ylabel("Predicted [GiB]")
        ax_pred.set_ylim(0, global_ymax)
    elif ax_pred is not None:
        ax_pred.text(
            0.5, 0.5, "No prediction",
            ha="center", va="center", transform=ax_pred.transAxes
        )
        ax_pred.set_ylabel("Predicted [GiB]")
        ax_pred.set_ylim(0, global_ymax)

    # ----------------------- Events ------------------------
    _draw_events(ax_evt, event_spans or [])
    ax_evt.text(
        0.0, -0.35, "[ms]",
        transform=ax_evt.transAxes,
        ha="left", va="top", fontsize=8
    )

    # ---- Save figure WITHOUT legends ----
    save_to_no_legend = save_to.with_name(save_to.stem + "_nolegend" + save_to.suffix)
    fig.savefig(save_to_no_legend, dpi=300, bbox_inches="tight", pad_inches=0.05)
    pdf_path_no_legend = save_to_no_legend.with_suffix(".pdf")
    fig.savefig(pdf_path_no_legend, bbox_inches="tight", pad_inches=0.05)

    # ----------------------- Legends ----------------------
    _add_legends(fig, order, event_spans or [])
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])

    # ---- Save figure WITH legends ----
    fig.savefig(save_to, dpi=300, bbox_inches="tight", pad_inches=0.05)
    pdf_path = save_to.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)

    plt.close(fig)

# ======================================================================
# 補助関数
# ======================================================================


def _apply_skip(
    df_act: pd.DataFrame,
    df_pred: pd.DataFrame | None,
    spans: list[EventSpan],
    profiler_warmup: int,
):
    """profiler_warmup 分をスキップし、0-base に揃える
       + 末尾を *最後の optimize* で切る
    """
    # ------------------------------------------------------------
    # 0. warm-up 無し
    # ------------------------------------------------------------
    if profiler_warmup == 0:
        return df_act, df_pred, spans

    # ------------------------------------------------------------
    # 1. skip する境界（optimize 終端）を求める
    # ------------------------------------------------------------
    end_ts = [ev.t1_ms for ev in spans if ev.label == "optimize"]
    if len(end_ts) < profiler_warmup:
        raise ValueError(
            f"profiler_warmup={profiler_warmup} だが optimize が {len(end_ts)} 回しか無い"
        )
    t_skip = end_ts[profiler_warmup - 1]          # ← 切り捨て境界

    # ------------------------------------------------------------
    # 2. Actual : 直前スナップショットを 0 ms 行として復元
    # ------------------------------------------------------------
    last_before = df_act[df_act["Timestamp"] < t_skip].tail(1)
    df_act = df_act.loc[df_act["Timestamp"] >= t_skip].copy()
    df_act["Timestamp"] -= t_skip
    if not last_before.empty:
        snap0 = last_before.iloc[0].copy()
        snap0["Timestamp"] = 0.0
        df_act = pd.concat([snap0.to_frame().T, df_act], ignore_index=True)

    # ------------------------------------------------------------
    # 3. Predicted : 単に時刻を合わせる
    # ------------------------------------------------------------
    if df_pred is not None:
        df_pred = df_pred.loc[df_pred["Timestamp"] >= t_skip].copy()
        df_pred["Timestamp"] -= t_skip

    # ------------------------------------------------------------
    # 4. EventSpan も切り詰め & 0-base
    # ------------------------------------------------------------
    new_spans: list[EventSpan] = []
    for ev in spans:
        if ev.t1_ms < t_skip:
            continue
        new_spans.append(
            EventSpan(ev.label, ev.cat, ev.t0_ms - t_skip, ev.t1_ms - t_skip)
        )

    # ------------------------------------------------------------
    # 5. ★ 尻を “最後の optimize” で切る ★
    # ------------------------------------------------------------
    if new_spans:
        t_end = new_spans[-1].t1_ms

        # ---- Actual ----
        df_act = df_act.loc[df_act["Timestamp"] <= t_end].copy()
        if df_act["Timestamp"].iloc[-1] < t_end:
            snap_end = df_act.iloc[-1].copy()
            snap_end["Timestamp"] = t_end
            df_act = pd.concat([df_act, snap_end.to_frame().T], ignore_index=True)

        # ---- Predicted ----
        if df_pred is not None:
            df_pred = df_pred.loc[df_pred["Timestamp"] <= t_end].copy()
            if (not df_pred.empty) and df_pred["Timestamp"].iloc[-1] < t_end:
                snap_end = df_pred.iloc[-1].copy()
                snap_end["Timestamp"] = t_end
                df_pred = pd.concat([df_pred, snap_end.to_frame().T], ignore_index=True)

    return df_act, df_pred, new_spans


# ------------------------------ Plot helpers ---------------------------

def _stack_fill(ax, df: pd.DataFrame, order: Sequence[str]):
    """カテゴリーごとに塗り分けてスタック面グラフを描く"""

    bottom = pd.Series(0.0, index=df.index)
    for cat in order:
        if cat not in df.columns:
            continue
        ax.fill_between(
            df["Timestamp"], bottom, bottom + df[cat],
            color=COLORS[cat], edgecolor=COLORS[cat],
            linewidth=0.4, alpha=0.8,
        )
        bottom += df[cat]


# ------------------------------------------------------------------
#  汎用: 破線＋ラベル描画
# ------------------------------------------------------------------
def _draw_peak_line(ax, y, y_max, label, color, x_frac):
    """
    共通ヘルパ：水平破線と数値ラベルを描く。

    Parameters
    ----------
    ax : plt.Axes
        描画対象の Axes。
    y : float
        線を引く y 座標。
    y_max : float
        その段の y-limit 上限。
    label : str
        表示するテキスト。
    color : str
        線と文字の色。
    x_frac : float
        x 軸全体に対するラベル x 位置 (0.0–1.0)。
    """
    x_pos = ax.get_xlim()[1] * x_frac
    delta = y_max / 100
    ax.axhline(y, ls="--", lw=1, color=color)
    ax.text(x_pos, y + delta, label, ha="center", color=color, fontsize=9)


def _draw_alloc_line(ax, alloc_gib, y_max):
    """実測 max_alloc を青破線で表示（右寄せラベル）"""
    _draw_peak_line(
        ax, alloc_gib, y_max,
        f"max_alloc:    {alloc_gib:.2f}", "red", x_frac=0.75
    )


def _draw_pred_total_line(ax, df_pred: pd.DataFrame, order: Sequence[str]):
    """予測カテゴリ総和の折れ線を Actual 段に描画"""

    total = df_pred[order].sum(axis=1)
    ax.plot(df_pred["Timestamp"], total, ls="--", lw=1.2, color="orange")


def _draw_events(ax, spans: Sequence[EventSpan]):
    """一番下の段にイベント区間を broken_barh で描画"""

    for ev in spans:
        ax.broken_barh(
            [(ev.t0_ms, ev.duration_ms)],
            (0, 0.8),
            facecolors=EVENT_SPECS.get(ev.label, {}).get("color", "lightgreen"),
            alpha=0.8,
        )
    ax.set_ylim(-0.2, 1)
    ax.set_yticks([])
    ax.grid(False)


def _add_legends(fig: plt.Figure, order: Sequence[str], spans: Sequence[EventSpan]):
    """メモリ要素とイベント凡例をそれぞれ別の場所に配置"""

    # ---- Memory categories legend (top, horizontal) ----
    mem_handles = [
        Patch(facecolor=COLORS[c], edgecolor=COLORS[c], label=c) for c in order
    ]
    fig.legend(
        handles=mem_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.9),  # タイトルの真下あたり
        ncol=len(mem_handles),
        frameon=False,
        fontsize=8,
    )

    # ---- Event labels legend (bottom, horizontal) ----
    evt_labels = {ev.label for ev in spans}
    if evt_labels:
        evt_handles = [
            Patch(facecolor=EVENT_SPECS[l]["color"], edgecolor="none", alpha=0.5, label=l)
            for l in evt_labels
        ]
        fig.legend(
            handles=evt_handles,
            loc="lower center",
            bbox_to_anchor=(0.53, 0.05),  # x軸のさらに下
            ncol=len(evt_handles),
            frameon=False,
            fontsize=8,
        )
