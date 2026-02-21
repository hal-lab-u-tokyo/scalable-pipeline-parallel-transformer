import json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, List

from .constants import EVENT_SPECS

# torch.profilerが出力したJSONトレースファイルを解析してconstants.pyで定義されたイベントの開始、終了をまとめる
# constants.pyファイルで定義された特定のパターンを抽出し、それぞれのイベントの名前、開始時刻、終了時刻を抽出する

# 抽出したイベントの情報を整理して格納するデータ構造
@dataclass(slots=True)
class EventSpan:
    label: str #ここにforward_one_chunkとかが入る
    cat:   str #ここにuser_annotationが入る
    t0_ms: float #イベントの開始時刻
    t1_ms: float #イベントの終了時刻

    #イベントの所要時間を計算
    @property
    def duration_ms(self) -> float:
        return self.t1_ms - self.t0_ms

# constatns.pyのパターンに一致する全てのイベントのEventSpanを格納したリストを返す
def parse_event_spans(
    trace_path: Path,
    *,
    patterns: Mapping[str, str] | None = None,
) -> List[EventSpan]:

    # patternを使用するか、constants.pyからインポートしたEVENT_SPECSから作成する辞書を使うか
    patt = patterns or {k: v["pattern"] for k, v in EVENT_SPECS.items()}

    with trace_path.open(encoding="utf-8") as f:
        events = json.load(f)["traceEvents"]

    spans: list[EventSpan] = []

    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        if cat != "user_annotation":
            continue

        label = _match_label(ev.get("name", ""), patt)
        if label is None:
            continue

        start_us = ev["ts"]
        end_us   = start_us + ev["dur"]
        spans.append(EventSpan(label, cat, start_us/1e3, end_us/1e3))

    active: dict[int, tuple[str, float, str]] = {}
    for ev in events:
        ph = ev.get("ph")
        if ph not in {"B", "E"}:
            continue

        eid = ev["id"]
        if ph == "B":
            active[eid] = (ev["name"], ev["ts"], ev.get("cat", ""))
        elif eid in active:                     # "E"
            name, start_us, cat = active.pop(eid)
            if cat != "user_annotation":
                continue

            label = _match_label(name, patt)
            if label is None:
                continue
            spans.append(EventSpan(label, cat, start_us/1e3, ev["ts"]/1e3))

    # sort & zero-base
    spans.sort(key=lambda s: s.t0_ms)
    if spans:
        base = spans[0].t0_ms
        for s in spans:
            s.t0_ms -= base
            s.t1_ms -= base
    return spans


# name(イベント名)がconstants.pyで定義されたパターンのいずれかに一致するかどうかを判定する
# nameがforward_one_chunkという文字列を含んでいればforward_one_chunkというラベル(lbl)を返す
def _match_label(name: str, patt: Mapping[str, str]) -> str | None:
    for lbl, regex in patt.items():
        if re.search(regex, name):
            return lbl
    return None
