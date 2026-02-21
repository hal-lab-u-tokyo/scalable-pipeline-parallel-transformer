# ---------- memory timeline categories ------------------------------------
# どの操作がどの程度メモリ使用しているかとか実行時間を可視化し、測定しているためのパラメータ
BASE_CATS = [ 
    "CATEGORY_0",
    "PARAMETER",
    "OPTIMIZER_STATE",
    "INPUT",
    "TEMPORARY",
    "ACTIVATION",
    "GRADIENT",
    "AUTOGRAD_DETAIL",
    "Unknown",
]
DEFAULT_ORDER = [
    "PARAMETER",
    "OPTIMIZER_STATE",
    "INPUT",
    "GRADIENT",
    "AUTOGRAD_DETAIL",
    "ACTIVATION",
    "TEMPORARY",
    "Unknown",
]

COLORS = {
    "PARAMETER":       "darkgreen",
    "OPTIMIZER_STATE": "goldenrod",
    "GRADIENT":        "mediumblue",
    "INPUT":           "black",
    "ACTIVATION":      "red",
    "AUTOGRAD_DETAIL": "royalblue",
    "TEMPORARY":       "mediumpurple",
    "Unknown":         "grey",
}
# ---------- user-annotation patterns & colours ----------------------------
EVENT_SPECS = {
    "dataload":           {"color": "lightgrey",      "pattern": r"## dataload ##"},
    # "forward":            {"color": "lightcoral",     "pattern": r"## forward ##"},
    # "losscomp":           {"color": "khaki",          "pattern": r"## losscomp ##"},
    # "backward":           {"color": "lightskyblue",   "pattern": r"## backward ##"},
    "send_labels":        {"color": "lightgreen",      "pattern": r"## send_labels ##"},
    "recv_labels":        {"color": "lightgreen",      "pattern": r"## recv_labels ##"},
    "forward_one_chunk":  {"color": "lightcoral",     "pattern": r"## forward_one_chunk ##"},
    "backward_one_chunk": {"color": "lightskyblue",   "pattern": r"## backward_one_chunk ##"},
    "optimize":           {"color": "mediumseagreen", "pattern": r"## optimize ##"},
}