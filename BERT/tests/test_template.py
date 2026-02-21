# tests/test_template.py
import itertools, subprocess, yaml, pathlib, pytest, shlex, os, signal, sys

HERE = pathlib.Path(__file__).parent
CFG = yaml.safe_load(open(HERE / "quick_test.yaml"))

# ★ is_pareprop も含めてマトリックスを作成
matrix = list(
    itertools.product(
        CFG["parallel"],
        CFG["reversible"],
        CFG["is_pareprop"], # ★ 追加
        CFG["autocast_dtype"],
        CFG["dataset"],
        CFG["batch_size"],
    )
)

def _mkid(val):
    if isinstance(val, dict):
        return val["mode"]
    # ★ is_pareprop の ID を追加 (rev=True の場合のみ意味を持つ)
    if isinstance(val, bool) and "is_pareprop" in CFG and val in CFG["is_pareprop"]:
         # この引数が is_pareprop か reversible か区別できないため、より工夫が必要かも
         # 例: pytest.param を使うか、ID生成を修正
         # ここでは簡易的に rev/std/pareprop/seq のように表示を試みる (要調整)
         pass # _mkid の修正は複雑なので一旦保留 (pytest がデフォルトIDを生成)
    elif isinstance(val, bool):
         return "rev" if val else "std"
    return str(val)

# ★ is_pareprop を parametrize に追加
@pytest.mark.parametrize(
    "par_cfg,reversible,is_pareprop,autocast_dtype,dataset,batch_size", # ★ 追加
    matrix,
    # ids=lambda v: _mkid(v), # ID生成は一旦 pytest にお任せ
)
def test_template_smoke(
    par_cfg, reversible, is_pareprop, autocast_dtype, dataset, batch_size, tmp_path # ★ 追加
):
    """launch torchrun and expect exit‑0 within given time"""

    # ★ is_pareprop=True なのに reversible=False は無効なのでスキップ
    if is_pareprop and not reversible:
        pytest.skip("is_pareprop=True requires reversible=True")

    cmd = [
        "torchrun",
        f"--nproc_per_node={par_cfg['nproc']}",
        # ★★★ エントリーポイントをあなたの BERT 実験用に変更 ★★★
        # 例: "-m", "my_bert_experiment.run" や "run_bert_exp.py" など
        "run_bert_experiment.py", # 仮のファイル名
        "--parallel-mode", par_cfg["mode"],
        "--batch-size", str(batch_size),
        "--autocast-dtype", autocast_dtype,
        "--dataset", dataset,
        "--num-epochs", str(CFG["num_epochs"]),
        "--seed", "42",
        # ★ BERT モデルパラメータを YAML から読み込んで追加
        "--model", CFG["model"],
        "--num-hidden-layers", str(CFG["num_hidden_layers"]),
        "--hidden-size", str(CFG["hidden_size"]),
        "--num-attention-heads", str(CFG["num_attention_heads"]),
        "--max-position-embeddings", str(CFG["max_position_embeddings"]),
        "--tokenizer-name", CFG["tokenizer_name"],
        # ★ テスト用にデータセットをサブセット化するフラグ (args.py に --use-subset が必要)
        "--use-subset", # 必要ならコメント解除
        # ★ チェックポイント関連のフラグ (args.py に必要)
        # "--save-checkpoint", # テストでは通常不要
        # "--load-checkpoint", # テストでは通常不要
        # ★ 出力ディレクトリを一時ディレクトリに (args.py に --log-rdir などが必要)
        "--log-rdir", str(tmp_path / "logs"),
        "--checkpoint-rdir", str(tmp_path / "checkpoints"),
        "--profile-rdir", str(tmp_path / "profiles"),
    ]
    # ★ reversible フラグを追加
    if reversible:
        cmd.append("--reversible")
        # ★ is_pareprop フラグを追加 (reversible の場合のみ)
        if is_pareprop:
            cmd.append("--is-pareprop")

    log = tmp_path / "run.log"
    print("$", " ".join(shlex.quote(c) for c in cmd), file=sys.stderr)

    try:
        proc = subprocess.run(
            cmd,
            stdout=open(log, "w"),
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONWARNINGS": "ignore"},
            # ★ タイムアウトを設定
            timeout=CFG["timeout"],
        )
    # ★ タイムアウト時の例外処理を追加
    except subprocess.TimeoutExpired:
         print(log.read_text(), file=sys.stderr)
         pytest.fail(f"Timeout ({CFG['timeout']}s exceeded)", pytrace=False)

    if proc.returncode == -signal.SIGKILL:
        pytest.fail("Killed by SIGKILL (likely OOM)", pytrace=False)
    if proc.returncode == -signal.SIGTERM:
        pytest.fail("Killed by SIGTERM (time‑limit?)", pytrace=False)

    if proc.returncode != 0:
        print(log.read_text(), file=sys.stderr)

    assert proc.returncode == 0, f"non‑zero exit ({proc.returncode})"