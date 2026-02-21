#モデルのチェックポイント(重みパラメータ)を保存および読み込みする

from __future__ import annotations
from pathlib import Path
from typing import Union
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP # FSDP1
from torch.distributed.fsdp import FSDPModule  # FSDP2
from .config import ParMode, GlobalConfig
from .env import EnvContext

#実験設定に基づきチェックポイントを保存するディレクトリのパスを生成する
def _ckpt_dir(cfg: GlobalConfig, env: EnvContext) -> Path:
    rev_part = "rev" if cfg.reversible else "std" #reversibleがtrueならrev, そうでなければstd(standard)を代入
    ldir = f"{cfg.par_mode.value}_{rev_part}_{cfg.autocast_dtype}_{cfg.dataset}_{cfg.seed:04d}" #ディレクトリ名を並列化の種類、revかstd、データ型、データセット名、実験の乱数シードとする
    return (cfg.checkpoint_rdir / ldir).resolve() #実験設定cfgにあるチェックポイントのディレクトリと今作ったディレクトリ名を結合して絶対パスに変換する

#チェックポイントのファイル名を生成
def _file_name(cfg: GlobalConfig, env: EnvContext, suffix: str = "latest") -> str:
    if cfg.par_mode in {ParMode.PP, ParMode.FSDP}: #もし並列化モードがPPやFSDPの場合
        return f"{suffix}_rank{env.rank}.pth" #ファイル名にはGPUの番号であるrankを含める(PPやFSDPは重みを各GPUに分割するから)
    return f"{suffix}.pth"

#モデルのチェックポイントをファイルに保存する
def save_checkpoint(
    model: Union[torch.nn.Module, DDP, FSDPModule],
    cfg: GlobalConfig,
    env: EnvContext,
    *,
    suffix: str = "latest",
) -> None:

    #ファイル名を生成して保存先のディレクトリパスを取得し、mkdirで生成
    ckpt_dir = _ckpt_dir(cfg, env)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    #保存ファイルの完全なパスを生成
    path = ckpt_dir / _file_name(cfg, env, suffix)

    #重みパラメータなどのモデルの状態を取得
    if isinstance(model, DDP):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    #パスにモデルの重みを保存し、保存が完了したことをログに出力
    torch.save(state, path)
    env.proc_logger.info("Checkpoint saved  → %s", path.as_posix())

#ファイルからモデルの状態を読み込む
def load_checkpoint(
    model: Union[torch.nn.Module, DDP, FSDPModule],
    cfg: GlobalConfig,
    env: EnvContext,
    *,
    suffix: str = "latest",
    strict: bool = True,
) -> bool:

    #パスを指定
    path = _ckpt_dir(cfg, env) / _file_name(cfg, env, suffix)
    if not path.is_file():
        env.proc_logger.warning("Checkpoint %s not found; start from scratch", path)
        return False

    #重みを読み込む
    state = torch.load(path, map_location="cpu")

    #読み込みを行う
    if isinstance(model, DDP):
        model.module.load_state_dict(state, strict=strict)
    else:
        model.load_state_dict(state, strict=strict)

    env.proc_logger.info("Checkpoint loaded  ← %s", path.as_posix())
    return True