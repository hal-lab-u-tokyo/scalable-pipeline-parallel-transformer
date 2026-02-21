import os
from dataclasses import dataclass
from pathlib import Path
import enum
import torch
import argparse

#args.pyから受け取った引数を扱いやすいデータ構造(Globalconfig)に変換する
class ExpMode(enum.Enum):
    TRAINING = "training"
    PROFILING = "profiling"
    ACTV_ERR = "actv-err"
    ACC = "acc"


class ParMode(enum.Enum):
    NONE = "none"
    DDP = "ddp"
    FSDP = "fsdp"
    PP = "pp"

# (BlockType は args.py にないので削除)

@dataclass(frozen=True)
class GlobalConfig:
    # 実験モード・並列化設定
    exp_mode: ExpMode
    par_mode: ParMode
    is_pareprop: bool
    reversible: bool
    inverse: bool
    checkpointing: bool # Activation Checkpointing

    # モデル設定
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    max_position_embeddings: int

    # データセット・学習設定
    autocast_dtype: torch.dtype
    dataset: str
    debug_subset: bool
    batch_size: int
    microbatch_size: int
    num_microbatches: int
    num_epochs: int
    lr: float
    gamma: float
    num_workers: int
    seed: int
    
    # パス・I/O設定
    dataset_rdir: Path
    log_rdir: Path
    checkpoint_rdir: Path
    profile_rdir: Path
    load_checkpoint: bool
    save_checkpoint: bool

    # ID設定
    exp_id: str
    env_id: str


def build_global_config(args: argparse.Namespace) -> GlobalConfig:
    """
    コマンドライン引数から全体設定オブジェクトを構築する
    """

    # 文字列をtorch.dtypeオブジェクトに変換
    dtype_table = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    try:
        dtype = dtype_table[args.autocast_dtype]
    except KeyError:
        raise ValueError(f"無効なデータ型です: {args.autocast_dtype}")

    # パイプライン並列時のバッチサイズ整合性チェック
    if args.microbatch_size * args.num_microbatches != args.batch_size:
        raise ValueError(
            "パイプライン並列では、batch_size は "
            "microbatch_size * num_microbatches と一致する必要があります"
        )

    # 実験を識別するための一意なIDを生成
    exp_id = ""
    exp_id += f"{args.exp_mode}_"
    exp_id += f"{args.par_mode}_"
    # モデルのタイプを追加 (変更点)
    if args.reversible:
        if args.is_pareprop:
            exp_id += "pareprop_"
            
        elif args.inverse:
            exp_id += "inverse"
        else:
            exp_id += "rev-seq_"
    elif args.checkpointing:
        exp_id += "wcp_"
    else:
        exp_id += "std_"
    exp_id += f"{args.autocast_dtype}_"
    exp_id += f"{args.dataset}_"
    exp_id += f"seed{args.seed:04d}"
    exp_id += f"numLayer{args.num_hidden_layers}"

    # SLURMなどのジョブスケジューラ情報から環境IDを生成
    env_id = f"{os.environ.get('SLURM_JOB_PARTITION', 'UNKNOWN')}"
    
    if args.is_pareprop and not args.reversible:
        raise ValueError("is_pareprop=True requires reversible=True")

    return GlobalConfig(
        exp_mode=ExpMode(args.exp_mode),
        par_mode=ParMode(args.par_mode),
        reversible=args.reversible,
        is_pareprop=args.is_pareprop,
        inverse=args.inverse, 
        checkpointing=args.checkpointing,
        num_hidden_layers=args.num_hidden_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        max_position_embeddings=args.max_position_embeddings,
        autocast_dtype=dtype,
        dataset=args.dataset,
        debug_subset=args.debug_subset,
        batch_size=args.batch_size,
        microbatch_size=args.microbatch_size,
        num_microbatches=args.num_microbatches,
        num_epochs=args.num_epochs,
        lr=args.lr,
        gamma=args.gamma,
        num_workers=8,
        seed=args.seed,
        dataset_rdir=Path("/dataset"), 
        log_rdir=Path("logs"),
        checkpoint_rdir=Path("checkpoints"),
        profile_rdir=Path("profiles"),
        load_checkpoint=args.load_checkpoint,
        save_checkpoint=args.save_checkpoint,
        exp_id=exp_id,
        env_id=env_id,
    )