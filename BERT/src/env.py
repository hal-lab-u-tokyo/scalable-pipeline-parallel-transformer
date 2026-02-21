from __future__ import annotations
from dataclasses import dataclass
import logging, os, sys
import torch, torch.distributed as dist
from typing import Optional
from .config import ParMode, GlobalConfig
from .logger import build_logger

#分散学習に関する情報を一箇所にまとめる入れ物
@dataclass
class EnvContext:
    world_size: int #GPUの数
    rank: int #全GPUの中での自分のID
    local_rank: int
    device: torch.device
    proc_logger: logging.Logger
    node_logger: Optional[logging.Logger] = None
    main_logger: Optional[logging.Logger] = None
    pg: Optional[dist.ProcessGroup] = None

    #world_sizeが1より大きいか、すなわち分散学習中かどうか判定
    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1 

    def barrier(self):
        if self.is_distributed:
            dist.barrier()

    def close(self):
        """call once, on every rank, after training finishes"""
        if self.is_distributed and dist.is_initialized():
            self.barrier()
            dist.destroy_process_group()
            if self.proc_logger:
                self.proc_logger.info("Process group destroyed.")


def build_env(cfg: GlobalConfig) -> EnvContext:
    """
    Build the environment context for distributed training.
    """

    if cfg.par_mode == ParMode.NONE:
        world_size = 1
        rank = 0
        local_rank = 0
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        pg = None
    else:
        if not dist.is_initialized():
            try:
                #分散学習なら、これから分散学習を開始すると宣言し、GPU間の通信路を確保する
                dist.init_process_group("nccl", init_method="env://")
            except Exception as e:
                print(f"[EnvBuilder] init_process_group failed: {e}", file=sys.stderr)
                sys.exit(1)

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.getenv("LOCAL_RANK", "-1"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        pg = dist.group.WORLD

    master_rank = world_size - 1  # ajusting for pp

    log_dir = cfg.log_rdir / cfg.exp_id / cfg.env_id
    os.makedirs(log_dir, exist_ok=True)

    proc_logger = build_logger(log_dir, f"proc_{rank}", console_output=False)

    node_logger = None
    if local_rank == 0:
        node_logger = build_logger(log_dir, "node", console_output=False)

    main_logger = None
    if rank == master_rank:
        main_logger = build_logger(log_dir, "main", console_output=True)

    return EnvContext(
        world_size, rank, local_rank, device, proc_logger, node_logger, main_logger, pg
    )