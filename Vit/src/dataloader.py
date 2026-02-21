from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from .config import ParMode, GlobalConfig
from .env import EnvContext
from .dataset_registry import load_dataset


@dataclass
class LoaderBundle:
    train_loader: DataLoader | None
    val_loader: DataLoader | None
    train_sampler: Sampler | None
    val_sampler: Sampler | None
    num_batches_train: int
    num_batches_val: int


def build_loaders(cfg: GlobalConfig, env: EnvContext) -> LoaderBundle:

    need_dataset = cfg.par_mode in {ParMode.NONE, ParMode.DDP, ParMode.FSDP} or (
        cfg.par_mode == ParMode.PP and env.rank == 0
    )
    need_sampler = cfg.par_mode in {ParMode.DDP, ParMode.FSDP} and env.is_distributed

    if cfg.par_mode in {ParMode.NONE, ParMode.DDP, ParMode.FSDP}:
        assert cfg.batch_size % env.world_size == 0, (
            f"Batch size {cfg.batch_size} must be divisible by world size {env.world_size} in parallel mode {cfg.par_mode}."
        )
        local_batch_size = cfg.batch_size // env.world_size
    elif cfg.par_mode == ParMode.PP:
        local_batch_size = cfg.batch_size
    else:
        raise ValueError(f"Unsupported parallel mode: {cfg.par_mode}")

    if need_dataset:
        ds_train, ds_val = load_dataset(
            cfg.dataset,
            cfg.dataset_rdir,
            subset_N=256 * local_batch_size if cfg.debug_subset else None,
        )
        train_sampler = (
            DistributedSampler(ds_train, shuffle=True) if need_sampler else None
        )
        val_sampler = (
            DistributedSampler(ds_val, shuffle=False) if need_sampler else None
        )

        train_loader = DataLoader(
            ds_train,
            batch_size=local_batch_size,
            drop_last=cfg.par_mode == ParMode.PP,
            shuffle=not need_sampler,
            sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            prefetch_factor=8,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=local_batch_size,
            drop_last=cfg.par_mode == ParMode.PP,
            shuffle=False,
            sampler=val_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            prefetch_factor=8,
            persistent_workers=True,
        )
        num_batches_train = len(train_loader)
        num_batches_val = len(val_loader)
    else:
        train_loader = val_loader = train_sampler = val_sampler = None
        num_batches_train = num_batches_val = 0

    if cfg.par_mode == ParMode.PP and env.is_distributed:
        t = torch.tensor(
            [num_batches_train, num_batches_val], device=env.device, dtype=torch.int64
        )
        dist.broadcast(t, src=0)
        num_batches_train, num_batches_val = t.tolist()

    return LoaderBundle(
        train_loader,
        val_loader,
        train_sampler,
        val_sampler,
        num_batches_train,
        num_batches_val,
    )
