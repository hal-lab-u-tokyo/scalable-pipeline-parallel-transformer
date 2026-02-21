from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List, Callable
import torch
import torch.distributed as dist
from torch.profiler import record_function
from torch.optim.lr_scheduler import LRScheduler
import math

from .config import ParMode, GlobalConfig
from .env import EnvContext
from .dataloader import LoaderBundle
from .dataset_registry import DatasetSpec
from .metric_tracker import MetricTracker

from pipelining import (
    PipelineStage,  # original pytorch impl
    RevPipelineStage,  # our impl
    Schedule1F1B,  # for training
    _ScheduleForwardOnly,  # for inference
)

@dataclass
class PPContext:
    first_rank  : int
    last_rank   : int
    stage       : Union[PipelineStage, RevPipelineStage]
    sched_train : Schedule1F1B
    sched_valid : _ScheduleForwardOnly

@dataclass
class Engine:
    cfg: GlobalConfig
    env: EnvContext
    d_spec: DatasetSpec
    l_bndl: LoaderBundle
    model: torch.nn.Module
    loss_fn: torch.nn.Module
    tracker: MetricTracker
    optimizer: torch.optim.Optimizer
    lr_sched: Optional[LRScheduler] = None
    pp_ctx: Optional[PPContext] = None
    on_batch_start  : List[Callable] = field(default_factory=list, repr=False)
    on_batch_end    : List[Callable] = field(default_factory=list, repr=False)
    on_epoch_start  : List[Callable] = field(default_factory=list, repr=False)
    on_epoch_end    : List[Callable] = field(default_factory=list, repr=False)

    def _run_cbs(self, hooks: List[Callable], *args, **kwargs):
        for hook in hooks:
            if callable(hook):
                hook(*args, **kwargs)
            else:
                raise ValueError(f"Hook {hook} is not callable")

    def _clip_gradients(self):
        """Perform gradient clipping."""
        max_norm = 1.0
        if self.cfg.par_mode == ParMode.PP and self.cfg.num_microbatches > 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.div_(self.cfg.num_microbatches)      
        
        with record_function("## clip_grad ##"):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            

    def step(self, epoch_idx: int, is_train: bool, batch_limit: Optional[int] = None):
        self._run_cbs(self.on_epoch_start, epoch_idx, is_train)
        self.tracker.reset("train" if is_train else "valid", epoch_idx)

        self.model.train() if is_train else self.model.eval()
        if self.cfg.par_mode in {ParMode.DDP, ParMode.FSDP}:
            if is_train:
                if self.l_bndl.train_sampler:
                    self.l_bndl.train_sampler.set_epoch(epoch_idx)
            else:
                if self.l_bndl.val_sampler:
                    self.l_bndl.val_sampler.set_epoch(epoch_idx)

        # Run epoch
        if self.cfg.par_mode in {ParMode.NONE, ParMode.DDP, ParMode.FSDP}:
            self._std_step(is_train, batch_limit)
        elif self.cfg.par_mode in {ParMode.PP}:
            if self.pp_ctx:
                if self.pp_ctx.stage.is_first:
                    self._step_pp_first(is_train, batch_limit)
                elif self.pp_ctx.stage.is_last:
                    self._step_pp_last(is_train, batch_limit)
                else:
                    self._step_pp_middle(is_train, batch_limit)
            else:
                raise ValueError("PPContext is not initialized")
        else:
            raise ValueError(f"Invalid parallelization mode: {self.cfg.par_mode.value}")

        self._run_cbs(self.on_epoch_end, epoch_idx, is_train)
        self.tracker.sync()
        self.tracker.log_epoch()
        
        if is_train and self.lr_sched:
            self.lr_sched.step()

    # --- Standard Step (Non-Pipeline) ---
    def _std_step(self, is_train: bool, batch_limit: Optional[int] = None):
        loader = self.l_bndl.train_loader if is_train else self.l_bndl.val_loader

        for batch_idx, batch in enumerate(loader):
            self._run_cbs(self.on_batch_start, batch_idx, is_train)
            with record_function("## dataload ##"):                
                # --- 変更点: 画像入力に対応 ---
                # batchのキーはDataset/Collatorの実装に依存しますが、ここでは 'image' と仮定
                # 辞書ではなくタプルで返ってくる場合は inputs, labels = batch としてください
                inputs = batch[0].to(self.env.device, non_blocking=True)
                labels = batch[1].to(self.env.device, non_blocking=True)
                # ViTでは通常attention_maskは不要（固定長パッチなら）

            with torch.enable_grad() if is_train else torch.no_grad():
                with record_function("## forward ##"):
                    # ViTモデルへの入力
                    outputs = self.model(inputs)
                with record_function("## losscomp ##"):
                    batch_loss = self.loss_fn(outputs, labels)

            if is_train:
                with record_function("## backward ##"):
                    batch_loss.backward()
                    
                self._clip_gradients()
                
                with record_function("## optimize ##"):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self._run_cbs(self.on_batch_end, batch_idx, is_train)
            self.tracker.update(batch_loss.detach(), outputs.detach(), labels)
            if batch_idx % 64 == 0:
                self.tracker.log_batch(batch_idx, len(loader))
            
            if batch_limit is not None and batch_idx >= batch_limit - 1:
                break
    
    # --- Pipeline: First Stage ---
    def _step_pp_first(self, is_train: bool, batch_limit: Optional[int] = None):
        loader = self.l_bndl.train_loader if is_train else self.l_bndl.val_loader
        p_sched = self.pp_ctx.sched_train if is_train else self.pp_ctx.sched_valid

        for batch_idx, batch in enumerate(loader):
            self._run_cbs(self.on_batch_start, batch_idx, is_train)

            with record_function("## dataload ##"):
                # --- 変更点: 画像入力に対応 ---
                inputs = batch[0].to(self.env.device, non_blocking=True)
                labels = batch[1].to(self.env.device, non_blocking=True)

            # Send labels to the last rank
            with record_function("## send_labels ##"):
                request = torch.distributed.isend(tensor=labels, dst=self.pp_ctx.last_rank)
                request.wait()

            with torch.enable_grad() if is_train else torch.no_grad():
                with record_function("## pipeline_step ##"):
                   # --- 変更点: attention_mask削除 ---
                   # Scheduleのstepに画像を渡す
                   p_sched.step(inputs)

            if is_train:
                self._clip_gradients()
                
                with record_function("## optimize ##"):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self._run_cbs(self.on_batch_end, batch_idx, is_train)

            if batch_limit is not None and batch_idx >= batch_limit - 1:
                break
            
            del inputs
            del labels
            del batch
    
    # --- Pipeline: Middle Stage (変更なし: 入力データに依存しない) ---
    def _step_pp_middle(self, is_train: bool, batch_limit: Optional[int] = None):
        num_batches = (
            self.l_bndl.num_batches_train if is_train else self.l_bndl.num_batches_val
        )
        p_sched = self.pp_ctx.sched_train if is_train else self.pp_ctx.sched_valid

        for batch_idx in range(num_batches):
            self._run_cbs(self.on_batch_start, batch_idx, is_train)

            with torch.enable_grad() if is_train else torch.no_grad():
                with record_function("## pipeline_step ##"):
                    p_sched.step() 

            if is_train:
                self._clip_gradients()
                
                with record_function("## optimize ##"):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            self._run_cbs(self.on_batch_end, batch_idx, is_train)

            if batch_limit is not None and batch_idx >= batch_limit - 1:
                break

    # --- Pipeline: Last Stage (変更なし: Label受信ロジックは同じ) ---
    def _step_pp_last(self, is_train: bool, batch_limit: Optional[int] = None):
        num_batches = (
            self.l_bndl.num_batches_train if is_train else self.l_bndl.num_batches_val
        )
        p_sched = self.pp_ctx.sched_train if is_train else self.pp_ctx.sched_valid
        labels = torch.zeros(self.cfg.batch_size, dtype=torch.int64).to(self.env.device)

        for batch_idx in range(num_batches):
            self._run_cbs(self.on_batch_start, batch_idx, is_train)
            
            with record_function("## recv_labels ##"):
                request = dist.irecv(tensor=labels, src=self.pp_ctx.first_rank)
                request.wait()

            mb_losses = [] if is_train else None
            with torch.enable_grad() if is_train else torch.no_grad():
                with record_function("## pipeline_step ##"):
                    outputs = p_sched.step(target=labels if is_train else None, losses=mb_losses)
            
            batch_loss = sum(mb_losses) / len(mb_losses) if is_train else self.loss_fn(outputs, labels)   
            
            current_loss_val = batch_loss.item() if isinstance(batch_loss, torch.Tensor) else batch_loss
            if self.env.main_logger:
                if math.isnan(batch_loss) or math.isinf(batch_loss):
                     self.env.main_logger.error(
                        f"🚨 [Rank {self.env.rank}] Loss EXPLOSION at Batch {batch_idx}! Value: {current_loss_val}"
                    )
                else:
                    self.env.main_logger.info(
                        f"[Rank {self.env.rank}][Batch {batch_idx}/{num_batches}] Loss: {current_loss_val:.6f}"
                    )
            
            if is_train:
                self._clip_gradients()
                
                with record_function("## optimize ##"):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self._run_cbs(self.on_batch_end, batch_idx, is_train)
            self.tracker.update(batch_loss, outputs.detach(), labels)
            if batch_idx % 64 == 0:
                self.tracker.log_batch(batch_idx, num_batches)

            if batch_limit is not None and batch_idx >= batch_limit - 1:
                break

def build_engine(
    cfg: GlobalConfig,
    env: EnvContext,
    d_spec: DatasetSpec,
    l_bndl: LoaderBundle,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_sched: Optional[LRScheduler] = None,
    pp_ctx: Optional[PPContext] = None,
) -> Engine:

    tracker = MetricTracker(
        d_spec.num_classes,
        (1, 5),
        cfg.num_epochs,
        env.device,
        env.main_logger,
    )

    if cfg.par_mode in {ParMode.PP}:
        if pp_ctx is None:
            raise ValueError("PPContext is not initialized")

    if env.main_logger:
        env.main_logger.info("===== Engine Setup (ViT) =====")
        env.main_logger.info(f"Parallelization mode: {cfg.par_mode.value}")

    return Engine(
        cfg=cfg,
        env=env,
        d_spec=d_spec,
        l_bndl=l_bndl,
        model=model,
        loss_fn=loss_fn,
        tracker=tracker,
        optimizer=optimizer,
        lr_sched=lr_sched,
        pp_ctx=pp_ctx if cfg.par_mode == ParMode.PP else None,
    )