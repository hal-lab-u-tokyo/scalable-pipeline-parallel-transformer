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
from .dataset_registry import DatasetSpecRevised
from .metric_tracker import MetricTracker

from pipelining import (
    PipelineStage,  # original pytorch impl
    RevPipelineStage,  # our impl
    Schedule1F1B,  # for training
    _ScheduleForwardOnly,  # for inference
)

#パイプライン並列の時のみ必要な情報（最初のGPU番号、最後のGPU番号、パイプラインステージ本体)
@dataclass
class PPContext:
    first_rank  : int
    last_rank   : int
    stage       : Union[PipelineStage, RevPipelineStage]
    sched_train : Schedule1F1B
    sched_valid : _ScheduleForwardOnly

#これまでの学習と評価に必要なオブジェクトを全て集めた道具箱
@dataclass
class Engine:
    cfg: GlobalConfig
    env: EnvContext
    d_spec: DatasetSpecRevised
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
        # Configにmax_grad_normが設定されていれば使い、なければデフォルト1.0
        max_norm = 1.0
        # 追加: パイプライン並列の場合、勾配をマイクロバッチ数で割って平均化する
        if self.cfg.par_mode == ParMode.PP and self.cfg.num_microbatches > 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.div_(self.cfg.num_microbatches)      
        
        with record_function("## clip_grad ##"):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            

    def step(self, epoch_idx: int, is_train: bool, batch_limit: Optional[int] = None):

        # initialize epoch
        self._run_cbs(self.on_epoch_start, epoch_idx, is_train)
        self.tracker.reset("train" if is_train else "valid", epoch_idx)

        self.model.train() if is_train else self.model.eval()
        if self.cfg.par_mode in {ParMode.DDP, ParMode.FSDP}:
            if is_train:
                if self.l_bndl.train_sampler:
                    self.l_bndl.train_sampler.set_epoch(epoch_idx) #各エポックでデータをシャッフルしたサンプラー
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

        # finalize epoch
        self._run_cbs(self.on_epoch_end, epoch_idx, is_train)
        self.tracker.sync()
        self.tracker.log_epoch()
        
        if is_train and self.lr_sched:
            self.lr_sched.step()

    # TODO: scaler
    # scaler自体は行けるが，分散環境におけるtorch.amp.GradScalerはunchargedpぽい
    # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-gpus
    # scaler = torch.amp.GradScaler("cuda") if autocast_dtype == torch.float16 else None
    # ...
    # if scaler:
    #     scaler.scale(batch_loss).backward()
    #     scaler.step(optimizer)
    #     scaler.update()
    # else:
    #     batch_loss.backward()
    #     optimizer.step()

    #パイプライン並列以外の標準的な学習と評価ループ
    def _std_step(self, is_train: bool, batch_limit: Optional[int] = None):
        loader = self.l_bndl.train_loader if is_train else self.l_bndl.val_loader

        #データローダーからバッチを取り出す
        for batch_idx, batch in enumerate(loader):
            self._run_cbs(self.on_batch_start, batch_idx, is_train)
            with record_function("## dataload ##"):                
                # inputs, labels = inputs.to(self.env.device), labels.to(self.env.device)
                input_ids = batch['input_ids'].to(self.env.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.env.device, non_blocking=True)
                labels = batch['labels'].to(self.env.device, non_blocking=True)

            with torch.enable_grad() if is_train else torch.no_grad(): #is_trainがTrueにより勾配計算を有効に
                with record_function("## forward ##"):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
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
    
    #GPUの最初の方の処理
    def _step_pp_first(self, is_train: bool, batch_limit: Optional[int] = None):
        loader = self.l_bndl.train_loader if is_train else self.l_bndl.val_loader #データローダーを受け取る
        p_sched = self.pp_ctx.sched_train if is_train else self.pp_ctx.sched_valid #パイプラインスケジュールの設定を受け取る

        for batch_idx, batch in enumerate(loader):
            self._run_cbs(self.on_batch_start, batch_idx, is_train)

            with record_function("## dataload ##"):
                # inputs, labels = inputs.to(self.env.device), labels.to(self.env.device)
                input_ids = batch["input_ids"].to(self.env.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.env.device, non_blocking=True)
                labels = batch["labels"].to(self.env.device, non_blocking=True)

            # We need to send labels to the last rank for loss computation
            with record_function("## send_labels ##"): #最後のGPU(損失計算者)にはあらかじめ答えとなるlabelを送っとく
                request = torch.distributed.isend(tensor=labels, dst=self.pp_ctx.last_rank)
                request.wait()

            with torch.enable_grad() if is_train else torch.no_grad():
                with record_function("## pipeline_step ##"):
                   p_sched.step(input_ids, attention_mask) #Transformerの最初の方のモデル部分の出力を行う、逆伝播の時はp_shedの制御により逆伝播が自分に返ってくるまで待機する

            if is_train:
                # Since schedr step includes loss backward,
                # we have no need to call loss.backward() here
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
            
    #GPUの中間の人たちの仕事
    def _step_pp_middle(self, is_train: bool, batch_limit: Optional[int] = None):
        num_batches = (
            self.l_bndl.num_batches_train if is_train else self.l_bndl.num_batches_val
        )
        p_sched = self.pp_ctx.sched_train if is_train else self.pp_ctx.sched_valid

        for batch_idx in range(num_batches):
            self._run_cbs(self.on_batch_start, batch_idx, is_train)

            with torch.enable_grad() if is_train else torch.no_grad():
                with record_function("## pipeline_step ##"):
                    p_sched.step() #firstと同様前のGPUの出力を待ち、次の方へ自身のTransformerの出力を送っていく

            if is_train:
                # Since schedr step includes loss backward,
                # we have no need to call loss.backward() here
                self._clip_gradients()
                
                with record_function("## optimize ##"):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            self._run_cbs(self.on_batch_end, batch_idx, is_train)

            if batch_limit is not None and batch_idx >= batch_limit - 1:
                break

    def _step_pp_last(self, is_train: bool, batch_limit: Optional[int] = None):
        num_batches = (
            self.l_bndl.num_batches_train if is_train else self.l_bndl.num_batches_val
        )
        p_sched = self.pp_ctx.sched_train if is_train else self.pp_ctx.sched_valid
        labels = torch.zeros(self.cfg.batch_size, dtype=torch.int64).to(self.env.device)

        for batch_idx in range(num_batches):
            self._run_cbs(self.on_batch_start, batch_idx, is_train)
            
            with record_function("## recv_labels ##"):
                request = dist.irecv(tensor=labels, src=self.pp_ctx.first_rank) #firstが送ってきたlabelを受け取る
                request.wait()

            # In validation, we use the _ScheduleForwardOnly schedule
            # Since it does not accept target and losses as arguments of step() and does not return loss, we need to pass them as None and handle loss computation separately.
            mb_losses = [] if is_train else None
            with torch.enable_grad() if is_train else torch.no_grad():
                with record_function("## pipeline_step ##"):
                    outputs = p_sched.step(target=labels if is_train else None, losses=mb_losses) #p_sched.step()の中でbackwardを呼び出す
            
            batch_loss = sum(mb_losses) / len(mb_losses) if is_train else self.loss_fn(outputs, labels)   
            
            current_loss_val = batch_loss.item() if isinstance(batch_loss, torch.Tensor) else batch_loss
            if self.env.main_logger:
                # 1. NaN / Inf (爆発) のチェック
                if math.isnan(batch_loss) or math.isinf(batch_loss):
                     self.env.main_logger.error(
                        f"🚨 [Rank {self.env.rank}] Loss EXPLOSION detected at Batch {batch_idx}! "
                        f"Value: {current_loss_val}"
                    )
                # 2. 毎バッチLossを表示 (デバッグ用)
                else:
                    self.env.main_logger.info(
                        f"[Rank {self.env.rank}][Batch {batch_idx}/{num_batches}] Loss: {current_loss_val:.6f}"
                    )
            
            if is_train:
                # Since schedr step includes loss backward,
                # we have no need to call loss.backward() here
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

#Engineオブジェクトを構築する、cfgやenvなどの全体設定やデータセット、モデルや学習ロジックを全て受け取りEngineオブジェクトを返す
def build_engine(
    cfg: GlobalConfig,
    env: EnvContext,
    d_spec: DatasetSpecRevised,
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
        env.main_logger.info("===== Engine Setup =====")
        env.main_logger.info(f"Parallelization mode: {cfg.par_mode.value}")
        # TODO: print some setup information

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