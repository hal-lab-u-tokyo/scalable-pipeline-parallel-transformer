from __future__ import annotations

from abc import ABC, abstractmethod
import time
from collections import defaultdict 
from typing import Optional

import torch
import torch.nn as nn

from src.config import GlobalConfig, ParMode, ExpMode
from src.env import EnvContext
from src.dataloader import build_loaders, LoaderBundle
from src.dataset_registry import DatasetSpec, REGISTRY
from src.engine import Engine, build_engine, PPContext
from src.logging_utils import (
    log_sys_info,
    log_exp_config,
    log_node_info,
    log_gpu_static_info,
    log_model_info,
)
from src.seed import set_seed
from src.ckpt_io import save_checkpoint, load_checkpoint

from models.vit_stages import (
    ViTFullStage,
    ViTFirstStage,
    ViTMidStage,
    ViTLastStage,
)

from models.normal_vit_stages import (
    NormalViTFullStage,
    NormalViTFirstStage,
    NormalViTMidStage,
    NormalViTLastStage,
)

# --- パイプライン関連のインポート (変更なし) ---
from pipelining import (
    PipelineStage,
    PipelineStageWithCP,
    RevPipelineStage,
    RevPipelineStageWithPareprop, # 改造された _RevPipelineStageBase を使う
    Schedule1F1B,
    _ScheduleForwardOnly,
)

class Experiment(ABC):
    
    def __init__(self, cfg: GlobalConfig, env: EnvContext):
        self.cfg, self.env = cfg, env
        self.d_spec: DatasetSpec = REGISTRY[cfg.dataset]
        self.l_bndl: Optional[LoaderBundle] = None
        self.engine: Optional[Engine] = None
        self.exp_start_time = time.time()

        set_seed(cfg.seed + env.rank, deterministic=True)

        if env.main_logger:
            log_sys_info(env.main_logger)
            log_exp_config(env.main_logger, cfg)
        if env.node_logger:
            log_node_info(env.node_logger)
        env.proc_logger.info(
            "Global Rank: %s, Local Rank: %s", env.rank, env.local_rank
        )
        log_gpu_static_info(env.proc_logger, env.device)       

    def before_run(self):
        self._build_data()
        self._build_model_and_engine()

    @abstractmethod
    def run(self): # pragma: no cover – abstract
        """Execute the core workload (training / profiling etc)."""

    def after_run(self):
        """Shared teardown: checkpoint, final log, env close."""
        if self.cfg.save_checkpoint and self.engine is not None:
            save_checkpoint(self.engine.model, self.cfg, self.env)
        if self.env.main_logger:
            self.env.main_logger.info("===== Experiment Complete =====")
            self.env.main_logger.info(
                "Total time taken: %.2f seconds", time.time() - self.exp_start_time
            )
        self.env.close()
        
    def _build_data(self):
        self.l_bndl = build_loaders(self.cfg, self.env)
        
    
    def _build_model_and_engine(self):
        cfg, env = self.cfg, self.env
        assert self.l_bndl is not None, "Data must be built first"
        # ★ GlobalConfig から BERT の設定を取得
        num_hidden_layers = cfg.num_hidden_layers
        
        # ★ パイプライン並列時のレイヤー数計算
        if cfg.par_mode == ParMode.PP:
            assert num_hidden_layers % env.world_size == 0, (
                f"num_hidden_layers {num_hidden_layers} must be divisible by world_size {env.world_size}"
            )
            num_blocks_per_stage = num_hidden_layers // env.world_size # ★ ステージごとのブロック数
        else:
            num_blocks_per_stage = num_hidden_layers # 非PP時は全レイヤー

        # ★ BERT ステージ共通の引数を準備
        model_kwargs = dict(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_attention_heads,
            num_blocks=num_blocks_per_stage,
            enable_amp=(cfg.autocast_dtype != torch.float32),
            autocast_dtype=cfg.autocast_dtype,
        )

        # ★ モデルクラスを BERT 用に変更 & 引数を調整
        model: nn.Module # 型ヒント
        if cfg.par_mode in {ParMode.NONE, ParMode.DDP, ParMode.FSDP}:
            # BertFullStage は num_classes も必要
            if cfg.reversible:
                if cfg.inverse:
                    
                    full_model_kwargs = model_kwargs.copy()
                    full_model_kwargs.update(
                        num_classes=self.d_spec.num_classes, # ★ データセットから取得
                        # BertFullStage は Embedding も持つので必要な引数を追加
                        img_size=cfg.img_size,
                        patch_size=cfg.patch_size,
                        in_chans=cfg.in_chans,
                    )
                    # model = BertFullStageWithInverse(**full_model_kwargs).to(env.device)
                    
                else:
                    full_model_kwargs = model_kwargs.copy()
                    full_model_kwargs.update(
                        num_classes=self.d_spec.num_classes, # ★ データセットから取得
                        # BertFullStage は Embedding も持つので必要な引数を追加
                        is_pareprop=cfg.is_pareprop,
                        img_size=cfg.img_size,
                        patch_size=cfg.patch_size,
                        in_chans=cfg.in_chans,
                    )
                    model = ViTFullStage(**full_model_kwargs).to(env.device)
            else:
                full_model_kwargs = model_kwargs.copy()
                full_model_kwargs.update(
                    num_classes=self.d_spec.num_classes, # ★ データセットから取得
                    # BertFullStage は Embedding も持つので必要な引数を追加
                    img_size=cfg.img_size,
                    patch_size=cfg.patch_size,
                    in_chans=cfg.in_chans,
                )      
                model =  NormalViTFullStage(**full_model_kwargs).to(env.device)        
                                       

        elif cfg.par_mode == ParMode.PP:
            if env.rank == 0:
                if cfg.reversible:
                    if cfg.inverse:
                        
                        # BertFirstStage は Embedding を持つので固有の引数を追加
                        first_stage_kwargs = model_kwargs.copy()
                        first_stage_kwargs.update(
                            img_size=cfg.img_size,
                            patch_size=cfg.patch_size,
                            in_chans=cfg.in_chans,
                        )
                        # model = BertFirstStageWithInverse(**first_stage_kwargs).to(env.device)
                        
                    else:
                        # BertFirstStage は Embedding を持つので固有の引数を追加
                        first_stage_kwargs = model_kwargs.copy()
                        first_stage_kwargs.update(
                            is_pareprop=cfg.is_pareprop,
                            img_size=cfg.img_size,
                            patch_size=cfg.patch_size,
                            in_chans=cfg.in_chans,
                        )
                        model =  ViTFirstStage(**first_stage_kwargs).to(env.device)
                        
                else:
                    first_stage_kwargs = model_kwargs.copy()
                    first_stage_kwargs.update(
                        img_size=cfg.img_size,
                        patch_size=cfg.patch_size,
                        in_chans=cfg.in_chans,
                    )
                    model = NormalViTFirstStage(**first_stage_kwargs).to(env.device)                                      

            elif env.rank == env.world_size - 1:
                if cfg.reversible:
                    if cfg.inverse:                     
                        # BertLastStage は Head を持つので num_classes が必要
                        last_stage_kwargs = model_kwargs.copy()
                        last_stage_kwargs.update(num_classes=self.d_spec.num_classes)
                        # model = BertLastStageWithInverse(**last_stage_kwargs).to(env.device)
                        
                    else:
                        last_stage_kwargs = model_kwargs.copy()
                        last_stage_kwargs.update(num_classes=self.d_spec.num_classes, is_pareprop=cfg.is_pareprop)
                        model = ViTLastStage(**last_stage_kwargs).to(env.device)
                        
                
                else:
                    last_stage_kwargs = model_kwargs.copy()
                    last_stage_kwargs.update(num_classes=self.d_spec.num_classes)
                    model = NormalViTLastStage(**last_stage_kwargs).to(env.device)                    
                        
                
            else:
                # BertMidStage は共通引数のみ
                if cfg.reversible:
                    if cfg.inverse:
                        raise NotImplementedError("Inverse mode for ViT is not implemented yet.")
                         # model = BertMidStageWithInverse(**model_kwargs).to(env.device)
                        
                    else:
                        mid_stage_kwargs = model_kwargs.copy()
                        mid_stage_kwargs.update(is_pareprop=cfg.is_pareprop)
                        model = ViTMidStage(**mid_stage_kwargs).to(env.device)
                        
                else:
                    model = NormalViTMidStage(**model_kwargs).to(env.device)
                                           
                        
        else:
            raise ValueError(cfg.par_mode)

        if env.main_logger:
            log_model_info(env.main_logger, model)

        if cfg.load_checkpoint:
            load_checkpoint(model, cfg, env)

        # -------------- parallel wrappers (★ 変更なし) ------------------
        pp_ctx = None
        if cfg.par_mode == ParMode.NONE:
            pass

        elif cfg.par_mode == ParMode.DDP:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[env.local_rank])

        elif cfg.par_mode == ParMode.FSDP:
            from torch.distributed.fsdp import fully_shard
            model = fully_shard(model, reshard_after_forward=True)
            
        elif cfg.par_mode == ParMode.PP:
            stage_kwargs = dict(
                submodule=model, # BertFirst/Mid/LastStage インスタンス
                stage_index=env.rank,
                num_stages=env.world_size,
                device=env.device,
            )
            
            if cfg.reversible:
                if cfg.inverse:
                    StageCI = RevPipelineStage
                    if cfg.exp_mode == ExpMode.ACTV_ERR:
                        stage_kwargs["eval_actv_err"] = True
                        
                else:
                    StageCI = RevPipelineStageWithPareprop
                    if cfg.exp_mode == ExpMode.ACTV_ERR:
                        stage_kwargs["eval_actv_err"] = True
                    
            elif cfg.checkpointing:
                StageCI = PipelineStageWithCP
            else:
                StageCI = PipelineStage
            
            # ★ stage オブジェクトを作成 (内部で改造版 _RevPipelineStageBase が使われる)
            self.stage = StageCI(**stage_kwargs)

            # スケジュール設定 (損失関数はBERTでも CrossEntropy で良い場合が多い)
            schd_kwargs = dict(
                stage=self.stage,
                n_microbatches=cfg.num_microbatches, # ★ GlobalConfig から取得
                loss_fn=torch.nn.CrossEntropyLoss(),
            )
            sched_train = Schedule1F1B(**schd_kwargs)
            sched_valid = _ScheduleForwardOnly(**schd_kwargs)
            
            # PPContext を作成 (変更なし)
            pp_ctx = PPContext(
                first_rank=0,
                last_rank=env.world_size - 1,
                stage=self.stage,
                sched_train=sched_train,
                sched_valid=sched_valid,
            )
        else:
            raise ValueError(cfg.par_mode)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01, eps=1e-6)
        lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.gamma)
        
        # ★ build_engine はインターフェースが変わらないのでそのまま使える
        self.engine = build_engine(
            cfg,
            env,
            self.d_spec,
            self.l_bndl,
            model, # ラップされた BERT モデル
            torch.nn.CrossEntropyLoss(), # ★ BERT 分類でも使える損失関数
            optimizer,
            lr_sched=lr_sched,
            pp_ctx=pp_ctx,
        )