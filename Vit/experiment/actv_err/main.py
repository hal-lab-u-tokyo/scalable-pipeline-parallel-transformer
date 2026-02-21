from ..base import Experiment
from .. import register
from src.config import ParMode, ExpMode

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import torch.distributed as dist  # 追加: 同期用

@register("actv-err")
class ActvErrExperiment(Experiment):

    def before_run(self):
        # ... (既存のチェック処理はそのまま) ...
        if self.cfg.exp_mode != ExpMode.ACTV_ERR:
            raise RuntimeError("Experiment mode must be ExpMode.ACTV_ERR")
        if self.cfg.par_mode != ParMode.PP:
            raise RuntimeError("ParMode must be PP")
        if not self.cfg.reversible:
            raise RuntimeError("reversible must be True")
        
        super().before_run()

    def run(self):
        assert self.engine is not None
        # batch_limit=1 で1ステップだけ実行して誤差を計測
        self.engine.step(0, is_train=True, batch_limit=1)

    def after_run(self):
        super().after_run()

        if getattr(self, "stage", None) is None:
            raise RuntimeError("Experiment stage is not set.")
    
        log_dir = Path(self.cfg.log_rdir) / self.cfg.exp_id / self.cfg.env_id
        log_dir.mkdir(parents=True, exist_ok=True)

        # 1. 各ランクでCSVを保存
        df = pd.DataFrame(
            {
                "mse": self.stage.mse_stats,
                "cos": self.stage.cos_stats
            }
        )
        csv_path = log_dir / f"actv_err_{self.stage.stage_index}.csv"
        df.to_csv(csv_path, index=False)

        if self.env.main_logger:
            self.env.main_logger.info(
                "[Rank %s] wrote metrics to %s (%d samples)", 
                self.env.rank, csv_path, len(df)
            )

        # 【重要】全ランクが書き込み終わるのを待機
        # これがないと Rank 0 が他ランクのファイルを読み込めない
        if dist.is_initialized():
            dist.barrier()

        # Rank 0 以外はここで終了
        if self.env.rank != 0:
            return

        # --- 以下 Rank 0 のみの処理 ---

        # Matplotlib のバックエンドを非GUI用に設定（サーバー対策）
        matplotlib.use("Agg") 

        csv_paths = sorted(log_dir.rglob("actv_err_*.csv"))
        if not csv_paths:
            print("[rank-0] no CSV found for plotting")
            return

        dfs = []
        for p in csv_paths:
            try:
                d = pd.read_csv(p)
            except Exception as exc:
                print(f"[rank-0] WARN: failed to read {p}: {exc}")
                continue
            
            if d.empty or set(d.columns) <= {"Unnamed: 0"}:
                continue
            
            # ファイル名から stage ID を抽出
            try:
                stage_idx = int(p.stem.split("_")[-1])
            except ValueError:
                stage_idx = -1
            d["stage"] = stage_idx
            dfs.append(d)

        if not dfs:
            print("[rank-0] no valid CSV for plotting")
            return

        data = pd.concat(dfs, ignore_index=True).sort_values("stage")

        def _save(col: str, ylabel: str, stem: Path) -> None:
            fig, ax = plt.subplots(figsize=(6, 4)) # サイズを少し調整
            
            # データが空でないか確認
            if col not in data.columns:
                print(f"[rank-0] Column {col} not found in data")
                return

            data.boxplot(column=col, by="stage", ax=ax)

            ax.set_title("")
            fig.suptitle("") # 自動タイトルの削除
            ax.set_xlabel("Pipeline Stage ID")
            ax.set_ylabel(ylabel)
            ax.ticklabel_format(useOffset=False, style="plain", axis="y")
            fig.tight_layout()

            for ext in ("png", "pdf"):
                out = stem.with_suffix(f".{ext}")
                fig.savefig(out, dpi=150 if ext == "png" else None)
                print(f"[rank-0] saved {out}")

            plt.close(fig)

        _save("mse", "MSE", log_dir.with_name(log_dir.name + "_mse"))
        _save("cos", "Cosine Similarity", log_dir.with_name(log_dir.name + "_cos"))