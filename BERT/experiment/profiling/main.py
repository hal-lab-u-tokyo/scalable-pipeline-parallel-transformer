import torch
from torch.profiler import profile, ProfilerActivity

from .. import register
from ..base import Experiment

from .assets import extract_png_from_html
from .events import parse_event_spans
from .constants import EVENT_SPECS
from .plot import plot_memory_compare
from ._plot import plot_memory
from .memory_predictor_prev import build_pred_timeline
from .memory_predictor import build_pred_max
import math
from collections import defaultdict
import json


@register("profiling")
class ProfilingExperiment(Experiment):
    """Run a single training batch under *torch.profiler* for analysis."""

    # experiment/profiling/main.py

    # データローダーなどのセッティング+プロファイラの初期化と起動
    def before_run(self):
        torch.cuda.reset_peak_memory_stats(device=self.env.device)

        # ★ 1. 最初にsuper().before_run()を呼び出し、
        #    engine とデータローダー (l_bndl) を初期化させます。
        super().before_run()
        assert self.engine is not None

        # 実際に利用可能なバッチ数を取得します。
        num_available_batches = self.engine.l_bndl.num_batches_train
        if num_available_batches < 1:
            raise ValueError("Profiler requires at least 1 batch.")

        # ★ 3. プロファイラのスケジュールを、利用可能なバッチ数に
        #    基づいて動的に決定します。
        
        # 元の希望スケジュール
        # 合計で(4+4+4)*1 = 12バッチ分必要
        target_wait = 4 #最初の4バッチを無視
        target_warmup = 4 #次の4バッチは測定の準備(warmup)に使う マイクロバッチではなく普通のバッチ
        target_active = 4 #次の4バッチを実際のプロファイル対象
        target_repeat = 1 #このサイクルを1回やる
        
        total_target_steps = (target_wait + target_warmup + target_active) * target_repeat
        
        if num_available_batches < total_target_steps:
            # 利用可能なバッチ数が少なすぎる場合 (例: 3 < 12)
            # スケジュールをダウングレードして、利用可能な全バッチを使います。
            # (最低でも warmup=1, active=1 を確保)
            print(f"[Profiler WARNING] Not enough batches ({num_available_batches}) for schedule ({total_target_steps}). "
                  f"Forcing schedule to fit {num_available_batches} batches.")
            
            self.prof_wait = 0
            self.prof_warmup = 1
            self.prof_active = num_available_batches - 1 # 3バッチなら active=2
            self.prof_repeat = 1
            if self.prof_active < 1: # 1バッチしかない場合
                self.prof_warmup = 0
                self.prof_active = 1
        else:
            # バッチ数が十分にある場合は、希望のスケジュールを使用
            self.prof_wait = target_wait
            self.prof_warmup = target_warmup
            self.prof_active = target_active
            self.prof_repeat = target_repeat
            
        # ★ 4. 最終的に実行する合計ステップ数を保存
        self.total_profile_steps = (self.prof_wait + self.prof_warmup + self.prof_active) * self.prof_repeat

        # ★ 5. 動的に決定したスケジュールでプロファイラを初期化
        self.prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=self.prof_wait, 
                warmup=self.prof_warmup, 
                active=self.prof_active, 
                repeat=self.prof_repeat
            ),
            record_shapes=True, #テンソル形状を記録
            profile_memory=True, #メモリ割り当てを記録
            with_stack=True, #どのPython/C++コードが実行されたかを記録
        )

        def _prof_cb(batch_idx, is_train, **_):
            if is_train:
                self.prof.step()

        # プロファイラを起動
        self.prof.start()
        # バッチが開始するたびに_prob_cbを呼び出し、プロファイラを使う
        self.engine.on_batch_start.append(_prof_cb)
        
    def run(self):
        assert self.engine is not None
        total_profile_steps = (self.prof_wait + self.prof_warmup + self.prof_active) * self.prof_repeat
        self.engine.step(0, is_train=True, batch_limit=total_profile_steps)

    def after_run(self):
        self.prof.stop() #プロファイラを終了
        if self.env.main_logger:
            print(torch.cuda.memory_summary(device=self.env.device, abbreviated=True))

        # 結果を保存するための一意ディレクトリパスをマイクロバッチサイズなdの設定に基づき作成
        settings = f"{self.cfg.microbatch_size}x{self.cfg.num_microbatches}"
        prof_dir = self.cfg.profile_rdir / self.cfg.exp_id / settings / self.cfg.env_id / f"rank{self.env.rank}"
        prof_dir.mkdir(parents=True, exist_ok=True)
        trace_json  = prof_dir / "trace.json"
        stacks_txt  = prof_dir / "stacks.txt"
        mem_html    = prof_dir / "memory.html"
        mem_json    = prof_dir / "memory.json"

        # プロファイラが収集したデータをファイルに書き出す
        self.prof.export_chrome_trace(str(trace_json))
        self.prof.export_stacks(str(stacks_txt))
        self.prof.export_memory_timeline(str(mem_html), self.env.device)
        self.prof.export_memory_timeline(str(mem_json), self.env.device)
        extract_png_from_html(mem_html, save_to=prof_dir/"memory.png")

        # trace_jsonファイルを解析し、forward_one_chunkのようなイベントの開始、終了時刻リストを取得
        spans = parse_event_spans(trace_json, patterns={k: v["pattern"] for k, v in EVENT_SPECS.items()})
        

        # (...) プロファイル結果のエクスポートとイベント抽出は変更なし
        # 0. 統計結果を格納するメインの辞書
        stats_to_save = {}
        
        if spans:
            # 最後のイベントが終わった時刻を取得し、総実行時間として代入
            total_profiled_duration_ms = max(s.t1_ms for s in spans)
            # プロファイラ設定から一回の繰り返しで計測対象としたステップ数(prof_active)とその繰り返し回数(prof_repeat)を取得し掛け合わせプロファイル期間中に実行された合計ステップ数を計算
            num_active_steps = self.prof_active * self.prof_repeat
            
            if num_active_steps > 0 and total_profiled_duration_ms > 0:
                # 1ステップあたりの平均経過時間を計算
                avg_step_time_ms = total_profiled_duration_ms / num_active_steps
                global_batch_size = self.cfg.batch_size # GlobalConfig から取得
                # 1秒ごとに何サンプル処理できたか、すなわちスループットを計算する
                throughput_samples_per_sec = (global_batch_size * num_active_steps) / (total_profiled_duration_ms / 1000.0)

                # 1. 総合スループットを辞書に保存
                overall_stats = {
                    "total_profiled_duration_ms": total_profiled_duration_ms,
                    "num_active_steps": num_active_steps,
                    "avg_step_time_ms": avg_step_time_ms,
                    "throughput_samples_per_sec": throughput_samples_per_sec
                }
                stats_to_save["overall_performance"] = overall_stats

                # 2. ログ表示 (内容は変更なし)
                if self.env.main_logger:
                    self.env.main_logger.info("--- 📈 Overall Performance (Active Steps) ---")
                    self.env.main_logger.info(f"  Total profiled duration: {total_profiled_duration_ms:.2f} ms")
                    self.env.main_logger.info(f"  Num active steps profiled: {num_active_steps}")
                    self.env.main_logger.info(f"  Avg step time: {avg_step_time_ms:.2f} ms/step")
                    self.env.main_logger.info(f"  Throughput: {throughput_samples_per_sec:.2f} samples/sec")
                    self.env.main_logger.info("-------------------------------------------------")
        
        # --- 観点②：Fwd/Bwd カーネル時間の計算 ---
        # forward_one_chunkなどの統計を計算する
        kernel_times_ms = defaultdict(list)
        for span in spans:
            kernel_times_ms[span.label].append(span.duration_ms)

        # 1. カーネル統計を計算し、辞書に保存
        kernel_stats = {}
        # kernel_timesは{"forward_one_chunk": [10.1, 10.2...]}のような辞書
        for label, times in kernel_times_ms.items():
            if times:
                count = len(times) # その操作が実行された回数を記録
                avg = sum(times) / count # その操作の平均計算時間を計算してavgに代入
                variance = sum((x - avg) ** 2 for x in times) / count # 標本分散を記録
                std = math.sqrt(variance) # 標準偏差
                
                kernel_stats[label] = {
                    "avg_ms": avg,
                    "std_ms": std,
                    "count": count
                }
        stats_to_save["kernel_time_statistics_ms"] = kernel_stats

        # 2. ログ表示 (内容は変更なし)
        if self.env.main_logger:
            self.env.main_logger.info("--- ⚙️ Kernel Time Statistics (ms) ---")
            # ログ表示のために、計算済みの辞書をループ
            for label, stats in kernel_stats.items():
                self.env.main_logger.info(f"  [{label}]:")
                self.env.main_logger.info(f"    Avg  : {stats['avg_ms']:.4f} ms")
                self.env.main_logger.info(f"    Std  : {stats['std_ms']:.4f} ms")
                self.env.main_logger.info(f"    Count: {stats['count']}")
            self.env.main_logger.info("-------------------------------------------")

        # 3. ★ 統計結果をJSONファイルに書き込む ★
        stats_json_path = prof_dir / "performance_stats.json"
        try:
            with open(stats_json_path, 'w', encoding='utf-8') as f:
                json.dump(stats_to_save, f, indent=4, ensure_ascii=False)
            if self.env.main_logger:
                self.env.main_logger.info(f"📈 Performance stats saved to: {stats_json_path}")
        except Exception as e:
            if self.env.main_logger:
                self.env.main_logger.error(f"Failed to save performance stats: {e}")
                
        # # メモリタイムライン予測(モデル構造や設定に基づき予測メモリ使用量を作成)
        # ts_pred, traces_pred = build_pred_timeline(
        #      spans,
        #      self.cfg,
        #      self.env,
        #      self.d_spec,
        #      self.engine.model,
        # )

        # # 予測される最大メモリ使用量を計算
        # pred_max = build_pred_max(
        #     model=self.engine.model, # モデルオブジェクトはそのまま渡す
        #     # --- BERT 固有の引数を渡す ---
        #     num_blocks_per_stage=self.cfg.num_hidden_layers // self.env.world_size, 
        #     hidden_size=self.cfg.hidden_size,                 # config から取得
        #     sequence_length=self.cfg.max_position_embeddings, # config から取得
        #     num_attention_heads=self.cfg.num_attention_heads, # config から取得
        #     vocab_size=self.d_spec.vocab_size,                # データセット仕様から取得
        #     # --- 共通の引数 ---
        #     stage_id=self.env.rank,
        #     num_stages=self.env.world_size,
        #     global_batch_size=self.cfg.batch_size,     # ★ 引数名を明確化 (Global Batch Size)
        #     microbatch_size=self.cfg.microbatch_size,
        #     is_rev=self.cfg.reversible,
        #     is_pareprop=self.cfg.is_pareprop, # ★ is_pareprop フラグを追加
        #     is_ckpt=self.cfg.checkpointing,
        #     # ★ データセット仕様の shape はテキストでは使わないので削除
        #     # num_classes は head の次元で model から取得できるはずなので削除 (必要なら d_spec から渡す)
        # )

        # グラフ描画 (引数自体は変更なし、pred_max の中身が変わる)
        # グラフの横軸は時間、縦軸はGPUメモリ使用量
        # profilerが計測した実際のメモリ使用量、予測メモリ使用量、実行イベント(その瞬間に実行されていたforward_one_chunkとかの処理)をプロットする
        plot_memory(
            json_path=mem_json, #実際に測定されたメモリのタイムライン
            save_to=prof_dir / "timeline.png",
            peak_alloc=torch.cuda.max_memory_allocated(device=self.env.device), #実際の最大メモリ使用量
            #peak_reserved=torch.cuda.max_memory_reserved(device=self.env.device),
            profiler_warmup=self.prof_warmup, # warmup=4 に設定したので 3 or 4 が適切? 要確認
            event_spans=spans,
        )
        
        # (...) 終了処理は変更なし                                                     