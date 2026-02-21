from __future__ import annotations
import torch.multiprocessing as mp
from args import parse_args
from src.config import build_global_config
from src.env import build_env
from experiment import EXP_REGISTRY

#python main.py --model RevBERT --dataset imdb ... のようにargs.pyに従ってコードを実行
def main():
    args = parse_args() #args.pyを呼び出し、自分がコマンドラインから渡した実験の設定（どのモデルを使うか、バッチサイズはいくつか)を読み取る
    cfg = build_global_config(args) #読み取った設定をプログラムで使いやすい形式(cfgオブジェクト)に変換
    env = build_env(cfg) #GPUの分散学習などの設定の処理をここで行う
    exp = EXP_REGISTRY[cfg.exp_mode.value](cfg, env) #実験の種類を設定
    #実験の実行
    exp.before_run() 
    exp.run()
    exp.after_run()
    
if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) is None:
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            # 既に設定されている場合などの競合をキャッチ
            print("Could not set start method 'fork', possibly already set.")
        except RecursionError:
            pass
    
    main()