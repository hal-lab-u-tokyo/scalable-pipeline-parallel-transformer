import os
import argparse
from datasets import load_dataset
from transformers import BertTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()
    
    # 保存先作成
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving to: {args.outdir}")

    # 1. トークナイザー保存
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained(os.path.join(args.outdir, "tokenizer"))
    
    # 2. データセット保存 (前処理なしで保存し、ロード時に処理させるのが無難)
    ds = load_dataset("imdb")
    ds.save_to_disk(os.path.join(args.outdir, "imdb_raw"))
    print("Done!")

if __name__ == "__main__":
    main()