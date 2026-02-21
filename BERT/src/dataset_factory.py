# ファイル名: dataset_factory.py (★ Hugging Face datasets を使うように修正)
import os
import torch
# ★ torchtext と tqdm は不要になったため削除
# from torchtext.datasets import IMDB
# from tqdm import tqdm
from transformers import BertTokenizer
# ★ Hugging Face datasets から必要なモジュールをインポート
from datasets import load_dataset, Dataset
from datasets import load_from_disk
from functools import partial # ★ ヘルパー関数用にインポート
from typing import Dict, Any # ★ 型ヒント用


# --- ★ IMDBBertDataset クラスは不要になったため削除 ---
# Hugging Face の .map() と .set_format() がこのクラスの役割を
# より高速に（バッチ処理で）実行してくれるため、削除します。

def _imdb(root: str, tokenizer_name: str, max_length: int):
    # 保存先パス
    raw_path = os.path.join(root, "imdb_raw")
    tok_path = os.path.join(root, "tokenizer")

    # A. スパコン上の保存済みデータがある場合
    if os.path.exists(raw_path):
        print(f"Loading local data from {raw_path}")
        tokenizer = BertTokenizer.from_pretrained(tok_path)
        raw_dataset = load_from_disk(raw_path)
    # B. ない場合 (ローカルPC等)
    else:
        print("Downloading from Hub...")
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        raw_dataset = load_dataset("imdb", cache_dir=root)

    # 共通処理: トークン化
    # ※ _tokenize_function はご自身のコードにあるものを使ってください
    tokenize_fn = partial(_tokenize_function, tokenizer=tokenizer, max_length=max_length)
    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    
    return tokenized_dataset["train"], tokenized_dataset["test"], tokenizer

# --- ★ datasets.map() で使用するトークン化関数 ---
def _tokenize_function(
    batch: Dict[str, Any], # ★ .map(batched=True) から渡されるバッチ
    tokenizer: BertTokenizer,
    max_length: int
) -> Dict[str, Any]:
    """
    Hugging Face datasets の .map() メソッドでバッチ処理するための関数。
    'text' フィールドをトークン化し、'label' を 'labels' にリネームする。
    """
    # テキストをトークン化
    encoding = tokenizer(
        batch['text'],
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length", # バッチ単位ではなく、グローバルな max_length でパディング
        truncation=True,
        return_tensors=None, # .map() の中は 'pt' ではなく Python リスト/int を返す
    )
    
    # 元の 'label' フィールド (0 or 1) を 'labels' というキーにコピー
    # (古い IMDBBertDataset が 'labels' というキーを生成していたため、互換性を維持)
    encoding['labels'] = batch['label']
    return encoding





# --- ★ _get_dummy_data (変更なし) ---
# (★ この関数は _debug で使われます)
def _get_dummy_data():
    """30件のダミーテキストとラベルを生成"""
    # (ラベル[0 or 1], テキスト) のタプルリスト
    data = [
        (1, "This movie was fantastic! Loved every minute."),
        (0, "Absolutely terrible. Waste of time."),
        (1, "A masterpiece of cinema. Brilliant acting."),
        (0, "I fell asleep halfway through. So boring."),
        (1, "Incredible story and visuals. Highly recommended."),
        (0, "Poorly written and badly directed."),
        (1, "Best film I've seen this year."),
        (0, "Don't bother watching this."),
        (1, "A truly heartwarming and beautiful film."),
        (0, "I wanted my money back. Awful."),
        (1, "Just wow! The plot twists were amazing."),
        (0, "Complete garbage. Zero stars."),
        (1, "Hilarious and witty. A must-see comedy."),
        (0, "The acting was wooden and unbelievable."),
        (1, "I cried, I laughed. What an experience."),
        (0, "One of the worst movies ever made."),
        (1, "Visually stunning and emotionally powerful."),
        (0, "This makes no sense. Confusing plot."),
        (1, "A great movie for the whole family."),
        (0, "I've seen better acting in a high school play."),
        (1, "Superb performance by the lead actor."),
        (0, "Avoid at all costs. Truly dreadful."),
        (1, "Captivating from start to finish."),
        (0, "A predictable and shallow story."),
        (1, "The soundtrack was amazing too!"),
        (0, "I can't believe I sat through this."),
        (1, "A fresh and original concept."),
        (0, "Nothing new here. Very derivative."),
        (1, "Two thumbs up! Will watch again."),
        (0, "Horrible. Just horrible.")
    ]
    return data


# --- ★ _debug 関数を Hugging Face datasets を使うように修正 ---
def _debug(root: str, tokenizer_name: str, max_length: int):
    """
    デバッグ用の小さなダミーデータセット(30件)を 'datasets' を使って生成する。
    """
    print(f"--- Using DEBUG dataset (30 samples) ---")
    
    print(f"Loading tokenizer: {tokenizer_name}...")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # 1. ダミーデータを取得
    raw_data = _get_dummy_data()
    
    # 2. Python リストから Hugging Face Dataset オブジェクトを作成
    #    ( { 'text': [...], 'label': [...] } の辞書形式に変換)
    data_dict = {
        "text": [text for label, text in raw_data],
        "label": [label for label, text in raw_data]
    }
    raw_dataset = load_dataset.from_dict(data_dict)
    
    # 3. .map() 用の関数を準備
    tokenize_fn = partial(
        _tokenize_function,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # 4. トークン化
    tokenized_dataset = raw_dataset.map(
        tokenize_fn,
        batched=True, # 30件だけでも batched=True は動作します
        remove_columns=["text"],
    )
    
    # 5. PyTorch 用にフォーマット
    tokenized_dataset.set_format(
        type="torch",
        columns=['input_ids', 'attention_mask', 'labels']
    )
    
    print("--- DEBUG dataset created. Using same data for train and val. ---")
    # 訓練用と検証用に同じオブジェクトを返す
    return tokenized_dataset, tokenized_dataset, tokenizer