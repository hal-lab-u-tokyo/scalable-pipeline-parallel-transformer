from dataclasses import dataclass, field # ★ field を追加
from typing import Callable, Tuple, Optional
from torch.utils.data import Dataset, Subset
from transformers import BertTokenizer, BertConfig# ★ インポート

# --- ★ 修正した dataset_factory からインポート ---
from .dataset_factory import _imdb, _debug


# --- ★ DatasetSpec (修正版) ---
@dataclass(frozen=True)
class DatasetSpecRevised:
    num_classes: int
    builder_fn: Callable[[str, str, int], tuple[Dataset, Dataset, BertTokenizer]]
    tokenizer_name: str # ★ トークナイザー名を保持
    # --- BERT 固有情報は __init__ で計算 ---
    vocab_size: int = field(init=False)
    type_vocab_size: int = field(init=False)
    shape: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        # DatasetSpec 作成時にトークナイザーを一時的にロードして情報を取得
        try:
            tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
            config = BertConfig.from_pretrained(self.tokenizer_name)
            # オブジェクトに値を設定 (frozen=True なので __setattr__ を使う)
            object.__setattr__(self, 'vocab_size', tokenizer.vocab_size)
            object.__setattr__(self, 'type_vocab_size', config.type_vocab_size)
            del tokenizer, config # 不要になったら削除
        except Exception as e:
            print(f"Warning: Failed to load tokenizer '{self.tokenizer_name}' during DatasetSpec init: {e}")
            object.__setattr__(self, 'vocab_size', -1) # エラー時は -1 など
            object.__setattr__(self, 'type_vocab_size', -1)


# --- ★ REGISTRY を新しい Spec で定義 ---
REGISTRY: dict[str, DatasetSpecRevised] = {
    # キーにトークナイザー情報を含めるか、Spec 内に持たせるか。Spec 内に持たせた。
    "imdb": DatasetSpecRevised(
        num_classes=2,
        builder_fn=_imdb,
        tokenizer_name="bert-base-uncased" # ★ 使用するトークナイザーを指定
    ),
    # 他のBERTデータセット (例: GLUE) も同様に追加可能
    "debug": DatasetSpecRevised(
        num_classes=2, # ダミーデータも2クラス (pos/neg)
        builder_fn=_debug, # 上で定義した _debug 関数を指定
        tokenizer_name="bert-base-uncased" # imdb と同じトークナイザーを使用
    )  
}

# --- ★ load_dataset を修正 ---
def load_dataset(name: str, root: str, tokenizer_name: str, max_length: int, subset_N: int | None = None):
    """データセット名、ルートパス、トークナイザー名、最大長に基づいてデータセットをロード"""
    try:
        spec = REGISTRY[name]
        # ★ spec に保存された tokenizer_name と引数が一致するか確認 (任意)
        if spec.tokenizer_name != tokenizer_name:
             print(f"Warning: Loading dataset '{name}' with tokenizer '{tokenizer_name}' "
                   f"but registry specifies '{spec.tokenizer_name}'.")
    except KeyError:
        raise KeyError(f"データセット '{name}' はREGISTRYに登録されていません。利用可能なデータセット: {list(REGISTRY.keys())}")
    
    # ★ builder_fn に tokenizer_name と max_length を渡す
    ds_tr, ds_va, tokenizer = spec.builder_fn(root, tokenizer_name, max_length)
    
    # サブセット処理 (変更なし)
    if subset_N:
        ds_tr = Subset(ds_tr, range(min(len(ds_tr), subset_N)))
        ds_va = Subset(ds_va, range(min(len(ds_va), subset_N)))
        
    # ★ tokenizer も返す
    return ds_tr, ds_va, tokenizer, spec # spec も返して vocab_size などを使えるようにする