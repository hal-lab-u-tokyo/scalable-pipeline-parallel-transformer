from transformers import PreTrainedTokenizer
import torch
from .config import GlobalConfig

class TokenizerCollate:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[tuple[str, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        生のテキストとラベルのバッチを受け取り、トークナイズしてテンソルに変換する

        Args:
            batch: [(text1, label1), (text2, label2), ...] という形式のリスト

        Returns:
            (input_ids, labels) のテンソルタプル
        """
        # batchをテキストのリストとラベルのリストに分解
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Tokenizerでテキストを一括処理
        inputs = self.tokenizer(
            texts,
            padding="max_length", # バッチ内で最長の文に合わせるか、max_lengthで固定
            truncation=True,      # max_lengthを超える文を切り捨てる
            max_length=self.max_length,
            return_tensors="pt"   # PyTorchのテンソルで返す
        )

        # ラベルもテンソルに変換
        label_tensor = torch.tensor(labels, dtype=torch.long)

        # inputs['input_ids'] と labels を返す
        return inputs['input_ids'], label_tensor