import csv, datetime, time
from pathlib import Path

import torch
import torch.distributed as dist

def topk_correct_per_class(outputs, labels, num_classes, topk=(1,)):
    """
    Count the number of correct top-k predictions for each class.

    Args:
        outputs (torch.Tensor): Model outputs (logits), shape [N, C]
        labels (torch.Tensor): Ground truth labels, shape [N]
        num_classes (int): Number of classes
        topk (tuple of int): Specifies the top-k predictions to evaluate.

    Returns:
        List[torch.Tensor]: For each k in topk, returns a tensor of shape [num_classes]
                            where each element is the number of samples of that class
                            correctly predicted (i.e. the true label appears in the top-k predictions).
    """
    with torch.no_grad():
        maxk = max(topk)
        k_for_topk = min(maxk, outputs.shape[1])
        _, pred = outputs.topk(k_for_topk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))  # shape: [maxk, N]

        results = []
        for k in topk:
            sample_correct = correct[:k].sum(dim=0) > 0
            per_class = torch.bincount(
                labels, weights=sample_correct.int(), minlength=num_classes
            )
            results.append(per_class)
        return results


class MetricTracker:
    def __init__(self, num_classes, topk, num_epochs, device, logger=None):
        self.num_classes = num_classes
        self.topk = topk
        self.num_epochs = num_epochs
        self.device = device
        self.logger = logger

        self.phase = "train"  # train or valid
        self.epoch_idx = 0
        self.start_time = time.time()

        # batch stats
        self.batch_size = 0
        self.batch_loss = torch.zeros(1, device=device)
        self.batch_count_per_class = torch.zeros(num_classes, device=device)
        self.batch_topk_correct_per_class = [
            torch.zeros(num_classes, device=device) for _ in topk
        ]

        # epoch stats
        self.epoch_loss = torch.zeros(1, device=device)
        self.epoch_count = torch.zeros(1, device=device, dtype=torch.long)
        self.epoch_count_per_class = torch.zeros(num_classes, device=device)
        self.epoch_topk_correct_per_class = [
            torch.zeros(num_classes, device=device) for _ in topk
        ]

        self.csv_writer = None
        if logger is not None:
        # if logger is not None and logger.name.endswith("main"):
            csv_path = Path(logger.name).with_suffix("")
            csv_path = csv_path.parent / "metrics.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_fp = open(csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_fp)
            self.csv_writer.writerow(["phase","epoch","loss","acc1","acc5","timestamp",])


    def reset(self, phase, epoch_idx):
        self.phase = phase
        self.epoch_idx = epoch_idx
        self.start_time = time.time()

        self.batch_loss = torch.zeros(1, device=self.device)
        self.batch_count_per_class.zero_()
        for buf in self.batch_topk_correct_per_class:
            buf.zero_()

        self.epoch_loss.zero_()
        self.epoch_count.zero_()
        self.epoch_count_per_class.zero_()
        for buf in self.epoch_topk_correct_per_class:
            buf.zero_()

    @torch.no_grad()
    def update(self, batch_loss, outputs, labels):
        self.batch_size = labels.size(0)

        # update batch stats
        self.batch_loss = batch_loss
        self.batch_count_per_class = torch.bincount(labels, minlength=self.num_classes)
        self.batch_topk_correct_per_class = topk_correct_per_class(
            outputs, labels, self.num_classes, self.topk
        )

        # update epoch stats
        self.epoch_loss += batch_loss * self.batch_size
        self.epoch_count += self.batch_size
        self.epoch_count_per_class += self.batch_count_per_class
        for e, b in zip(
            self.epoch_topk_correct_per_class, self.batch_topk_correct_per_class
        ):
            e += b

    @torch.no_grad()
    def sync(self):
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(self.epoch_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.epoch_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.epoch_count_per_class, op=dist.ReduceOp.SUM)
            for buf in self.epoch_topk_correct_per_class:
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)

    @torch.no_grad()
    def log_batch(self, batch_idx, num_batches):
        if self.logger is None:
            return
        log_str = ""
        log_str += f"[{self.phase}]"
        log_str += f"[Epoch {self.epoch_idx+1}/{self.num_epochs}]"
        log_str += f"[Batch {batch_idx+1}/{num_batches}]"
        log_str += f" "
        log_str += f"Loss: {self.batch_loss:.4f}, "
        for k, correct in zip(self.topk, self.batch_topk_correct_per_class):
            acc = correct.sum().item() / self.batch_size * 100.0
            log_str += (
                f"Acc@{k}: {acc:.2f}% ({int(correct.sum().item())}/{self.batch_size}), "
            )
        elapsed_time = time.time() - self.start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        log_str += f"Elapsed: {elapsed_str}, "
        batch_per_second = batch_idx / elapsed_time
        remaining_batches = num_batches - batch_idx
        eta_seconds = (
            remaining_batches / batch_per_second if batch_per_second > 0 else 0
        )
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        log_str += f"ETA: {eta_str}"
        self.logger.info(log_str)

    @torch.no_grad()
    def log_epoch(self):
        if self.logger is None:
            return
        log_str = ""
        log_str += f"[{self.phase}]"
        log_str += f"[Epoch {self.epoch_idx+1}/{self.num_epochs}]"
        log_str += f" "
        log_str += f"Loss: {self.epoch_loss.item() / self.epoch_count.item():.4f}, "
        for k, correct in zip(self.topk, self.epoch_topk_correct_per_class):
            acc = correct.sum().item() / self.epoch_count.item() * 100.0
            log_str += f"Acc@{k}: {acc:.2f}% ({int(correct.sum().item())}/{self.epoch_count.item()}), "
        per_class_acc = (
            self.epoch_topk_correct_per_class[0] / self.epoch_count_per_class
        )
        mean_acc = per_class_acc.mean().item() * 100.0
        log_str += f"mean: {mean_acc:.2f}%, "
        max_idx, max_acc = (
            per_class_acc.argmax().item(),
            per_class_acc.max().item() * 100.0,
        )
        log_str += f"max: {max_acc:.2f}% ({max_idx}), "
        min_idx, min_acc = (
            per_class_acc.argmin().item(),
            per_class_acc.min().item() * 100.0,
        )
        log_str += f"min: {min_acc:.2f}% ({min_idx})"
        self.logger.info(log_str)

        if self.csv_writer:
            # ★ 修正: Acc@1 を計算
            acc1 = self.epoch_topk_correct_per_class[0].sum().item() / self.epoch_count.item()
            
            # ★ 修正: Acc@5 (インデックス 1) が存在するか確認し、存在しない場合は 0.0 を書き込む
            acc5 = 0.0
            if len(self.epoch_topk_correct_per_class) > 1:
                acc5 = self.epoch_topk_correct_per_class[1].sum().item() / self.epoch_count.item()

            self.csv_writer.writerow([
                self.phase,
                self.epoch_idx,
                (self.epoch_loss / self.epoch_count).item(),
                acc1, # 修正した Acc@1
                acc5, # 修正した Acc@5
                datetime.datetime.now().isoformat(timespec="seconds"),
            ])