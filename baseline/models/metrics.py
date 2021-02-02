"""
Classification metrics
"""
import pdb
from typing import Any, Callable, Optional

import torch
from pytorch_lightning.metrics import Metric


class MyAccuracy(Metric):
    def __init__(
        self,
        threshold: float = 0.5,
        compute_on_step: bool = True,
        dist_sync_on_step=False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            # dist_sync_fn=dist_sync_fn,
        )

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.threshold = threshold

    def _input_format(self, preds: torch.Tensor, target: torch.Tensor):
        if not (
            len(preds.shape) == len(target.shape)
            or len(preds.shape) == len(target.shape) + 1
        ):
            raise ValueError(
                "preds and target must have same number of dimensions,"
                "or one additional dimension for preds"
            )

        if len(preds.shape) == len(target.shape) + 1:
            # multi class probabilites
            preds = torch.argmax(preds, dim=1)

        if len(preds.shape) == len(target.shape) and preds.dtype == torch.float:
            # binary or multilabel probablities
            preds = (preds >= self.threshold).long()
        return preds, target

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(logits, target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return 100.0 * self.correct / self.total


class MyTopK(Metric):
    def __init__(
        self,
        topk: int = 5,
        compute_on_step: bool = True,
        dist_sync_on_step=False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            # dist_sync_fn=dist_sync_fn,
        )

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.topk = topk

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        assert len(logits) == len(target)
        _, pred = torch.topk(logits, self.topk, dim=1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        self.correct += correct.reshape(-1).int().sum()
        self.total += target.numel()

    def compute(self):
        return 100.0 * self.correct / self.total


if __name__ == "__main__":
    # Test topk
    topk = 5
    nclasses = 10
    nsamples = 20
    mytopk = MyTopK(topk)

    results = []
    for _ in range(50):
        logits = torch.rand((nsamples, nclasses))
        target = torch.randint(high=nclasses, size=(nsamples,))
        bs = target.numel()
        _, pred = torch.topk(logits, topk, dim=1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct.reshape(-1).float().sum()
        result = correct_k.mul_(100.0 / bs).item()
        print(result)
        results.append(result)
        assert 0.0 <= result <= 100.0
        mytopk(logits, target)

    # mytopk = MyTopK(topk)
    # for _ in range(50):
    #     logits = torch.rand((nsamples, nclasses))
    #     target = torch.randint(high=nclasses, size=(nsamples,))
    #     mytopk(logits, target)

    result = mytopk.compute()
    print(f"Top{topk} result: {result}")
    assert result.item() == sum(results) / len(results)
