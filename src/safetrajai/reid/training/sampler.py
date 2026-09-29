"""每批 P 個身分 × K 張的取樣（design D1）；身分張數不足 K 時重複取樣。"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class PKBatchSampler(Sampler[list[int]]):
    def __init__(self, labels: Sequence[int], p: int, k: int, seed: int):
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self.by_label[int(label)].append(index)
        if len(self.by_label) < p:
            raise ValueError(f"身分數 {len(self.by_label)} 少於每批身分數 P={p}")
        self.p, self.k, self.seed = p, k, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.by_label) // self.p

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(sorted(self.by_label))
        for start in range(0, len(self) * self.p, self.p):
            batch = []
            for label in order[start:start + self.p]:
                pool = self.by_label[int(label)]
                batch.extend(int(i) for i in rng.choice(pool, self.k, replace=len(pool) < self.k))
            yield batch
