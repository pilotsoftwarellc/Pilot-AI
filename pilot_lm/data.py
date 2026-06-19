from pathlib import Path

import numpy as np
import torch


class TokenBatcher:
    def __init__(self, data_dir: str | Path, block_size: int, batch_size: int, device: str):
        data_dir = Path(data_dir)
        self.train = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        for split_name, split_data in (("train", self.train), ("val", self.val)):
            needed = block_size + 2
            if len(split_data) < needed:
                raise ValueError(
                    f"{split_name}.bin has {len(split_data)} tokens; need at least {needed} "
                    f"for block_size={block_size}."
                )

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        high = len(data) - self.block_size - 1
        ix = torch.randint(high, (self.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy(np.array(data[i : i + self.block_size], dtype=np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(np.array(data[i + 1 : i + 1 + self.block_size], dtype=np.int64))
                for i in ix
            ]
        )

        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y
