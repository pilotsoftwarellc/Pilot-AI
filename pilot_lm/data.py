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


class SupervisedTokenBatcher:
    def __init__(self, data_dir: str | Path, block_size: int, batch_size: int, device: str):
        data_dir = Path(data_dir)
        self.train_x = np.memmap(data_dir / "train_input_ids.bin", dtype=np.uint16, mode="r")
        self.train_y = np.memmap(data_dir / "train_labels.bin", dtype=np.int32, mode="r")
        self.val_x = np.memmap(data_dir / "val_input_ids.bin", dtype=np.uint16, mode="r")
        self.val_y = np.memmap(data_dir / "val_labels.bin", dtype=np.int32, mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        for split_name, ids, labels in (
            ("train", self.train_x, self.train_y),
            ("val", self.val_x, self.val_y),
        ):
            if len(ids) != len(labels):
                raise ValueError(f"{split_name} ids/labels length mismatch: {len(ids)} != {len(labels)}")
            needed = block_size + 2
            if len(ids) < needed:
                raise ValueError(
                    f"{split_name} supervised bins have {len(ids)} tokens; need at least {needed} "
                    f"for block_size={block_size}."
                )

    def _sample_window(self, ids, labels) -> tuple[np.ndarray, np.ndarray]:
        high = len(ids) - self.block_size - 1
        for _ in range(64):
            i = int(torch.randint(high, (1,)).item())
            y = np.array(labels[i + 1 : i + 1 + self.block_size], dtype=np.int64)
            if np.any(y != -100):
                x = np.array(ids[i : i + self.block_size], dtype=np.int64)
                return x, y
        i = int(torch.randint(high, (1,)).item())
        return (
            np.array(ids[i : i + self.block_size], dtype=np.int64),
            np.array(labels[i + 1 : i + 1 + self.block_size], dtype=np.int64),
        )

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.train_x if split == "train" else self.val_x
        labels = self.train_y if split == "train" else self.val_y
        xs, ys = zip(*(self._sample_window(ids, labels) for _ in range(self.batch_size)))
        x = torch.stack([torch.from_numpy(row) for row in xs])
        y = torch.stack([torch.from_numpy(row) for row in ys])

        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y
