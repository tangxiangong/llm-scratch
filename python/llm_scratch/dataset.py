from typing import override

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import GPTTokenizer


class GPTDatasetV1(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, txt: str, tokenizer: GPTTokenizer, max_length: int, stride: int):
        super().__init__()
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataset_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
):
    tokenizer = GPTTokenizer()
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


if __name__ == "__main__":
    from .utils import load_data

    raw_text = load_data()
    dataloader = create_dataset_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)  # pyright: ignore[reportAny]
    print(first_batch)  # pyright: ignore[reportAny]
