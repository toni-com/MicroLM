import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class MicroDataset(Dataset):
    def __init__(self, text_data: list[str], stoi: dict, block_size: int):
        full_text = "".join(text_data)

        self.data = torch.tensor([stoi[c] for c in full_text], dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size

        if start_idx + self.block_size + 1 > len(self.data):
            start_idx = 0

        chunk = self.data[start_idx : start_idx + self.block_size + 1]
        X = chunk[:-1]
        y = chunk[1:]

        return X, y


def get_micro_transformer(dataset: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    unique_chars = set([c for element in dataset for c in element])
    stoi = {c: i for i, c in enumerate(unique_chars)}
    itos = {i: c for i, c in enumerate(unique_chars)}

    return stoi, itos


def get_micro_dataset() -> list[str]:
    dataset = load_dataset("roneneldan/TinyStories", split="train[:3%]")
    return list(dataset["text"])
