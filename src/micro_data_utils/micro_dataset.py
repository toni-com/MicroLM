from typing import Optional

from datasets import load_dataset
import torch


def get_micro_dataset() -> list[str]:
    dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")
    return list(dataset["text"])


def get_micro_transformer(dataset: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    unique_chars = set([c for element in dataset for c in element])
    stoi = {c: i + 2 for i, c in enumerate(unique_chars)}
    itos = {i + 2: c for i, c in enumerate(unique_chars)}

    stoi["<sos>"], itos[0] = 0, "<sos>"
    stoi["<eos>"], itos[1] = 1, "<eos>"
    return stoi, itos


def micro_transform(dataset: list[str], stoi: dict[str, int]) -> list[list[int]]:
    result_data = []
    for story in dataset:
        story_result = []
        for c in story:
            story_result.append(stoi[c])
        result_data.append(story_result)
    return result_data


def build_dataset(
    text_data: list[str],
    block_size: int,
    stoi: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    X, y = [], []

    for text in text_data:
        context_window = [0] * block_size
        for c in text:
            # record data
            c_idx = stoi[c]
            X.append(context_window.copy())
            y.append(c_idx)

            # update window
            context_window.pop(0)
            context_window.append(c_idx)

    X, y = torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    return X, y


def micro_transform_and_split_data(
    dataset: list[str], block_size: int, stoi: dict[str, int], split: list[float]
) -> tuple[torch.Tensor, ...]:
    assert len(split) == 3 and sum(split) == 1

    train_p, val_p, test_p = split
    total_length = len(dataset)
    train_indices = int(total_length * train_p)
    val_indices = int(train_indices + total_length * val_p)
    test_indices = int(val_indices + total_length * test_p)

    print(f"total length: {total_length}")
    print(f"train: {train_indices}, val: {val_indices}, test: {test_indices}")

    X_train, y_train = build_dataset(
        text_data=dataset[:train_indices], block_size=block_size, stoi=stoi
    )
    X_val, y_val = build_dataset(
        text_data=dataset[train_indices:val_indices], block_size=block_size, stoi=stoi
    )
    X_test, y_test = build_dataset(
        text_data=dataset[val_indices:test_indices], block_size=block_size, stoi=stoi
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
