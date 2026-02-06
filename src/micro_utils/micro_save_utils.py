import datetime
import json
import os
import pickle

import torch
from torch import nn


def get_output_names() -> str:
    output_path = "src/micro_model/models"
    dir_path = f"{output_path}/{datetime.datetime.now().strftime('%d-%m__%H-%M')}"

    os.makedirs(dir_path, exist_ok=True)

    return dir_path


def save_model(model: nn.Module, output_dir: str) -> None:
    file_path = f"{output_dir}/model.pth"
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), file_path)
    print(f"\nModel saved to {file_path}")


def save_hyperparameters(
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    block_size: int,
    hidden_size: int,
    embedding_size: int,
    time: float,
) -> None:
    file_path = f"{output_dir}/params.json"
    os.makedirs(output_dir, exist_ok=True)

    hyper_params = {
        "date": datetime.datetime.now().strftime("%d-%m-%Y - %H:%M"),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "block_size": block_size,
        "hidden_size": hidden_size,
        "embedding_size": embedding_size,
        "time": time,
    }

    with open(file_path, "w") as f:
        json.dump(hyper_params, f, indent=4)
    print(f"Hyperparameters saved to {file_path}")


def save_losses(
    output_dir: str,
    train_loss: list[float],
    val_loss: list[float],
    test_loss: list[float],
) -> None:
    file_path = f"{output_dir}/losses.pickle"
    os.makedirs(output_dir, exist_ok=True)

    losses = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
    }

    with open(file_path, "wb") as f:
        pickle.dump(losses, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Losses saved to {file_path}")
