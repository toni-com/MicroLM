import datetime
import json
import os

import torch
from torch import nn


def save_model(model: nn.Module):
    output_path = "src/micro_model/models"
    dir_path = f"{output_path}/{datetime.datetime.now().strftime('%d-%m__%H-%M')}"
    file_path = f"{dir_path}/model.pth"

    os.makedirs(dir_path, exist_ok=True)

    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


def save_hyperparameters(
    epochs: int,
    batch_size: int,
    lr: float,
    block_size: int,
    hidden_size: int,
    embedding_size: int,
    time: float,
    train_loss: list[float],
    val_loss: list[float],
    test_loss: list[float],
) -> None:
    now = datetime.datetime.now()
    output_path = "src/micro_model/models"
    dir_path = f"{output_path}/{now.strftime('%d-%m__%H-%M')}"
    file_path = f"{dir_path}/params.json"

    os.makedirs(dir_path, exist_ok=True)

    hyper_params = {
        "date": now.strftime("%d-%m-%Y - %H:%M"),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "block_size": block_size,
        "hidden_size": hidden_size,
        "embedding_size": embedding_size,
        "time": time,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
    }

    with open(file_path, "w") as f:
        json.dump(hyper_params, f, indent=4)
    print(f"Hyperparameters saved to {file_path}")
