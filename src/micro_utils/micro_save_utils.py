import datetime
import json
import os

import torch
from torch import nn


def save_model(model: nn.Module):
    now = datetime.datetime.now()

    output_path = "src/micro_model/models"
    os.makedirs(output_path, exist_ok=True)
    full_path = f"{output_path}/{now.strftime('%d-%m__%H-%M')}.pth"

    torch.save(model.state_dict(), full_path)
    print(f"Model saved to {full_path}")


def save_hyperparameters(
    epochs: int, batch_size: int, lr: float, block_size: int, hidden_size: int, embedding_size: int, time: float
) -> None:
    now = datetime.datetime.now()
    output_path = "src/micro_model/models"

    os.makedirs(output_path, exist_ok=True)
    full_path = f"{output_path}/{now.strftime('%d-%m__%H-%M')}.json"

    hyper_params = {
        "date": now.strftime("%d-%m-%Y - %H:%M"),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "block_size": block_size,
        "hidden_size": hidden_size,
        "embedding_size": embedding_size,
        "time": time,
    }

    with open(full_path, "w") as f:
        json.dump(hyper_params, f, indent=4)
    print(f"Hyperparameters saved to {full_path}")
