import datetime
import json
import os
import pickle
import random
from typing import Any

import numpy as np
import torch
from torch import nn


def set_seed(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    hyper_params: dict[str, Any],
) -> None:
    file_path = f"{output_dir}/params.json"
    os.makedirs(output_dir, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(hyper_params, f, indent=4)
    print(f"Hyperparameters saved to {file_path}")


def save_checkpoint(model: nn.Module, itos: dict, stoi: dict, hyper_params: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    file_path = f"{output_dir}/checkpoint.pth"

    checkpoint = {"model_state_dict": model.state_dict(), "itos": itos, "stoi": stoi, "config": hyper_params}

    torch.save(checkpoint, file_path)
    print(f"\nCheckpoint saved to {file_path}")


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
