import datetime
import itertools
import json
import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from engine.train import train, evaluate_one_epoch
from micro_model.micro_model import MicroModel
from micro_data_utils.micro_dataset import MicroDataset, get_micro_dataset, get_micro_transformer
from micro_utils.micro_save_utils import set_seed


def get_loaders(data, stoi, batch_size, block_size):
    """recreating loaders since block_size might change per run."""
    n = len(data)
    train_data = data[: int(n * 0.7)]
    val_data = data[int(n * 0.7) : int(n * 0.85)]
    test_data = data[int(n * 0.85) :]

    # Create Datasets
    train_ds = MicroDataset(train_data, stoi, block_size)
    val_ds = MicroDataset(val_data, stoi, block_size)
    test_ds = MicroDataset(test_data, stoi, block_size)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def run_experiment(config, data_args):
    set_seed(12)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running: {config}")

    # init params
    train_dl, val_dl, test_dl = get_loaders(
        data=data_args["full_data"],
        stoi=data_args["stoi"],
        batch_size=config["batch_size"],
        block_size=config["block_size"],
    )

    # init model
    model = MicroModel(
        vocab_size=len(data_args["stoi"]),
        embed_dims=config["embed_dim"],
        block_size=config["block_size"],
        hidden_dims=config["hidden_dim"],
    ).to(device)

    # train
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    train_losses, val_losses = train(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        epochs=config["epochs"],
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
        loss_fn=loss_fn,
        device=device,
        use_tqdm=False,
    )

    # test
    test_losses = evaluate_one_epoch(model, test_dl, loss_fn, device)

    return {
        **config,
        "avg_train_loss": sum(train_losses) / len(train_losses),
        "avg_val_loss": sum(val_losses) / len(val_losses),
        "avg_test_loss": sum(test_losses) / len(test_losses),
    }


def load_data_and_shuffle(full_data):
    # shuffles data ONCE
    set_seed(12)
    random.shuffle(full_data)
    return full_data


def main():
    # laod data
    full_data = get_micro_dataset()
    if True:
        full_data = full_data[: (len(full_data) // 2)]
    full_data = load_data_and_shuffle(full_data)
    stoi, _ = get_micro_transformer(full_data)
    data_args = {"full_data": full_data, "stoi": stoi}

    # init hyperparams
    param_grid = {
        "epochs": [2],
        "lr": [0.001],
        "batch_size": [64],
        "block_size": [32, 64],
        "hidden_dim": [64, 128],
        "embed_dim": [32, 64, 128],
    }

    keys, values = zip(*param_grid.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # run loop
    results = []
    for i, config in enumerate(configs):
        print(f"--- Tuning {i+1}/{len(configs)} ---")
        try:
            result = run_experiment(config, data_args)
            results.append(result)
        except Exception as e:
            print(f"Run failed: {e}")

    results = sorted(results, key=lambda x: x["avg_test_loss"])
    # save results
    os.makedirs("src/results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%d%m_%H%M")

    with open(f"src/results/tuning_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=4)

    if results:
        best = min(results, key=lambda x: x["avg_test_loss"])
        print("\n--- Best Run ---")
        print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
