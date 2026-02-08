import datetime
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from micro_utils.micro_parser_utils import read_train_args
from engine.train import train, evaluate_one_epoch
from micro_model.micro_model import MicroModel
from micro_utils.micro_save_utils import save_checkpoint, save_hyperparameters, save_losses, get_output_names, set_seed
from micro_data_utils.micro_dataset import MicroDataset, get_micro_dataset, get_micro_transformer


def main() -> None:
    set_seed(12)

    # read args
    epochs, batch_size, lr, block_size, hidden_size, embedding_size, should_save, test_run = read_train_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Running on the following parameters: \n"
        f"device: {device}\n"
        f"epochs: {epochs}\n"
        f"batch_size: {batch_size}\n"
        f"lr: {lr}\n"
        f"block_size: {block_size}\n"
        f"hidden_size: {hidden_size}\n"
        f"embedding_size: {embedding_size}\n"
    )

    # initialize micro_data_utils
    full_data = get_micro_dataset()
    if test_run:
        full_data = full_data[0 : int(len(full_data) * 0.2)]
    stoi, itos = get_micro_transformer(full_data)

    test_dataloader, train_dataloader, val_dataloader = load_data(full_data, stoi, batch_size, block_size)

    # init models and params
    micro_model = MicroModel(
        vocab_size=len(stoi),
        embed_dims=embedding_size,
        block_size=block_size,
        hidden_dims=hidden_size,
    )
    micro_model = micro_model.to(device)

    optimizer = torch.optim.AdamW(params=micro_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # train model
    time_before = datetime.datetime.now()
    train_loss, val_loss = train(
        model=micro_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        use_tqdm=True,
    )
    time_after = datetime.datetime.now()

    # test model
    test_loss = evaluate_one_epoch(model=micro_model, val_dataloader=test_dataloader, loss_fn=loss_fn, device=device)

    print(f"Average training loss: {sum(train_loss) / len(train_loss):.3f}")
    print(f"Average val loss: {sum(val_loss) / len(val_loss):.3f}")
    print(f"Average test loss: {sum(test_loss) / len(test_loss):.3f}")

    # save model
    if should_save:
        output_dir = get_output_names()
        hyper_params = {
            "date": datetime.datetime.now().strftime("%d-%m-%Y - %H:%M"),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "block_size": block_size,
            "hidden_dim": hidden_size,
            "embed_dim": embedding_size,
            "time": time_after.second - time_before.second,
        }

        save_checkpoint(model=micro_model, itos=itos, stoi=stoi, hyper_params=hyper_params, output_dir=output_dir)
        save_hyperparameters(hyper_params=hyper_params, output_dir=output_dir)
        save_losses(train_loss=train_loss, val_loss=val_loss, test_loss=test_loss, output_dir=output_dir)


def load_data(full_data, stoi, batch_size, block_size) -> tuple[DataLoader, DataLoader, DataLoader]:
    random.shuffle(full_data)

    split_idx_train = int(len(full_data) * 0.7)
    split_idx_val = int(len(full_data) * 0.85)

    train_text = full_data[:split_idx_train]
    val_text = full_data[split_idx_train:split_idx_val]
    test_text = full_data[split_idx_val:]

    train_dataset = MicroDataset(text_data=train_text, stoi=stoi, block_size=block_size)
    val_dataset = MicroDataset(text_data=val_text, stoi=stoi, block_size=block_size)
    test_dataset = MicroDataset(text_data=test_text, stoi=stoi, block_size=block_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"Training data size: {len(train_dataloader)}",
        f"Validation data size: {len(val_dataloader)}",
        f"Test data size: {len(test_dataloader)}",
    )
    return test_dataloader, train_dataloader, val_dataloader


if __name__ == "__main__":
    main()
