import torch
from torch.utils.data import DataLoader, TensorDataset

from micro_data_utils.micro_dataset import (
    get_micro_dataset,
    get_micro_transformer,
    micro_transform,
    micro_transform_and_split_data,
)
from micro_utils.micro_parser_utils import read_args
from engine.train import train
from micro_model.micro_model import MicroModel


def main() -> None:

    # read args
    epochs, batch_size, lr, block_size, hidden_size, embedding_size = read_args()

    # initialize micro_data_utils
    full_data = get_micro_dataset()

    stoi, itos = get_micro_transformer(full_data)

    X_train, y_train, X_val, y_val, X_test, y_test = micro_transform_and_split_data(
        dataset=full_data, block_size=block_size, stoi=stoi, split=[0.7, 0.15, 0.15]
    )

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    micro_model = MicroModel(
        vocab_size=len(stoi),
        embed_dims=embedding_size,
        block_size=block_size,
        hidden_dims=hidden_size,
    )

    optimizer = torch.optim.SGD(micro_model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train(
        model=micro_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()
