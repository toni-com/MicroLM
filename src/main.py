import torch
from torch import nn
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
    full_data = full_data[0 : int(len(full_data) * 0.3)]
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
    micro_model = micro_model.to(device)

    optimizer = torch.optim.SGD(micro_model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_fn = nn.CrossEntropyLoss()

    train_loss, val_loss = train(
        model=micro_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
    )

    print(f"Average training loss: {sum(train_loss) / len(train_loss):.3f}")
    print(f"Average val loss: {sum(val_loss) / len(val_loss):.3f}")


if __name__ == "__main__":
    main()
