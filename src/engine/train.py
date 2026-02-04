from torch import nn, optim
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
):

    for epoch in range(epochs):
        pass
