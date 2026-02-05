from torch import nn, optim
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    device,
) -> tuple[list[float], list[float]]:

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
        )

        val_loss = val_one_epoch(model=model, val_dataloader=val_dataloader, loss_fn=loss_fn, device=device)

        train_losses.extend(train_loss)
        val_losses.extend(val_loss)

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"Average train loss last epoch: {sum(train_loss) / len(train_loss):.3f}")
            print(f"Average validation loss last epoch: {sum(val_loss) / len(val_loss):.3f}")

    return train_losses, val_losses


def train_one_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    device: str,
) -> list[float]:
    train_losses = []

    for X, y in train_dataloader:
        X = X.to(device)
        y = y.to(device)

        # forward pass
        y_pred = model(X)

        # backward pass
        optimizer.zero_grad()
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # track loss
        train_losses.append(loss.item())

    return train_losses


def val_one_epoch(model: nn.Module, val_dataloader: DataLoader, loss_fn: nn.Module, device: str) -> list[float]:
    val_losses = []
    for X, y in val_dataloader:
        X = X.to(device)
        y = y.to(device)

        # forward pass
        y_pred = model(X)

        # track loss
        loss = loss_fn(y_pred, y)
        val_losses.append(loss.item())

    return val_losses
