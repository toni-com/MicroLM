import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    device: str,
    use_tqdm: bool = True,
) -> tuple[list[float], list[float]]:

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        looping_iterator = tqdm(train_dataloader, desc=f"Training Epoch: {epoch+1}") if use_tqdm else train_dataloader

        train_loss = train_one_epoch(
            model=model,
            train_dataloader=looping_iterator,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        val_loss = evaluate_one_epoch(model=model, val_dataloader=val_dataloader, loss_fn=loss_fn, device=device)

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"Average train loss last epoch: {sum(train_loss) / len(train_loss):.3f}")
            print(f"Average validation loss last epoch: {sum(val_loss) / len(val_loss):.3f}")

        train_losses.extend(train_loss)
        val_losses.extend(val_loss)

        scheduler.step()

    return train_losses, val_losses


def train_one_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
) -> list[float]:
    train_losses = []

    model.train()
    for X, y in train_dataloader:
        X = X.to(device)
        y = y.to(device)

        # forward pass
        y_pred = model(X)  # Batch x Vocab_Size

        # backward pass
        last_target = y[:, -1]
        loss = loss_fn(y_pred, last_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # track loss
        train_losses.append(loss.item())

    avg_loss = sum(train_losses) / len(train_losses)
    return [avg_loss]


def evaluate_one_epoch(model: nn.Module, val_dataloader: DataLoader, loss_fn: nn.Module, device: str) -> list[float]:
    model.eval()
    losses = []

    with torch.no_grad():
        for X, y in val_dataloader:
            X = X.to(device)
            y = y.to(device)

            # forward pass
            y_pred = model(X)
            last_target = y[:, -1]

            # track loss
            loss = loss_fn(y_pred, last_target)
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    return [avg_loss]
