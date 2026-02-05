import datetime
import os

import torch
from torch import optim, nn


def save_model(model: nn.Module):
    now = datetime.datetime.now()

    output_path = "src/micro_model/models"
    os.makedirs(output_path, exist_ok=True)
    full_path = f"{output_path}/{now.strftime('%d-%m__%H-%M')}.pth"

    torch.save(model.state_dict(), full_path)
