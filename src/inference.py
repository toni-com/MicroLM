from typing import Any

import torch

from micro_utils.micro_parser_utils import read_inference_args
from micro_model.micro_model import MicroModel


def main():

    checkpoint_path, prompt = read_inference_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, itos, stoi, config = load_model(checkpoint_path, device=device)

    output = generate(
        model=model,
        prompt=prompt,
        stoi=stoi,
        itos=itos,
        block_size=config["block_size"],
        max_new_tokens=250,
        device=device,
    )

    print(output)


def generate(
    model: MicroModel, prompt: str, stoi: dict, itos: dict, block_size: int, max_new_tokens: int, device: torch.device
):
    prompt_ids = [stoi[c] for c in prompt]

    if len(prompt_ids) < block_size:
        padding = [0] * (block_size - len(prompt_ids))
        context_window = padding + prompt_ids
    else:
        context_window = prompt_ids[len(prompt_ids) - block_size :]

    out = prompt_ids[:]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ix = torch.tensor([context_window]).to(device)
            logits = model(input_ix)
            probs = torch.softmax(logits, dim=1)

            # sample
            ix = torch.multinomial(probs, num_samples=1).item()
            out.append(ix)
            context_window = context_window[1:] + [ix]

            if ix == 0:
                break

    return "".join([itos[i] for i in out])


def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[MicroModel, dict[int, str], dict[str, int], dict[str, Any]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"{checkpoint_path} not found")

    config = checkpoint["config"]
    itos = checkpoint["itos"]
    stoi = checkpoint["stoi"]
    state_dict = checkpoint["model_state_dict"]

    model = MicroModel(
        vocab_size=len(stoi),
        embed_dims=config["embed_dim"],
        block_size=config["block_size"],
        hidden_dims=config["hidden_dim"],
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, itos, stoi, config


if __name__ == "__main__":
    main()
