from datasets import load_dataset


def get_micro_dataset() -> list[str]:
    dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")
    return list(dataset["text"])


def get_micro_transformer(dataset: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    unique_chars = set(dataset)
    stoi = {c: i + 2 for i, c in enumerate(unique_chars)}
    itos = {i + 2: c for i, c in enumerate(unique_chars)}

    stoi["<sos>"], itos[0] = 0, "<sos>"
    stoi["<eos>"], itos[1] = 1, "<eos>"
    return stoi, itos


def micro_transform(dataset: list[str], stoi: dict[str, int]) -> list[list[int]]:
    result_data = []
    for story in dataset:
        story_result = []
        for c in story:
            story_result.append(stoi[c])
        result_data.append(story_result)
    return result_data
