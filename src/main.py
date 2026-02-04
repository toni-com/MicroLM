from data.micro_dataset import get_micro_dataset, get_micro_transformer, micro_transform
from torch.utils.data import TensorDataset


def main() -> None:

    # initialize data
    full_data = get_micro_dataset()

    stoi, itos = get_micro_transformer(full_data)
    tensor_input = micro_transform(full_data, stoi)


if __name__ == "__main__":
    main()
