from micro_data_utils.micro_dataset import (
    get_micro_dataset,
    get_micro_transformer,
    micro_transform,
    micro_transform_and_split_data,
)


def main() -> None:

    # initialize micro_data_utils
    full_data = get_micro_dataset()

    stoi, itos = get_micro_transformer(full_data)

    X_train, y_train, X_val, y_val, X_test, y_test = micro_transform_and_split_data(
        dataset=full_data, block_size=12, stoi=stoi, split=[0.7, 0.15, 0.15]
    )
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)


if __name__ == "__main__":
    main()
