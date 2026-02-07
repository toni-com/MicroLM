import pickle

from micro_utils.micro_parser_utils import read_eval_args
import matplotlib.pyplot as plt


def main():
    results_path = read_eval_args()

    plot_loss(results_path)
    plt.tight_layout()
    plt.show()


def plot_loss(results_path):
    try:
        with open(results_path, "rb") as f:
            results = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Results file {results_path} not found.")

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot([i for i in range(len(results["train_loss"]))], results["train_loss"])
    ax[0, 0].set_title("Training Loss")
    ax[0, 1].plot([i for i in range(len(results["val_loss"]))], results["val_loss"])
    ax[0, 1].set_title("Validation Loss")
    ax[1, 0].plot([i for i in range(len(results["test_loss"]))], results["test_loss"])
    ax[1, 0].set_title("Test Loss")


if __name__ == "__main__":
    main()
