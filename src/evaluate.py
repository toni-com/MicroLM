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

    fig = plt.figure(figsize=(6, 6))
    plt.title("Loss Curve over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot([i for i in range(len(results["train_loss"]))], results["train_loss"], label="Train", color="blue")
    plt.plot([i for i in range(len(results["val_loss"]))], results["val_loss"], label="Validation", color="red")
    plt.axhline(y=results["test_loss"], label="Test", color="green", linestyle="--")
    plt.legend()
    fig.tight_layout()


if __name__ == "__main__":
    main()
