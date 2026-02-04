import argparse


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--block_size", type=int, default=32, help="Block size.")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size.")
    parser.add_argument("embedding_size", type=int, default=10, help="Embedding size.")

    args = parser.parse_args()

    return (
        args.epochs,
        args.batch_size,
        args.lr,
        args.block_size,
        args.hidden_size,
        args.embedding_size,
    )
