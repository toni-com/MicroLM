import argparse


def read_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", required=False, type=int, default=10, help="Number of Epochs.")
    parser.add_argument("--batch_size", required=False, type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", required=False, type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--block_size", required=False, type=int, default=32, help="Block size.")
    parser.add_argument("--hidden_size", required=False, type=int, default=128, help="Hidden size.")
    parser.add_argument("--embedding_size", required=False, type=int, default=64, help="Embedding size.")
    parser.add_argument("--save", required=False, type=bool, default=True, help="Save model.")
    parser.add_argument("--test_run", required=False, type=bool, default=False, help="Test run.")
    parser.add_argument(
        "--dataset",
        required=False,
        type=str,
        default="roneneldan/TinyStories",
        help="Dataset name",
    )
    parser.add_argument("--patience", required=False, type=int, default=4, help="Early stopping patience.")

    args = parser.parse_args()

    return (
        args.epochs,
        args.batch_size,
        args.lr,
        args.block_size,
        args.hidden_size,
        args.embedding_size,
        args.save,
        args.test_run,
        args.dataset,
        args.patience,
    )


def read_inference_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=False, type=str, default=None, help="Path to saved model.")
    parser.add_argument("--prompt", required=False, type=str, default=None, help="Prompt sentence.")
    args = parser.parse_args()
    return args.model_path, args.prompt


def read_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=False, type=str, default=None, help="Path to pickled results.")
    args = parser.parse_args()
    return args.results_path
