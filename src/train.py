import os
import pandas as pd

from argparse import ArgumentParser
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from definitions import ROOT_PATH
from src.models import Scorer, TfidfScorer, BertScorer


def create_submission(scorer: Scorer, df: pd.DataFrame, name_postfix: str = ""):
    scores = scorer.score_news(test_df["text"])
    df["score"] = scores

    df.to_csv(os.path.join(ROOT_PATH, f"artifacts/submission{name_postfix}.csv"), index=False)


def calc_metrics(preds, target):
    return {
        "recall": recall_score(preds, target),
        "precision": precision_score(preds, target),
        "f1_score": f1_score(preds, target),
        "accuracy_score": accuracy_score(preds, target),
    }


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", type=str, default="bert", choices={"bert", "tf-idf"}, help="Which model to use.")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--train_size", type=float, default=0.8, help="Train size when splitting data.")
    args.add_argument("--bs", type=int, default=32, help="Batch size for bert training.")
    args.add_argument("--n_epochs", type=int, default=4, help="Number of epochs.")
    args.add_argument("--use_gpu", action="store_true", help="Whether to switch to gpu for training.")
    args = args.parse_args()

    train_df = pd.read_csv(os.path.join(ROOT_PATH, "data/train_data.csv"))
    test_df = pd.read_csv(os.path.join(ROOT_PATH, "data/test_data.csv"))

    X, y = train_df["sentence"], train_df["label"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=args.train_size, shuffle=True, random_state=args.seed)

    scorer = BertScorer() if args.model == "bert" else TfidfScorer(args.seed)
    scorer.fit(X_train, y_train, X_val, y_val, n_epochs=args.n_epochs, batch_size=args.bs, use_gpu=args.use_gpu)

    if args.model == "tf-idf":
        train_metrics = calc_metrics(scorer.score_sentences(X_train), y_train)
        test_metrics = calc_metrics(scorer.score_sentences(X_val), y_val)

        print("Train:")
        for metric, val in train_metrics.items():
            print(f"{metric}: {val:.3f}")

        print("\nVal:")
        for metric, val in test_metrics.items():
            print(f"{metric}: {val:.3f}")

    print("Done!")
