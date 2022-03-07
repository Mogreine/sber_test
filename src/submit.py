import os
import pandas as pd

from argparse import ArgumentParser

from definitions import ROOT_PATH
from src.models import Scorer, TfidfScorer, BertScorer


def create_submission(scorer: Scorer, df: pd.DataFrame, name_postfix: str = ""):
    scores = scorer.score_news(test_df["text"])
    df["score"] = scores

    df.to_csv(os.path.join(ROOT_PATH, f"artifacts/submission{name_postfix}.csv"), index=False)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", type=str, default="bert", choices={"bert", "tf-idf"}, help="Which model to use.")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--bs", type=int, default=16, help="Batch size for bert training.")
    args.add_argument("--n_epochs", type=int, default=4, help="Number of epochs.")
    args.add_argument("--use_gpu", action="store_true", help="Whether to switch to gpu for training.")
    args = args.parse_args()

    train_df = pd.read_csv(os.path.join(ROOT_PATH, "data/train_data.csv"))
    test_df = pd.read_csv(os.path.join(ROOT_PATH, "data/test_data.csv"))

    X, y = train_df["sentence"], train_df["label"]

    scorer = BertScorer() if args.model == "bert" else TfidfScorer(args.seed)
    scorer.fit(X, y, n_epochs=args.n_epochs, batch_size=args.bs, use_gpu=args.use_gpu)

    if args.use_gpu and isinstance(scorer, BertScorer):
        scorer.model.to("cuda")

    create_submission(scorer, test_df, f"_{args.model}")

    print("Done!")
