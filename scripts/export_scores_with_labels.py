import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import load_data
from train_anollm import get_run_name


def normalize_model_name(model):
    if model == "smol":
        return "HuggingFaceTB/SmolLM-135M"
    if model == "smol-360":
        return "HuggingFaceTB/SmolLM-360M"
    if model == "smol-1.7b":
        return "HuggingFaceTB/SmolLM-1.7B"
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="transaction")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--n_splits", type=int, default=1)
    parser.add_argument("--split_idx", type=int, default=0)
    parser.add_argument("--setting", type=str, default="semi_supervised")
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binning", type=str, default="standard")
    parser.add_argument("--n_buckets", type=int, default=10)
    parser.add_argument("--remove_feature_name", action="store_true")
    parser.add_argument("--model", type=str, default="distilgpt2")
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--random_init", action="store_true", default=False)
    parser.add_argument("--no_random_permutation", action="store_true", default=False)
    parser.add_argument("--exp_dir", type=Path, default=None)
    parser.add_argument("--score_path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    args.model = normalize_model_name(args.model)
    if args.exp_dir is None:
        args.exp_dir = Path("exp") / args.dataset / args.setting / f"split{args.n_splits}" / f"split{args.split_idx}"

    run_name = get_run_name(args)
    if args.score_path is None:
        args.score_path = args.exp_dir / "scores" / f"{run_name}.npy"
    if args.output is None:
        args.output = args.exp_dir / "scores" / f"{args.dataset}_scores_with_labels.csv"

    _, X_test, _, y_test = load_data(args)
    scores = np.load(args.score_path)
    if len(scores) != len(y_test):
        raise ValueError(
            f"Score length ({len(scores)}) does not match test label length ({len(y_test)}). "
            "Check that the score file was produced with the same dataset/split args."
        )

    output_df = pd.DataFrame(
        {
            "row_index": X_test.index.to_numpy(),
            "anomaly_score": scores,
            "is_anomaly": y_test.astype(int),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)

    print(f"Wrote {len(output_df)} rows to {args.output}")
    print(output_df["is_anomaly"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
