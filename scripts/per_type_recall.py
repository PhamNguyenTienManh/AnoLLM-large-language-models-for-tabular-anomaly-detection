"""Per-anomaly-type detection rate using the official averaged anomaly scores.

Flags the top-`contamination` fraction of test rows as anomalies (the same
contamination-based threshold used in evaluate_robust.py) and reports, for each
ground-truth anomaly_type, how many of its frauds were caught. This exposes
which borderline fraud types are genuinely hard for the model.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--exp_dir", type=Path, required=True)
    args = parser.parse_args()

    scores_csv = args.exp_dir / "scores" / "transaction_scores_with_labels.csv"
    df = pd.read_csv(scores_csv)  # row_index, anomaly_score, is_anomaly

    # Map row_index -> transaction_id via the original (un-split) dataframe order.
    X, y = load_dataset("transaction", args.data_dir)
    id_series = X["transaction id"] if "transaction id" in X.columns else X.iloc[:, 0]
    id_by_index = id_series.to_dict()
    df["transaction_id"] = df["row_index"].map(id_by_index)

    meta = pd.read_csv(Path(args.data_dir) / "transaction" / "transaction_meta.csv")
    type_by_id = dict(zip(meta["transaction_id"].astype(str), meta["anomaly_type"]))
    df["anomaly_type"] = df["transaction_id"].astype(str).map(type_by_id).fillna("unknown")

    contamination = df["is_anomaly"].mean()
    thr = np.quantile(df["anomaly_score"], 1 - contamination)
    df["pred"] = (df["anomaly_score"] >= thr).astype(int)

    fraud = df[df["is_anomaly"] == 1]
    print("=" * 64)
    print(f"Per-type detection rate (threshold = top {contamination*100:.1f}%):")
    print(f"  {'anomaly_type':<28}{'n':>5}{'caught':>8}{'recall':>9}")
    for atype, grp in fraud.groupby("anomaly_type"):
        caught = int(grp["pred"].sum())
        print(f"  {atype:<28}{len(grp):>5}{caught:>8}{caught/len(grp):>9.3f}")
    print("-" * 64)
    print(f"  {'ALL FRAUD':<28}{len(fraud):>5}{int(fraud['pred'].sum()):>8}"
          f"{fraud['pred'].mean():>9.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
