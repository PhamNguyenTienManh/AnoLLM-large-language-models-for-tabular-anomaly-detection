"""Explain *why* AnoLLM flags a transaction as fraud.

AnoLLM's anomaly score is the negative log-likelihood of the serialized row.
decision_function(..., feature_wise=True) returns that NLL broken down per
feature, shape (n_test, n_features, n_permutations). A feature whose value the
model finds "surprising" contributes a high NLL.

To turn raw per-feature NLL into an attribution we standardize each feature
against the distribution of NLL it has on *normal* test rows:

    z_feature(row) = (nll_feature(row) - mean_normal_feature) / std_normal_feature

The feature with the largest z is the main reason the row was flagged. We then
cross-check the dominant feature against the ground-truth anomaly_type sidecar
(e.g. `subtle_amount` rows should be driven by the `amount` feature).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssl_patch import patch_ssl_context

patch_ssl_context()

from src.data_utils import load_data, get_text_columns, get_max_length_dict
from train_anollm import get_run_name


def normalize_model_name(model):
    mapping = {
        "smol": "HuggingFaceTB/SmolLM-135M",
        "smol-360": "HuggingFaceTB/SmolLM-360M",
        "smol-1.7b": "HuggingFaceTB/SmolLM-1.7B",
    }
    return mapping.get(model, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="transaction")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--n_splits", type=int, default=1)
    parser.add_argument("--split_idx", type=int, default=0)
    parser.add_argument("--setting", type=str, default="semi_supervised")
    parser.add_argument("--train_ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binning", type=str, default="standard")
    parser.add_argument("--n_buckets", type=int, default=10)
    parser.add_argument("--remove_feature_name", action="store_true")
    parser.add_argument("--model", type=str, default="distilgpt2")
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--random_init", action="store_true", default=False)
    parser.add_argument("--no_random_permutation", action="store_true", default=False)
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--n_permutations", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    args.model = normalize_model_name(args.model)
    run_name = get_run_name(args)
    model_path = args.exp_dir / "models" / f"{run_name}.pt"
    score_dir = args.exp_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data(args)
    y_test = np.asarray(y_test).astype(int)
    columns = list(X_test.columns)

    # ---- load the trained model (CPU) ----
    from anollm import AnoLLM

    max_length_dict = get_max_length_dict(args.dataset)
    text_columns = get_text_columns(args.dataset)
    model = AnoLLM(
        args.model,
        efficient_finetuning="lora" if args.lora else "",
        model_path=model_path,
        max_length_dict=max_length_dict,
        textual_columns=text_columns,
        no_random_permutation=args.no_random_permutation,
        bf16=False,
    )
    model.load_from_state_dict(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.to(device)

    # ---- per-feature anomaly scores ----
    fw = model.decision_function(
        X_test, n_permutations=args.n_permutations,
        batch_size=args.batch_size, device=device, feature_wise=True,
    )  # (n_test, n_features, n_perm)
    feat_scores = fw.mean(axis=2)  # (n_test, n_features)

    # ---- standardize each feature against the normal baseline ----
    normal_mask = y_test == 0
    mu = feat_scores[normal_mask].mean(axis=0)
    sigma = feat_scores[normal_mask].std(axis=0) + 1e-6
    z = (feat_scores - mu) / sigma  # (n_test, n_features)

    # ---- attach ground-truth anomaly_type (analysis only) ----
    meta_path = Path(args.data_dir) / args.dataset / "transaction_meta.csv"
    id_col = "transaction id" if "transaction id" in columns else columns[0]
    tx_ids = X_test[id_col].astype(str).to_numpy()
    type_lookup = {}
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        type_lookup = dict(zip(meta["transaction_id"].astype(str), meta["anomaly_type"]))
    anomaly_type = np.array([type_lookup.get(t, "unknown") for t in tx_ids])

    top_idx = z.argmax(axis=1)
    top_feature = np.array([columns[i] for i in top_idx])

    # ---- per-row attribution table ----
    out = pd.DataFrame({"row_index": X_test.index.to_numpy(),
                        "transaction_id": tx_ids,
                        "is_anomaly": y_test,
                        "anomaly_type": anomaly_type,
                        "total_score": feat_scores.sum(axis=1),
                        "top_feature": top_feature,
                        "top_z": z[np.arange(len(z)), top_idx]})
    for j, c in enumerate(columns):
        out[f"z__{c}"] = z[:, j]
    attr_path = score_dir / "feature_attribution.csv"
    out.to_csv(attr_path, index=False)

    # ===================== printed summary =====================
    fraud = y_test == 1
    print("=" * 72)
    print(f"Feature attribution  |  test={len(y_test)}  fraud={fraud.sum()}")
    print("-" * 72)
    print("Mean standardized contribution (z) per feature  [fraud vs normal]:")
    order = np.argsort(-z[fraud].mean(axis=0))
    print(f"  {'feature':<18}{'fraud z':>10}{'normal z':>10}")
    for j in order:
        print(f"  {columns[j]:<18}{z[fraud, j].mean():>10.3f}{z[normal_mask, j].mean():>10.3f}")

    print("-" * 72)
    print("How often each feature is the top driver of a flagged fraud:")
    tf = pd.Series(top_feature[fraud]).value_counts()
    for name, cnt in tf.items():
        print(f"  {name:<18}{cnt:>5}  ({cnt / fraud.sum() * 100:4.1f}%)")

    if type_lookup:
        print("-" * 72)
        print("Dominant feature per ground-truth anomaly_type:")
        df = out[out["is_anomaly"] == 1]
        for atype, grp in df.groupby("anomaly_type"):
            top = grp["top_feature"].value_counts()
            frac = top.iloc[0] / len(grp) * 100
            print(f"  {atype:<28} n={len(grp):<4} -> {top.index[0]} "
                  f"({frac:.0f}% of rows), meanZ={grp['top_z'].mean():.2f}")

    print("-" * 72)
    print("Example flagged transactions (highest total score):")
    ex = out[out["is_anomaly"] == 1].sort_values("total_score", ascending=False).head(5)
    for _, r in ex.iterrows():
        contribs = sorted(
            ((c, r[f"z__{c}"]) for c in columns), key=lambda kv: -kv[1])[:3]
        reason = ", ".join(f"{c}(z={zz:.1f})" for c, zz in contribs)
        print(f"  {r['transaction_id']}  [{r['anomaly_type']}] -> {reason}")

    print("=" * 72)
    print(f"Wrote {attr_path}")


if __name__ == "__main__":
    main()
