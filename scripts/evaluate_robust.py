"""Robust evaluation of AnoLLM transaction scores.

Instead of reporting a single ROC-AUC / F1 number, this script quantifies how
trustworthy the result is:

  * point metrics (ROC-AUC, Average Precision, F1@k, Precision@k, Recall@k)
  * bootstrap 95% confidence intervals (resampling the test set)
  * variance across column-order permutations (from the raw per-permutation scores)
  * a threshold sweep (best-F1 threshold + contamination-based threshold)
  * an ablation over the number of permutations used to average the score

It reuses load_data() so the test labels line up exactly with the saved scores,
and writes everything to <exp_dir>/scores/robust_metrics.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn import metrics

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import load_data
from train_anollm import get_run_name


def normalize_model_name(model):
    mapping = {
        "smol": "HuggingFaceTB/SmolLM-135M",
        "smol-360": "HuggingFaceTB/SmolLM-360M",
        "smol-1.7b": "HuggingFaceTB/SmolLM-1.7B",
    }
    return mapping.get(model, model)


def f1_at_k(y_true, y_score, seed=0):
    """F1 using the top-k highest scores as positives, where k = #true anomalies.

    This is the threshold-free F1@k used in src/get_results.py:tabular_metrics.
    Ties are broken by a seeded shuffle so ordering does not bias argpartition.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    idx = rng.permutation(n)
    yt = y_true[idx].astype(int)
    ys = y_score[idx]
    k = int((yt == 1).sum())
    if k == 0 or k == n:
        return 0.0, 0.0, 0.0
    top = np.argpartition(ys, -k)[-k:]
    yp = np.zeros_like(yt)
    yp[top] = 1
    p, r, f1, _ = metrics.precision_recall_fscore_support(
        yt, yp, average="binary", zero_division=0
    )
    return float(f1), float(p), float(r)


def point_metrics(y_true, y_score):
    return {
        "roc_auc": float(metrics.roc_auc_score(y_true, y_score)),
        "average_precision": float(metrics.average_precision_score(y_true, y_score)),
        "f1_at_k": f1_at_k(y_true, y_score)[0],
        "precision_at_k": f1_at_k(y_true, y_score)[1],
        "recall_at_k": f1_at_k(y_true, y_score)[2],
    }


def bootstrap_ci(y_true, y_score, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    acc = {"roc_auc": [], "average_precision": [], "f1_at_k": []}
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        ys = y_score[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        acc["roc_auc"].append(metrics.roc_auc_score(yt, ys))
        acc["average_precision"].append(metrics.average_precision_score(yt, ys))
        acc["f1_at_k"].append(f1_at_k(yt, ys, seed=b)[0])
    ci = {}
    for k, v in acc.items():
        v = np.asarray(v)
        ci[k] = {
            "mean": float(v.mean()),
            "std": float(v.std()),
            "ci95_low": float(np.percentile(v, 2.5)),
            "ci95_high": float(np.percentile(v, 97.5)),
        }
    ci["n_bootstrap"] = int(len(acc["roc_auc"]))
    return ci


def permutation_variance(y_true, raw_scores):
    """raw_scores: (n_test, n_perm). Metric spread when using a single permutation."""
    n_perm = raw_scores.shape[1]
    per = {"roc_auc": [], "average_precision": [], "f1_at_k": []}
    for j in range(n_perm):
        s = raw_scores[:, j]
        per["roc_auc"].append(metrics.roc_auc_score(y_true, s))
        per["average_precision"].append(metrics.average_precision_score(y_true, s))
        per["f1_at_k"].append(f1_at_k(y_true, s, seed=j)[0])
    out = {"n_permutations": int(n_perm)}
    for k, v in per.items():
        v = np.asarray(v)
        out[k] = {"mean": float(v.mean()), "std": float(v.std()),
                  "min": float(v.min()), "max": float(v.max())}
    return out


def n_permutation_ablation(y_true, raw_scores):
    """Average the first k permutations and measure the metric, for k=1,2,4,8,...."""
    n_perm = raw_scores.shape[1]
    ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= n_perm]
    if n_perm not in ks:
        ks.append(n_perm)
    rows = []
    for k in ks:
        s = raw_scores[:, :k].mean(axis=1)
        rows.append({
            "n_perm": int(k),
            "roc_auc": float(metrics.roc_auc_score(y_true, s)),
            "average_precision": float(metrics.average_precision_score(y_true, s)),
            "f1_at_k": f1_at_k(y_true, s)[0],
        })
    return rows


def threshold_analysis(y_true, y_score):
    contamination = float(y_true.mean())
    # Best-F1 threshold over the full PR curve.
    prec, rec, thr = metrics.precision_recall_curve(y_true, y_score)
    f1_curve = np.divide(2 * prec * rec, prec + rec,
                         out=np.zeros_like(prec), where=(prec + rec) > 0)
    best = int(np.argmax(f1_curve))
    best_thr = float(thr[min(best, len(thr) - 1)])
    # Contamination-based threshold: flag the top `contamination` fraction.
    cont_thr = float(np.quantile(y_score, 1 - contamination))
    yp = (y_score >= cont_thr).astype(int)
    p, r, f1, _ = metrics.precision_recall_fscore_support(
        y_true, yp, average="binary", zero_division=0)
    return {
        "contamination": contamination,
        "best_f1": float(f1_curve[best]),
        "best_f1_threshold": best_thr,
        "best_f1_precision": float(prec[best]),
        "best_f1_recall": float(rec[best]),
        "contamination_threshold": cont_thr,
        "contamination_f1": float(f1),
        "contamination_precision": float(p),
        "contamination_recall": float(r),
    }


def maybe_plot(y_true, y_score, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"(skip plots, matplotlib unavailable: {exc})")
        return
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve")
    plt.savefig(out_dir / "roc_curve.png", dpi=120, bbox_inches="tight"); plt.close()

    prec, rec, _ = metrics.precision_recall_curve(y_true, y_score)
    plt.figure(); plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve")
    plt.savefig(out_dir / "pr_curve.png", dpi=120, bbox_inches="tight"); plt.close()

    plt.figure()
    plt.hist(y_score[y_true == 0], bins=40, alpha=0.6, label="normal", density=True)
    plt.hist(y_score[y_true == 1], bins=40, alpha=0.6, label="fraud", density=True)
    plt.xlabel("anomaly score"); plt.ylabel("density"); plt.legend()
    plt.title("Score distribution"); plt.savefig(out_dir / "score_hist.png",
                                                  dpi=120, bbox_inches="tight"); plt.close()
    print(f"Saved plots to {out_dir}")


def fmt_ci(d):
    return f"{d['mean']:.4f}  (95% CI {d['ci95_low']:.4f}-{d['ci95_high']:.4f})"


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
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    args.model = normalize_model_name(args.model)
    run_name = get_run_name(args)
    score_dir = args.exp_dir / "scores"
    mean_path = score_dir / f"{run_name}.npy"
    raw_path = score_dir / f"raw_{run_name}.npy"

    _, X_test, _, y_test = load_data(args)
    y_test = np.asarray(y_test).astype(int)
    scores = np.load(mean_path)
    if len(scores) != len(y_test):
        raise ValueError(f"score len {len(scores)} != label len {len(y_test)}")

    report = {
        "run_name": run_name,
        "n_test": int(len(y_test)),
        "n_fraud": int(y_test.sum()),
        "n_normal": int((y_test == 0).sum()),
        "point_metrics": point_metrics(y_test, scores),
        "bootstrap": bootstrap_ci(y_test, scores, n_boot=args.n_bootstrap, seed=args.seed),
        "threshold": threshold_analysis(y_test, scores),
    }

    if raw_path.exists():
        raw = np.load(raw_path)
        report["permutation_variance"] = permutation_variance(y_test, raw)
        report["n_permutation_ablation"] = n_permutation_ablation(y_test, raw)
    else:
        print(f"(no raw permutation file at {raw_path}; skipping permutation analysis)")

    maybe_plot(y_test, scores, score_dir)

    out_path = score_dir / "robust_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ---- human-readable summary ----
    pm = report["point_metrics"]
    bs = report["bootstrap"]
    th = report["threshold"]
    print("=" * 70)
    print(f"Robust evaluation  |  test={report['n_test']}  "
          f"fraud={report['n_fraud']}  normal={report['n_normal']}")
    print("-" * 70)
    print("Point estimates:")
    print(f"  ROC-AUC            : {pm['roc_auc']:.4f}")
    print(f"  Average Precision  : {pm['average_precision']:.4f}")
    print(f"  F1@k               : {pm['f1_at_k']:.4f}  "
          f"(P={pm['precision_at_k']:.4f}, R={pm['recall_at_k']:.4f})")
    print("Bootstrap (resample test set):")
    print(f"  ROC-AUC            : {fmt_ci(bs['roc_auc'])}")
    print(f"  Average Precision  : {fmt_ci(bs['average_precision'])}")
    print(f"  F1@k               : {fmt_ci(bs['f1_at_k'])}")
    if "permutation_variance" in report:
        pv = report["permutation_variance"]
        print(f"Across {pv['n_permutations']} permutations (single-perm spread):")
        print(f"  ROC-AUC            : {pv['roc_auc']['mean']:.4f} "
              f"+/- {pv['roc_auc']['std']:.4f} "
              f"[{pv['roc_auc']['min']:.4f}, {pv['roc_auc']['max']:.4f}]")
        print(f"  F1@k               : {pv['f1_at_k']['mean']:.4f} "
              f"+/- {pv['f1_at_k']['std']:.4f}")
        print("n_permutation ablation (averaged score):")
        for row in report["n_permutation_ablation"]:
            print(f"  n_perm={row['n_perm']:>2}  ROC-AUC={row['roc_auc']:.4f}  "
                  f"AP={row['average_precision']:.4f}  F1@k={row['f1_at_k']:.4f}")
    print("Threshold analysis:")
    print(f"  best-F1            : {th['best_f1']:.4f} "
          f"(P={th['best_f1_precision']:.4f}, R={th['best_f1_recall']:.4f})")
    print(f"  F1 @ contamination : {th['contamination_f1']:.4f} "
          f"(P={th['contamination_precision']:.4f}, R={th['contamination_recall']:.4f})")
    print("=" * 70)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
