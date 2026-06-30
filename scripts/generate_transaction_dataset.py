import argparse
import csv
import random
import shutil
from pathlib import Path


COLUMNS = [
    "transaction_id",
    "user_id",
    "amount",
    "currency",
    "merchant",
    "category",
    "country",
    "device",
    "hour",
    "payment_method",
    "is_anomaly",
]

# Sidecar file: transaction_id -> anomaly_type. Used ONLY for analysis / ablation,
# never fed to the model (transaction.csv keeps the original 11 columns).
META_COLUMNS = ["transaction_id", "anomaly_type"]


NORMAL_COUNTRIES = ["VN", "US", "SG", "JP", "KR", "TH"]
RARE_COUNTRIES = ["AQ", "IR", "KP", "SY", "ZZ"]
NORMAL_MERCHANTS = [
    "FreshMart",
    "BookNest",
    "CloudCafe",
    "MetroRide",
    "QuickPay",
    "StyleHub",
    "GameBox",
    "HomePlus",
]
RARE_MERCHANTS = ["ShadowWire", "CryptoMule", "NightVault", "PhantomShop"]
NORMAL_DEVICES = ["ios", "android", "web", "pos_terminal"]
RARE_DEVICES = ["unknown_emulator", "rooted_phone", "tor_browser", "foreign_pos"]
CATEGORIES = ["grocery", "transport", "food", "retail", "entertainment", "utilities"]
ODD_CATEGORIES = ["crypto", "wire_transfer", "luxury"]
PAYMENT_METHODS = ["card", "wallet", "bank_transfer", "cash_on_delivery"]
RARE_PAYMENTS = ["prepaid_card", "crypto_wallet"]
CURRENCIES = ["VND", "USD", "SGD", "JPY"]

# Daytime hour weights (7..22). Night hours are added separately with a small
# probability so that "late hour" is no longer a perfect fraud indicator.
DAY_HOURS = list(range(7, 23))
DAY_WEIGHTS = [2, 3, 4, 6, 7, 8, 8, 8, 7, 6, 6, 5, 4, 3, 2, 1]


def normal_amount():
    # Heavier tail (sigma 0.7) and no hard cap, so a small fraction of normal
    # transactions reach 400-900 and genuinely overlap the "subtle" fraud range.
    return round(random.lognormvariate(3.6, 0.7), 2)


def normal_row(index):
    # ~3% of normal transactions happen at night, ~2% from a rare country.
    if random.random() < 0.03:
        hour = random.choice([0, 1, 2, 3, 4, 5, 6, 23])
    else:
        hour = random.choices(DAY_HOURS, weights=DAY_WEIGHTS, k=1)[0]

    country = random.choice(NORMAL_COUNTRIES)
    if random.random() < 0.02:
        country = random.choice(RARE_COUNTRIES)

    return {
        "transaction_id": f"TX{index:06d}",
        "user_id": f"U{random.randint(1, 80):04d}",
        "amount": normal_amount(),
        "currency": random.choices(CURRENCIES, weights=[70, 15, 10, 5], k=1)[0],
        "merchant": random.choice(NORMAL_MERCHANTS),
        "category": random.choice(CATEGORIES),
        "country": country,
        "device": random.choice(NORMAL_DEVICES),
        "hour": hour,
        "payment_method": random.choice(PAYMENT_METHODS),
        "is_anomaly": 0,
        "anomaly_type": "normal",
    }


# Fraud types. The list mixes "borderline" frauds that overlap normal behaviour
# (subtle_amount, odd_hour_local, merchant_category_mismatch, velocity_like)
# with a few more separable ones, so the task is no longer trivially solvable.
ANOMALY_TYPES = [
    "subtle_amount",
    "odd_hour_local",
    "merchant_category_mismatch",
    "velocity_like",
    "large_amount",
    "rare_country",
    "rare_device",
    "combined",
]
ANOMALY_WEIGHTS = [22, 20, 16, 12, 10, 8, 7, 5]


def anomaly_row(index):
    row = normal_row(index)
    anomaly_type = random.choices(ANOMALY_TYPES, weights=ANOMALY_WEIGHTS, k=1)[0]

    if anomaly_type == "subtle_amount":
        # Sits in the upper tail of normal amounts -> hard to separate.
        row["amount"] = round(random.uniform(250, 600), 2)

    elif anomaly_type == "odd_hour_local":
        # Domestic country, but unusual hour + slightly elevated amount.
        row["hour"] = random.choice([1, 2, 3, 4])
        row["amount"] = round(random.uniform(200, 500), 2)

    elif anomaly_type == "merchant_category_mismatch":
        # Familiar merchant, but an out-of-context category / payment method.
        row["category"] = random.choice(ODD_CATEGORIES)
        if random.random() < 0.5:
            row["payment_method"] = random.choice(RARE_PAYMENTS)

    elif anomaly_type == "velocity_like":
        # Looks almost normal: same kind of fields, only a mild combination of
        # slightly-high amount + off hour. Intentionally hard (tests recall).
        row["amount"] = round(random.uniform(150, 350), 2)
        row["hour"] = random.choice([5, 6, 23])

    elif anomaly_type == "large_amount":
        # Reduced from the old 1500-12000 range; now partly overlaps the tail.
        row["amount"] = round(random.uniform(800, 3000), 2)

    elif anomaly_type == "rare_country":
        row["country"] = random.choice(RARE_COUNTRIES)
        row["amount"] = round(random.uniform(200, 900), 2)

    elif anomaly_type == "rare_device":
        row["device"] = random.choice(RARE_DEVICES)
        row["payment_method"] = random.choice(RARE_PAYMENTS)

    else:  # combined -> several strong signals at once (kept easy on purpose)
        row["amount"] = round(random.uniform(1500, 6000), 2)
        row["country"] = random.choice(RARE_COUNTRIES)
        row["merchant"] = random.choice(RARE_MERCHANTS)
        row["device"] = random.choice(RARE_DEVICES)
        row["hour"] = random.choice([0, 1, 2, 3, 4])
        row["payment_method"] = random.choice(RARE_PAYMENTS)

    row["is_anomaly"] = 1
    row["anomaly_type"] = anomaly_type
    return row


def generate(output_path, n_normal, n_anomaly, seed):
    random.seed(seed)
    rows = [normal_row(i + 1) for i in range(n_normal)]
    rows.extend(anomaly_row(n_normal + i + 1) for i in range(n_anomaly))
    random.shuffle(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    meta_path = output_path.parent / "transaction_meta.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    for cache_path in [
        output_path.parent / "data.pkl",
        output_path.parent / "semi_supervised",
        output_path.parent / "unsupervised",
    ]:
        try:
            if cache_path.is_dir():
                shutil.rmtree(cache_path)
            elif cache_path.exists():
                cache_path.unlink()
        except PermissionError:
            print(f"Warning: could not remove cache {cache_path}; transaction.csv will still be used directly.")

    # Quick sanity report on the normal/fraud amount overlap.
    normal_amounts = sorted(r["amount"] for r in rows if r["is_anomaly"] == 0)
    fraud_amounts = sorted(r["amount"] for r in rows if r["is_anomaly"] == 1)

    def pct(values, q):
        if not values:
            return float("nan")
        return values[min(len(values) - 1, int(q * len(values)))]

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Wrote anomaly-type sidecar to {meta_path}")
    print(f"Normal: {n_normal}, anomaly: {n_anomaly}")
    print(
        "Normal amount  p50/p95/max: {:.1f} / {:.1f} / {:.1f}".format(
            pct(normal_amounts, 0.50), pct(normal_amounts, 0.95), normal_amounts[-1]
        )
    )
    print(
        "Fraud  amount  min/p50/p95: {:.1f} / {:.1f} / {:.1f}".format(
            fraud_amounts[0], pct(fraud_amounts, 0.50), pct(fraud_amounts, 0.95)
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/transaction/transaction.csv"))
    parser.add_argument("--n_normal", type=int, default=3000)
    parser.add_argument("--n_anomaly", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output, args.n_normal, args.n_anomaly, args.seed)


if __name__ == "__main__":
    main()
