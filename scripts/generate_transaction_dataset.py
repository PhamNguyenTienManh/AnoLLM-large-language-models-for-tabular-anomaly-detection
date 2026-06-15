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
PAYMENT_METHODS = ["card", "wallet", "bank_transfer", "cash_on_delivery"]
CURRENCIES = ["VND", "USD", "SGD", "JPY"]


def normal_row(index):
    amount = round(random.lognormvariate(3.5, 0.55), 2)
    return {
        "transaction_id": f"TX{index:06d}",
        "user_id": f"U{random.randint(1, 80):04d}",
        "amount": min(amount, 320.0),
        "currency": random.choices(CURRENCIES, weights=[70, 15, 10, 5], k=1)[0],
        "merchant": random.choice(NORMAL_MERCHANTS),
        "category": random.choice(CATEGORIES),
        "country": random.choice(NORMAL_COUNTRIES),
        "device": random.choice(NORMAL_DEVICES),
        "hour": random.choices(range(7, 23), weights=[2, 3, 4, 6, 7, 8, 8, 8, 7, 6, 6, 5, 4, 3, 2, 1], k=1)[0],
        "payment_method": random.choice(PAYMENT_METHODS),
        "is_anomaly": 0,
    }


def anomaly_row(index):
    row = normal_row(index)
    anomaly_type = random.choice(
        [
            "large_amount",
            "midnight_foreign",
            "rare_merchant",
            "rare_device",
            "combined",
        ]
    )

    if anomaly_type == "large_amount":
        row["amount"] = round(random.uniform(1500, 9800), 2)
    elif anomaly_type == "midnight_foreign":
        row["hour"] = random.choice([0, 1, 2, 3])
        row["country"] = random.choice(RARE_COUNTRIES)
    elif anomaly_type == "rare_merchant":
        row["merchant"] = random.choice(RARE_MERCHANTS)
        row["category"] = random.choice(["crypto", "wire_transfer", "luxury"])
    elif anomaly_type == "rare_device":
        row["device"] = random.choice(RARE_DEVICES)
        row["payment_method"] = random.choice(["prepaid_card", "crypto_wallet"])
    else:
        row["amount"] = round(random.uniform(2500, 12000), 2)
        row["country"] = random.choice(RARE_COUNTRIES)
        row["merchant"] = random.choice(RARE_MERCHANTS)
        row["device"] = random.choice(RARE_DEVICES)
        row["hour"] = random.choice([0, 1, 2, 3, 4])
        row["payment_method"] = random.choice(["prepaid_card", "crypto_wallet"])

    row["is_anomaly"] = 1
    return row


def generate(output_path, n_normal, n_anomaly, seed):
    random.seed(seed)
    rows = [normal_row(i + 1) for i in range(n_normal)]
    rows.extend(anomaly_row(n_normal + i + 1) for i in range(n_anomaly))
    random.shuffle(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
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

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Normal: {n_normal}, anomaly: {n_anomaly}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/transaction/transaction.csv"))
    parser.add_argument("--n_normal", type=int, default=1080)
    parser.add_argument("--n_anomaly", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output, args.n_normal, args.n_anomaly, args.seed)


if __name__ == "__main__":
    main()
