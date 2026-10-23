"""
sample_data.py
--------------
Generate a synthetic subscription-business customer dataset with a realistic
churn signal. Deliberately *not* the telecom schema — it shows the engine
working on columns it has never seen, and doubles as test data.

Usage:
    python -m src.sample_data --output ui/examples/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_customers(n: int = 3000, seed: int = 7, labelled: bool = True) -> pd.DataFrame:
    """
    Build n synthetic customers. Churn is driven by short tenure, month-to-month
    contracts, support tickets, inactivity, and price — plus noise.
    """
    rng = np.random.default_rng(seed)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    plan     = rng.choice(["Basic", "Standard", "Premium"], n, p=[0.4, 0.4, 0.2])
    payment  = rng.choice(["Card", "Bank transfer", "Invoice", "PayPal"], n)
    region   = rng.choice(["North", "South", "East", "West"], n)

    tenure   = rng.integers(1, 72, n)
    base     = {"Basic": 20, "Standard": 45, "Premium": 80}
    monthly  = np.array([base[p] for p in plan]) + rng.normal(0, 6, n).round(2)
    tickets  = rng.poisson(1.2, n)
    inactive = rng.exponential(10, n).round().astype(int)
    total    = (monthly * tenure).round(2).astype(object)
    total[rng.random(n) < 0.01] = " "     # blank values, like real billing exports

    logit = (
        -1.0
        - 0.04 * tenure
        + 1.4 * (contract == "Month-to-month")
        - 0.8 * (contract == "Two year")
        + 0.35 * tickets
        + 0.05 * inactive
        + 0.012 * (monthly - 45)
        + 0.4 * (payment == "Invoice")
        + rng.normal(0, 0.6, n)
    )
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame({
        "customer_id":          [f"C{100000 + i}" for i in range(n)],
        "tenure_months":        tenure,
        "contract":             contract,
        "plan":                 plan,
        "monthly_charges":      monthly,
        "total_charges":        total,
        "payment_method":       payment,
        "region":               region,
        "support_tickets_90d":  tickets,
        "days_since_last_login": inactive,
    })
    if labelled:
        df["churned"] = np.where(churned, "Yes", "No")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write synthetic example CSVs")
    parser.add_argument("--output", default="ui/examples/", help="Output directory")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    make_customers(3000, seed=7).to_csv(out / "sample_customers.csv", index=False)
    make_customers(500, seed=99, labelled=False).to_csv(out / "customers_to_score.csv", index=False)
    print(f"Wrote {out / 'sample_customers.csv'} and {out / 'customers_to_score.csv'}")
