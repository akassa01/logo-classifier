"""
Enrich labels.csv with country, year_founded, and brand_name from the
People Data Labs companies dataset cached by kagglehub.

Join key: domain (exact match after lowercasing both sides).
New columns added to labels.csv:
  - brand_name   : company name from the dataset
  - country      : country from the dataset
  - year_founded : year founded from the dataset
"""

import sys
from pathlib import Path

import pandas as pd

# Reuse the cache path defined in kaggle.py
sys.path.insert(0, str(Path(__file__).parent))
from kaggle import _find_dataset_csv

LABELS_CSV = Path(__file__).parent.parent / "labels.csv"


def main() -> None:
    labels = pd.read_csv(LABELS_CSV)
    print(f"Labels rows: {len(labels):,}")

    csv_path = _find_dataset_csv()
    print(f"Loading companies dataset: {csv_path}")
    companies = pd.read_csv(
        csv_path,
        usecols=["name", "domain", "country", "year founded"],
        low_memory=False,
    )
    print(f"Companies rows: {len(companies):,}")

    # Normalise domain for joining
    labels["_domain_key"] = labels["domain"].str.strip().str.lower()
    companies["_domain_key"] = companies["domain"].str.strip().str.lower()

    # Keep one row per domain (largest company wins — dataset is sorted by size desc)
    companies_dedup = companies.drop_duplicates(subset="_domain_key", keep="first")

    lookup = companies_dedup[["_domain_key", "name", "country", "year founded"]].rename(
        columns={"name": "brand_name", "year founded": "year_founded"}
    )

    enriched = labels.merge(lookup, on="_domain_key", how="left")
    enriched = enriched.drop(columns=["_domain_key"])

    matched = enriched["country"].notna().sum()
    print(f"Matched {matched:,} / {len(enriched):,} rows ({matched/len(enriched):.1%})")

    enriched.to_csv(LABELS_CSV, index=False)
    print(f"Saved enriched labels to {LABELS_CSV}")


if __name__ == "__main__":
    main()
