"""
Compare industry distribution between enriched_labels.csv and the cached Kaggle
companies dataset, matched by domain.
"""

import pandas as pd
from pathlib import Path

ENRICHED = Path(__file__).parent.parent / "enriched_labels.csv"
KAGGLE_CSV = (
    Path.home()
    / ".cache/kagglehub/datasets/peopledatalabssf"
    / "free-7-million-company-dataset/versions/1/companies_sorted.csv"
)


def load_data():
    enriched = pd.read_csv(ENRICHED, usecols=["domain", "sector"])
    enriched["domain"] = enriched["domain"].str.lower().str.strip()

    kaggle = pd.read_csv(KAGGLE_CSV, usecols=["domain", "industry"], low_memory=False)
    kaggle["domain"] = kaggle["domain"].str.lower().str.strip()
    kaggle = kaggle.dropna(subset=["domain", "industry"])
    # keep first occurrence per domain (dataset has duplicates)
    kaggle = kaggle.drop_duplicates(subset="domain")

    return enriched, kaggle


def industry_balance(enriched: pd.DataFrame, kaggle: pd.DataFrame) -> pd.DataFrame:
    merged = enriched.merge(kaggle, on="domain", how="inner")

    total = len(merged)
    counts = (
        merged.groupby("industry")
        .agg(count=("domain", "count"), sectors=("sector", lambda s: ", ".join(sorted(s.unique()))))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    counts["pct"] = (counts["count"] / total * 100).round(2)
    return counts, merged, total


def main():
    print("Loading data...")
    enriched, kaggle = load_data()

    n_enriched = len(enriched)
    n_matched = enriched["domain"].isin(kaggle["domain"]).sum()

    print(f"\nEnriched labels : {n_enriched:,} rows")
    print(f"Matched to Kaggle: {n_matched:,} rows ({n_matched/n_enriched*100:.1f}%)")
    print(f"Unmatched        : {n_enriched - n_matched:,} rows\n")

    counts, merged, total = industry_balance(enriched, kaggle)

    print(f"Industry balance across {total:,} matched companies")
    print(f"{'Industry':<50} {'Count':>7}  {'%':>6}  Sectors")
    print("-" * 100)
    for _, row in counts.iterrows():
        print(f"{row['industry']:<50} {row['count']:>7}  {row['pct']:>5.2f}%  {row['sectors']}")

    out_path = Path(__file__).parent / "industry_balance_report.csv"
    counts.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    print("\n--- Sector distribution within matched set ---")
    sector_counts = merged["sector"].value_counts()
    for sector, n in sector_counts.items():
        print(f"  {sector:<40} {n:>7}  ({n/total*100:.1f}%)")


if __name__ == "__main__":
    main()
