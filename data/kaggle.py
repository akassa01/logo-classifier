import os
import glob
import kagglehub
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# kagglehub uses KAGGLE_TOKEN for the new KGAT_* token format
token = os.getenv("KAGGLE_API_TOKEN")
if token and not os.getenv("KAGGLE_TOKEN"):
    os.environ["KAGGLE_TOKEN"] = token


# ---------------------------------------------------------------------------
# Industry normalization
# ---------------------------------------------------------------------------

# Exact-duplicate renames: map the messier variant to the canonical one
INDUSTRY_RENAMES: dict[str, str] = {
    "nonprofit organization management": "non-profit organization management",
    "airlines/aviation": "aviation & aerospace",
    "law practice": "legal services",
}

# Full industry → sector mapping
INDUSTRY_TO_SECTOR: dict[str, str] = {
    # Technology
    "information technology and services": "Technology",
    "computer software": "Technology",
    "internet": "Technology",
    "computer hardware": "Technology",
    "computer networking": "Technology",
    "computer & network security": "Technology",
    "wireless": "Technology",
    "information services": "Technology",
    "semiconductors": "Technology",
    "computer games": "Technology",
    # Marketing & Communications
    "marketing and advertising": "Marketing & Communications",
    "public relations and communications": "Marketing & Communications",
    "market research": "Marketing & Communications",
    # Construction & Infrastructure
    "construction": "Construction & Infrastructure",
    "civil engineering": "Construction & Infrastructure",
    "building materials": "Construction & Infrastructure",
    "facilities services": "Construction & Infrastructure",
    # Management Consulting
    "management consulting": "Management Consulting",
    # Real Estate
    "real estate": "Real Estate",
    "commercial real estate": "Real Estate",
    # Financial Services  (accounting folded in)
    "financial services": "Financial Services",
    "banking": "Financial Services",
    "investment management": "Financial Services",
    "investment banking": "Financial Services",
    "capital markets": "Financial Services",
    "venture capital & private equity": "Financial Services",
    "insurance": "Financial Services",
    "accounting": "Financial Services",
    # Health & Wellness
    "health, wellness and fitness": "Health & Wellness",
    "hospital & health care": "Health & Wellness",
    "medical practice": "Health & Wellness",
    "mental health care": "Health & Wellness",
    "alternative medicine": "Health & Wellness",
    "veterinary": "Health & Wellness",
    "pharmaceuticals": "Health & Wellness",
    # Design & Creative 
    "design": "Design & Creative",
    "graphic design": "Design & Creative",
    "photography": "Design & Creative",
    "fine art": "Design & Creative",
    "arts and crafts": "Design & Creative",
    "architecture & planning": "Design & Creative",
    "animation": "Design & Creative",
    # Education
    "education management": "Education",
    "higher education": "Education",
    "primary/secondary education": "Education",
    "e-learning": "Education",
    "professional training & coaching": "Education",
    "libraries": "Education",
    "program development": "Education",
    # Non-profit & Social
    "non-profit organization management": "Non-profit & Social",
    "civic & social organization": "Non-profit & Social",
    "philanthropy": "Non-profit & Social",
    "religious institutions": "Non-profit & Social",
    "fund-raising": "Non-profit & Social",
    "individual & family services": "Non-profit & Social",
    "museums and institutions": "Non-profit & Social",
    # Human Resources & Staffing
    "human resources": "Human Resources & Staffing",
    "staffing and recruiting": "Human Resources & Staffing",
    "outsourcing/offshoring": "Human Resources & Staffing",
    # Government & Public Sector
    "government administration": "Government & Public Sector",
    "government relations": "Government & Public Sector",
    "public policy": "Government & Public Sector",
    "political organization": "Government & Public Sector",
    "international affairs": "Government & Public Sector",
    "legislative office": "Government & Public Sector",
    "military": "Government & Public Sector",
    "public safety": "Government & Public Sector",
    "law enforcement": "Government & Public Sector",
    "judiciary": "Government & Public Sector",
    "defense & space": "Government & Public Sector",
    "think tanks": "Government & Public Sector",
    "international trade and development": "Government & Public Sector",
    # Legal
    "legal services": "Legal",
    "alternative dispute resolution": "Legal",
    # Energy & Environment
    "oil & energy": "Energy & Environment",
    "renewables & environment": "Energy & Environment",
    "environmental services": "Energy & Environment",
    "utilities": "Energy & Environment",
    "mining & metals": "Energy & Environment",
    # Food & Agriculture
    "food & beverages": "Food & Agriculture",
    "food production": "Food & Agriculture",
    "restaurants": "Food & Agriculture",
    "farming": "Food & Agriculture",
    "dairy": "Food & Agriculture",
    "fishery": "Food & Agriculture",
    "ranching": "Food & Agriculture",
    "supermarkets": "Food & Agriculture",
    "wine and spirits": "Food & Agriculture",
    # Transportation & Logistics
    "transportation/trucking/railroad": "Transportation & Logistics",
    "logistics and supply chain": "Transportation & Logistics",
    "maritime": "Transportation & Logistics",
    "warehousing": "Transportation & Logistics",
    "package/freight delivery": "Transportation & Logistics",
    "aviation & aerospace": "Transportation & Logistics",
    "import and export": "Transportation & Logistics",
    # Manufacturing
    "mechanical or industrial engineering": "Manufacturing",
    "electrical/electronic manufacturing": "Manufacturing",
    "machinery": "Manufacturing",
    "industrial automation": "Manufacturing",
    "chemicals": "Manufacturing",
    "plastics": "Manufacturing",
    "packaging and containers": "Manufacturing",
    "paper & forest products": "Manufacturing",
    "glass, ceramics & concrete": "Manufacturing",
    "textiles": "Manufacturing",
    "printing": "Manufacturing",
    "shipbuilding": "Manufacturing",
    "railroad manufacture": "Manufacturing",
    "automotive": "Manufacturing",
    # Retail & Consumer
    "retail": "Retail & Consumer",
    "apparel & fashion": "Retail & Consumer",
    "cosmetics": "Retail & Consumer",
    "luxury goods & jewelry": "Retail & Consumer",
    "furniture": "Retail & Consumer",
    "consumer goods": "Retail & Consumer",
    "consumer electronics": "Retail & Consumer",
    "sporting goods": "Retail & Consumer",
    "wholesale": "Retail & Consumer",
    "business supplies and equipment": "Retail & Consumer",
    "tobacco": "Retail & Consumer",
    # Media & Entertainment
    "media production": "Media & Entertainment",
    "entertainment": "Media & Entertainment",
    "broadcast media": "Media & Entertainment",
    "music": "Media & Entertainment",
    "publishing": "Media & Entertainment",
    "online media": "Media & Entertainment",
    "newspapers": "Media & Entertainment",
    "motion pictures and film": "Media & Entertainment",
    "performing arts": "Media & Entertainment",
    "writing and editing": "Media & Entertainment",
    # Sports, Recreation & Travel
    "sports": "Sports, Recreation & Travel",
    "recreational facilities and services": "Sports, Recreation & Travel",
    "leisure, travel & tourism": "Sports, Recreation & Travel",
    "hospitality": "Sports, Recreation & Travel",
    "gambling & casinos": "Sports, Recreation & Travel",
    "events services": "Sports, Recreation & Travel",
    # Telecommunications
    "telecommunications": "Telecommunications",
    # Research & Science
    "research": "Research & Science",
    "nanotechnology": "Research & Science",
    "biotechnology": "Research & Science",
    "medical devices": "Research & Science",
    # Professional Services
    "consumer services": "Professional Services",
    "security and investigations": "Professional Services",
    "translation and localization": "Professional Services",
    "executive office": "Professional Services",
    "non-profit organization management": "Non-profit & Social",  # keep alias safe
}


def normalize_industries(df: pd.DataFrame, industry_col: str = "industry") -> pd.DataFrame:
    """
    Applies two transformations to *df* in-place (on a copy):
      1. Renames exact-duplicate industry labels to their canonical form.
      2. Adds a 'sector' column derived from the normalized industry name.

    Safe to call on both the full 7M-row dataset and the summary counts frame.
    """
    df = df.copy()
    df[industry_col] = df[industry_col].replace(INDUSTRY_RENAMES)
    df["sector"] = df[industry_col].map(INDUSTRY_TO_SECTOR).fillna("Other")
    return df


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def download_dataset() -> str:
    print("Downloading dataset...")
    path = kagglehub.dataset_download("peopledatalabssf/free-7-million-company-dataset")
    print(f"Dataset path: {path}")
    return path


def analyze_industries(dataset_path: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    csv_path = csv_files[0]
    print(f"Reading: {csv_path}")

    # Only load the two columns we need — keeps memory usage low for 7M rows
    df = pd.read_csv(csv_path, usecols=["domain", "industry"], low_memory=False)
    print(f"Total rows loaded: {len(df):,}")

    # Drop rows with no domain entry
    df = df[df["domain"].notna() & (df["domain"].str.strip() != "")]
    print(f"Rows with domain: {len(df):,}")

    # Drop rows with no industry entry
    df = df[df["industry"].notna() & (df["industry"].str.strip() != "")]
    print(f"Rows with industry: {len(df):,}")

    # Normalize industry labels and add sector column
    df = normalize_industries(df)

    # Count entries per industry
    industry_counts = (
        df.groupby(["industry", "sector"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    print(f"Unique industry values (after normalization): {len(industry_counts):,}")

    return industry_counts


def main():
    dataset_path = download_dataset()
    industry_counts = analyze_industries(dataset_path)

    output_path = os.path.join(os.path.dirname(__file__), "industry_counts.csv")
    industry_counts.to_csv(output_path, index=False)
    print(f"Exported to: {output_path}")


if __name__ == "__main__":
    main()
