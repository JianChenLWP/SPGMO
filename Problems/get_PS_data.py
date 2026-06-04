import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# Settings
# ============================================================

OUTPUT_DIR = Path("ff_benchmark_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USE_PRICE_RELATIVE = True
# True:  use x_t = 1 + r_t, matching "monthly price relative sequences"
# False: use monthly returns r_t directly

APPLY_PAPER_START_DATES = True
# True:  use the start dates shown in the screenshot
# False: use the full available sample from Kenneth French's files


BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

DATASETS = {
    "FF25": {
        "region": "US",
        "assets": 25,
        "start": "1971-07-01",
        "url": f"{BASE_URL}/25_Portfolios_5x5_CSV.zip",
        "description": "25 portfolios formed on ME and BE/ME",
    },
    "FF25EU": {
        "region": "EU",
        "assets": 25,
        "start": "1990-11-01",
        "url": f"{BASE_URL}/Europe_25_Portfolios_ME_Prior_12_2_CSV.zip",
        "description": "25 European portfolios formed on ME and prior return",
    },
    "FF32": {
        "region": "US",
        "assets": 32,
        "start": "1971-07-01",
        "url": f"{BASE_URL}/32_Portfolios_ME_BEME_INV_2x4x4_CSV.zip",
        "description": "32 portfolios formed on ME, BE/ME, and investment",
    },
    "FF48": {
        "region": "US",
        "assets": 48,
        "start": "1971-07-01",
        "url": f"{BASE_URL}/48_Industry_Portfolios_CSV.zip",
        "description": "48 industry portfolios",
    },
    "FF100": {
        "region": "US",
        "assets": 100,
        "start": "1971-07-01",
        "url": f"{BASE_URL}/100_Portfolios_10x10_CSV.zip",
        "description": "100 portfolios formed on ME and BE/ME",
    },
    "FF100MEOP": {
        "region": "US",
        "assets": 100,
        "start": "1971-07-01",
        "url": f"{BASE_URL}/100_Portfolios_ME_OP_10x10_CSV.zip",
        "description": "100 portfolios formed on ME and operating profitability",
    },
}


# ============================================================
# Helper functions
# ============================================================

def download_zip_text(url: str) -> str:
    """Download a zipped CSV file from Kenneth French's website and return the CSV text."""
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_files = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError(f"No CSV file found in zip: {url}")
        csv_name = csv_files[0]
        raw = zf.read(csv_name)

    return raw.decode("utf-8", errors="ignore")


def parse_first_monthly_return_table(text: str) -> pd.DataFrame:
    """
    Parse the first monthly return table in a Kenneth French CSV file.

    The first table is usually the value-weighted return table.
    Dates are YYYYMM. Returns are percentages in the raw file.
    Missing values such as -99.99 and -999 are set to NaN.
    """
    lines = text.splitlines()

    monthly_pattern = re.compile(r"^\s*\d{6}\s*,")
    first_data_idx = None

    for i, line in enumerate(lines):
        if monthly_pattern.match(line):
            first_data_idx = i
            break

    if first_data_idx is None:
        raise ValueError("Could not find monthly data rows with YYYYMM format.")

    # The header line is the nearest non-empty line before the first data row.
    header_idx = first_data_idx - 1
    while header_idx >= 0 and lines[header_idx].strip() == "":
        header_idx -= 1

    table_lines = [lines[header_idx]]

    i = first_data_idx
    while i < len(lines) and monthly_pattern.match(lines[i]):
        table_lines.append(lines[i])
        i += 1

    df = pd.read_csv(io.StringIO("\n".join(table_lines)))
    df = df.rename(columns={df.columns[0]: "date"})

    df["date"] = df["date"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("date")

    # Convert all columns to numeric.
    df = df.apply(pd.to_numeric, errors="coerce")

    # Kenneth French missing-value codes.
    df = df.mask(df <= -99)

    # Drop rows with missing portfolio returns.
    df = df.dropna(axis=0, how="any")

    # Convert percentage returns to decimal returns.
    df = df / 100.0

    # Clean column names.
    df.columns = [str(c).strip() for c in df.columns]

    return df


def load_dataset(name: str, meta: dict) -> pd.DataFrame:
    """Download and clean one dataset."""
    text = download_zip_text(meta["url"])
    returns = parse_first_monthly_return_table(text)

    if APPLY_PAPER_START_DATES:
        returns = returns.loc[pd.to_datetime(meta["start"]):]

    if USE_PRICE_RELATIVE:
        data = 1.0 + returns
    else:
        data = returns

    return data


def compute_and_save_statistics(name: str, data: pd.DataFrame, meta: dict):
    """
    Compute and save mean vector, variance vector, and covariance matrix.

    Outputs:
    - monthly data matrix
    - mean vector
    - variance vector
    - covariance matrix
    - pickle file for easy Python loading
    """
    dataset_dir = OUTPUT_DIR / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    mean_vec = data.mean(axis=0)
    var_vec = data.var(axis=0, ddof=1)
    cov_mat = data.cov(ddof=1)

    # Save CSV files.
    data.to_csv(dataset_dir / f"{name}_monthly_data.csv")
    mean_vec.to_csv(dataset_dir / f"{name}_mean.csv", header=["mean"])
    var_vec.to_csv(dataset_dir / f"{name}_variance.csv", header=["variance"])
    cov_mat.to_csv(dataset_dir / f"{name}_covariance.csv")

    # Save a pickle object for easy calling.
    stats = {
        "name": name,
        "region": meta["region"],
        "description": meta["description"],
        "data_type": "price_relative" if USE_PRICE_RELATIVE else "return",
        "start_date": data.index.min(),
        "end_date": data.index.max(),
        "months": data.shape[0],
        "assets": data.shape[1],
        "data": data,
        "mean": mean_vec,
        "variance": var_vec,
        "covariance": cov_mat,
        "source_url": meta["url"],
    }

    pd.to_pickle(stats, dataset_dir / f"{name}_stats.pkl")

    return stats


def format_month(dt) -> str:
    return pd.to_datetime(dt).strftime("%b/%Y")


# ============================================================
# Main
# ============================================================

all_stats = {}
summary_rows = []

for name, meta in DATASETS.items():
    print(f"Processing {name} ...")

    data = load_dataset(name, meta)

    if data.shape[1] != meta["assets"]:
        print(
            f"Warning: {name} expected {meta['assets']} assets, "
            f"but parsed {data.shape[1]} columns."
        )

    stats = compute_and_save_statistics(name, data, meta)
    all_stats[name] = stats

    summary_rows.append({
        "Data Set": name,
        "Region": stats["region"],
        "Description": stats["description"],
        "Time": f"{format_month(stats['start_date'])} ~ {format_month(stats['end_date'])}",
        "Months": stats["months"],
        "Assets": stats["assets"],
        "Source": stats["source_url"],
    })

summary = pd.DataFrame(summary_rows)

# Save the summary table.
summary.to_csv(OUTPUT_DIR / "dataset_information.csv", index=False)
summary.to_latex(
    OUTPUT_DIR / "dataset_information.tex",
    index=False,
    escape=False,
    caption="Information of six benchmark data sets from real-world financial markets.",
    label="tab:dataset_information",
)

# Save all statistics together.
pd.to_pickle(all_stats, OUTPUT_DIR / "all_stats.pkl")

print("\nSummary table:")
print(summary[["Data Set", "Region", "Time", "Months", "Assets"]])

print(f"\nAll files have been saved to: {OUTPUT_DIR.resolve()}")
