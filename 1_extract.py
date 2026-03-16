"""
Step 1: Extract data from Apple Numbers file → enriched_raw.csv
Reads: Const (IMDb ID), Your Rating, Title
"""
import numbers_parser
import pandas as pd
from config import NUMBERS_FILE, RAW_CSV


def extract():
    print(f"Reading: {NUMBERS_FILE}")
    doc = numbers_parser.Document(NUMBERS_FILE)
    sheet = doc.sheets[0]
    table = sheet.tables[0]

    headers = [table.cell(0, c).value for c in range(table.num_cols)]
    print(f"Columns found: {headers}")

    rows = []
    for r in range(1, table.num_rows):
        row = {headers[c]: table.cell(r, c).value for c in range(table.num_cols)}
        rows.append(row)

    df = pd.DataFrame(rows)

    # Keep only the columns we need
    df = df[["Const", "Your Rating", "Title"]].copy()

    # Drop rows with no rating or no IMDb ID
    df = df.dropna(subset=["Const", "Your Rating"])
    df["Your Rating"] = df["Your Rating"].astype(float)

    # Deduplicate on IMDb ID (keep first occurrence)
    df = df.drop_duplicates(subset=["Const"])

    df.to_csv(RAW_CSV, index=False)
    print(f"Saved {len(df)} rows → {RAW_CSV}")
    print(df.head())
    return df


if __name__ == "__main__":
    extract()
