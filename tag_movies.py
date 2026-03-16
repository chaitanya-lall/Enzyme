"""
Batch LLM tagging pipeline — run ONCE to generate data/movie_tags.csv.

Loads both Chai's and Noel's enriched CSVs, deduplicates by Const (IMDb ID),
then calls Gemini Flash once per unique movie to assign 5 categorical tags.

Resumable: skips movies that already have at least one valid tag (non-zero one-hot row).
Falls back to Groq if Gemini fails.
"""
from __future__ import annotations

import time
import pandas as pd

from config import OMDB_CSV, NOEL_OMDB_CSV, TAG_CSV
from tag_features import call_gemini_tagger, encode_tags, ALL_TAG_COLS

SLEEP_BETWEEN = 6.5  # 6.5s between calls = ~9 RPM, under the 10 RPM limit


def load_unique_movies() -> pd.DataFrame:
    """Combine both enriched CSVs, return unique movies by Const."""
    chai_df = pd.read_csv(OMDB_CSV)
    noel_df = pd.read_csv(NOEL_OMDB_CSV)
    all_df  = pd.concat([chai_df, noel_df], ignore_index=True)
    all_df  = all_df.dropna(subset=["Const"])
    all_df  = all_df[all_df["Const"].str.strip() != ""]
    all_df  = all_df.drop_duplicates(subset=["Const"])
    return all_df.reset_index(drop=True)


def main():
    movies = load_unique_movies()
    print(f"Unique movies across both datasets: {len(movies)}")

    # Resume: only skip movies that SUCCESSFULLY got tags (at least 1 non-zero one-hot)
    try:
        existing = pd.read_csv(TAG_CSV)
        # Ensure all tag columns are present (fill missing with 0)
        for col in ALL_TAG_COLS:
            if col not in existing.columns:
                existing[col] = 0
        # A row is "done" only if it has at least one valid tag
        success_mask = existing[ALL_TAG_COLS].sum(axis=1) > 0
        done_ids = set(existing.loc[success_mask, "Const"].values)
        failed_ids = set(existing.loc[~success_mask, "Const"].values)
        print(f"Already successfully tagged: {len(done_ids)}")
        print(f"Previously failed (will retry): {len(failed_ids)}")
        # Drop failed rows from existing so they get re-added cleanly
        existing = existing[success_mask].reset_index(drop=True)
    except FileNotFoundError:
        existing = pd.DataFrame()
        done_ids = set()

    pending = movies[~movies["Const"].isin(done_ids)].reset_index(drop=True)
    total   = len(pending)
    if total == 0:
        print("All movies already tagged!")
        return

    est_minutes = round(total * SLEEP_BETWEEN / 60, 1)
    print(f"Tagging {total} movies — estimated {est_minutes} min at {SLEEP_BETWEEN}s/call\n")

    new_rows: list[dict] = []
    fail_count = 0

    for i, (_, row) in enumerate(pending.iterrows()):
        rec   = row.to_dict()
        const = rec["Const"]
        title = rec.get("Title", "?")

        tags_dict = call_gemini_tagger(rec)
        encoded   = encode_tags(tags_dict)

        if not tags_dict:
            fail_count += 1

        out_row = {
            "Const": const,
            "Title": title,
            **tags_dict,
            **encoded,
        }
        new_rows.append(out_row)

        tag_summary = ", ".join(f"{k}={v}" for k, v in tags_dict.items()) or "FAILED"
        print(f"[{i+1}/{total}] {title} → {tag_summary}")

        # Save incrementally every 20 movies and at the very end
        if (i + 1) % 20 == 0 or i == total - 1:
            batch_df = pd.DataFrame(new_rows)
            combined = (
                pd.concat([existing, batch_df], ignore_index=True)
                if not existing.empty else batch_df
            )
            # Ensure all tag cols present
            for col in ALL_TAG_COLS:
                if col not in combined.columns:
                    combined[col] = 0
            combined.to_csv(TAG_CSV, index=False)
            existing  = combined
            new_rows  = []
            saved_ok  = int((combined[ALL_TAG_COLS].sum(axis=1) > 0).sum())
            pct_done  = round((len(done_ids) + i + 1) / len(movies) * 100)
            print(f"  → Saved {len(combined)} rows ({saved_ok} with valid tags, {pct_done}% complete)")

        if i < total - 1:
            time.sleep(SLEEP_BETWEEN)

    print(f"\nTagging complete! Results saved to {TAG_CSV}")
    print(f"Success: {len(existing) - fail_count} | Failed: {fail_count}")


if __name__ == "__main__":
    main()
