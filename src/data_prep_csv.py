from pathlib import Path
import pandas as pd

INPUT_CSV = Path("data/raw/archive/enron_spam_data.csv")
OUTPUT_CSV = Path("data/processed/enron.csv")


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}. Make sure it is in data/raw/archive/"
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # Fill missing values BEFORE any string operations
    df["Subject"] = df["Subject"].fillna("").astype(str)
    df["Message"] = df["Message"].fillna("").astype(str)
    df["Spam/Ham"] = df["Spam/Ham"].fillna("").astype(str)

    # Build text: Subject + Message
    df["text"] = (df["Subject"].str.strip() + "\n" + df["Message"].str.strip()).str.strip()

    # Labels
    df["label"] = df["Spam/Ham"].str.lower().str.strip()

    # Keep only spam/ham labels
    out = df[df["label"].isin(["spam", "ham"])][["text", "label"]].copy()

    # Drop only rows where text is truly empty
    out = out[out["text"].str.len() > 0]

    # Remove duplicates
    # out = out.drop_duplicates(subset=["text"])

    print("Label distribution:")
    print(out["label"].value_counts())

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned dataset to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

