"""Answer the test set from train.csv alone, with no model at all."""

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANSWER_COLS = ["A", "B", "C", "D", "E"]


def clean(text: str) -> str:
    text = str(text).lower()  # type: ignore
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_core_question(prompt: str) -> str:
    """Extract core question by removing template prefixes and suffixes."""
    if ":" in prompt:
        q = prompt.split(":", 1)[1].strip()
    else:
        q = prompt.strip()

    if "?" in q:
        q = q.rsplit("?", 1)[0].strip() + "?"
    return q


def core_key(prompt: str) -> str:
    """Retrieval key: the core question, normalised."""
    return clean(extract_core_question(prompt))


def option_sequence(row: pd.Series) -> str:
    return "|||".join(clean(row[col]) for col in ANSWER_COLS)


def build_sequence_lookup(train: pd.DataFrame) -> dict[str, str]:
    grouped = train.groupby("_sequence")["answer"].agg(lambda s: s.mode().iat[0])
    return grouped.to_dict()


def build_core_lookup(train: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for _, row in train.iterrows():
        letter = str(row["answer"]).strip().upper()
        lookup.setdefault(row["_core"], clean(row[letter]))
    return lookup


def letter_prior(train: pd.DataFrame) -> list[str]:
    counts = Counter(train["answer"].astype(str))
    return [letter for letter, _ in counts.most_common()]


def solve_row(
    row: pd.Series,
    sequence_lookup: dict[str, str],
    core_lookup: dict[str, str],
    prior: list[str],
) -> tuple[str, str]:
    """Return (predicted letter, which tier produced it)."""
    letter = sequence_lookup.get(row["_sequence"])
    if letter is not None:
        return letter, "exact option sequence"

    answer_text = core_lookup.get(row["_core"])
    if answer_text is not None:
        for col in ANSWER_COLS:
            if clean(row[col]) == answer_text:
                return col, "core question + answer text"

    return prior[0], "fallback (most frequent letter)"


def rank_top3(letter: str, prior: list[str]) -> str:
    """Put the predicted letter first, then fill from the global prior."""
    ranked = [letter] + [other for other in prior if other != letter]
    return " ".join(ranked[:3])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_core"] = df["prompt"].apply(core_key)
    df["_sequence"] = df.apply(option_sequence, axis=1)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "dataset/train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "dataset/test.csv")
    default_out = ROOT / "outputs/lookup_submission.csv"
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()

    for path in (args.train, args.test):
        if not path.exists():
            raise FileNotFoundError(f"required dataset not found: {path}")

    train = prepare(pd.read_csv(args.train))
    test = prepare(pd.read_csv(args.test))

    sequence_lookup = build_sequence_lookup(train)
    core_lookup = build_core_lookup(train)
    prior = letter_prior(train)

    predictions: list[str] = []
    tiers: Counter = Counter()
    for _, row in test.iterrows():
        letter, tier = solve_row(row, sequence_lookup, core_lookup, prior)
        predictions.append(rank_top3(letter, prior))
        tiers[tier] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"id": test["id"], "Prediction": predictions})
    submission.to_csv(args.out, index=False)

    print(f"train rows {len(train)}")
    print(f"  distinct option sequences : {len(sequence_lookup)}")
    print(f"  distinct core questions   : {len(core_lookup)}\n")
    print("how each test row was answered:")
    for tier, count in tiers.most_common():
        print(f"  {count:3d}  ({count / len(test):5.1%})  {tier}")
    print(f"\nwrote {args.out}")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
