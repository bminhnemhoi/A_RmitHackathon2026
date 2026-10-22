"""Validate a Challenge 1 submission CSV (id,label) against sample_submission.csv.

Exit code 0 = valid, 1 = invalid. Checks only the format, not prediction quality.

Usage:
    python scripts/validate_submission.py --sample data/sample_submission.csv --submission outputs/submission.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

LABELS = ("BENIGN", "JAILBREAK", "PROMPT_INJECTION")
EXPECTED_HEADER = ["id", "label"]


def read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows[0], rows[1:]


def validate(sample: Path, submission: Path, expected_rows: int | None = 6000) -> list[str]:
    """Return a list of error messages; an empty list means the submission is valid."""
    errors: list[str] = []
    try:
        s_header, s_rows = read_rows(sample)
        header, rows = read_rows(submission)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return [str(exc)]

    if [h.strip() for h in header] != EXPECTED_HEADER:
        errors.append(f"header must be {EXPECTED_HEADER}, got {header}")
    if [h.strip() for h in s_header] != EXPECTED_HEADER:
        errors.append(f"sample header unexpected: {s_header}")

    if expected_rows is not None and len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} data rows, got {len(rows)}")
    if len(rows) != len(s_rows):
        errors.append(f"row count {len(rows)} != sample row count {len(s_rows)}")

    bad_width = [i + 2 for i, r in enumerate(rows) if len(r) != 2]
    if bad_width:
        errors.append(f"{len(bad_width)} rows do not have exactly 2 fields (first at line {bad_width[0]})")
        return errors

    ids = [r[0].strip() for r in rows]
    sample_ids = [r[0].strip() for r in s_rows]
    if any(not i for i in ids):
        errors.append("empty id found")
    dup = [i for i, c in Counter(ids).items() if c > 1]
    if dup:
        errors.append(f"{len(dup)} duplicated ids, e.g. {dup[:3]}")
    missing = set(sample_ids) - set(ids)
    extra = set(ids) - set(sample_ids)
    if missing:
        errors.append(f"{len(missing)} sample ids missing, e.g. {sorted(missing)[:3]}")
    if extra:
        errors.append(f"{len(extra)} unknown ids, e.g. {sorted(extra)[:3]}")
    if not missing and not extra and ids != sample_ids:
        errors.append("ids are not in the same order as sample_submission.csv")

    bad_labels = Counter(r[1] for r in rows if r[1] not in LABELS)
    if bad_labels:
        errors.append(f"invalid labels (must be exactly one of {LABELS}): {dict(bad_labels.most_common(5))}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=6000, help="use -1 to skip the fixed row count check")
    args = parser.parse_args(argv)

    expected = None if args.expected_rows < 0 else args.expected_rows
    errors = validate(args.sample, args.submission, expected)
    if errors:
        print("INVALID submission:")
        for e in errors:
            print(f"  - {e}")
        return 1

    _, rows = read_rows(args.submission)
    dist = Counter(r[1] for r in rows)
    print(f"VALID submission: {len(rows)} rows")
    for label in LABELS:
        print(f"  {label:<17} {dist.get(label, 0):>5}  ({dist.get(label, 0) / len(rows):.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
