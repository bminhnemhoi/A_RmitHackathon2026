import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_submission import main, validate  # noqa: E402

IDS = ["t000006", "t000007", "t000008"]


def write(path: Path, rows: list[list[str]], header: str = "id,label") -> Path:
    path.write_text(header + "\n" + "".join(",".join(r) + "\n" for r in rows), encoding="utf-8")
    return path


def sample(tmp_path: Path) -> Path:
    return write(tmp_path / "sample.csv", [[i, "BENIGN"] for i in IDS])


def test_valid(tmp_path):
    sub = write(tmp_path / "sub.csv", [[IDS[0], "BENIGN"], [IDS[1], "PROMPT_INJECTION"], [IDS[2], "JAILBREAK"]])
    assert validate(sample(tmp_path), sub, expected_rows=3) == []
    assert main(["--sample", str(sample(tmp_path)), "--submission", str(sub), "--expected-rows", "3"]) == 0


def test_bom_is_accepted(tmp_path):
    sub = tmp_path / "sub.csv"
    sub.write_text("﻿id,label\n" + "".join(f"{i},BENIGN\n" for i in IDS), encoding="utf-8")
    assert validate(sample(tmp_path), sub, expected_rows=3) == []


def test_invalid_label(tmp_path):
    sub = write(tmp_path / "sub.csv", [[IDS[0], "benign"], [IDS[1], "BENIGN"], [IDS[2], "ATTACK"]])
    errors = validate(sample(tmp_path), sub, expected_rows=3)
    assert any("invalid labels" in e for e in errors)


def test_missing_row(tmp_path):
    sub = write(tmp_path / "sub.csv", [[IDS[0], "BENIGN"], [IDS[1], "BENIGN"]])
    errors = validate(sample(tmp_path), sub, expected_rows=3)
    assert any("expected 3 data rows" in e for e in errors)
    assert any("missing" in e for e in errors)


def test_wrong_order(tmp_path):
    sub = write(tmp_path / "sub.csv", [[IDS[1], "BENIGN"], [IDS[0], "BENIGN"], [IDS[2], "BENIGN"]])
    assert any("same order" in e for e in validate(sample(tmp_path), sub, expected_rows=3))


def test_duplicate_and_unknown_id(tmp_path):
    sub = write(tmp_path / "sub.csv", [[IDS[0], "BENIGN"], [IDS[0], "BENIGN"], ["t999999", "BENIGN"]])
    errors = validate(sample(tmp_path), sub, expected_rows=3)
    assert any("duplicated" in e for e in errors)
    assert any("unknown ids" in e for e in errors)


def test_wrong_header_and_extra_column(tmp_path):
    sub = write(tmp_path / "sub.csv", [[i, "BENIGN", "0.9"] for i in IDS], header="id,target,score")
    errors = validate(sample(tmp_path), sub, expected_rows=3)
    assert any("header" in e for e in errors)
    assert any("exactly 2 fields" in e for e in errors)
    assert main(["--sample", str(sample(tmp_path)), "--submission", str(sub), "--expected-rows", "3"]) == 1
