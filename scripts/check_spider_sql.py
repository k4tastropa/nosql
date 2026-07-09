from __future__ import annotations

import json
import sqlite3
from pathlib import Path


PROCESSED_DIR = Path("/mnt/nosql/data/processed/spider")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def check_file(path: Path) -> int:
    failures = 0
    checked = 0

    for line_number, row in iter_jsonl(path):
        checked += 1

        db_path = Path(row["db_path"])
        sql = row["sql"]

        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute(sql).fetchall()
        except Exception as exc:
            failures += 1
            #print(f"{path.name}:{line_number} {row['id']} failed")
            #print(f"db: {db_path}")
            #print(f"sql: {sql}")
            #print(f"error: {exc}")
            #print()

    # print(f"{path.name}: checked={checked}, failures={failures}")
    return failures


def main() -> None:
    failures = 0
    failures += check_file(PROCESSED_DIR / "train.jsonl")
    failures += check_file(PROCESSED_DIR / "dev.jsonl")

    if failures:
        print(f"SQL check completed with {failures} failures")

if __name__ == "__main__":
    main()