from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from nosql.spider.paths import get_spider_paths


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def check_file(path: Path) -> tuple[int, int, list[dict]]:
    failures = []
    checked = 0

    for line_number, row in iter_jsonl(path):
        checked += 1

        db_path = Path(row["db_path"])
        sql = row["sql"]

        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute(sql).fetchall()
        except Exception as exc:
            failures.append(
                {
                    "file": path.name,
                    "line": line_number,
                    "id": row["id"],
                    "split": row["split"],
                    "db_id": row["db_id"],
                    "db_path": str(db_path),
                    "question": row["question"],
                    "sql": sql,
                    "error": str(exc),
                }
            )

    print(f"{path.name}: checked={checked}, failures={len(failures)}")
    return checked, len(failures), failures


def write_failures(path: Path, failures: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")


def main() -> None:
    paths = get_spider_paths()
    all_failures = []

    _, _, train_failures = check_file(paths.train_jsonl)
    _, _, dev_failures = check_file(paths.dev_jsonl)

    all_failures.extend(train_failures)
    all_failures.extend(dev_failures)

    write_failures(paths.sql_failures, all_failures)

    print(f"SQL failure report: {paths.sql_failures}")
    print(f"SQL check completed with {len(all_failures)} failures")


if __name__ == "__main__":
    main()
