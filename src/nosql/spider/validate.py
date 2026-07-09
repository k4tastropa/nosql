from __future__ import annotations

import json
from pathlib import Path

from nosql.spider.paths import get_spider_paths


REQUIRED_KEYS = (
    "id",
    "split",
    "db_id",
    "question",
    "schema",
    "input",
    "target",
    "sql",
    "db_path",
)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            yield line_number, json.loads(line)


def validate_file(path: Path) -> int:
    errors = 0
    rows = 0

    for line_number, row in iter_jsonl(path):
        rows += 1

        for key in REQUIRED_KEYS:
            if key not in row:
                print(f"{path.name}:{line_number} missing key: {key}")
                errors += 1

        if not row.get("input", "").strip():
            print(f"{path.name}:{line_number} empty input")
            errors += 1

        if not row.get("target", "").strip():
            print(f"{path.name}:{line_number} empty target")
            errors += 1

        db_path = Path(row.get("db_path", ""))
        if not db_path.exists():
            print(f"{path.name}:{line_number} missing db file: {db_path}")
            errors += 1

    print(f"{path.name}: checked {rows} rows, errors={errors}")
    return errors


def main() -> None:
    paths = get_spider_paths()
    total_errors = 0
    total_errors += validate_file(paths.train_jsonl)
    total_errors += validate_file(paths.dev_jsonl)

    if total_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
