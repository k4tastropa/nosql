from __future__ import annotations

import json
from pathlib import Path

from nosql.spider.paths import get_spider_paths


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    paths = get_spider_paths()

    train = load_json(paths.train_json)
    dev = load_json(paths.dev_json)
    tables = load_json(paths.tables_json)

    sqlite_files = list(paths.database_dir.glob("*/*.sqlite"))

    print(f"Spider root: {paths.raw_root}")
    print(f"Train examples: {len(train)}")
    print(f"Dev examples: {len(dev)}")
    print(f"Tables/schemas: {len(tables)}")
    print(f"SQLite databases: {len(sqlite_files)}")

    sample = train[0]

    print("\nSample:")
    print(f"db_id: {sample['db_id']}")
    print(f"question: {sample['question']}")
    print(f"sql: {sample['query']}")

    db_path = paths.database_dir / sample["db_id"] / f"{sample['db_id']}.sqlite"
    print(f"sqlite: {db_path}")
    print(f"sqlite_exists: {db_path.exists()}")


if __name__ == "__main__":
    main()
