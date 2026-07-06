from __future__ import annotations

import json
import os
from pathlib import Path 

DEFAULT_SPIDER_ROOT = "/mnt/nosql/data/raw/spider/spider_data"

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main() -> None:
    spider_root = Path(os.getenv("SPIDER_ROOT", DEFAULT_SPIDER_ROOT))

    train_path = spider_root / "train_spider.json"
    dev_path = spider_root / "dev.json"
    tables_path = spider_root / "tables.json"
    database_dir = spider_root / "database"

    train = load_json(train_path)
    dev = load_json(dev_path)
    tables = load_json(tables_path)

    sqlite_files = list(database_dir.glob("*/*.sqlite"))

    print(f"Spider root: {spider_root}")
    print(f"Train examples: {len(train)}")
    print(f"Dev examples: {len(dev)}")
    print(f"Tables/schemas: {len(tables)}")
    print(f"SQLite databases: {len(sqlite_files)}")

    sample = train[0]

    print("\nSample:")
    print(f"db_id: {sample['db_id']}")
    print(f"question: {sample['question']}")
    print(f"sql: {sample['query']}")

    db_path = database_dir / sample["db_id"] / f"{sample['db_id']}.sqlite"
    print(f"sqlite: {db_path}")
    print(f"sqlite_exists: {db_path.exists()}")

if __name__ == "__main__":
    main()