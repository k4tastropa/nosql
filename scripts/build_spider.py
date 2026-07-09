from __future__ import annotations

import json 
import os 
from pathlib import Path

DEFAULT_SPIDER_ROOT = "/mnt/nosql/data/raw/spider/spider_data"
DEFAULT_OUTPUT_DIR = "/mnt/nosql/data/processed/spider"

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def build_schema_map(tables: list[dict]) -> dict[str, str]:
    schemas = {}

    for db in tables:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        columns = db["column_names_original"]

        table_columns: dict[int, list[str]] = {
            index: [] for index in range(len(table_names))
        }

        for table_index, column_name in columns:
            if table_index == -1:
                continue

            table_columns[table_index].append(column_name)
        
        lines = []

        for table_index, table_name in enumerate(table_names):
            cols = ", ".join(table_columns[table_index])
            lines.append(f"table {table_name}({cols})")

        schemas[db_id] = "\n".join(lines)
    
    return schemas

def build_rows(split: str, examples: list[dict], database_dir: Path, schema_map: dict[str, str]) -> list[dict]:
    rows = []

    for index, example in enumerate(examples):
        db_id = example["db_id"]
        db_path = database_dir / db_id / f"{db_id}.sqlite"

        rows.append(
            {
                "id": f"spider-{split}-{index:06d}",
                "split": split,
                "db_id": db_id,
                "question": example["question"],
                "schema": schema_map[db_id],
                "sql": example["query"],
                "db_path": str(db_path)
            }
        )

    return rows

def main() -> None:
    spider_root = Path(os.getenv("SPIDER_ROOT", DEFAULT_SPIDER_ROOT))
    output_dir = Path(os.getenv("SPIDER_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

    database_dir = spider_root / "database"

    train = load_json(spider_root / "train_spider.json")
    dev = load_json(spider_root / "dev.json")
    tables = load_json(spider_root / "tables.json")

    schema_map = build_schema_map(tables)

    train_rows = build_rows("train", train, database_dir, schema_map)
    dev_rows = build_rows("dev", dev, database_dir, schema_map)

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "dev.jsonl", dev_rows)

    print(f"Wrote {len(train_rows)} train rows")
    print(f"Wrote {len(dev_rows)} dev rows")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()