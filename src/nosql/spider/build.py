from __future__ import annotations

import json
from pathlib import Path

from nosql.spider.paths import get_spider_paths


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


def build_rows(
    split: str,
    examples: list[dict],
    database_dir: Path,
    schema_map: dict[str, str],
) -> list[dict]:
    rows = []

    for index, example in enumerate(examples):
        db_id = example["db_id"]
        db_path = database_dir / db_id / f"{db_id}.sqlite"

        input_text = (
            f"### Schema\n"
            f"{schema_map[db_id]}\n\n"
            f"### Question\n"
            f"{example['question']}\n\n"
            f"### SQL\n"
        )

        rows.append(
            {
                "id": f"spider-{split}-{index:06d}",
                "split": split,
                "db_id": db_id,
                "question": example["question"],
                "schema": schema_map[db_id],
                "input": input_text,
                "target": example["query"],
                "sql": example["query"],
                "db_path": str(db_path),
            }
        )

    return rows


def main() -> None:
    paths = get_spider_paths()

    train = load_json(paths.train_json)
    dev = load_json(paths.dev_json)
    tables = load_json(paths.tables_json)

    schema_map = build_schema_map(tables)

    train_rows = build_rows("train", train, paths.database_dir, schema_map)
    dev_rows = build_rows("dev", dev, paths.database_dir, schema_map)

    write_jsonl(paths.train_jsonl, train_rows)
    write_jsonl(paths.dev_jsonl, dev_rows)

    print(f"Wrote {len(train_rows)} train rows")
    print(f"Wrote {len(dev_rows)} dev rows")
    print(f"Output: {paths.processed_dir}")


if __name__ == "__main__":
    main()
