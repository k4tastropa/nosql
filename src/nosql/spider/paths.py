from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_DIR = Path("/mnt/nosql/data")


@dataclass(frozen=True)
class SpiderPaths:
    data_dir: Path
    raw_root: Path
    database_dir: Path
    processed_dir: Path
    train_json: Path
    dev_json: Path
    tables_json: Path
    train_jsonl: Path
    dev_jsonl: Path
    sql_failures: Path


def get_spider_paths() -> SpiderPaths:
    data_dir = Path(os.getenv("NOSQL_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
    raw_root = data_dir / "raw" / "spider" / "spider_data"
    processed_dir = data_dir / "processed" / "spider"

    return SpiderPaths(
        data_dir=data_dir,
        raw_root=raw_root,
        database_dir=raw_root / "database",
        processed_dir=processed_dir,
        train_json=raw_root / "train_spider.json",
        dev_json=raw_root / "dev.json",
        tables_json=raw_root / "tables.json",
        train_jsonl=processed_dir / "train.jsonl",
        dev_jsonl=processed_dir / "dev.jsonl",
        sql_failures=processed_dir / "sql_failures.jsonl",
    )
