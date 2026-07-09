from __future__ import annotations

import json
from pathlib import Path

from nosql.spider.paths import get_spider_paths


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sft_split(input_path: Path, output_path: Path) -> int:
    rows = []

    for row in iter_jsonl(input_path):
        rows.append(
            {
                "id": row["id"],
                "input": row["input"],
                "target": row["target"],
            }
        )

    write_jsonl(output_path, rows)
    return len(rows)


def main() -> None:
    processed_dir = get_spider_paths().processed_dir
    output_dir = processed_dir.parent / "spider_sft"

    train_count = build_sft_split(
        processed_dir / "train.jsonl",
        output_dir / "train.jsonl",
    )
    dev_count = build_sft_split(
        processed_dir / "dev.jsonl",
        output_dir / "dev.jsonl",
    )

    print(f"Wrote {train_count} train SFT rows")
    print(f"Wrote {dev_count} dev SFT rows")
    print(f"Output: {output_dir}")
