# nosql

Experimental project to distill an LLM from Qwen or Devstral into a tiny text-to-SQL specialist, then quantize it aggressively, eventually aiming to fit under 25MB while beating the teacher model at that specific task.

Docs live in [`docs/`](docs/).

## Current pipeline

Spider is used as the initial text-to-SQL dataset.

```bash
uv run nosql spider prepare
```

This builds processed JSONL files from the raw Spider dataset, validates the rows, checks gold SQL execution against SQLite databases, and writes a SQL failure report.

By default, the pipeline reads and writes under `/mnt/nosql/data`. Set
`NOSQL_DATA_DIR` to use another data root.

Expected data layout:

```bash
$NOSQL_DATA_DIR/raw/spider/spider_data
$NOSQL_DATA_DIR/processed/spider
```
