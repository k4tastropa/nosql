# nosql

Experimental project to distill an LLM from Qwen or Devstral into a tiny text-to-SQL specialist, then quantize it aggressively, eventually aiming to fit under 25MB while beating the teacher model at that specific task.

Docs live in [`docs/`](docs/).

## Runtime setup

I cannot realistically run this project end-to-end on my local machine, so the workflow is built around AWS from the start.

The setup is split between a cheap CPU instance and a GPU instance:

```txt
t3.large
  CPU development box
  used for repo work, data preparation, validation, Docker testing, and pipeline development

g6.xlarge
  GPU runner
  used for tokenizer/model work, teacher generation, fine-tuning experiments, and later quantization
````

Both instances have their own root volumes. The important project state lives on a separate EBS volume mounted at:

```txt
/mnt/nosql
```

That shared volume holds datasets, processed JSONL files, model caches, checkpoints, generated teacher outputs, and experiment artifacts. Docker is used so the same project commands can run on both the CPU dev box and the GPU box without depending on whatever Python environment happens to be installed on the instance.


## Current pipeline

Spider is used as the initial text-to-SQL dataset.

```bash
uv run nosql spider prepare
```

This builds processed JSONL files from the raw Spider dataset, validates the rows, checks gold SQL execution against SQLite databases, and writes a SQL failure report.

By default, the pipeline reads and writes under `/mnt/nosql/data`. Set `NOSQL_DATA_DIR` to use another data root.

Expected data layout:

```bash
$NOSQL_DATA_DIR/raw/spider/spider_data
$NOSQL_DATA_DIR/processed/spider
```

Build the minimal supervised fine-tuning dataset:

```bash
uv run nosql spider build-sft
```

This writes SFT-ready JSONL files under:

```bash
$NOSQL_DATA_DIR/processed/spider_sft
```

## Docker

Build:

```bash
docker build -t nosql:dev .
```

Run Spider prepare:

```bash
docker run --rm -it \
  -v /mnt/nosql/data:/mnt/nosql/data \
  -e NOSQL_DATA_DIR=/mnt/nosql/data \
  nosql:dev \
  uv run nosql spider prepare
```

Run SFT build:

```bash
docker run --rm -it \
  -v /mnt/nosql/data:/mnt/nosql/data \
  -e NOSQL_DATA_DIR=/mnt/nosql/data \
  nosql:dev \
  uv run nosql spider build-sft
```
