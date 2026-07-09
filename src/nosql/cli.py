from __future__ import annotations

from pathlib import Path
import sys

import typer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_spider, check_spider_sql, inspect_spider, validate_spider

app = typer.Typer(no_args_is_help=True)
spider_app = typer.Typer(no_args_is_help=True)

app.add_typer(spider_app, name="spider")


@spider_app.command("inspect")
def spider_inspect() -> None:
    inspect_spider.main()


@spider_app.command("build")
def spider_build() -> None:
    build_spider.main()


@spider_app.command("validate")
def spider_validate() -> None:
    validate_spider.main()


@spider_app.command("check-sql")
def spider_check_sql() -> None:
    check_spider_sql.main()


@spider_app.command("prepare")
def spider_prepare() -> None:
    build_spider.main()
    validate_spider.main()
    check_spider_sql.main()


def main() -> None:
    app()