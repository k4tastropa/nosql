from __future__ import annotations

import typer

from nosql.spider import build, check_sql, inspect, validate, sft

app = typer.Typer(no_args_is_help=True)
spider_app = typer.Typer(no_args_is_help=True)

app.add_typer(spider_app, name="spider")


@spider_app.command("inspect")
def spider_inspect() -> None:
    inspect.main()


@spider_app.command("build")
def spider_build() -> None:
    build.main()


@spider_app.command("validate")
def spider_validate() -> None:
    validate.main()


@spider_app.command("check-sql")
def spider_check_sql() -> None:
    check_sql.main()

@spider_app.command("build-sft")
def spider_build_sft() -> None:
    sft.main()

@spider_app.command("prepare")
def spider_prepare() -> None:
    build.main()
    validate.main()
    check_sql.main()


def main() -> None:
    app()
