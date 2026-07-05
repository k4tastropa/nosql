import typer
from rich import print
from pathlib import Path

from nosql.config import load_config

app = typer.Typer(help="NoSQL: tiny text-to-SQL distillation lab.")


@app.callback()
def root() -> None:
    """NoSQL command line interface."""


@app.command()
def doctor() -> None:
    print("[green]NoSQL project is ready.[/green]")

@app.command("config")
def config_show(path: Path = Path("configs/default.yml")) -> None:
    settings = load_config(path)
    print(settings.model_dump_json(indent=2))


def main() -> None:
    app()