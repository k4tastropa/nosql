import typer
from rich import print

app = typer.Typer(help="NoSQL: tiny text-to-SQL distillation lab.")


@app.callback()
def root() -> None:
    """NoSQL command line interface."""


@app.command()
def doctor() -> None:
    print("[green]NoSQL project is ready.[/green]")


def main() -> None:
    app()