from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def hello() -> None:
    print("nosql")


def main() -> None:
    app()


if __name__ == "__main__":
    main()