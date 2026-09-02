"""Entry point for the 2025 journal figure workspace."""

from __future__ import annotations

from .style import apply_journal_style


def main() -> None:
    figure_dir = apply_journal_style()
    print(f"Journal figure directory: {figure_dir}")


if __name__ == "__main__":
    main()
