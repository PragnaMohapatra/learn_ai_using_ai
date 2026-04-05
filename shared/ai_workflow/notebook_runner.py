"""Shared notebook execution utilities reused across episodes."""

from datetime import datetime
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient


def run_notebook(
    notebook_path: str,
    output_path: str | None = None,
    timeout: int = 300,
    kernel_name: str = "python3",
) -> dict:
    """Execute a Jupyter notebook and save the result."""
    nb_path = Path(notebook_path)

    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    if output_path is None:
        output_path = str(nb_path.parent / f"{nb_path.stem}_executed{nb_path.suffix}")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )

    start_time = time.time()
    errors: list[str] = []

    try:
        client.execute()
    except Exception as exc:  # pragma: no cover - report the notebook error instead
        errors.append(str(exc))

    elapsed = round(time.time() - start_time, 2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    total_cells = len(nb.cells)
    code_cells = sum(1 for cell in nb.cells if cell.cell_type == "code")

    return {
        "timestamp": datetime.now().isoformat(),
        "notebook": notebook_path,
        "output": output_path,
        "total_cells": total_cells,
        "code_cells": code_cells,
        "execution_time_seconds": elapsed,
        "errors": errors,
        "success": len(errors) == 0,
    }
