"""
Notebook Runner Module — Decode AI Using AI, Episode 1
Executes Jupyter notebooks programmatically using nbclient.
Saves executed notebooks with outputs for review.
"""

import nbformat
from nbclient import NotebookClient
from pathlib import Path
from datetime import datetime
import time


def run_notebook(
    notebook_path: str,
    output_path: str | None = None,
    timeout: int = 300,
    kernel_name: str = "python3",
) -> dict:
    """
    Execute a Jupyter notebook and save the result.

    Args:
        notebook_path: Path to the .ipynb file
        output_path:   Where to save the executed notebook (default: _executed suffix)
        timeout:       Max seconds per cell
        kernel_name:   Jupyter kernel to use

    Returns:
        Report dict with execution metadata
    """
    nb_path = Path(notebook_path)

    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    # Default output path: same name with _executed suffix
    if output_path is None:
        output_path = str(nb_path.parent / f"{nb_path.stem}_executed{nb_path.suffix}")

        # Read the notebook
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Execute
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )

    start_time = time.time()
    errors = []

    try:
        client.execute()
    except Exception as e:
        errors.append(str(e))

    elapsed = round(time.time() - start_time, 2)

        # Save executed notebook
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    # Count cells
    total_cells = len(nb.cells)
    code_cells = sum(1 for c in nb.cells if c.cell_type == "code")

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





