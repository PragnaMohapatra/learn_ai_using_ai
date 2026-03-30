# Episode 1: Build From Scratch Guide

**Data Cleaner + Notebook Runner CLI — Your First Step in the AI Learning Journey**

---

## Why Episode 1? Why This Project?

Every AI project — whether it's training a model, building a RAG pipeline, or fine-tuning an LLM — starts with the same unglamorous reality: **messy data and unrepeatable workflows**.

Before you can do anything intelligent, you need to:

1. **Clean your data** — Real-world data has typos, blanks, duplicates, inconsistent formatting
2. **Automate your notebooks** — Clicking "Run All" in Jupyter doesn't scale
3. **Containerize everything** — "It works on my machine" kills collaboration
4. **Build a CLI** — Pipelines need commands, not GUIs

This episode teaches you the **invisible backbone** that every future episode depends on:

*   **Python project structure:** Every ML repo is organized this way
*   **pandas data cleaning:** You'll clean datasets before every training run
*   **Programmatic notebook execution:** Automate experiment tracking and reporting
*   **Docker containerization:** Reproducible environments for training and inference
*   **CLI design with argparse:** Pipeline orchestration starts with commands
*   **Makefile shortcuts:** One command to run your entire workflow

**If you skip this, everything else breaks.** Episode 2 (embeddings), Episode 3 (vector search), Episode 4 (RAG) — they all assume you can clean data, run notebooks, and work inside Docker.

---

## What We're Building

A Python CLI with two commands, fully containerized in Docker:

```text
ai-workflow clean   <input.csv> <output.csv> --missing drop|fill --json
ai-workflow run-nb  <notebook.ipynb> --output <path> --timeout 300 --json
```

**The `clean` command:**
- Reads a messy CSV
- Normalizes headers (`  Name  ` → `name`)
- Trims whitespace from all values
- Removes duplicate rows
- Handles missing values (drop or fill)
- Saves clean CSV + prints a JSON report

**The `run-nb` command:**
- Executes a Jupyter notebook programmatically (no browser)
- Saves the executed notebook with all cell outputs
- Reports execution time, cell counts, and errors

---

## Prerequisites

- Docker Desktop
- `make` (comes pre-installed on macOS)
- A terminal and text editor (VS Code recommended)

That's it. **Python, pip, pandas, Jupyter — everything else lives inside Docker.** You don't install anything on your Mac except Docker.

---

## Step 1 — Create the Project Structure

Every Python project needs a clean folder layout. This one follows conventions you'll see in real ML repos.

```sh
mkdir -p episode-01/{src,data/raw,data/cleaned,notebooks,logs,tests}
cd episode-01
```

**Why this structure?**

*   **`data/raw/`**: Untouched source data (Training data, downloaded datasets)
*   **`data/cleaned/`**: Processed output (Feature-engineered data ready for models)
*   **`notebooks/`**: Exploration and analysis (Experiment notebooks, EDA)
*   **`src/`**: Reusable Python modules (Model code, data pipelines, utilities)
*   **`logs/`**: Execution records (Training logs, metrics, experiment tracking)
*   **`tests/`**: Automated tests (Validate data pipelines, model outputs)

---

## Step 2 — Create the Messy Sample Data

Real data is never clean. This CSV is intentionally dirty so we have something meaningful to fix.

Create `data/raw/customers.csv`:

```csv
  Name  , Email , Age,  City  , Signup Date , Purchase Amount
Alice Johnson, alice@example.com, 29, San Francisco, 2025-01-15, 120.50
 Bob Smith , bob@example.com, , New York, 2025-02-20, 89.99
Alice Johnson, alice@example.com, 29, San Francisco, 2025-01-15, 120.50
  Charlie Brown ,charlie@example.com, 35,  Los Angeles , 2025-03-10, 
Dave Wilson, dave@example.com, 42, Chicago, 2025-04-05, 200.00
 Eve Davis , , 28, Seattle, 2025-05-12, 55.75
Frank Miller, frank@example.com, 31, Austin, 2025-06-01, 175.25
, grace@example.com, 27, Denver, 2025-07-18, 95.00
 Bob Smith , bob@example.com, , New York, 2025-02-20, 89.99
Hannah Lee, hannah@example.com, 33, Portland, 2025-08-22, 310.00
Ivan Torres, ivan@example.com, 45, Miami, 2025-09-30, 
Jane Kim, jane@example.com, 26, Boston, 2025-10-11, 67.50
```

**Every problem in this data is deliberate:**

*   **Headers have spaces:** `clean_headers()` strips and lowercases them.
*   **Values have whitespace:** `trim_strings()` strips all strings.
*   **Duplicate rows:** `drop_duplicates()` removes exact copies of Alice and Bob.
*   **Missing Data:** Missing ages, missing purchases, and missing emails are dropped or filled via `handle_missing()`.

12 rows, 8 problems. After cleaning with `--missing drop`, only 5 rows survive.

---

## Step 3 — Create the Python Package Init

```sh
touch src/__init__.py
```

This empty file tells Python that `src/` is a package so you can write `from src.cleaner import clean_csv`. Without it, Python won't recognize your imports.

**AI relevance:** Every ML library you'll use (`transformers`, `langchain`, `torch`) uses this same `__init__.py` pattern.

---

## Step 4 — Write the Data Cleaner (`src/cleaner.py`)

This is the core module. It takes messy data and makes it usable.

```python
import pandas as pd
from pathlib import Path
from datetime import datetime

def clean_headers(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    return df

def trim_strings(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df

def drop_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    return df, before - len(df)

def handle_missing(df, strategy="drop"):
    missing_count = int(df.isnull().sum().sum())
    if strategy == "drop":
        df = df.dropna()
    elif strategy == "fill":
        for col in df.columns:
            if df[col].dtype in ("float64", "int64"):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("UNKNOWN")
    return df, missing_count

def clean_csv(input_path, output_path, missing_strategy="drop"):
    df = pd.read_csv(input_path)
    original = len(df)
    
    df = clean_headers(df)
    df = trim_strings(df)
    df, dropped = drop_duplicates(df)
    df, missing = handle_missing(df, strategy=missing_strategy)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return {
        "original_rows": original,
        "cleaned_rows": len(df),
        "duplicates_dropped": dropped,
        "missing_handled": missing
    }
```

**Understanding the design:**

Each function does exactly one thing. This is intentional. In AI workflows, you'll often need to swap out a single step without rewriting everything. This exact pattern: **read → transform → save → report** is how every data preprocessing pipeline works.

---

## Step 5 — Write the Notebook Runner (`src/notebook_runner.py`)

This module lets you execute Jupyter notebooks from the command line.

```python
import nbformat
from nbclient import NotebookClient
from pathlib import Path

def run_notebook(notebook_path, output_path=None, timeout=300):
    nb_path = Path(notebook_path)
    if not output_path:
        output_path = str(nb_path.parent / f"{nb_path.stem}_executed{nb_path.suffix}")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(nb_path.parent)}})
    
    client.execute()

    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    return {"status": "success", "notebook": notebook_path, "output": output_path}
```

**Why automate notebook execution?**

In AI work, you'll have notebooks that generate training reports after each model run, produce evaluation metrics, and create visualizations. Running them by hand doesn't scale.

---

## Step 6 — Write the CLI Entry Point (`src/cli.py`)

This ties everything together. Tools like `typer` and `click` are fancier, but `argparse` ships with Python — zero dependencies.

```python
import argparse
import json
import sys
from src.cleaner import clean_csv
from src.notebook_runner import run_notebook

def main():
    parser = argparse.ArgumentParser(prog="ai-workflow")
    subparsers = parser.add_subparsers(dest="command")

    clean_p = subparsers.add_parser("clean")
    clean_p.add_argument("input")
    clean_p.add_argument("output")
    clean_p.add_argument("--missing", choices=["drop", "fill"], default="drop")
    clean_p.add_argument("--json", action="store_true")

    nb_p = subparsers.add_parser("run-nb")
    nb_p.add_argument("notebook")
    nb_p.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.command == "clean":
        res = clean_csv(args.input, args.output, args.missing)
        if args.json: print(json.dumps(res, indent=2))
        else: print(f"Cleaned {res['original_rows']} -> {res['cleaned_rows']} rows.")
    elif args.command == "run-nb":
        res = run_notebook(args.notebook, args.output)
        print(f"Executed: {res['output']}")

if __name__ == "__main__":
    main()
```

---

## Step 7 — Containerize with Docker

These are the Python packages Docker installs inside the container. Create `requirements.txt`:

```text
pandas==2.2.3
nbformat==5.10.4
nbclient==0.10.2
ipykernel==6.29.5
jupyterlab
```

Next, create your `Dockerfile`:

```dockerfile
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m ipykernel install --user --name python3

COPY src/ ./src/
COPY data/ ./data/
COPY notebooks/ ./notebooks/

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
```

**How Docker Fixes "Works on my machine" issues:**
*   **Issue:** Python version conflicts. **Fix:** Container has exactly Python 3.12.
*   **Issue:** Dependency hell. **Fix:** `requirements.txt` is frozen and installed once.
*   **Issue:** Onboarding takes hours. **Fix:** Run one command and you're done.

---

## Step 8 — `docker-compose.yml`

Create `docker-compose.yml` to set up two parallel services:

```yaml
services:
  cli:
    build: .
    volumes:
      - ./data:/workspace/data
      - ./notebooks:/workspace/notebooks
      - ./src:/workspace/src
    entrypoint: ["python", "-m", "src.cli"]

  jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ./data:/workspace/data
      - ./notebooks:/workspace/notebooks
      - ./src:/workspace/src
    entrypoint: []
    command: >
      jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=decodeai
```

The **volumes** are the secret sauce. Without volumes, files inside the container disappear when it stops. With volumes, cleaned CSVs, executed notebooks, and logs map directly to your Mac.

---

## Step 9 — The Automation Native: Makefile

Create a `Makefile` (Make sure to use real tabs for indents beneath the commands!):

```makefile
build:
    docker compose build

run-clean:
    docker compose run --rm cli clean /workspace/data/raw/customers.csv /workspace/data/cleaned/customers_clean.csv --missing drop

run-notebook:
    docker compose run --rm cli run-nb /workspace/notebooks/analyze_customers.ipynb

run-all: run-clean run-notebook

jupyter:
    docker compose up jupyter
```

---

## What You've Learned (and What's Next)

### Skills from this episode:

*   **Project structure:** Organized folders for data, code, notebooks.
*   **Data cleaning:** Built a multi-step pandas pipeline.
*   **Notebook automation:** Executed `.ipynb` files without a browser.
*   **Docker:** Containerized the entire workflow.
*   **CLI design:** Built clean orchestration commands.

### What's coming:

*   **Episode 2 (Text embeddings):** Clean data → convert to vectors
*   **Episode 3 (Vector search):** Embeddings → similarity search
*   **Episode 4 (RAG pipeline):** Search → retrieval-augmented generation

Every future episode assumes you can clean data, run notebooks, and work inside Docker. That's why this is Episode 1.