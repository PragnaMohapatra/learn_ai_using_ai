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

| Skill You'll Learn | Why It Matters for AI |
|---|---|
| Python project structure | Every ML repo is organized this way |
| pandas data cleaning | You'll clean datasets before every training run |
| Programmatic notebook execution | Automate experiment tracking and reporting |
| Docker containerization | Reproducible environments for training and inference |
| CLI design with argparse | Pipeline orchestration starts with commands |
| Makefile shortcuts | One command to run your entire workflow |

**If you skip this, everything else breaks.** Episode 2 (embeddings), Episode 3 (vector search), Episode 4 (RAG) — they all assume you can clean data, run notebooks, and work inside Docker.

---

## What We're Building

A Python CLI with two commands, fully containerized in Docker:

```
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

- Docker Desktop ([install guide](https://docs.docker.com/get-docker/))
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

```
episode-01/
├── data/
│   ├── raw/          # Messy input data goes here
│   └── cleaned/      # Cleaned output lands here
├── notebooks/        # Jupyter notebooks
├── src/              # Python source code
├── logs/             # Execution logs (future use)
└── tests/            # Unit tests (future use)
```

**Why this structure?**

| Folder | Purpose | AI Relevance |
|---|---|---|
| `data/raw/` | Untouched source data | Training data, downloaded datasets |
| `data/cleaned/` | Processed output | Feature-engineered data ready for models |
| `notebooks/` | Exploration and analysis | Experiment notebooks, EDA |
| `src/` | Reusable Python modules | Model code, data pipelines, utilities |
| `logs/` | Execution records | Training logs, metrics, experiment tracking |
| `tests/` | Automated tests | Validate data pipelines, model outputs |

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

| # | Problem | Where | What Our Cleaner Does |
|---|---|---|---|
| 1 | Headers have spaces | `  Name  `, `  City  ` | `clean_headers()` strips and lowercases |
| 2 | Values have whitespace | ` Bob Smith ` | `trim_strings()` strips all strings |
| 3 | Duplicate row | Alice Johnson row 2 = row 4 | `drop_duplicates()` removes copies |
| 4 | Duplicate row | Bob Smith row 3 = row 10 | `drop_duplicates()` removes copies |
| 5 | Missing age | Bob Smith, age is blank | `handle_missing()` drops or fills |
| 6 | Missing purchase | Charlie Brown, Ivan Torres | `handle_missing()` drops or fills |
| 7 | Missing email | Eve Davis has no email | `handle_missing()` drops or fills |
| 8 | Missing name | Row 8, only has grace@example.com | `handle_missing()` drops or fills |

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

Create `src/cleaner.py`:

```python
"""
Data Cleaner Module — Decode AI Using AI, Episode 1
Cleans messy CSV files: trims whitespace, normalizes headers,
drops duplicates, handles missing values, and standardizes types.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase headers, replace spaces/special chars with underscores."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def trim_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all string columns."""
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    return df


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows. Returns cleaned df and count of dropped rows."""
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    return df, dropped


def handle_missing(df: pd.DataFrame, strategy: str = "drop") -> tuple[pd.DataFrame, int]:
    """
    Handle missing values.
    Strategies: 'drop' — remove rows with any NaN
                'fill' — fill numeric with median, strings with 'UNKNOWN'
    """
    missing_before = int(df.isnull().sum().sum())

    if strategy == "drop":
        df = df.dropna()
    elif strategy == "fill":
        for col in df.columns:
            if df[col].dtype in ("float64", "int64"):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("UNKNOWN")

    return df, missing_before


def generate_report(
    original_rows: int,
    cleaned_rows: int,
    duplicates_dropped: int,
    missing_handled: int,
    output_path: str,
) -> dict:
    """Generate a summary report of the cleaning operation."""
    return {
        "timestamp": datetime.now().isoformat(),
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "duplicates_dropped": duplicates_dropped,
        "missing_values_handled": missing_handled,
        "output_file": output_path,
        "rows_removed_total": original_rows - cleaned_rows,
    }


def clean_csv(input_path: str, output_path: str, missing_strategy: str = "drop") -> dict:
    """
    Full cleaning pipeline:
    1. Read CSV
    2. Clean headers
    3. Trim strings
    4. Drop duplicates
    5. Handle missing values
    6. Save cleaned CSV
    7. Return report
    """
    df = pd.read_csv(input_path)
    original_rows = len(df)

    df = clean_headers(df)
    df = trim_strings(df)
    df, duplicates_dropped = drop_duplicates(df)
    df, missing_handled = handle_missing(df, strategy=missing_strategy)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return generate_report(
        original_rows=original_rows,
        cleaned_rows=len(df),
        duplicates_dropped=duplicates_dropped,
        missing_handled=missing_handled,
        output_path=output_path,
    )
```

**Understanding the design:**

Each function does exactly one thing. This is intentional — in AI workflows, you'll often need to swap out a single step (e.g., change how missing values are handled) without rewriting everything.

| Function | Input | Output | Example |
|---|---|---|---|
| `clean_headers()` | DataFrame | DataFrame with clean column names | `  Name  ` → `name` |
| `trim_strings()` | DataFrame | DataFrame with trimmed strings | `" Bob "` → `"Bob"` |
| `drop_duplicates()` | DataFrame | (DataFrame, count of dropped rows) | 12 rows → 10 rows, dropped 2 |
| `handle_missing()` | DataFrame + strategy | (DataFrame, count of missing values) | Drop rows or fill blanks |
| `generate_report()` | Counts | Summary dict | `{"original_rows": 12, "cleaned_rows": 5, ...}` |
| `clean_csv()` | File paths | Saved CSV + report dict | Orchestrates all of the above |

**AI relevance:** This exact pattern — read → transform → save → report — is how every data preprocessing pipeline works, whether you're cleaning text for NLP or images for computer vision.

---

## Step 5 — Write the Notebook Runner (`src/notebook_runner.py`)

This module lets you execute Jupyter notebooks from the command line. No browser, no clicking, no manual steps.

Create `src/notebook_runner.py`:

```python
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
```

**Why automate notebook execution?**

In AI work, you'll have notebooks that:
- Generate training reports after each model run
- Produce evaluation metrics and visualizations
- Create dataset summaries before and after preprocessing

Running them by hand doesn't scale. This module lets you trigger them from a script, a Makefile, or a CI/CD pipeline.

---

## Step 6 — Write the CLI Entry Point (`src/cli.py`)

This ties everything together. It's the command users actually type.

Create `src/cli.py`:

```python
"""
CLI Entry Point — Decode AI Using AI, Episode 1
Commands:
  clean   — Clean a CSV file
  run-nb  — Execute a Jupyter notebook
"""

import argparse
import json
import sys
from src.cleaner import clean_csv
from src.notebook_runner import run_notebook


def cmd_clean(args):
    """Handle the 'clean' subcommand."""
    print(f"🧹 Cleaning: {args.input}")
    print(f"   Strategy: {args.missing}")
    print(f"   Output:   {args.output}")
    print()

    report = clean_csv(args.input, args.output, missing_strategy=args.missing)

    print("✅ Cleaning complete!")
    print(f"   Original rows:      {report['original_rows']}")
    print(f"   Cleaned rows:       {report['cleaned_rows']}")
    print(f"   Duplicates dropped: {report['duplicates_dropped']}")
    print(f"   Missing handled:    {report['missing_values_handled']}")
    print(f"   Saved to:           {report['output_file']}")

    if args.json:
        print("\n📊 JSON Report:")
        print(json.dumps(report, indent=2))

    return report


def cmd_run_nb(args):
    """Handle the 'run-nb' subcommand."""
    print(f"📓 Running notebook: {args.notebook}")
    print()

    report = run_notebook(
        notebook_path=args.notebook,
        output_path=args.output,
        timeout=args.timeout,
    )

    status = "✅ Success" if report["success"] else "❌ Failed"
    print(f"{status}")
    print(f"   Code cells:      {report['code_cells']}")
    print(f"   Execution time:  {report['execution_time_seconds']}s")
    print(f"   Output saved to: {report['output']}")

    if report["errors"]:
        print(f"   Errors: {report['errors']}")

    if args.json:
        print("\n📊 JSON Report:")
        print(json.dumps(report, indent=2))

    return report


def main():
    parser = argparse.ArgumentParser(
        prog="ai-workflow",
        description="🤖 Decode AI Using AI — Episode 1: Data Cleaner + Notebook Runner",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- clean command ---
    clean_parser = subparsers.add_parser("clean", help="Clean a messy CSV file")
    clean_parser.add_argument("input", help="Path to input CSV file")
    clean_parser.add_argument("output", help="Path to save cleaned CSV")
    clean_parser.add_argument(
        "--missing",
        choices=["drop", "fill"],
        default="drop",
        help="Strategy for missing values: drop rows or fill (default: drop)",
    )
    clean_parser.add_argument(
        "--json", action="store_true", help="Print JSON report",
    )

    # --- run-nb command ---
    nb_parser = subparsers.add_parser("run-nb", help="Execute a Jupyter notebook")
    nb_parser.add_argument("notebook", help="Path to .ipynb file")
    nb_parser.add_argument(
        "--output", default=None, help="Path to save executed notebook",
    )
    nb_parser.add_argument(
        "--timeout", type=int, default=300,
        help="Timeout per cell in seconds (default: 300)",
    )
    nb_parser.add_argument(
        "--json", action="store_true", help="Print JSON report",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "clean":
        cmd_clean(args)
    elif args.command == "run-nb":
        cmd_run_nb(args)


if __name__ == "__main__":
    main()
```

**Why argparse?**

In later episodes, you'll build more complex CLIs (e.g., `embed`, `search`, `chat`). Starting with argparse now teaches the pattern. Tools like `typer` and `click` are fancier, but argparse ships with Python — zero dependencies.

---

## Step 7 — Create the Analysis Notebook

This notebook reads the cleaned CSV and produces summary stats. It's what the `run-nb` command will execute.

Create `notebooks/analyze_customers.ipynb` with these cells:

**Cell 1 — Markdown:**

```markdown
# Customer Data Analysis
**Decode AI Using AI — Episode 1**

This notebook reads the cleaned customer data and produces basic analysis.
```

**Cell 2 — Code: Load the data**

```python
import pandas as pd

df = pd.read_csv('/workspace/data/cleaned/customers_clean.csv')
print(f'Loaded {len(df)} rows')
df.head()
```

**Cell 3 — Code: Dataset summary**

```python
print('=== Dataset Summary ===')
print(f'Total customers: {len(df)}')
print(f'Columns: {list(df.columns)}')
print()
df.describe()
```

**Cell 4 — Code: City distribution**

```python
print('=== Customers by City ===')
city_counts = df['city'].value_counts()
print(city_counts)
```

**Cell 5 — Code: Purchase statistics**

```python
if 'purchase_amount' in df.columns:
    print('=== Purchase Stats ===')
    print(f'Average purchase: ${df["purchase_amount"].mean():.2f}')
    print(f'Max purchase:     ${df["purchase_amount"].max():.2f}')
    print(f'Min purchase:     ${df["purchase_amount"].min():.2f}')
    print(f'Total revenue:    ${df["purchase_amount"].sum():.2f}')
```

**Cell 6 — Code: Final summary**

```python
summary = {
    'total_customers': len(df),
    'avg_age': round(df['age'].mean(), 1) if 'age' in df.columns else None,
    'avg_purchase': round(df['purchase_amount'].mean(), 2) if 'purchase_amount' in df.columns else None,
    'cities': df['city'].nunique() if 'city' in df.columns else None,
}
print('=== Final Summary ===')
for k, v in summary.items():
    print(f'  {k}: {v}')
```

> **Tip:** You can create this notebook in Jupyter Lab (`make jupyter`) or save it as raw JSON. The key point is that `run-nb` will execute it without a browser.

---

## Step 8 — Create `requirements.txt`

These are the Python packages Docker installs inside the container.

Create `requirements.txt`:

```
pandas==2.2.3
nbformat==5.10.4
nbclient==0.10.2
ipykernel==6.29.5
jupyterlab
```

| Package | Version | Why | AI Relevance |
|---|---|---|---|
| `pandas` | 2.2.3 | Read and clean CSV files | The #1 data tool in ML |
| `nbformat` | 5.10.4 | Read/write `.ipynb` files | Notebook automation |
| `nbclient` | 0.10.2 | Execute notebooks programmatically | CI/CD for experiments |
| `ipykernel` | 6.29.5 | Python kernel for notebook execution | Required by nbclient |
| `jupyterlab` | latest | Interactive notebook UI | Exploration and EDA |

---

## Step 9 — Write the Dockerfile

This builds a container with Python 3.12, all dependencies, and your source code. Nobody on your team needs to install Python or pip — just Docker.

Create `Dockerfile`:

```dockerfile
# Start from official Python 3.12 (slim = smaller image, no extras)
FROM python:3.12-slim

# Labels (metadata, not functional)
LABEL maintainer="Decode AI Using AI"
LABEL description="Episode 1: Data cleaning and notebook automation CLI"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /workspace

# Install dependencies first (layer caching — reinstalls only when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the IPython kernel so notebooks can execute
RUN python -m ipykernel install --user --name python3

# Copy project source code
COPY src/ ./src/

# Copy sample data and notebooks
COPY data/ ./data/
COPY notebooks/ ./notebooks/

# Default entrypoint: the CLI
ENTRYPOINT ["python", "-m", "src.cli"]

# Default command: show help
CMD ["--help"]
```

**Why Docker for AI?**

| Problem Without Docker | How Docker Fixes It |
|---|---|
| "Works on my machine" | Same image runs everywhere |
| Python version conflicts | Container has exactly Python 3.12 |
| Dependency hell | `requirements.txt` is frozen and installed once |
| GPU driver mismatches (later episodes) | NVIDIA Docker handles GPU passthrough |
| Onboarding takes hours | `make build` and you're done |

---

## Step 10 — Write `docker-compose.yml`

The Dockerfile builds one image. `docker-compose.yml` defines **services** — different ways to use that image.

Create `docker-compose.yml`:

```yaml
services:
  # --- CLI tool: data cleaner + notebook runner ---
  cli:
    build: .
    container_name: decode-ai-cli
    volumes:
      - ./data:/workspace/data
      - ./notebooks:/workspace/notebooks
      - ./logs:/workspace/logs
      - ./src:/workspace/src
    entrypoint: ["python", "-m", "src.cli"]

  # --- Jupyter Lab: for interactive exploration ---
  jupyter:
    build: .
    container_name: decode-ai-jupyter
    ports:
      - "8888:8888"
    volumes:
      - ./data:/workspace/data
      - ./notebooks:/workspace/notebooks
      - ./src:/workspace/src
    entrypoint: []
    command: >
      jupyter lab
      --ip=0.0.0.0
      --port=8888
      --no-browser
      --allow-root
      --NotebookApp.token=decodeai
```

### Understanding volumes (the key concept)

```
Your Mac                          Inside Container
─────────────────                 ─────────────────
./data/raw/customers.csv    →    /workspace/data/raw/customers.csv
./data/cleaned/             ←    /workspace/data/cleaned/  (output lands here)
./src/cleaner.py            →    /workspace/src/cleaner.py (live reload)
```

The arrow goes both ways — changes on either side are instantly visible. You can edit `src/cleaner.py` in VS Code and re-run the container without rebuilding.

- **Without volumes:** files inside the container disappear when it stops
- **With volumes:** cleaned CSVs, executed notebooks, and logs persist on your Mac

### Why two services from one image?

| Service | Use Case | How You Run It |
|---|---|---|
| `cli` | Automated pipeline — clean data, run notebooks | `docker compose run --rm cli clean ...` |
| `jupyter` | Interactive exploration — try things, visualize | `docker compose up jupyter` → open browser |

In a real AI workflow, you use Jupyter to **experiment**, then move proven code into the CLI for **automation**.

> **Important:** YAML uses spaces for indentation, not tabs.

---

## Step 11 — Write the Makefile

Nobody wants to type `docker compose run --rm cli clean /workspace/data/raw/customers.csv /workspace/data/cleaned/customers_clean.csv --missing drop --json` every time.

Create `Makefile` (**must use tabs for indentation, not spaces**):

```makefile
.PHONY: build run-clean run-clean-fill run-notebook run-all jupyter down clean help

build:
    docker compose build

run-clean:
    docker compose run --rm cli clean \
        /workspace/data/raw/customers.csv \
        /workspace/data/cleaned/customers_clean.csv \
        --missing drop --json

run-clean-fill:
    docker compose run --rm cli clean \
        /workspace/data/raw/customers.csv \
        /workspace/data/cleaned/customers_clean.csv \
        --missing fill --json

run-notebook:
    docker compose run --rm cli run-nb \
        /workspace/notebooks/analyze_customers.ipynb \
        --json

run-all: run-clean run-notebook

jupyter:
    docker compose up jupyter

down:
    docker compose down

clean:
    docker compose down --rmi all --volumes --remove-orphans

help:
    @echo ""
    @echo "Decode AI Using AI — Episode 1"
    @echo "===================================="
    @echo "  make build          Build the Docker image"
    @echo "  make run-clean      Clean CSV (drop missing rows)"
    @echo "  make run-clean-fill Clean CSV (fill missing values)"
    @echo "  make run-notebook   Execute the analysis notebook"
    @echo "  make run-all        Full pipeline: clean + notebook"
    @echo "  make jupyter        Launch Jupyter Lab (port 8888)"
    @echo "  make down           Stop all containers"
    @echo "  make clean          Remove images and volumes"
    @echo "  make help           Show this help"
    @echo ""
```

> **Critical:** The file must be named exactly `Makefile` (capital M). Lines under each target must be indented with **a real tab character**, not spaces. This is the #1 cause of `missing separator` errors.

---

## Step 12 — Build and Run Everything

Make sure Docker Desktop is running and you're inside `episode-01/`.

### 12a: Build the image

```sh
make build
```

Takes 1–2 minutes the first time. You should see:

```
 => [2/7] COPY requirements.txt .
 => [3/7] RUN pip install --no-cache-dir -r requirements.txt
 => [4/7] RUN python -m ipykernel install --user --name python3
 => [5/7] COPY src/ ./src/
```

If it ends without errors, your image is built. Subsequent builds are cached and take seconds.

### 12b: Test the CLI

```sh
docker compose run --rm cli --help
```

Expected:

```
🤖 Decode AI Using AI — Episode 1: Data Cleaner + Notebook Runner

positional arguments:
  {clean,run-nb}  Available commands
    clean         Clean a messy CSV file
    run-nb        Execute a Jupyter notebook
```

If you see this, your CLI is alive inside Docker.

### 12c: Clean the data

```sh
make run-clean
```

Expected:

```
🧹 Cleaning: /workspace/data/raw/customers.csv
   Strategy: drop
   Output:   /workspace/data/cleaned/customers_clean.csv

✅ Cleaning complete!
   Original rows:      12
   Cleaned rows:       5
   Duplicates dropped: 2
   Missing handled:    5
   Saved to:           /workspace/data/cleaned/customers_clean.csv
```

12 rows in, 5 survived. Verify on your Mac:

```sh
cat data/cleaned/customers_clean.csv
```

Clean headers, no whitespace, no blanks, no duplicates. The file was created inside Docker but appeared on your Mac because of volumes.

### 12d: Run the notebook

```sh
make run-notebook
```

Expected:

```
📓 Running notebook: /workspace/notebooks/analyze_customers.ipynb

✅ Success
   Code cells:      5
   Execution time:  ~3-5s
   Output saved to: /workspace/notebooks/analyze_customers_executed.ipynb
```

Check the output:

```sh
ls notebooks/
```

Two files:
- `analyze_customers.ipynb` — your original (untouched)
- `analyze_customers_executed.ipynb` — with all cell outputs filled in

### 12e: Run the full pipeline

```sh
make run-all
```

Runs clean → notebook in sequence. One command, complete pipeline.

### 12f: Try the fill strategy

```sh
make run-clean-fill
```

Instead of dropping rows, missing numbers get the median, missing strings get `"UNKNOWN"`. You'll see ~10 rows instead of 5:

```sh
cat data/cleaned/customers_clean.csv
```

### 12g: Launch Jupyter Lab

```sh
make jupyter
```

Open `http://localhost:8888/?token=decodeai` in your browser. Stop with `Ctrl+C` or `make down`.

---

## Final Project Structure

```
episode-01/
├── data/
│   ├── raw/
│   │   └── customers.csv              # Messy input (12 rows, 8 problems)
│   └── cleaned/
│       └── customers_clean.csv        # Clean output (generated)
├── notebooks/
│   ├── analyze_customers.ipynb               # Source notebook
│   └── analyze_customers_executed.ipynb      # Executed output (generated)
├── src/
│   ├── __init__.py                    # Package marker (empty)
│   ├── cleaner.py                     # Data cleaning pipeline
│   ├── cli.py                         # CLI entry point (argparse)
│   └── notebook_runner.py             # Programmatic notebook execution
├── logs/
├── tests/
├── Dockerfile                         # Container definition
├── docker-compose.yml                 # Service definitions (cli + jupyter)
├── Makefile                           # Command shortcuts
├── requirements.txt                   # Python dependencies
├── README.md                          # Quick-start guide
└── BUILD_GUIDE.md                     # This file
```

---

## What You've Learned (and What's Next)

### Skills from this episode:

| Skill | What You Did | Where It Shows Up in AI |
|---|---|---|
| Project structure | Organized folders for data, code, notebooks | Every ML repo follows this pattern |
| Data cleaning | Built a multi-step pandas pipeline | Preprocessing before training |
| Notebook automation | Executed `.ipynb` files without a browser | Experiment reporting, CI/CD |
| Docker | Containerized the entire workflow | Reproducible training environments |
| CLI design | Built `clean` and `run-nb` commands | Pipeline orchestration |
| Makefile | One-command shortcuts | `make train`, `make evaluate`, `make deploy` |

### What's coming:

| Episode | Topic | Builds On |
|---|---|---|
| **2** | Text embeddings | Clean data → convert to vectors |
| **3** | Vector search | Embeddings → similarity search |
| **4** | RAG pipeline | Search → retrieval-augmented generation |
| **5** | Fine-tuning | Clean data → train your own model |

Every future episode assumes you can clean data, run notebooks, and work inside Docker. That's why this is Episode 1.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `make: *** No rule to make target 'build'` | Not in `episode-01/` or `Makefile` doesn't exist (must be capital M) |
| `missing separator in Makefile` | Indentation must be **tabs**, not spaces |
| `ModuleNotFoundError: No module named 'src'` | `src/__init__.py` is missing — create it with `touch src/__init__.py` |
| `No kernel named python3 found` | Rebuild with `make build` |
| `FileNotFoundError: customers.csv` | `data/raw/customers.csv` doesn't exist |
| `docker: command not found` | Docker Desktop isn't running — open it first |
| Port 8888 already in use | Stop other Jupyter instances with `make down` |

**If Makefile isn't found:**

```sh
pwd                    # Should end in episode-01
ls -la Makefile        # Must exist, capital M
mv makefile Makefile   # Fix if lowercase
make help              # Test it
```