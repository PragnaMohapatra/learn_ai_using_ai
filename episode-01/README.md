# Episode 1: Python for AI Workflows

**Data Cleaner + Notebook Runner CLI — all in Docker**

> The invisible backbone of every AI project: a repeatable, containerized
> workflow for cleaning data and automating notebook execution.

## What We're Building

A Python CLI with two commands:
- `clean` — Takes a messy CSV → normalizes headers, trims whitespace, removes duplicates, handles missing values → outputs clean CSV + report
- `run-nb` — Executes a Jupyter notebook programmatically → saves output notebook with results + execution metadata

## Prerequisites

- Docker Desktop ([install guide](https://docs.docker.com/get-docker/))
- `make` (optional, but recommended — comes pre-installed on macOS/Linux)

That's it. Python, pip, Jupyter — everything else lives inside Docker.

## Project Structure

```
episode-01/
├── data/
│   ├── raw/              # Messy source data
│   │   └── customers.csv
│   └── cleaned/          # Output from the clean command
│       └── customers_clean.csv
├── notebooks/
│   └── analyze_customers.ipynb   # Analysis notebook
├── src/
│   ├── __init__.py
│   ├── cleaner.py        # Data cleaning logic
│   ├── cli.py            # CLI entry point
│   └── notebook_runner.py # Notebook execution logic
├── logs/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

## Step-by-Step: Run the Project End to End

### Step 1 — Clone the Repository and Navigate to Episode 1

```sh
git clone <your-repo-url>
cd learn_ai_using_ai/episode-01
```

### Step 2 — Build the Docker Image

This installs Python 3.12, all dependencies (`pandas`, `nbformat`, `nbclient`, `ipykernel`, `jupyterlab`), and copies your source code into the container.

```sh
make build
```

Or without `make`:

```sh
docker compose build
```

### Step 3 — View CLI Help

Verify everything is wired up by printing the CLI help:

```sh
docker compose run --rm cli --help
```

You should see the two available commands: `clean` and `run-nb`.

### Step 4 — Clean the Raw CSV (Drop Missing Rows)

This reads the messy `data/raw/customers.csv`, normalizes headers, trims whitespace, removes duplicates, drops rows with missing values, and saves the result to `data/cleaned/customers_clean.csv`.

```sh
make run-clean
```

Or without `make`:

```sh
docker compose run --rm cli clean \
  /workspace/data/raw/customers.csv \
  /workspace/data/cleaned/customers_clean.csv \
  --missing drop --json
```

**What happens under the hood:**
1. Headers like `Name  ` → `name`, `Signup Date` → `signup_date`
2. Whitespace is trimmed from all string values
3. Duplicate rows are removed (e.g., Alice Johnson and Bob Smith appear twice)
4. Rows with any missing value are dropped (`--missing drop`)
5. A JSON report is printed with row counts and stats

### Step 5 (Alternative) — Clean with Fill Strategy

Instead of dropping rows with missing data, fill numeric columns with the median and string columns with `"UNKNOWN"`:

```sh
make run-clean-fill
```

Or without `make`:

```sh
docker compose run --rm cli clean \
  /workspace/data/raw/customers.csv \
  /workspace/data/cleaned/customers_clean.csv \
  --missing fill --json
```

### Step 6 — Run the Analysis Notebook

Execute the Jupyter notebook programmatically. It reads the cleaned CSV and produces summary statistics (customer counts, purchase stats, city distribution).

```sh
make run-notebook
```

Or without `make`:

```sh
docker compose run --rm cli run-nb \
  /workspace/notebooks/analyze_customers.ipynb \
  --json
```

The executed notebook (with cell outputs) is saved as `notebooks/analyze_customers_executed.ipynb`.

### Step 7 — Run the Full Pipeline (Clean + Notebook)

Run both steps in sequence:

```sh
make run-all
```

This first cleans the CSV (drop strategy), then executes the notebook against the cleaned data.

### Step 8 (Optional) — Launch Jupyter Lab

For interactive exploration, start Jupyter Lab on port 8888:

```sh
make jupyter
```

Or without `make`:

```sh
docker compose up jupyter
```

Then open your browser at:

```
http://localhost:8888/?token=decodeai
```

### Step 9 — Stop and Clean Up

Stop all running containers:

```sh
make down
```

Remove all built images, volumes, and orphan containers:

```sh
make clean
```

## Quick Reference: All Make Commands

| Command              | Description                          |
|----------------------|--------------------------------------|
| `make build`         | Build the Docker image               |
| `make run-clean`     | Clean CSV (drop missing rows)        |
| `make run-clean-fill`| Clean CSV (fill missing values)      |
| `make run-notebook`  | Execute the analysis notebook        |
| `make run-all`       | Full pipeline: clean + notebook      |
| `make jupyter`       | Launch Jupyter Lab (port 8888)       |
| `make down`          | Stop all containers                  |
| `make clean`         | Remove images and volumes            |
| `make help`          | Show available commands              |

## Status

🔨 Building step by step...

## Build From Scratch Guide

Want to build this entire project from zero and understand why this is Episode 1? See the **[Build From Scratch Guide](BUILD_GUIDE.md)** — a complete walkthrough covering every file, every line, and every command.


