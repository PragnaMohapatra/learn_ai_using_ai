# Decode AI Using AI: Episode 1

**Data Cleaner + Notebook Runner CLI — Your First Step in the AI Learning Journey**

---

## Why We're Building This Series

Most AI learning is passive: watch a video, read a paper, copy a notebook. This series — **Decode AI Using AI** — is different. Every episode:

*   **Solves a real problem** with a working prototype.
*   **Uses AI to build AI** — LLMs, code assistants, and AI tools are used throughout the build process itself.
*   **Ships something** — every episode ends with a published artifact: a blog post, a repo, an app.
*   **Compounds** — each episode builds on the last. By Episode 15, you will have a full AI product portfolio. 

This series is for people who want to stop watching AI happen and start building it.

---

## The 15 Episodes Begin Here: Python for AI Workflows

Every AI project — whether it's training a model, building a RAG pipeline, or fine-tuning an LLM — starts with the same unglamorous reality: **messy data and unrepeatable workflows**. 

**The Problem:** Raw data is dirty. Furthermore, running Jupyter notebooks manually is error-prone and not reproducible. You need a repeatable pipeline before any real AI work can begin.

**The Prototype (v1):** We are building a Dockerized CLI (`src/cli.py`) with two subcommands:
1. `clean` — reads a raw CSV, normalizes headers, trims whitespace, drops duplicates, handles missing values, and writes a cleaned CSV.
2. `run-nb` — executes a Jupyter notebook programmatically and returns a JSON report.

This episode teaches you the **invisible backbone** that every future episode depends on:

*   **Python project structure:** Every ML repo is organized this way
*   **pandas data cleaning:** You'll clean datasets before every training run
*   **Programmatic notebook execution:** Automate experiment tracking and reporting
*   **Docker containerization:** Reproducible environments for training and inference
*   **CLI design with argparse:** Pipeline orchestration starts with commands
*   **Makefile shortcuts:** One command to run your entire workflow

**If you skip this, everything else breaks.** Upcoming episodes assume you can clean data, run notebooks, and work inside Docker. Let's start building.

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