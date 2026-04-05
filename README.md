# Decode AI Using AI

> Learn AI by building real prototypes.

A 15-episode, project-based series where every topic produces a working prototype,
a blog post, and a podcast episode. No theory — just build, measure, improve.

## Episodes

| # | Topic | Status |
|---|-------|--------|
| 1 | Python for AI Workflows — Data Cleaner + Notebook Runner CLI | 🔨 In Progress |
| 2 | Tabular Data Wrangling | ⏳ Upcoming |
| 3 | Linear Regression Workshop | ⏳ Upcoming |
| 4 | Classification Workshop | ⏳ Upcoming |
| 5 | Evaluation & Metrics Deep Dive | ⏳ Upcoming |
| 6 | Neural Network Basics | ⏳ Upcoming |
| 7 | NLP Basics | ⏳ Upcoming |
| 8 | LLM Prompting | ⏳ Upcoming |
| 9 | RAG Pipeline | ⏳ Upcoming |
| 10 | AI Agents | ⏳ Upcoming |
| 11 | Recommendation System | ⏳ Upcoming |
| 12 | Deployment Workshop | ⏳ Upcoming |
| 13 | Rebuild Episode (3 or 4) | ⏳ Upcoming |
| 14 | Rebuild Episode (8 or 9) | ⏳ Upcoming |
| 15 | Capstone | ⏳ Upcoming |

## How to Use

Each episode keeps its own README, data, and notebooks, while **common reusable workflow code lives in `shared/ai_workflow/`** so later episodes can build on earlier ones without copying files around.

```bash
cd episode-01
make build
make run-all
```

## Shared Foundation

As the series grows, common building blocks move to the repo root:

- `shared/ai_workflow/cleaner.py` — reusable CSV cleaning pipeline
- `shared/ai_workflow/notebook_runner.py` — reusable notebook execution logic

This keeps the series compounding: Episode 2 reuses Episode 1 instead of duplicating it.

## Series Rules

1. Build first, read only when blocked
2. Record one metric per episode
3. Everything runs in Docker — no local installs
4. Repeat projects later to measure improvement
