# Episode 2: Data Analysis for AI

This episode **builds on Episode 1 instead of copying it**.

## Why copying was a bad idea

Copying `cleaner.py`, `notebook_runner.py`, and Docker workflow files into every episode creates:

- duplicate bugs
- duplicate fixes
- inconsistent behavior between episodes
- unnecessary maintenance

## Better pattern

Common code now lives in the repo-level `shared/ai_workflow/` package.

That means:

- Episode 1 can keep cleaning and notebook execution logic in one place
- Episode 2 can import the same shared functions and add new analysis features
- future episodes can reuse the same foundation without copying files around

## Reuse model

- **Episode 1** produces clean data and repeatable execution
- **Episode 2** consumes that clean data and generates deeper analysis
- **shared/ai_workflow/** holds the reusable building blocks

## Current workflow

Episode 2 now does three useful things with the cleaned dataset from Episode 1:

- profiles column types and missing values
- renders an HTML report with charts
- can optionally ask an LLM for a short narrative summary of the dataset

## Run it

```bash
docker compose run --rm cli analyze \
	/workspace/episode-01/data/cleaned/customers_clean.csv \
	/workspace/episode-02/data/reports \
	--charts
```

If you want an AI-written summary in the report, set `OPENAI_API_KEY` and add `--ai-summary`.

The CLI keeps terminal output concise and writes full artifacts to files in `data/reports/`:

- `basic_profile.json`
- `basic_profile.html`
