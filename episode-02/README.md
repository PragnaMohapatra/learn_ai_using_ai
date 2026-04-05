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

Next step: implement `src/eda.py` and the `analyze` workflow.
