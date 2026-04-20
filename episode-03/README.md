# Episode 3: First ML Model — Spam or Review Classifier

This episode introduces a simple machine learning classifier.

## Goal

Build a first baseline model that can classify:

- spam vs not spam, or
- positive vs negative review

## What this episode will cover

- loading labeled data
- splitting train and test sets
- converting text into features
- training a simple classifier
- evaluating model performance
- saving the trained model

## Project structure

- `Dockerfile` — container setup
- `docker-compose.yml` — local development workflow
- `Makefile` — common commands
- `requirements.txt` — Python dependencies
- `src/` — application code
- `tests/` — test suite
- `data/` — datasets, outputs, and saved models

## Data source

For the first classifier, this episode uses the **UCI SMS Spam Collection**:

- Dataset page: `https://archive.ics.uci.edu/dataset/228/sms+spam+collection`
- Direct download: `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`
- License: `CC BY 4.0`

Downloaded dataset files can be kept locally in `data/` for training and experimentation.

## Outcome

By the end of this episode, there will be a working end-to-end classifier with reproducible training and evaluation.