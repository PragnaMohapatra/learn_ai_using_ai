# Episode 3 : Learn AI by Doing It, Your First ML Model

**Train a spam classifier from scratch, step by step — and learn what every line of scikit-learn is actually doing.**

## Why DIY Projects in the Age of AI

The world is moving at a pace no curriculum can keep up with. Large language models write production code, generate test suites, and draft documentation in seconds. The traditional pillar of software engineering — deep mastery of one technology — is no longer sufficient on its own. Breadth is becoming pivotal.

When AI can write the code for you, the bottleneck shifts. What matters now is knowing what to ask for. Understanding **why** a train/test split is stratified, **what** TF-IDF actually computes, and **when** a 97% test accuracy is lying to you — that is what separates an engineer who can ship ML from one who is one `scikit-learn` tutorial deep.

Episode 1 ingested and cleaned raw data. Episode 2 profiled it with a deterministic EDA pipeline and an opt-in AI narrative. Episode 3 takes the next step: we train our first real ML model. A spam classifier. No deep learning. No GPU. No magic. Just `pandas`, `scikit-learn`, and the discipline of building it one layer at a time so every function earns its place.

## What We'll Cover

1. Why Start with a Classifier
2. Project Structure
3. Step 1 — Docker & Infrastructure
4. Step 2 — Get Real, Licensed Data
5. Step 3 — Load and Clean the Labeled Dataset
6. Step 4 — Split Train and Test (and Why Stratification Matters)
7. Step 5 — Turn Text into Numbers with TF-IDF
8. Step 6 — Train Two Baseline Models and Compare
9. Step 7 — Save the Artifacts
10. Step 8 — The Prediction Module
11. Step 9 — The CLI Entry Point
12. Step 10 — Tests (All 15 of Them)
13. Reality Check: 15 Real-World Messages
14. Exercise vs Production: What We Did Not Do
15. Glossary

---

## 1. Why Start with a Classifier

Classification is the "hello world" of supervised machine learning. You have inputs (SMS messages, reviews, emails) and you want a label (spam / ham, positive / negative, fraud / legit). The whole pipeline that powers much larger systems — data loading, feature engineering, train/test discipline, model selection, serialization, inference — is already present in the smallest classifier.

If you can build a spam classifier properly, you can build a next-token predictor, a review scorer, or a churn model. The shape of the code is the same. Only the math inside one call to `.fit()` changes.

We deliberately pick **text classification** because:

- Real datasets are free and licensed (UCI SMS Spam, IMDB reviews, Yelp polarity).
- Feature engineering (TF-IDF) is a concept worth understanding before you hide it behind an embedding model.
- A linear model on bag-of-words is genuinely competitive for short-text spam. You do not need a transformer to get 97% accuracy.

## 2. Project Structure

Episode 3 mirrors the layout we settled on in Episode 2 so the muscle memory transfers:

```
episode-03/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirments.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── training.py
│   └── prediction.py
├── tests/
│   ├── test_cli.py
│   ├── test_training.py
│   └── test_prediction.py
├── templates/
└── data/
    ├── SMSSpamCollection       # raw
    ├── sms_spam_clean.csv       # cleaned
    └── artifacts/
        ├── best_model.joblib
        ├── vectorizer.joblib
        └── metrics.json
```

Notice what is **not** there: no notebook, no experiment-tracking dashboard, no `model.pkl` checked into git. The deliverables are reproducible artifacts written to `data/artifacts/`. Everything else is code you can re-run.

## 3. Step 1 — Docker & Infrastructure

Same pattern as Episode 2. Docker is not ceremony — it is how you make sure the person cloning your repo in six months gets the same NumPy version that produced your 97% accuracy.

### Dockerfile

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace:/workspace/episode-03

WORKDIR /workspace/episode-03

COPY episode-03/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

COPY shared/ /workspace/shared/
COPY episode-03/ /workspace/episode-03/

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
```

The critical detail is `PYTHONPATH`. It lets `from src import train_classifier` and `from shared.ai_workflow import ...` both resolve the same way on your laptop and inside the container.

### docker-compose.yml

```yaml
services:
  cli:
    build:
      context: ..
      dockerfile: episode-03/Dockerfile
    container_name: decode-ai-episode-03-cli
    working_dir: /workspace/episode-03
    environment:
      PYTHONPATH: /workspace:/workspace/episode-03
    volumes:
      - ./:/workspace/episode-03
      - ../shared:/workspace/shared:ro
      - ../episode-01/data:/workspace/episode-01/data:ro
      - ../episode-02/data:/workspace/episode-02/data:ro
    command: ["--help"]
```

Again, `context: ..` is the parent directory, so Docker can see `shared/`, `episode-01/data/`, and `episode-02/data/` at build and run time. Earlier episodes become read-only dependencies for later ones.

### requirements.txt

Pinned so that a year from now the model scores are reproducible to the fourth decimal place:

```
pandas==2.2.3
scikit-learn==1.5.2
joblib==1.4.2
jinja2==3.1.4
pytest==8.3.5
```

### Makefile

Four commands you will actually type:

```makefile
build:
	docker compose build cli

test:
	docker compose run --rm --entrypoint python cli -m pytest -q

shell:
	docker compose run --rm cli bash

help:
	@echo "make build | make test | make shell"
```

## 4. Step 2 — Get Real, Licensed Data

Toy datasets teach toy lessons. We downloaded the **UCI SMS Spam Collection** (CC BY 4.0):

```bash
curl -L -o data/sms_spam_collection.zip \
  https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
unzip data/sms_spam_collection.zip -d data/
```

We get `SMSSpamCollection` — a tab-separated file with two columns: `label` (`ham` / `spam`) and `text`. 5,574 messages. About 13% spam. Real UK SMS traffic from the early 2000s.

One thing to flag early, because it matters for the "exercise vs production" discussion at the end: **this dataset is from 2002**. It knows nothing about WhatsApp, PayPal, crypto, or 2FA phishing. A model trained on it is a time capsule. That is fine for learning — not fine for shipping.

## 5. Step 3 — Load and Clean the Labeled Dataset

We build the pipeline one function at a time in `src/training.py`. Each function does one thing. Each is independently testable.

### Read the raw file

The dataset ships as tab-separated, but if we later re-save it as CSV, we want the same loader to handle both:

```python
def _read_input_dataset(input_path: str) -> pd.DataFrame:
    dataset_path = Path(input_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {input_path}")

    if dataset_path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(dataset_path)
    else:
        dataframe = pd.read_csv(
            dataset_path,
            sep="\t",
            header=None,
            names=["label", "text"],
        )

    if dataframe.empty:
        raise ValueError("Training dataset is empty.")

    return dataframe
```

Fail fast with a clear message. We will do that everywhere.

### Clean the labels and text

```python
REQUIRED_COLUMNS = {"text", "label"}

def clean_training_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Training dataset is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    cleaned = dataframe.loc[:, ["text", "label"]].copy()
    cleaned["text"] = (
        cleaned["text"].fillna("").astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    cleaned["label"] = (
        cleaned["label"].astype("string").str.strip().str.lower()
    )
    cleaned = cleaned[
        (cleaned["text"] != "")
        & cleaned["label"].notna()
        & (cleaned["label"] != "")
    ]
    cleaned = cleaned.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("Training dataset has no usable rows after cleaning.")

    return cleaned
```

What this actually does:

- Validates the columns exist — the model cannot guess what "spam" looks like if there is no `label` column.
- Collapses runs of whitespace (`"hello    world"` → `"hello world"`) so tokenization does not create bogus vocabulary entries.
- Lowercases labels so `"Spam"`, `"SPAM"`, and `"spam"` are the same class.
- Drops empty rows and exact duplicates. The SMS Spam Collection has a lot of duplicated messages — after dedup we go from 5,574 to **5,158** rows.

And `prepare_training_data` writes the cleaned CSV to disk so the rest of the pipeline has a stable input.

## 6. Step 4 — Split Train and Test (and Why Stratification Matters)

This is where the questions started flying, so let me reproduce the Q&A before the code.

**"What is stratify doing, exactly?"**

When you do a random 80/20 split on a dataset that is 87% ham and 13% spam, you can — by sheer bad luck — end up with a test set that is 95% ham. Your "accuracy" will look great (predict ham always → 95% right) but you have learned nothing about spam. Stratification says: *preserve the class ratio in both splits*. If the full dataset is 87/13, the training set is 87/13 and the test set is 87/13.

**"And random_state?"**

Train/test splits are random. `random_state=42` (or any fixed integer) pins the random number generator so that running the code twice gives the exact same split. Reproducibility beats cleverness.

**"What does `.fit()` actually do?"**

For a vectorizer: it **learns** — it scans the training text, builds the vocabulary, computes document frequencies. For a model: it learns parameters (weights for logistic regression, class-conditional word probabilities for naive Bayes). `.transform()` (or `.predict()`) then **applies** what was learned.

Now the code, with a safety net for tiny datasets we use in tests:

```python
from math import ceil
from sklearn.model_selection import train_test_split

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

def split_training_data(
    dataframe: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    if len(dataframe) < 2:
        raise ValueError("Need at least 2 rows to split training data.")

    X = dataframe["text"]
    y = dataframe["label"]

    stratify = None
    class_count = int(y.nunique())
    test_rows = ceil(len(dataframe) * test_size)
    train_rows = len(dataframe) - test_rows

    if (
        class_count > 1
        and y.value_counts().min() >= 2
        and test_rows >= class_count
        and train_rows >= class_count
    ):
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "stratified": stratify is not None,
    }
```

Why the gated `if`? Because our tests use a 4-row fixture and scikit-learn refuses to stratify when there are not enough samples per class to go around. Rather than catching the exception, we check the preconditions up front and silently fall back to a non-stratified split. On the real dataset (5,158 rows) the conditions are always satisfied — stratification kicks in.

## 7. Step 5 — Turn Text into Numbers with TF-IDF

Models do not read English. They consume vectors of floats. So we need a deterministic way to turn `"WIN a FREE iPhone now!!!"` into a row of numbers.

**TF-IDF** = *Term Frequency × Inverse Document Frequency*. In plain language:

- **TF** rewards a word that appears many times in the message you are scoring. A message that says "free" five times cares about "free".
- **IDF** penalizes a word that appears in almost every message. "The", "a", "is" show up everywhere — they carry no signal.
- **TF × IDF** gives a high score to words that are common *in this message* and rare *across the corpus*. "Free" in a casual "hi mom how are you" message is nothing; "free" in a message that also has "winner" and "click" is everything.

scikit-learn does the whole thing in one object:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(
    X_train: pd.Series,
    X_test: pd.Series,
    *,
    max_features: int = 2000,
) -> dict:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    return {
        "vectorizer": vectorizer,
        "X_train_features": X_train_features,
        "X_test_features": X_test_features,
        "train_shape": tuple(X_train_features.shape),
        "test_shape": tuple(X_test_features.shape),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
    }
```

Two details that matter for understanding and for production:

- **`fit_transform` on train, `transform` on test.** We **never** fit the vectorizer on the test set. If we did, the vocabulary would include words the model was never trained on, and our "held-out" accuracy would be a lie. The test set must be a stranger to everything in the pipeline.
- **`stop_words="english"` and `max_features=2000`** are two knobs. Stopwords remove the "the / is / a" noise. `max_features=2000` keeps only the 2,000 most informative tokens, which is enough for SMS and keeps the matrix small. These are reasonable defaults, not sacred values.

After this step, `X_train_features` is a sparse matrix of shape `(4126, 2000)` — 4,126 training messages, each represented by a 2,000-dimensional TF-IDF vector.

## 8. Step 6 — Train Two Baseline Models and Compare

A single-model ML tutorial is a trap. You never know whether the number you got is good or bad without a reference. So we train two classic text classifiers and compare them:

- **Logistic Regression** — a linear model. For each word, learn a weight; sum the weights of the words present; squash through a sigmoid to get P(spam).
- **Multinomial Naive Bayes** — a probabilistic model. Estimate P(word | spam) and P(word | ham) from training counts, then use Bayes' rule at prediction time. Famous for being strong on text out of the box.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def train_baseline_models(
    X_train_features, X_test_features, y_train, y_test,
) -> dict:
    if y_train.nunique() < 2:
        raise ValueError("Need at least 2 label classes to train.")

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "multinomial_nb": MultinomialNB(),
    }

    model_results, fitted_models = {}, {}
    best_model, best_accuracy = None, -1.0

    for name, model in candidates.items():
        model.fit(X_train_features, y_train)
        predictions = model.predict(X_test_features)
        accuracy = float(accuracy_score(y_test, predictions))
        model_results[name] = {
            "accuracy": round(accuracy, 4),
            "predictions": predictions.tolist(),
        }
        fitted_models[name] = model
        if accuracy > best_accuracy:
            best_model, best_accuracy = name, accuracy

    return {
        "models": model_results,
        "fitted_models": fitted_models,
        "best_model": best_model,
        "best_accuracy": round(best_accuracy, 4),
    }
```

Results on the 80/20 stratified split:

| Model                | Test accuracy |
|----------------------|---------------|
| Logistic Regression  | 0.9641        |
| **Multinomial NB**   | **0.9777**    |

Naive Bayes wins by a hair. That is very on-brand for short-text spam — it has been the textbook baseline since the 1990s and it still holds up against a linear model on bag-of-words features.

## 9. Step 7 — Save the Artifacts

The orchestrator `train_classifier` wires all the layers together and **saves everything we will need at inference time**:

```python
import json, joblib
from pathlib import Path

def train_classifier(input_csv: str, output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataframe = load_training_data(input_csv)
    labels = sorted(dataframe["label"].unique().tolist())
    split = split_training_data(dataframe)
    vec = vectorize_text(split["X_train"], split["X_test"])
    models = train_baseline_models(
        vec["X_train_features"], vec["X_test_features"],
        split["y_train"], split["y_test"],
    )

    best = models["fitted_models"][models["best_model"]]

    model_file = output_path / "best_model.joblib"
    vectorizer_file = output_path / "vectorizer.joblib"
    metrics_file = output_path / "metrics.json"

    joblib.dump(best, model_file)
    joblib.dump(vec["vectorizer"], vectorizer_file)
    metrics_file.write_text(json.dumps({
        "labels": labels,
        "train_rows": split["train_rows"],
        "test_rows": split["test_rows"],
        "stratified": split["stratified"],
        "vocabulary_size": vec["vocabulary_size"],
        "model_scores": {n: {"accuracy": d["accuracy"]}
                         for n, d in models["models"].items()},
        "best_model": models["best_model"],
        "best_accuracy": models["best_accuracy"],
    }, indent=2), encoding="utf-8")

    return {"status": "model_saved", ...}
```

The crucial point — and the one most tutorials silently get wrong — is that **you must save the vectorizer alongside the model**. The model learned weights for feature `#417`. Feature `#417` only means "the word *won*" because the vectorizer's vocabulary said so. Load the model without the vectorizer and your predictions are random noise. They travel together, or they do not travel.

## 10. Step 8 — The Prediction Module

`src/prediction.py` is intentionally small. It mirrors training as the inference side:

```python
from pathlib import Path
import joblib

def load_artifacts(model_dir: str) -> dict:
    artifacts_path = Path(model_dir)
    model_file = artifacts_path / "best_model.joblib"
    vectorizer_file = artifacts_path / "vectorizer.joblib"

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not vectorizer_file.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_file}")

    return {
        "model": joblib.load(model_file),
        "vectorizer": joblib.load(vectorizer_file),
    }


def predict_text(model_dir: str, text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    artifacts = load_artifacts(model_dir)
    features = artifacts["vectorizer"].transform([text])
    prediction = artifacts["model"].predict(features)[0]

    return {"model_dir": model_dir, "text": text, "prediction": str(prediction)}
```

Notice the order: `vectorizer.transform([text])` uses the **same vocabulary** the training data used, then `model.predict(features)` runs the exact classifier we trained. No `fit` anywhere. Inference is pure function application.

## 11. Step 9 — The CLI Entry Point

Same argparse pattern as Episode 2. Three subcommands: `train`, `evaluate`, `predict`.

```python
import argparse
from src import predict_text, train_classifier

def main():
    parser = argparse.ArgumentParser(
        prog="ai-workflow-ep3",
        description="Decode AI Using AI — Episode 3: First ML Model",
    )
    subparsers = parser.add_subparsers(dest="command")

    train_p = subparsers.add_parser("train")
    train_p.add_argument("input")
    train_p.add_argument("output")

    predict_p = subparsers.add_parser("predict")
    predict_p.add_argument("model")
    predict_p.add_argument("text")

    # ... evaluate subparser (placeholder for the next episode)

    args = parser.parse_args()

    if args.command == "train":
        r = train_classifier(args.input, args.output)
        print(f"Status:              {r['status']}")
        print(f"Train rows:          {r['train_rows']}")
        print(f"Logistic Regression: {r['model_scores']['logistic_regression']['accuracy']}")
        print(f"Multinomial NB:      {r['model_scores']['multinomial_nb']['accuracy']}")
        print(f"Best model:          {r['best_model']}")
    elif args.command == "predict":
        r = predict_text(args.model, args.text)
        print(f"Prediction: {r['prediction']}")
```

End-to-end from a shell:

```bash
python -m src.cli train data/sms_spam_clean.csv data/artifacts
python -m src.cli predict data/artifacts "WIN a FREE iPhone now!!! Click here"
# → Prediction: spam
```

## 12. Step 10 — Tests (All 15 of Them)

Fifteen tests across three files. They cover:

- `test_training.py`: `prepare_training_data` happy path, missing-column error, stratified split on a small fixture, TF-IDF shapes, two-model training returns a `best_model`, full `train_classifier` writes all three artifact files and produces `status="model_saved"`.
- `test_prediction.py`: `predict_text` returns `ham` or `spam` after training, raises `ValueError` on empty text, raises `FileNotFoundError` when the model directory does not contain artifacts.
- `test_cli.py`: `--help` works, `train` CLI prints the expected status and artifact paths, `predict` CLI (run after `train`) prints a `Prediction:` line, `evaluate` placeholder runs without crashing.

A representative one — the end-to-end test that would have caught every bug we actually hit during development:

```python
def test_train_classifier_saves_artifacts(tmp_path):
    raw = tmp_path / "sms.tsv"
    raw.write_text(
        "ham\they what's up\n"
        "spam\twin a free iphone now\n"
        "ham\tsee you at 5\n"
        "spam\tclaim your prize today\n",
        encoding="utf-8",
    )
    clean_csv = tmp_path / "clean.csv"
    prepare_training_data(str(raw), str(clean_csv))

    result = train_classifier(str(clean_csv), str(tmp_path / "artifacts"))

    assert result["status"] == "model_saved"
    assert Path(result["model_file"]).exists()
    assert Path(result["vectorizer_file"]).exists()
    assert Path(result["metrics_file"]).exists()
    assert result["best_model"] in {"logistic_regression", "multinomial_nb"}
```

`pytest -q` → **15 passed**.

## 13. Reality Check: 15 Real-World Messages

A held-out test accuracy of 97.77% looks glorious, but you have not really tested a model until you have fed it text your test set has never seen. So we did a sanity check: 15 hand-written modern messages, a mix of obvious spam, obvious ham, and borderline cases.

| # | Message (abbreviated)                                                                        | Expected | Predicted |
|---|----------------------------------------------------------------------------------------------|----------|-----------|
| 1 | "Your bank alert: a $40 charge at Starbucks was authorized on your card."                     | ham      | ham ✅    |
| 2 | "CONGRATULATIONS! You've won a $1000 gift card. Click http://bit.ly/xyz to claim."            | spam     | spam ✅   |
| 3 | "Hey, running 10 mins late, see you at the office."                                          | ham      | ham ✅    |
| 4 | "URGENT: your account has been suspended. Verify immediately: http://secure-login-bank.co"    | spam     | spam ✅   |
| 5 | "Interested in a senior backend role at Stripe? Happy to chat."                              | ham      | ham ✅    |
| 6 | "FREE PIZZA Friday at the office, come to the 4th floor kitchen at 12."                      | ham      | ham ✅    |
| 7 | "WhatsApp Gold is available now! Download the premium version before it's banned."            | spam     | **ham ❌** |
| 8 | "Mom, can you pick up milk on the way home?"                                                 | ham      | ham ✅    |
| … | (15 total)                                                                                   |          |           |

**Final score: 14 / 15 = 93%.**

The one miss — the WhatsApp Gold message — is the single most educational moment of the whole episode. Why did it fail?

Because our training data is **SMS from 2002**. The words `whatsapp`, `download`, `premium`, `banned`, and `gold` in this phishing sense simply do not exist in the UCI corpus. TF-IDF has no vocabulary entry for `whatsapp`. Logistic regression has no weight to pull. The model does the only thing it can do — score the message based on the words it *does* know, which here look pretty benign — and it says "ham".

This is the lesson: **model accuracy numbers describe the test set, not the world.** A 97.77% test accuracy means "of the messages very similar to the ones I trained on, I get 97.77% right". The moment the input distribution shifts — new slang, new product names, a new type of scam — accuracy degrades silently.

## 14. Exercise vs Production: What We Did Not Do

We built this as a learning exercise. If we were shipping this to real users, here is what would be different. Worth listing explicitly, because the gap is where the real engineering lives.

**Data**
- **Exercise**: one small labeled CSV, once.
- **Production**: continuous data collection, a proper labeling workflow (or active learning), class-imbalance handling (SMOTE, class weights, re-sampling), regular re-training as spam patterns evolve.

**Splitting**
- **Exercise**: a single 80/20 stratified split with `random_state=42`.
- **Production**: **k-fold cross-validation** for stable accuracy estimates; a **time-based split** (train on older data, test on newer) to simulate concept drift; a hold-out set that is never touched until final acceptance.

**Features**
- **Exercise**: TF-IDF with default tokenization, English stopwords, 2,000 features.
- **Production**: character n-grams (catches URL-obfuscated spam), domain-specific features (number of links, number of emojis, presence of phone numbers), possibly a pre-trained sentence embedding model as the feature layer.

**Models**
- **Exercise**: logistic regression and multinomial naive Bayes, default hyperparameters, pick the one with higher test accuracy.
- **Production**: hyperparameter search (`GridSearchCV`, `RandomizedSearchCV`, or Optuna), calibrated probabilities, threshold tuning for a target precision/recall, ensembling, and a fallback rule-based layer for high-risk decisions.

**Metrics**
- **Exercise**: accuracy.
- **Production**: **precision, recall, F1, PR-AUC, confusion matrix**, cost-weighted metrics (flagging a real message as spam is much worse than letting spam through, or vice versa, depending on product). Accuracy on an 87/13 class split is borderline useless by itself.

**Serving**
- **Exercise**: `joblib.load` a pickle in a CLI.
- **Production**: a versioned model registry, a feature store so the same transforms run at train and serve time, a stateless inference service (FastAPI + uvicorn), health checks, request/response logging, latency budgets.

**Monitoring**
- **Exercise**: a pytest suite.
- **Production**: data-drift and prediction-drift monitoring, shadow deployments, A/B tests, alerting when the daily spam rate deviates from expectation, periodic ground-truth sampling to catch silent degradation (our WhatsApp Gold miss, at scale).

**Orchestration**
- **Exercise**: `python -m src.cli train ...` from a Makefile.
- **Production**: a DAG in Airflow / Prefect / Dagster that cleans, trains, evaluates, promotes to the registry, and emits metrics on a schedule. Triggered by data landing, not by a human.

You do not need any of that today. You need to understand that everything we *did* build maps to a box in that bigger diagram. When someone later says "add cross-validation", you will know exactly which line to change.

## 15. Glossary

Plain-English definitions of every term this episode uses.

- **Classifier** — a model that takes an input and assigns it a discrete label. Here: a message → `ham` or `spam`.
- **Supervised learning** — training on examples where the correct label is known.
- **Ham / Spam** — the two labels in the SMS Spam Collection. *Ham* is the term of art for "not spam".
- **Training set** — the rows the model learns from.
- **Test set** — rows the model has never seen, used once at the end to estimate real-world performance.
- **Train/test split** — the act of partitioning a labeled dataset into training and test sets.
- **Stratified split** — a split that preserves the class ratio in both parts. Prevents a bad random draw from hiding all the spam in the training set.
- **`random_state`** — a seed for the random number generator. Fixing it makes random operations reproducible.
- **Feature** — a numeric input the model consumes. Here, each TF-IDF column is one feature.
- **Feature engineering** — the craft of turning raw inputs (text, timestamps, categories) into useful numeric features.
- **Bag of words** — representing a document as the multiset of its words, ignoring order. The base representation behind TF-IDF.
- **TF-IDF** — *Term Frequency × Inverse Document Frequency*. Scores a word higher when it is frequent in one document and rare across the corpus.
- **Stopwords** — very common words (*the*, *is*, *a*) removed because they carry no signal.
- **Vocabulary** — the sorted set of tokens the vectorizer has learned from the training data. Each vocabulary entry is one feature column.
- **Sparse matrix** — a matrix where almost all entries are zero, stored compactly. A typical SMS has a handful of words against a 2,000-word vocabulary → 99% zeros.
- **`.fit()`** — learn something from the training data (vocabulary for a vectorizer; weights for a model).
- **`.transform()`** — apply what was learned to new data. Does not update internal state.
- **`.fit_transform()`** — shortcut for `fit` then `transform` on the same data. Only valid on the **training** set.
- **`.predict()`** — produce labels for new inputs using the already-fitted model.
- **Logistic regression** — a linear classifier that predicts the log-odds of a class as a weighted sum of features, then squashes to a probability with the sigmoid function.
- **Multinomial naive Bayes** — a probabilistic text classifier that assumes word occurrences are conditionally independent given the class. Surprisingly strong baseline for short text.
- **Baseline model** — a simple model you train first to set a floor for performance. Any fancier approach must beat it.
- **Accuracy** — fraction of predictions that are correct. Easy to compute, misleading on imbalanced data.
- **Precision** — of the items we predicted positive, how many actually are. (Next episode's topic.)
- **Recall** — of the items that actually are positive, how many we caught.
- **F1** — harmonic mean of precision and recall.
- **Confusion matrix** — a 2×2 (or N×N) grid of true vs predicted labels. Tells you *what kind* of mistakes the model makes.
- **Overfitting** — the model memorizes the training set instead of generalizing. High train accuracy, low test accuracy.
- **Underfitting** — the model is too simple to capture the signal. Low accuracy on both train and test.
- **Concept drift / data drift** — the real-world distribution changes over time (new slang, new scam formats). Yesterday's model degrades without code changing.
- **`joblib`** — a lightweight serializer from the scikit-learn ecosystem. Faster than `pickle` for NumPy arrays; used here to save the model and vectorizer.
- **Artifact** — a file produced by a pipeline step that downstream steps (or production) consume. Here: `best_model.joblib`, `vectorizer.joblib`, `metrics.json`.
- **Pipeline** — a sequence of transformations and a final estimator, chained so that `fit` and `transform` flow in order. We built ours by hand; scikit-learn also has `sklearn.pipeline.Pipeline`.
- **k-fold cross-validation** — split the data into *k* parts, train on *k−1* and test on the remaining one, rotate *k* times, average the scores. Much more stable than a single split.
- **Hyperparameter** — a knob you set *before* training (`max_iter`, `max_features`, `test_size`). Contrasts with parameters the model learns itself.

## Wrapping Up

In Episode 3 we built our first end-to-end classifier:

1. **Data with a license.** The UCI SMS Spam Collection, downloaded, unzipped, and cleaned to 5,158 deduplicated rows.
2. **A cleaning pipeline** that collapses whitespace, normalizes labels, and drops empties and duplicates.
3. **A stratified 80/20 split** with an explicit safety net for the small-dataset edge cases that show up in tests.
4. **TF-IDF vectorization** with English stopwords and a 2,000-feature cap, fit on train and applied to test.
5. **Two baseline models** — logistic regression and multinomial naive Bayes — trained side by side; NB wins at **0.9777** test accuracy.
6. **Serialized artifacts** — model, vectorizer, and metrics — written together so inference is reproducible.
7. **A small prediction module** that validates input, reuses the saved vectorizer, and returns a label.
8. **A CLI** with `train`, `predict`, and a placeholder `evaluate` subcommand.
9. **15 passing tests** covering every layer.
10. **A real-world sanity check** — 14/15 correct on hand-written modern messages, with one miss that perfectly illustrates why held-out accuracy is not the same as production accuracy.

The exercise is small. The skeleton is the same one every production text-classification system has. When you read the next paper or the next service's model card, you will recognize every noun. That is the whole point of building from scratch.

Next up: Episode 4 — proper evaluation metrics (precision, recall, F1, confusion matrix), a Jinja-rendered evaluation report, and the first honest answer to the question "is this model actually good enough to ship?"

AI is as good as our knowledge.
