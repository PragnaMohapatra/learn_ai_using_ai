# Episode 3 — Learn AI by Doing It: Your First ML Model

*Train a spam classifier with scikit-learn — and pop the hood on every single number the model uses to make a prediction.*

**Source code:** [github.com/PragnaMohapatra/learn_ai_using_ai/tree/main/episode-03](https://github.com/PragnaMohapatra/learn_ai_using_ai/tree/main/episode-03)

---

## Why DIY Projects in the Age of AI

Large language models write production code, generate test suites, and draft documentation in seconds. Deep mastery of one framework is no longer the differentiator. **Knowing what to ask for** — and understanding the code the AI hands back — is.

Episode 1 ingested and cleaned raw data. Episode 2 profiled it and layered on an opt-in AI narrative. Episode 3 takes the next step: **we train our first real ML model.** A spam classifier. No deep learning. No GPU. Just `pandas`, `scikit-learn`, and the discipline of building one layer at a time so every function earns its place.

---

## What We'll Cover

1. Why start with a classifier
2. Project structure
3. Docker & infrastructure
4. Get real, licensed data
5. Load & clean the labeled dataset
6. Train/test split & stratification
7. TF-IDF: text → numbers
8. Two baseline models, side-by-side
9. Save the artifacts
10. The prediction module
11. The CLI entry point
12. Tests (all 15 of them)
13. **Under the hood: the math, end-to-end**
14. Reality check: 15 real-world messages
15. Exercise vs production
16. Glossary

---

## 1. Why Start with a Classifier

Classification is the "hello world" of supervised ML. Inputs (messages, reviews, emails) → labels (ham/spam, positive/negative, fraud/legit). The full pipeline that powers much larger systems — data loading, feature engineering, train/test discipline, model selection, serialization, inference — is already present in the smallest classifier.

Build a spam classifier properly and the shape of the code carries straight to a next-token predictor, a review scorer, or a churn model. Only the math inside one call to `.fit()` changes.

---

## 2. Project Structure

```
episode-03/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
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
    ├── sms_spam_clean.csv      # cleaned
    └── artifacts/
        ├── best_model.joblib
        ├── vectorizer.joblib
        └── metrics.json
```

No notebook, no dashboard, no model checked into git. The deliverables are reproducible artifacts in `data/artifacts/`.

---

## 3. Docker & Infrastructure

Same pattern as Episode 2. Docker is not ceremony — it is how a collaborator six months from now gets the same NumPy version that produced your 97% accuracy.

**Dockerfile**

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

**docker-compose.yml**

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

**requirements.txt**

```
pandas==2.2.3
scikit-learn==1.5.2
joblib==1.4.2
jinja2==3.1.4
pytest==8.3.5
```

---

## 4. Get Real, Licensed Data

Toy datasets teach toy lessons. We grab the **UCI SMS Spam Collection** (CC BY 4.0):

```bash
curl -L -o data/sms_spam_collection.zip \
  https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
unzip data/sms_spam_collection.zip -d data/
```

Tab-separated, two columns: `label` (ham/spam) and `text`. 5,574 messages. ~13% spam. Real UK SMS traffic.

> **Heads up:** this dataset is from 2002. No WhatsApp, no PayPal, no crypto. A model trained on it is a time capsule — fine for learning, dangerous for shipping. We revisit this in section 14.

---

## 5. Load & Clean the Labeled Dataset

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
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    cleaned["label"] = cleaned["label"].astype("string").str.strip().str.lower()
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

Validate columns · collapse whitespace · lowercase labels · drop empties & exact duplicates. After dedup we go from 5,574 rows to **5,158**.

---

## 6. Train/Test Split & Stratification

> **"What is `stratify` doing, exactly?"**
> On a dataset that is 87% ham / 13% spam, a random 80/20 split can — by bad luck — produce a test set that is 95% ham. Your "accuracy" then just rewards predicting ham always. Stratification *preserves the class ratio in both splits*.

> **"And `random_state`?"**
> A seed for the RNG. Fixing it makes the random split reproducible across runs.

> **"What does `.fit()` actually do?"**
> Vectorizer → scans training text, builds the vocabulary, computes document frequencies. Model → learns parameters (weights / probabilities). `.transform()` and `.predict()` just *apply* what was learned.

```python
from math import ceil
from sklearn.model_selection import train_test_split

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

def split_training_data(dataframe, *, test_size=DEFAULT_TEST_SIZE,
                       random_state=DEFAULT_RANDOM_STATE):
    X, y = dataframe["text"], dataframe["label"]
    stratify = None
    class_count = int(y.nunique())
    test_rows = ceil(len(dataframe) * test_size)
    train_rows = len(dataframe) - test_rows
    if (class_count > 1 and y.value_counts().min() >= 2
        and test_rows >= class_count and train_rows >= class_count):
        stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    return {"X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "stratified": stratify is not None}
```

---

## 7. TF-IDF: Text → Numbers

Models do not read English. They consume vectors of floats. **TF-IDF** = *Term Frequency × Inverse Document Frequency*:

- **TF** — a word that appears often in *this* message gets a higher score.
- **IDF** — a word that appears in almost every message ("the", "is") gets downweighted.
- **TF × IDF** — high when a word is frequent here but rare across the corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(X_train, X_test, *, max_features=2000):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X_train_features = vectorizer.fit_transform(X_train)   # learn + apply
    X_test_features  = vectorizer.transform(X_test)        # apply only
    return {"vectorizer": vectorizer,
            "X_train_features": X_train_features,
            "X_test_features": X_test_features}
```

> **Golden rule:** `fit_transform` on train, `transform` on test. If you `fit` on test, your "held-out" accuracy is a lie — the vocabulary has already seen the answers.

---

## 8. Two Baseline Models, Side-by-Side

Never train one model. You cannot tell whether the number you got is good without a reference.

- **Logistic Regression** — linear weights per word; sum them; squash with sigmoid → P(spam).
- **Multinomial Naive Bayes** — estimate P(word | spam) from counts; apply Bayes' rule at predict time.

| Model                  | Test accuracy |
|------------------------|---------------|
| Logistic Regression    | 0.9641        |
| **Multinomial NB**     | **0.9777** ← winner |

---

## 9. Save the Artifacts

```python
joblib.dump(best_model_obj,  output_path / "best_model.joblib")
joblib.dump(vec["vectorizer"], output_path / "vectorizer.joblib")
metrics_file.write_text(json.dumps(metrics, indent=2))
```

> **The thing most tutorials get wrong:** save the vectorizer *with* the model. Feature `#417` means "the word *won*" only because the vectorizer's vocabulary says so. Load the model without the vectorizer and your predictions are random noise.

---

## 10. The Prediction Module

```python
def load_artifacts(model_dir):
    p = Path(model_dir)
    return {"model": joblib.load(p / "best_model.joblib"),
            "vectorizer": joblib.load(p / "vectorizer.joblib")}

def predict_text(model_dir, text):
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")
    a = load_artifacts(model_dir)
    features = a["vectorizer"].transform([text])   # same vocabulary as training
    return {"prediction": str(a["model"].predict(features)[0])}
```

---

## 11. The CLI Entry Point

```bash
python -m src.cli train   data/sms_spam_clean.csv data/artifacts
python -m src.cli predict data/artifacts "WIN a FREE iPhone now!!! Click here"
# → Prediction: spam
```

---

## 12. Tests — All 15 of Them

15 tests across `test_training.py`, `test_prediction.py`, `test_cli.py`. Happy paths, empty-text errors, missing-artifact errors, CLI subprocess integration. `pytest -q` → **15 passed**.

---

## 13. Under the Hood: The Math, End-to-End

Everything up to here is plumbing. This section is the payoff: we take one real test message and walk it through every number the model touches — **TF-IDF storage, the fitted model's internal state, the Bayes rule computation, and the final probability**. Every value below is copy-pasted from a real run on the trained model.

### 13.1  How TF-IDF Is Stored

After `vectorizer.fit_transform(X_train)`, the returned object is a **sparse CSR matrix** — not a dense NumPy array. For our 4,126 training messages:

```
X_train_features.shape = (4126, 2000)     # 4126 rows · 2000 features
X_train_features.dtype = float64
type                   = scipy.sparse.csr_matrix
non-zero entries       = 25,070            # out of 4126 × 2000 = 8,252,000
density                = 0.00304           # ~0.3% filled — 99.7% zeros
```

If we stored 8.25M floats densely we would burn ~66 MB of RAM for a dataset that barely needs 200 KB. Sparse storage keeps only the non-zero `(row, col, value)` triples. CSR (Compressed Sparse Row) goes further — it groups values by row so `matrix[i]` is an O(1) slice, which is exactly what scikit-learn needs during `.fit()`.

### 13.2  What the Vectorized Data Actually Looks Like

Pick one real training row. Its text and every non-zero TF-IDF value:

```
text:
  "Sorry i cant take your call right now. It so happens that there r 2waxsto
   do wat you want. She can come and ill get her..."

non-zero features (23 out of 2000):
   idx   token          tf-idf
   ----  -----------    -------
    119  able           0.1534
    233  basic          0.1858
    350  care           0.1267
    413  come           0.0998
    471  currently      0.1622
    507  deliver        0.1858
    703  friday         0.1669
    774  guide          0.1763
    794  happens        0.1727
    875  ill            0.1379
    898  insurance      0.5575   ← highest (rare word → high IDF)
    945  just           0.0868
   1027  ll             0.1935
   1106  medical        0.3527
   1157  morning        0.1263
   1288  person         0.1420
   1462  right          0.3662
   1550  shopping       0.1411
   1613  sorry          0.1135
   1735  thats          0.1372
   1756  til            0.1459
   1885  want           0.1021
   1894  wat            0.1166
   ...
   1,977 other features = 0.0
```

Notice two things. First, **the vast majority of columns are zero**. Second, `insurance` gets the highest weight (0.5575) not because it appears many times — it appears once — but because it is *rare across the corpus*. IDF is doing its job.

### 13.3  What the Fitted Model Looks Like

After `model.fit(X_train_features, y_train)`, Multinomial Naive Bayes stores three things:

```
nb.classes_          = ['ham', 'spam']
nb.class_log_prior_  = [-0.1330, -2.0828]
nb.feature_log_prob_ = array of shape (2, 2000)   # log P(word | class)
```

The priors reverse-engineer to:

```
P(ham)  = exp(-0.1330) = 0.875    # 87.5% of training = ham
P(spam) = exp(-2.0828) = 0.125    # 12.5% of training = spam
```

`feature_log_prob_` is the real meat: for every (class, word) pair, the log probability of seeing that word given that class. We'll use it in 13.5.

### 13.4  A Test Message Meets the Vectorizer

Run a fresh message through the **exact same vectorizer** — using `.transform()`, never `.fit_transform()`:

```
text = "WIN a FREE iPhone now!!! Click http://bit.ly/xyz to claim your prize"
```

Three things happen in order: tokenize → drop stopwords → look up each token in the training vocabulary → compute TF-IDF. Tokens not in the 2,000-word vocabulary (`iphone`, `xyz`) are silently dropped. What survives:

| idx  | token | TF-IDF |
|------|-------|--------|
| 264  | bit   | 0.3747 |
| 391  | claim | 0.3301 |
| 397  | click | 0.4939 |
| 695  | free  | 0.2791 |
| 853  | http  | 0.4256 |
| 1369 | prize | 0.3475 |
| 1930 | win   | 0.3562 |

Seven features. The other 1,993 columns are zero. Those seven numbers are the entire fingerprint the model sees.

### 13.5  The Math — Naive Bayes, Step by Step

Bayes' rule for classification:

```
P(class | x) ∝ P(class) · ∏ᵢ P(wordᵢ | class)^xᵢ
```

Take logs to keep numbers sane:

```
log P(class | x) = log P(class) + Σᵢ xᵢ · log P(wordᵢ | class) + const
```

For our seven non-zero features, `feature_log_prob_` gives us:

| Token | TF-IDF (xᵢ) | log P(w given ham) | log P(w given spam) | Δ (spam − ham) |
|-------|-------------|--------------------|---------------------|----------------|
| bit   | 0.3747      | -6.549             | -8.208              | **-1.659**     |
| claim | 0.3301      | -9.168             | -5.339              | **+3.828**     |
| click | 0.4939      | -8.540             | -7.405              | **+1.135**     |
| free  | 0.2791      | -6.448             | -4.838              | **+1.610**     |
| http  | 0.4256      | -9.168             | -6.425              | **+2.743**     |
| prize | 0.3475      | -9.168             | -5.446              | **+3.721**     |
| win   | 0.3562      | -8.049             | -5.785              | **+2.264**     |

Six of seven features lean spam. `claim`, `prize`, and `http` are overwhelming — they almost never occur in ham, which makes `log P(word | ham)` extremely negative. Only `bit` tilts toward ham (it appears in casual messages like "a bit later").

Now plug it all into Bayes:

```
log P(ham  | x) = -0.1330  +  Σ xᵢ · log P(wᵢ | ham)   =  -21.586
log P(spam | x) = -2.0828  +  Σ xᵢ · log P(wᵢ | spam)  =  -18.617
```

Spam has the larger log-probability (less negative). Softmax to turn the two logs into proper posteriors:

```
P(ham  | x) = 0.0488    (4.88%)
P(spam | x) = 0.9512    (95.12%)
argmax      = 'spam'
```

> **Verification.** These numbers match `nb.predict_proba([...])` exactly: `[0.04884, 0.95116]`. Not approximately — to the 14th decimal place. *That* is what it means to understand your model.

### 13.6  The Math — Logistic Regression, Same Message

Different model, same input. Logistic regression stores one weight per feature plus an intercept:

```
lr.intercept_ = -2.6070          # bias toward ham
lr.coef_      = array of shape (1, 2000)   # positive → pushes toward 'spam'
```

For each non-zero feature, multiply `coef × tfidf` and sum:

| Token | TF-IDF | coef (spam) | product |
|-------|--------|-------------|---------|
| bit   | 0.3747 | -0.493      | -0.185  |
| claim | 0.3301 | +3.309      | +1.092  |
| click | 0.4939 | +0.436      | +0.216  |
| free  | 0.2791 | +3.025      | +0.844  |
| http  | 0.4256 | +1.825      | +0.777  |
| prize | 0.3475 | +2.524      | +0.877  |
| win   | 0.3562 | +2.292      | +0.816  |

```
z = intercept + Σ (coefᵢ · xᵢ) = -2.607 + 4.437 = 1.8303
P(spam | x) = σ(z) = 1 / (1 + e^-1.8303) = 0.8618   (86.18%)
```

Both models agree: **spam**. NB is more confident (95%) because it treats the seven "spammy" tokens as multiplicative evidence; LR treats them as additive log-odds. Two different worldviews, same answer.

> **This is the whole of "doing ML" in one test message:** text → tokenized → mapped to a sparse 2,000-dim TF-IDF vector → dotted against either log-probability tables (NB) or weight vectors (LR) → passed through softmax or sigmoid → argmax → label. Every production spam filter, every review classifier, every toxicity detector is a variation on this same machinery.

---

## 14. Reality Check: 15 Real-World Messages

A 97.77% test accuracy looks glorious. But you have not really tested a model until you hand it text your test set has never seen. We wrote 15 modern, hand-crafted messages and ran them through `predict_text`.

| # | Message (abbreviated) | Expected | Predicted |
|---|------------------------|----------|-----------|
| 1 | "Your bank alert: a $40 charge at Starbucks was authorized on your card." | ham | ham ✓ |
| 2 | "CONGRATULATIONS! You've won a $1000 gift card. Click http://bit.ly/xyz to claim." | spam | spam ✓ |
| 3 | "Hey, running 10 mins late, see you at the office." | ham | ham ✓ |
| 4 | "URGENT: your account has been suspended. Verify immediately: http://secure-login-bank.co" | spam | spam ✓ |
| 5 | "Interested in a senior backend role at Stripe? Happy to chat." | ham | ham ✓ |
| 6 | "FREE PIZZA Friday, 4th floor kitchen at 12." | ham | ham ✓ |
| 7 | "WhatsApp Gold is available now! Download the premium version before it's banned." | spam | **ham ✗** |
| 8 | "Mom, can you pick up milk on the way home?" | ham | ham ✓ |
| … | (15 total) | | |

**Final score: 14 / 15 = 93%.**

The WhatsApp Gold miss is the single most educational moment of the episode. Our training data is **SMS from 2002**. The words `whatsapp`, `download`, `premium`, `banned`, `gold` (in this phishing sense) do not exist in the UCI corpus — so they are silently dropped in 13.4 style. The model scores the message using the words it *does* know, which look benign. It says ham.

> **The lesson:** model accuracy describes the test set, not the world. 97.77% means "of messages very similar to training, I get 97.77% right". When the input distribution shifts — new slang, new product names, a new scam — accuracy degrades silently. This is *concept drift*, and monitoring for it is its own discipline.

---

## 15. Exercise vs Production: What We Did Not Do

Every choice we made has a production counterpart. Worth listing, because the gap is where the real engineering lives.

| Area | What we did (exercise) | Production |
|------|------------------------|------------|
| **Data** | One labeled CSV, once. | Continuous collection, labeling pipeline, class-imbalance handling (SMOTE, class weights), scheduled re-training. |
| **Splitting** | Single 80/20 stratified split, `random_state=42`. | k-fold CV for stable estimates; time-based splits to simulate drift; a final hold-out set never touched until acceptance. |
| **Features** | TF-IDF, English stopwords, 2,000 features. | Character n-grams (obfuscated URLs), domain features (link count, emoji count, phone numbers), sentence embeddings. |
| **Models** | LR + NB at defaults, pick higher accuracy. | `GridSearchCV` / Optuna, calibrated probabilities, threshold tuning, ensembling. |
| **Metrics** | Accuracy. | Precision, recall, F1, PR-AUC, confusion matrix, cost-weighted metrics. Accuracy on 87/13 is borderline useless alone. |
| **Serving** | `joblib.load` in a CLI. | Versioned model registry, feature store, stateless inference service (FastAPI/uvicorn), health checks, latency SLOs. |
| **Monitoring** | pytest. | Data/prediction drift monitoring, shadow deployments, A/B tests, alerting on daily spam rate anomalies. |
| **Orchestration** | `make train`. | Airflow / Prefect / Dagster DAG — clean → train → evaluate → promote, triggered by data landing. |

You don't need any of that today. You need to know that everything we *did* build maps to a box in that larger picture.

---

## 16. Glossary

**Classifier** — A model that maps an input to a discrete label. Here: message → ham / spam.

**Supervised learning** — Training on examples where the correct label is known.

**Ham / Spam** — The two labels in the SMS Spam Collection. *Ham* is the term of art for "not spam".

**Train / Test split** — Partition the labeled data — most goes to training, a held-out slice estimates real-world performance.

**Stratified split** — A split that preserves the class ratio in both parts.

**random_state** — A seed for the RNG. Fix it for reproducible splits.

**Feature** — A numeric input the model consumes. Here, each TF-IDF column is one feature.

**Bag of words** — Representing a document as the multiset of its words, ignoring order.

**TF-IDF** — *Term Frequency × Inverse Document Frequency*. Scores a word higher when frequent here and rare across the corpus.

**Stopwords** — Very common words (*the, is, a*) dropped because they carry no signal.

**Vocabulary** — The sorted set of tokens the vectorizer learned from training. Each entry is one feature column.

**Sparse matrix / CSR** — A matrix where almost all entries are zero, stored compactly. CSR groups non-zeros by row for fast slicing.

**.fit()** — Learn from training data (vocabulary for a vectorizer; weights / probabilities for a model).

**.transform()** — Apply what was learned to new data. No internal state changes.

**.fit_transform()** — Shortcut for fit then transform on the same data. Only valid on **training**.

**.predict()** — Produce labels for new inputs using the already-fitted model.

**Logistic regression** — Predicts log-odds as a weighted sum of features, squashed with the sigmoid to a probability.

**Multinomial naive Bayes** — Estimates P(word | class) from training counts and applies Bayes' rule at predict time. Surprisingly strong on short text.

**Prior / Posterior** — *Prior* P(class) is the class's frequency before seeing the message. *Posterior* P(class | x) is the updated belief after seeing it.

**Softmax / Sigmoid** — Functions that turn raw scores into probabilities. Softmax generalizes sigmoid to more than two classes.

**Baseline model** — A simple model trained first to set a performance floor.

**Accuracy** — Fraction of predictions that are correct. Misleading on imbalanced data.

**Precision / Recall / F1** — Metrics that separate "how clean were my positive predictions" from "how many positives did I catch". Topic of Episode 4.

**Confusion matrix** — An N×N grid of true vs predicted labels, showing *what kind* of mistakes the model makes.

**Overfitting / Underfitting** — Memorizing training data vs being too simple to capture signal.

**Concept drift / Data drift** — The world changes; yesterday's model degrades without code changing.

**joblib / Artifact** — `joblib` is a fast serializer for NumPy-heavy Python objects. An *artifact* is a file a pipeline step emits for the next step to consume.

**k-fold cross-validation** — Rotate through *k* partitions, averaging scores. Much more stable than a single split.

**Hyperparameter** — A knob you set *before* training (`max_iter`, `max_features`, `test_size`).

---

## Wrapping Up

In Episode 3 we built an end-to-end classifier — licensed data, cleaning, stratified split, TF-IDF, two baseline models, serialized artifacts, prediction module, CLI, 15 passing tests — and then pried open the fitted model so the final prediction was no longer magic but arithmetic. The exercise is small; the skeleton is the one every production text-classification system has.

Next up: **Episode 4 — proper evaluation metrics** (precision, recall, F1, confusion matrix), a Jinja-rendered report, and the first honest answer to "is this model actually good enough to ship?"

*AI is as good as our knowledge.*
