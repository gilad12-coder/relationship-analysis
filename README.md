# Relationship Analysis

Automatically classify relationship types in text using LLM-optimised prompts. Given a text snippet, the pipeline independently classifies it against six relationship labels — producing a multi-label binary output.

**Why?** Manually crafting classification prompts is slow and brittle. This pipeline uses [DSPy](https://dspy.ai/) with [GEPA](https://github.com/gepa-ai/gepa) (Genetic-Pareto) to evolve optimal prompts per label through a grid search over generation and reflection model pairs.

### Highlights

- **6 binary classifiers** — romantic, family, friendship, professional, unknown, irrelevant
- **Automatic prompt optimisation** via GEPA (Genetic-Pareto evolutionary search)
- **Grid search** across configured generation/reflection model pairs
- **Statistically sized selection holdout sets** — per-label holdout size targets enough positives for reliable F1 confidence intervals when data allows
- **Dual split support** — identical locked selection holdout sets for both DSPy prompt tuning and transformer fine-tuning
- **Serving endpoint** — FastAPI server loads optimised programs and classifies text via REST API

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Dataset](#dataset)
- [Training](#training)
- [Serving](#serving)
- [Pipeline Overview](#pipeline-overview)
- [Data Splitting](#data-splitting)
- [Project Structure](#project-structure)
- [Output](#output)
- [Configuration](#configuration)

---

## Prerequisites

- Python 3.10+
- OpenAI API key

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy the `.env` file and fill in your keys:

```bash
cp .env .env.local  # optional: keep a local override
```

Edit `.env`:

```
OPENAI_API_KEY=sk-...
LM_BASE_URL=              # leave empty for default OpenAI endpoint
```

## Dataset

Place your dataset at the path specified by `DATASET_PATH` in `training/config/settings.py` (default: `data/relationships.csv`).

**Supported formats:** `.xlsx`, `.csv`

**Required columns:**

| Column | Type | Description |
|---|---|---|
| `text` | string | The text to classify |
| `romantic` | bool | Romantic relationship present |
| `family` | bool | Family relationship present |
| `friendship` | bool | Friendship present |
| `professional` | bool | Professional relationship present |
| `unknown` | bool | Relationship present but type unclear |
| `irrelevant` | bool | No relationship content |

Label columns accept any of these as truthy: `True`, `"true"`, `"yes"`, `"YES"`, `"1"`, `1`.

## Quick Start

1. **Explore the data** — open `analysis.ipynb` and run the pre-training cells to inspect label distributions, class balance, and text characteristics.
2. **Train** — open `train.ipynb` and run all cells. This runs the full grid search (54 trials: 3 gen × 3 reflection × 6 labels), evaluates the best program per label on a locked selection holdout set, saves programs, and smoke-tests the API.
3. **Analyse results** — go back to `analysis.ipynb` and run the post-training cells for grid search heatmaps, confusion matrices, and error analysis.
4. **Serve** — run `python serve.py` to start the API.

## Training

**Recommended: use the notebook.** Open `train.ipynb` and run all cells — it walks through every pipeline step interactively and ends with API smoke tests.

Alternatively, use the CLI:

```bash
# Full grid search across configured model pairs
python train.py

# Override GEPA iteration budget
python train.py --auto heavy

# Skip grid search — use a fixed model pair
python train.py --gen_model openai/gpt-4o --reflection_model openai/o4-mini
```

After training, optimised programs are saved to `training/results/programs/` for serving.

## Serving

Start the FastAPI server (requires a completed training run):

```bash
python serve.py
```

The server starts on `http://localhost:8000`.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/classify` | Classify a text against all 6 labels |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive API documentation (Swagger UI) |

**Example request:**

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "She held his hand as they walked through the park"}'
```

**Example response:**

```json
{
  "labels": ["romantic"],
  "details": {
    "romantic": true,
    "family": false,
    "friendship": false,
    "professional": false,
    "unknown": false,
    "irrelevant": false
  }
}
```

## Pipeline Overview

```
 Dataset (.xlsx/.csv)
        |
        v
  1. Load & normalise labels
        |
        v
  2. Split per label ──> locked selection holdout set (shared)
        |                       |
        v                       v
   DSPy splits           Transformer splits
   (train / val)         (trainval only)
        |
        v
  3. Build dspy.Example objects
        |
        v
  4. GEPA grid search (or fixed pair)
     ┌──────────────────────────────────┐
     │  For each (gen_model, refl_model):│
     │    Compile ChainOfThought        │
     │    Score on val set (F1)         │
     │    Select best per label         │
     └──────────────────────────────────┘
        |
        v
  5. Evaluate best program per label
     on locked selection holdout set (once)
        |
        v
  6. Save optimised programs to disk
        |
        v
   training/results/
```

**Steps in detail:**

1. **Load** the dataset and normalise label columns to booleans.
2. **Split** the data per label. The selection holdout set is locked first from label statistics alone, then DSPy train/val splits and transformer trainval pools are produced from the remaining data.
3. **Build** `dspy.Example` objects from the DSPy splits.
4. **Optimise** prompts via GEPA grid search (or a fixed model pair). Each trial compiles a `ChainOfThought` program on the training set and scores it on the validation set.
5. **Evaluate** the best program per label on the locked selection holdout set exactly once.
6. **Save** the optimised programs to `training/results/programs/` with a manifest mapping each label to its selected models and dataset hash.

## Data Splitting

The split pipeline computes a per-label holdout size that targets enough positive examples for a reliable F1 confidence interval, and warns when the dataset is too sparse to meet that target. Both DSPy and transformer splits share the same locked selection holdout set:

- **DSPy splits** — large train (for GEPA reflection sampling), small val (for Pareto tracking).
- **Transformer splits** — full trainval pool only (you decide downstream how to split or use it).

Access via the `Splits` dataclass (dot notation):

```python
from training.data.split_pipeline import run

splits = run(df)

splits.romantic.holdout                   # shared selection holdout set
splits.romantic.dspy.train                # GEPA train
splits.romantic.dspy.val                  # GEPA val
splits.romantic.transformer.trainval      # full transformer train pool

# Key access also works for loops:
for label in LABELS:
    splits[label].holdout
```

## Project Structure

```
train.py                             Training entry point
serve.py                             Serving entry point
training/
    config/
        settings.py                  Tuneable pipeline settings
        constants.py                 Fixed project-wide constants
    labels/
        signatures.py                DSPy Signature classes (one per label)
        metrics.py                   GEPA metric functions and label registry
    main.py                          Training pipeline orchestration
    data/
        split_pipeline.py            Data splitting (both DSPy and transformer)
        dataset_builder.py           DataFrame -> dspy.Example conversion
    optimization/
        optimize.py                  Single GEPA run for one (label, model pair) trial
        grid_search.py               Grid search over all model pairs and labels
    evaluation/
        evaluate.py                  Final holdout-set evaluation
serving/
    config/
        settings.py                  Serving-specific settings
    app.py                           FastAPI application and endpoints
    main.py                          Uvicorn runner
analysis.ipynb                       Pre/post-training analysis notebook
train.ipynb                          Interactive training notebook
```

## Output

Results are written to the `training/results/` directory:

| File | Description |
|---|---|
| `grid_search_results.csv` | All trial results with val F1 scores |
| `evaluation.csv` | Holdout metrics for baseline and optimized runs (`stage` column) |
| `predictions/{stage}_{label}_predictions.csv` | Per-example predictions with gold, predicted, and correctness |
| `programs/{label}/` | Serialised optimised program per label (for serving) |
| `programs/manifest.json` | Maps each label to `gen_model`, `reflection_model`, and `dataset_hash` |
| `pipeline.log` | Full debug log |

## Configuration

All tuneable settings live in `training/config/settings.py`. Key options:

| Setting | Default | Description |
|---|---|---|
| `LABELS` | 6 relationship types | Classification targets |
| `GENERATION_MODELS` | gpt-4o-mini, gpt-4o, gpt-4.1-mini | Models used for generation |
| `REFLECTION_MODELS` | gpt-4o, o4-mini, gpt-4.1 | Models used for GEPA reflection |
| `GEPA_AUTO` | `"heavy"` | GEPA iteration budget (`light` / `medium` / `heavy`) |
| `CONFIDENCE_LEVEL` | 0.95 | Confidence level for holdout-set F1 CI sizing |

Fixed constants (not tuneable) live in `training/config/constants.py`.
