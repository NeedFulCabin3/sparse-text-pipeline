# Sparse Text Pipeline

> Memory-efficient text classification pipeline using hashing vectorization and custom scikit-learn feature transformers

## Overview
In natural language processing systems handling large-scale or streaming corpora, traditional vocabulary indexing scales linearly with unique token counts, frequently triggering out-of-memory errors during model training. This tool solves that operational friction by implementing a stateless text classification pipeline that bypasses explicit dictionary fitting entirely while extracting both semantic and structural textual features.

## How It Works
The execution pipeline relies on scikit-learn's object-oriented estimator framework to process text documents through a multi-stage architecture:
1. **Stateless Feature Extraction (`FeatureUnion`):** The input corpus is split across parallel transformers. The `HashingVectorizer` maps token unigrams and bigrams directly into fixed-size numeric indices using murmurhash3 without storing vocabulary dictionaries. Concurrently, `TextLengthTransformer` and `PunctuationDensityTransformer` compute structural text attributes.
2. **Numerical Scaling:** Custom structural features pass through `StandardScaler` configured with `with_mean=False` to preserve sparse matrix integrity.
3. **Multiclass Linear Classification:** The unified sparse matrix feeds into a `OneVsRestClassifier` wrapping an `SGDClassifier` optimized with logarithmic loss and L2 penalty.

## Key Features
- **Stateless Vectorization:** Eliminates memory overhead associated with vocabulary lookup tables.
- **Custom Structural Transformers:** Integrates domain-specific text metrics (character length, punctuation density) into the vector space.
- **Robust Multiclass Wrapper:** Scales binary stochastic gradient descent classifiers across multiple discrete target labels.
- **Pipeline Reproducibility:** Encapsulates preprocessing, feature union, scaling, and estimation into a single serializable object.

## Tech Stack & Core Dependencies Breakdown
- **Python 3.10+:** Target runtime environment utilizing modern type hinting and standard library modules (`string`, `sys`).
- **NumPy (`>=1.22.0`):** Provides high-performance multidimensional array operations for custom transformers.
- **scikit-learn (`>=1.1.0`):** Powers the pipeline architecture, feature unions, text vectorization, scaling, and linear classification models.

## Environment & Web-Based Quick Start

### Running via GitHub Codespaces
1. Navigate to the repository main page on GitHub.
2. Click the **Code** dropdown and select the **Codespaces** tab.
3. Click **Create codespace on main** to spin up a pre-configured cloud development container.
4. Once the terminal initializes, run the setup commands below.

### Local or Codespace Virtual Environment Setup
```bash
# Clone or open the repository workspace
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install numpy scikit-learn

# Execute the classification pipeline
python main.py
```

## Repository Structure

```bash
sparse-text-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── LICENSE
├── README.md
└── main.py
```

## Roadmap

**[ ] Type Stubs & Static Analysis:** Introduce strict type annotations across custom BaseEstimator subclasses and integrate mypy checking into the CI workflow.

**[ ] Async I/O Batch Streaming:** Implement asynchronous file readers and generator-based batch ingestion for multi-gigabyte text corpora.

**[ ] Structural Logging:** Replace standard print statements with structured JSON logging to track pipeline execution timings and feature matrix dimensions.