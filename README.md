# VespaTune

No code tool for training tabular models.

<div align="center">[![Deploy on Hugging Face Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-on-spaces-md.svg)](https://huggingface.co/spaces/vespa-engine/vespatune?duplicate=true)</div>

- **Web UI** for training, monitoring, and managing models
- Tune models directly from CSV files
- Real-time training progress with WebSocket updates
- Export models to ONNX format for deployment

## Installation

Install using pip:

```bash
pip install vespatune
```

## Quick Start

### Web UI (Recommended)

Start the web interface:

```bash
vespatune
```

This launches the VespaTune UI at `http://127.0.0.1:9999` where you can:
- Upload train/validation CSV files
- Configure model type, target columns, and hyperparameters
- Start training with real-time progress monitoring
- View trial results and metrics
- Download trained models and artifacts
- Manage multiple training runs

You can also specify host and port:

```bash
vespatune --host 0.0.0.0 --port 8080
```

### CLI

Train a model with explicit train/valid split:

```bash
vespatune train \
  --train_filename train.csv \
  --valid_filename valid.csv \
  --output outputs/my_model \
  --model xgboost
```

Or let VespaTune auto-split your data:

```bash
vespatune train \
  --train_filename data.csv \
  --output outputs/my_model \
  --model xgboost
```

Make predictions:

```bash
vespatune predict \
  --model_path outputs/my_model \
  --test_filename test.csv \
  --output_filename predictions.csv
```

Serve a trained model for predictions:

```bash
vespatune serve --model_path outputs/my_model --host 0.0.0.0 --port 8000
```

### Python API

```python
from vespatune import VespaTune

# With explicit validation file
vtune = VespaTune(
    train_filename="train.csv",
    valid_filename="valid.csv",
    output="outputs/my_model",
    model_type="xgboost",  # or "lightgbm" or "catboost"
    targets=["target"],
    num_trials=100,
    time_limit=3600,
)
vtune.train()

# Or with auto-split (no validation file needed)
vtune = VespaTune(
    train_filename="data.csv",
    output="outputs/my_model",
    model_type="xgboost",
    targets=["target"],
    num_trials=100,
)
vtune.train()
```

## Web UI Features

The web interface provides:

- **File Upload**: Drag and drop CSV files for training (validation file is optional)
- **Auto-Split**: If no validation file is provided, automatically splits training data
- **Auto Column Detection**: Automatically detects columns for target and ID selection
- **Model Selection**: Choose between XGBoost, LightGBM, or CatBoost
- **Real-time Monitoring**: Watch training progress with live trial updates via WebSocket
- **Metrics Visualization**: View loss curves and hyperparameter importance
- **Run Management**: Start, stop, and delete training runs
- **Artifact Downloads**: Download trained models, configs, and ONNX exports

## Parameters

### Required

| Parameter | Description |
|-----------|-------------|
| `train_filename` | Path to training CSV file |
| `output` | Path to output directory for model artifacts |

### Optional

| Parameter | Default | Description |
|-----------|---------|-------------|
| `valid_filename` | `None` | Path to validation CSV file (auto-splits training data if not provided) |
| `model_type` | `"xgboost"` | Model to use: `"xgboost"`, `"lightgbm"`, `"catboost"`, or `"logreg"` |
| `test_filename` | `None` | Path to test CSV file (predictions saved if provided) |
| `task` | `None` | `"classification"` or `"regression"` (auto-detected if not specified) |
| `idx` | `"id"` | Name of the ID column |
| `targets` | `["target"]` | List of target column names |
| `features` | `None` | List of feature columns (all non-id/target columns if not specified) |
| `categorical_features` | `None` | List of categorical columns (auto-detected if not specified) |
| `use_gpu` | `False` | Whether to use GPU for training |
| `seed` | `42` | Random seed for reproducibility |
| `num_trials` | `1000` | Number of Optuna trials for hyperparameter tuning |
| `time_limit` | `None` | Time limit for optimization in seconds |

## Supported Models

### XGBoost
- Default model with extensive hyperparameter search
- Supports GPU acceleration
- Best for general-purpose tasks

### LightGBM
- Native categorical feature support
- Fast training on large datasets
- Supports GPU acceleration

### CatBoost
- Best native categorical feature handling
- Robust to overfitting
- Supports GPU acceleration

### Logistic Regression
- Linear model for classification tasks only
- Searches over preprocessing (imputation, scaling) and regularization
- Fast training, interpretable coefficients

## Data Splitting

VespaTune supports two modes:

1. **Explicit split**: Provide both `train_filename` and `valid_filename`
2. **Auto-split**: Provide only `train_filename` - VespaTune automatically creates a 5-fold split and uses fold 0 (80% train, 20% valid)

For manual control over splits, use the splitter utility:

```bash
vespatune splitter \
  --data_filename data.csv \
  --output splits/ \
  --target target \
  --task classification \
  --num_folds 5
```

Or via Python:

```python
from vespatune import VespaTuneSplitter

splitter = VespaTuneSplitter(
    data_filename="data.csv",
    output="splits/",
    target="target",
    task="classification",
    num_folds=5,
)
splitter.split()
```

This creates `fold_0_train.csv`, `fold_0_valid.csv`, etc. for k-fold cross-validation.


## Prediction

### Using the trained model

```python
from vespatune import VespaTunePredict

predictor = VespaTunePredict(model_path="outputs/my_model")

# Predict on file
predictor.predict_file("test.csv", "predictions.csv")

# Predict single sample
prediction = predictor.predict_single({"feature1": 1.0, "feature2": "A"})
```

### Using ONNX model

```python
from vespatune import VespaTuneONNXPredict

predictor = VespaTuneONNXPredict(model_path="onnx_model/")

# Predict on file
predictor.predict_file("test.csv", "predictions.csv")

# Predict single sample
prediction = predictor.predict_single({"feature1": 1.0, "feature2": "A"})
```

### Standalone Preprocessing

Use `VespaTuneProcessor` when you want to preprocess data independently and pass it to an external ONNX runtime or inference system:

```python
from vespatune import VespaTuneProcessor
import onnxruntime as ort

# Load preprocessor from model or ONNX export directory
processor = VespaTuneProcessor(model_path="outputs/my_model")

# Transform DataFrame
processed = processor.transform(df)  # Returns float32 numpy array

# Transform single sample
processed = processor.transform_single({"feature1": 1.0, "feature2": "A"})

# Get feature metadata
processor.get_feature_names()        # Input feature names
processor.get_categorical_features() # Categorical feature names
processor.get_feature_names_out()    # Output feature names after transform
processor.get_input_schema()         # Pydantic schema for API validation

# Pass to ONNX runtime
session = ort.InferenceSession("model.onnx")
predictions = session.run(None, {"input": processed})
```

## CLI Reference

### Default (UI)

```bash
vespatune [--host HOST] [--port PORT]

options:
  --host                Host to serve on (default: 127.0.0.1)
  --port                Port to serve on (default: 9999)
  --version, -v         Display VespaTune version
```

### train

```bash
vespatune train --help

options:
  --train_filename      Path to training file (required)
  --valid_filename      Path to validation file (optional, auto-splits if not provided)
  --output              Path to output directory (required)
  --model               Model type: xgboost, lightgbm, catboost, logreg (default: xgboost)
  --test_filename       Path to test file
  --task                Task type: classification, regression
  --idx                 ID column name
  --targets             Target column(s), separate multiple by ';'
  --features            Feature columns, separate by ';'
  --use_gpu             Use GPU for training
  --seed                Random seed (default: 42)
  --num_trials          Number of Optuna trials (default: 100)
  --time_limit          Time limit in seconds
```

### predict

```bash
vespatune predict --help

options:
  --model_path          Path to trained model directory (required)
  --test_filename       Path to test file (required)
  --output_filename     Path to output predictions file (required)
```

### export

```bash
vespatune export --help

options:
  --model_path          Path to trained model directory (required)
  --output_dir          Path to ONNX output directory
```

### serve

```bash
vespatune serve --help

options:
  --model_path          Path to ONNX export directory
  --host                Host to bind (default: 127.0.0.1)
  --port                Port to bind (default: 9999)
  --workers             Number of workers (default: 1)
  --reload              Enable auto-reload for development
```

### splitter

```bash
vespatune splitter --help

options:
  --data_filename       Path to data file (required)
  --output              Path to output directory (required)
  --target              Target column name (required)
  --task                Task type: classification, regression (required)
  --num_folds           Number of folds (default: 5)
```

## Output Files

After training, the following files are created in the output directory:

| File | Description |
|------|-------------|
| `vtune_model.final` | Trained model |
| `vtune.config` | Model configuration |
| `vtune.best_params` | Best hyperparameters from Optuna |
| `vtune.preprocessor.joblib` | Fitted preprocessor (encoding, scaling, imputation) |
| `vtune.target_encoder` | Target encoder (for classification) |
| `params.db` | Optuna study database |
| `train.feather` | Processed training data |
| `valid.feather` | Processed validation data |
| `onnx/` | ONNX export directory (after export) |
| `_splits/` | Auto-generated train/valid splits (only if no validation file provided) |

## Example

```python
from vespatune import VespaTune

# Train with LightGBM
vtune = VespaTune(
    train_filename="data/train.csv",
    valid_filename="data/valid.csv",
    output="outputs/lgb_model",
    model_type="lightgbm",
    targets=["price"],
    task="regression",
    num_trials=200,
    time_limit=1800,
    use_gpu=False,
    seed=42,
)
vtune.train()
```
