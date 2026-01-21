# VespaTune

Gradient Boosting + Optuna: no brainer

- Auto train XGBoost, LightGBM, or CatBoost directly from CSV files
- Auto tune hyperparameters using Optuna
- Export models to ONNX format for deployment
- Serve trained models using FastAPI

NOTE: PRs are currently not accepted. If there are issues/problems, please create an issue.

## Installation

Install using pip:

```bash
pip install vespatune
```

## Quick Start

### CLI

Train a model:

```bash
vespatune train \
  --train_filename train.csv \
  --valid_filename valid.csv \
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

Export to ONNX:

```bash
vespatune export --model_path outputs/my_model
```

Serve the model:

```bash
vespatune serve --model_path outputs/my_model --host 0.0.0.0 --port 8000
```

### Python API

```python
from vespatune import VespaTune

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
```

## Parameters

### Required

| Parameter | Description |
|-----------|-------------|
| `train_filename` | Path to training CSV file |
| `valid_filename` | Path to validation CSV file |
| `output` | Path to output directory for model artifacts |

### Optional

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | `"xgboost"` | Model to use: `"xgboost"`, `"lightgbm"`, or `"catboost"` |
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

## Data Splitting

VespaTune uses an explicit train/validation split. If you have a single dataset, use the splitter utility:

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

## ONNX Export

Export trained models for deployment:

```python
from vespatune import VespaTuneExport

exporter = VespaTuneExport(model_path="outputs/my_model")
exporter.export_to_onnx(output_dir="onnx_model/", verify=True)
```

The export includes:
- ONNX model file(s)
- `metadata.json` with feature names and mappings
- Encoders for categorical features and targets

## Prediction

### Using the trained model

```python
from vespatune import VespaTunePredict

predictor = VespaTunePredict(model_path="outputs/my_model")
predictions = predictor.predict_file("test.csv")
```

### Using ONNX model

```python
from vespatune import VespaTuneONNXPredict

predictor = VespaTuneONNXPredict(model_path="onnx_model/")
predictions = predictor.predict_file("test.csv")
```

## CLI Reference

### train

```bash
vespatune train --help

options:
  --train_filename      Path to training file (required)
  --valid_filename      Path to validation file (required)
  --output              Path to output directory (required)
  --model               Model type: xgboost, lightgbm, catboost (default: xgboost)
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
  --model_path          Path to trained model directory (required)
  --host                Host to bind (default: 0.0.0.0)
  --port                Port to bind (default: 8000)
  --debug               Enable debug mode
```

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

# The following files are created in outputs/lgb_model/:
# - vtune.config          : Model configuration
# - vtune_model.final     : Trained model
# - vtune.best_params     : Best hyperparameters
# - vtune.categorical_encoder : Categorical feature encoder
# - vtune.target_encoder  : Target encoder (for classification)
# - params.db             : Optuna study database
# - train.feather         : Processed training data
# - valid.feather         : Processed validation data
```
