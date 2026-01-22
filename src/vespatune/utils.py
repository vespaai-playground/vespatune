import copy
import os
from functools import partial

import joblib
import numpy as np
import optuna
import pandas as pd

from .enums import ProblemType
from .logger import logger
from .metrics import Metrics
from .models import get_model


optuna.logging.set_verbosity(optuna.logging.INFO)


def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.info(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def get_eval_metric_and_direction(problem_type):
    """Get evaluation metric and optimization direction for problem type."""
    metric_map = {
        ProblemType.binary_classification: ("logloss", "minimize"),
        ProblemType.multi_class_classification: ("mlogloss", "minimize"),
        ProblemType.multi_label_classification: ("logloss", "minimize"),
        ProblemType.single_column_regression: ("rmse", "minimize"),
        ProblemType.multi_column_regression: ("rmse", "minimize"),
    }
    return metric_map.get(problem_type, ("rmse", "minimize"))


def use_predict_proba(problem_type):
    """Check if we should use predict_proba for this problem type."""
    return problem_type in (
        ProblemType.binary_classification,
        ProblemType.multi_class_classification,
        ProblemType.multi_label_classification,
    )


def get_categorical_indices(model_config):
    """Get indices of categorical features."""
    if not model_config.categorical_features:
        return None
    return [
        model_config.features.index(f)
        for f in model_config.categorical_features
        if f in model_config.features
    ]


def optimize(trial, model_config):
    """Optimization function for Optuna."""
    model_name = getattr(model_config, "model_type", "xgboost")
    problem_type_str = model_config.problem_type.name

    # Create model instance
    model = get_model(
        model_name=model_name,
        problem_type=problem_type_str,
        random_state=model_config.seed,
    )

    # Get hyperparameters
    params = model.get_params(trial, model_config)

    metrics = Metrics(model_config.problem_type)
    eval_metric, _ = get_eval_metric_and_direction(model_config.problem_type)

    # Load data
    train_df = pd.read_feather(os.path.join(model_config.output, "train.feather"))
    valid_df = pd.read_feather(os.path.join(model_config.output, "valid.feather"))

    xtrain = train_df[model_config.features].values
    xvalid = valid_df[model_config.features].values
    ytrain = train_df[model_config.targets].values
    yvalid = valid_df[model_config.targets].values

    # Get categorical feature indices
    cat_indices = get_categorical_indices(model_config)

    # Train model
    if model_config.problem_type in (
        ProblemType.multi_column_regression,
        ProblemType.multi_label_classification,
    ):
        ypred = []
        for idx in range(len(model_config.targets)):
            _model = get_model(
                model_name=model_name,
                problem_type=problem_type_str,
                random_state=model_config.seed,
            )
            _params = copy.deepcopy(params)
            _model.fit(
                xtrain,
                ytrain[:, idx],
                xvalid,
                yvalid[:, idx],
                _params,
                categorical_features=cat_indices,
            )
            if model_config.problem_type == ProblemType.multi_column_regression:
                ypred_temp = _model.predict(xvalid)
            else:
                ypred_temp = _model.predict_proba(xvalid)[:, 1]
            ypred.append(ypred_temp)
        ypred = np.column_stack(ypred)
    else:
        model.fit(
            xtrain,
            ytrain.ravel() if ytrain.ndim > 1 else ytrain,
            xvalid,
            yvalid.ravel() if yvalid.ndim > 1 else yvalid,
            params,
            categorical_features=cat_indices,
        )
        if use_predict_proba(model_config.problem_type):
            ypred = model.predict_proba(xvalid)
        else:
            ypred = model.predict(xvalid)

    metric_dict = metrics.calculate(yvalid, ypred)
    logger.info(f"Metrics: {metric_dict}")

    # Log all metrics to trial user_attrs for UI visualization
    for metric, value in metric_dict.items():
        trial.set_user_attr(metric, value)

    return metric_dict[eval_metric]


def train_model(model_config, callbacks=None):
    """Run hyperparameter optimization."""
    _, direction = get_eval_metric_and_direction(model_config.problem_type)

    optimize_func = partial(optimize, model_config=model_config)

    db_path = os.path.join(model_config.output, "params.db")
    study = optuna.create_study(
        direction=direction,
        study_name="vespatune",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )
    study.optimize(
        optimize_func,
        n_trials=model_config.num_trials,
        timeout=model_config.time_limit,
        callbacks=callbacks,
    )
    return study.best_params


def train_final_model(model_config, best_params):
    """Train a final model on all data (train + valid) using optimal hyperparameters."""
    logger.info("Training final model on all data with optimal parameters")

    model_name = getattr(model_config, "model_type", "xgboost")
    problem_type_str = model_config.problem_type.name

    best_params = copy.deepcopy(best_params)
    best_params.pop("early_stopping_rounds", None)  # Remove, not used in final training

    # Handle GPU settings for XGBoost
    if model_config.use_gpu and model_name == "xgboost":
        best_params["device"] = "cuda"
        best_params["tree_method"] = "hist"

    # Load and combine train + valid data
    train_df = pd.read_feather(os.path.join(model_config.output, "train.feather"))
    valid_df = pd.read_feather(os.path.join(model_config.output, "valid.feather"))
    full_train_df = pd.concat([train_df, valid_df], ignore_index=True)

    xtrain = full_train_df[model_config.features].values
    ytrain = full_train_df[model_config.targets].values

    # Get categorical feature indices
    cat_indices = get_categorical_indices(model_config)

    # For final model, set n_estimators/iterations
    if "n_estimators" not in best_params and "iterations" not in best_params:
        if model_name == "catboost":
            best_params["iterations"] = 10000
        else:
            best_params["n_estimators"] = 10000

    # Train final model
    if model_config.problem_type in (
        ProblemType.multi_column_regression,
        ProblemType.multi_label_classification,
    ):
        trained_models = []
        for idx in range(len(model_config.targets)):
            _model = get_model(
                model_name=model_name,
                problem_type=problem_type_str,
                random_state=model_config.seed,
            )
            _params = copy.deepcopy(best_params)

            # For final training without early stopping, we need a dummy validation set
            # Using a small portion of training data
            split_idx = int(len(xtrain) * 0.9)
            _model.fit(
                xtrain[:split_idx],
                ytrain[:split_idx, idx],
                xtrain[split_idx:],
                ytrain[split_idx:, idx],
                _params,
                categorical_features=cat_indices,
            )
            trained_models.append(_model.get_model())
        final_model = trained_models
    else:
        model = get_model(
            model_name=model_name,
            problem_type=problem_type_str,
            random_state=model_config.seed,
        )
        # For final training, use small validation split
        split_idx = int(len(xtrain) * 0.9)
        y = ytrain.ravel() if ytrain.ndim > 1 else ytrain
        model.fit(
            xtrain[:split_idx],
            y[:split_idx],
            xtrain[split_idx:],
            y[split_idx:],
            best_params,
            categorical_features=cat_indices,
        )
        final_model = model.get_model()

    # Save final model
    final_model_path = os.path.join(model_config.output, "vtune_model.final")
    joblib.dump(final_model, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Generate test predictions if test file provided
    if model_config.test_filename is not None:
        _generate_test_predictions(model_config, final_model)

    return final_model


def _generate_test_predictions(model_config, final_model):
    """Generate predictions on test set."""
    logger.info("Generating test predictions")
    test_df = pd.read_feather(os.path.join(model_config.output, "test.feather"))
    xtest = test_df[model_config.features].values
    test_ids = test_df[model_config.idx].values

    if model_config.problem_type in (
        ProblemType.multi_column_regression,
        ProblemType.multi_label_classification,
    ):
        test_preds = []
        for idx in range(len(final_model)):
            if model_config.problem_type == ProblemType.multi_column_regression:
                pred = final_model[idx].predict(xtest)
            else:
                pred = final_model[idx].predict_proba(xtest)[:, 1]
            test_preds.append(pred)
        test_preds = np.column_stack(test_preds)
    else:
        if use_predict_proba(model_config.problem_type):
            test_preds = final_model.predict_proba(xtest)
        else:
            test_preds = final_model.predict(xtest)

    # Save test predictions
    target_encoder = joblib.load(f"{model_config.output}/vtune.target_encoder")
    if target_encoder is None:
        test_preds_df = pd.DataFrame(test_preds, columns=model_config.targets)
    else:
        test_preds_df = pd.DataFrame(test_preds, columns=list(target_encoder.classes_))
    test_preds_df.insert(loc=0, column=model_config.idx, value=test_ids)
    test_preds_df.to_csv(
        os.path.join(model_config.output, "test_predictions.csv"), index=False
    )
    logger.info("Test predictions saved to test_predictions.csv")
