import copy
import os
from functools import partial

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from .enums import ProblemType
from .logger import logger
from .metrics import Metrics
from .params import get_params


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


def fetch_xgb_model_params(model_config):
    if model_config.problem_type == ProblemType.binary_classification:
        xgb_model = xgb.XGBClassifier
        use_predict_proba = True
        direction = "minimize"
        eval_metric = "logloss"
    elif model_config.problem_type == ProblemType.multi_class_classification:
        xgb_model = xgb.XGBClassifier
        use_predict_proba = True
        direction = "minimize"
        eval_metric = "mlogloss"
    elif model_config.problem_type == ProblemType.multi_label_classification:
        xgb_model = xgb.XGBClassifier
        use_predict_proba = True
        direction = "minimize"
        eval_metric = "logloss"
    elif model_config.problem_type == ProblemType.single_column_regression:
        xgb_model = xgb.XGBRegressor
        use_predict_proba = False
        direction = "minimize"
        eval_metric = "rmse"
    elif model_config.problem_type == ProblemType.multi_column_regression:
        xgb_model = xgb.XGBRegressor
        use_predict_proba = False
        direction = "minimize"
        eval_metric = "rmse"
    else:
        raise NotImplementedError

    return xgb_model, use_predict_proba, eval_metric, direction


def optimize(trial, xgb_model, use_predict_proba, eval_metric, model_config):
    params = get_params(trial, model_config)
    early_stopping_rounds = params.pop("early_stopping_rounds")

    metrics = Metrics(model_config.problem_type)

    # Load data
    train_df = pd.read_feather(os.path.join(model_config.output, "train.feather"))
    valid_df = pd.read_feather(os.path.join(model_config.output, "valid.feather"))

    xtrain = train_df[model_config.features]
    xvalid = valid_df[model_config.features]
    ytrain = train_df[model_config.targets].values
    yvalid = valid_df[model_config.targets].values

    # Train model
    model = xgb_model(
        random_state=model_config.seed,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )

    if model_config.problem_type in (
        ProblemType.multi_column_regression,
        ProblemType.multi_label_classification,
    ):
        ypred = []
        for idx in range(len(model_config.targets)):
            _m = copy.deepcopy(model)
            _m.fit(
                xtrain,
                ytrain[:, idx],
                eval_set=[(xvalid, yvalid[:, idx])],
                verbose=False,
            )
            if model_config.problem_type == ProblemType.multi_column_regression:
                ypred_temp = _m.predict(xvalid)
            else:
                ypred_temp = _m.predict_proba(xvalid)[:, 1]
            ypred.append(ypred_temp)
        ypred = np.column_stack(ypred)
    else:
        model.fit(
            xtrain,
            ytrain,
            eval_set=[(xvalid, yvalid)],
            verbose=False,
        )
        if use_predict_proba:
            ypred = model.predict_proba(xvalid)
        else:
            ypred = model.predict(xvalid)

    metric_dict = metrics.calculate(yvalid, ypred)
    logger.info(f"Metrics: {metric_dict}")
    return metric_dict[eval_metric]


def train_model(model_config):
    xgb_model, use_predict_proba, eval_metric, direction = fetch_xgb_model_params(
        model_config
    )

    optimize_func = partial(
        optimize,
        xgb_model=xgb_model,
        use_predict_proba=use_predict_proba,
        eval_metric=eval_metric,
        model_config=model_config,
    )
    db_path = os.path.join(model_config.output, "params.db")
    study = optuna.create_study(
        direction=direction,
        study_name="vespatune",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )
    study.optimize(
        optimize_func, n_trials=model_config.num_trials, timeout=model_config.time_limit
    )
    return study.best_params


def train_final_model(model_config, best_params):
    """Train a final model on all data (train + valid) using optimal hyperparameters."""
    logger.info("Training final model on all data with optimal parameters")

    best_params = copy.deepcopy(best_params)
    early_stopping_rounds = best_params.pop("early_stopping_rounds", None)

    if model_config.use_gpu is True:
        best_params["device"] = "cuda"
        best_params["tree_method"] = "hist"

    xgb_model, use_predict_proba, eval_metric, _ = fetch_xgb_model_params(model_config)

    # Load and combine train + valid data
    train_df = pd.read_feather(os.path.join(model_config.output, "train.feather"))
    valid_df = pd.read_feather(os.path.join(model_config.output, "valid.feather"))
    full_train_df = pd.concat([train_df, valid_df], ignore_index=True)

    xtrain = full_train_df[model_config.features]
    ytrain = full_train_df[model_config.targets].values

    # For final model, we use a fixed n_estimators since we don't have a validation set
    # Use the value from best_params or a reasonable default
    if "n_estimators" not in best_params:
        best_params["n_estimators"] = 10000

    # Train final model
    model = xgb_model(
        random_state=model_config.seed,
        eval_metric=eval_metric,
        **best_params,
    )

    if model_config.problem_type in (
        ProblemType.multi_column_regression,
        ProblemType.multi_label_classification,
    ):
        trained_models = []
        for idx in range(len(model_config.targets)):
            _m = copy.deepcopy(model)
            _m.fit(xtrain, ytrain[:, idx], verbose=False)
            trained_models.append(_m)
        final_model = trained_models
    else:
        model.fit(xtrain, ytrain, verbose=False)
        final_model = model

    # Save final model
    final_model_path = os.path.join(model_config.output, "vtune_model.final")
    joblib.dump(final_model, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Generate test predictions if test file provided
    if model_config.test_filename is not None:
        logger.info("Generating test predictions")
        test_df = pd.read_feather(os.path.join(model_config.output, "test.feather"))
        xtest = test_df[model_config.features]
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
            if use_predict_proba:
                test_preds = final_model.predict_proba(xtest)
            else:
                test_preds = final_model.predict(xtest)

        # Save test predictions
        target_encoder = joblib.load(f"{model_config.output}/vtune.target_encoder")
        if target_encoder is None:
            test_preds_df = pd.DataFrame(test_preds, columns=model_config.targets)
        else:
            test_preds_df = pd.DataFrame(
                test_preds, columns=list(target_encoder.classes_)
            )
        test_preds_df.insert(loc=0, column=model_config.idx, value=test_ids)
        test_preds_df.to_csv(
            os.path.join(model_config.output, "test_predictions.csv"), index=False
        )
        logger.info("Test predictions saved to test_predictions.csv")

    return final_model


# Keep for backward compatibility but not used in simplified flow
def predict_model(model_config, best_params):
    """Deprecated: Use train_final_model instead."""
    pass
