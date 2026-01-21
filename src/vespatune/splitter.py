import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target

from .logger import logger


@dataclass
class VespaTuneSplitter:
    input_filename: str
    output_dir: str
    num_folds: int = 5
    targets: Optional[List[str]] = None
    task: Optional[str] = None
    seed: int = 42

    def __post_init__(self):
        if self.targets is None:
            self.targets = ["target"]

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

    def _determine_split_strategy(self, df):
        """Determine whether to use stratified or regular k-fold."""
        if self.task == "regression":
            logger.info("Task: regression (user specified)")
            return "stratified_bins"
        elif self.task == "classification":
            logger.info("Task: classification (user specified)")
            return "stratified"
        else:
            # Auto-detect using sklearn's type_of_target
            target_values = df[self.targets].values
            target_type = type_of_target(target_values)
            logger.info(f"Auto-detected target type: {target_type}")

            if target_type == "continuous":
                logger.info("Task: single column regression (auto-detected)")
                return "stratified_bins"
            elif target_type == "continuous-multioutput":
                logger.info("Task: multi-column regression (auto-detected)")
                return "regular"
            elif target_type == "binary":
                logger.info("Task: binary classification (auto-detected)")
                return "stratified"
            elif target_type == "multiclass":
                logger.info("Task: multi-class classification (auto-detected)")
                return "stratified"
            elif target_type == "multilabel-indicator":
                logger.info("Task: multi-label classification (auto-detected)")
                return "regular"
            else:
                logger.warning(f"Unknown target type: {target_type}, using regular k-fold")
                return "regular"

    def split(self) -> List[dict]:
        """Split the data into k folds and save train/valid files for each fold."""
        logger.info(f"Reading data from {self.input_filename}")
        df = pd.read_csv(self.input_filename)
        logger.info(f"Data shape: {df.shape}")

        strategy = self._determine_split_strategy(df)
        logger.info(f"Using split strategy: {strategy}")

        df["_fold"] = -1

        if strategy == "stratified":
            y = df[self.targets].values
            kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            for fold, (_, valid_idx) in enumerate(kf.split(X=df, y=y)):
                df.loc[valid_idx, "_fold"] = fold

        elif strategy == "stratified_bins":
            # Bin continuous targets for stratification
            y = df[self.targets].values.ravel()
            num_bins = min(10, int(np.floor(1 + np.log2(len(df)))))
            bins = pd.cut(y, bins=num_bins, labels=False)
            kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            for fold, (_, valid_idx) in enumerate(kf.split(X=df, y=bins)):
                df.loc[valid_idx, "_fold"] = fold

        else:  # regular
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            for fold, (_, valid_idx) in enumerate(kf.split(X=df)):
                df.loc[valid_idx, "_fold"] = fold

        # Save fold files
        fold_files = []
        for fold in range(self.num_folds):
            train_df = df[df["_fold"] != fold].drop(columns=["_fold"]).reset_index(drop=True)
            valid_df = df[df["_fold"] == fold].drop(columns=["_fold"]).reset_index(drop=True)

            train_path = os.path.join(self.output_dir, f"train_fold_{fold}.csv")
            valid_path = os.path.join(self.output_dir, f"valid_fold_{fold}.csv")

            train_df.to_csv(train_path, index=False)
            valid_df.to_csv(valid_path, index=False)

            fold_files.append(
                {
                    "fold": fold,
                    "train_path": train_path,
                    "valid_path": valid_path,
                    "train_size": len(train_df),
                    "valid_size": len(valid_df),
                }
            )

            logger.info(f"Fold {fold}: train={len(train_df)} samples, valid={len(valid_df)} samples")

        logger.info(f"Successfully created {self.num_folds} fold splits in {self.output_dir}")
        return fold_files


def split_data(
    input_filename: str,
    output_dir: str,
    num_folds: int = 5,
    targets: Optional[List[str]] = None,
    task: Optional[str] = None,
    seed: int = 42,
) -> List[dict]:
    """Split data into k folds for cross-validation.

    Args:
        input_filename: Path to input CSV file.
        output_dir: Directory to save fold files.
        num_folds: Number of folds (default: 5).
        targets: Target column name(s) for stratification.
        task: Task type ('classification' or 'regression') for stratification strategy.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with fold info (fold number, train/valid paths, sizes).
    """
    splitter = VespaTuneSplitter(
        input_filename=input_filename,
        output_dir=output_dir,
        num_folds=num_folds,
        targets=targets,
        task=task,
        seed=seed,
    )
    return splitter.split()
