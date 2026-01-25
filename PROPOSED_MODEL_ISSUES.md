# Proposed GitHub Issues: Model Additions for VespaTune

This document contains proposed GitHub issues for adding new model types to VespaTune. Each issue includes a title, description, rationale, and implementation considerations.

---

## Issue 1: Add Random Forest Support

**Title:** Add Random Forest model support

**Labels:** enhancement, model

**Description:**

Add support for Random Forest (both RandomForestClassifier and RandomForestRegressor from scikit-learn) as a model option in VespaTune.

**Rationale:**

- Random Forest is one of the most popular and widely-used ensemble methods for tabular data
- Provides good baseline performance with minimal hyperparameter tuning
- Robust to overfitting and works well on small to medium-sized datasets
- Interpretable through feature importances
- No need for feature scaling (unlike logistic regression)
- Naturally handles non-linear relationships
- Complements existing gradient boosting models (XGBoost, LightGBM, CatBoost)

**Implementation Considerations:**

- Create `randomforest_model.py` in `src/vespatune/models/`
- Implement `RandomForestModel` class inheriting from `BaseModel`
- Implement `RandomForestPreprocessor` class
- Hyperparameters to tune:
  - `n_estimators`: number of trees (50-1000)
  - `max_depth`: maximum depth of trees (3-50, or None)
  - `min_samples_split`: minimum samples to split a node (2-20)
  - `min_samples_leaf`: minimum samples in leaf node (1-10)
  - `max_features`: number of features for best split ('sqrt', 'log2', None)
  - `criterion`: split quality measure ('gini', 'entropy' for classification; 'squared_error', 'absolute_error' for regression)
  - `bootstrap`: whether to use bootstrap samples (True/False)
- Native categorical feature handling: Use `OrdinalEncoder` in preprocessor
- Supports both classification and regression
- ONNX export: Use `sklearn-onnx` converter
- Set `supports_categorical = False` (requires encoding)
- Set `supports_gpu = False`
- Set `searches_preprocessing = False`

---

## Issue 2: Add Neural Network (MLPClassifier/MLPRegressor) Support

**Title:** Add Multi-layer Perceptron (Neural Network) model support

**Labels:** enhancement, model, neural-network

**Description:**

Add support for Multi-layer Perceptron (MLP) models using scikit-learn's MLPClassifier and MLPRegressor.

**Rationale:**

- Provides a simple neural network option without requiring PyTorch/TensorFlow
- Can capture complex non-linear patterns in tabular data
- Good for medium-sized datasets with complex feature interactions
- Fits well with scikit-learn ecosystem already used in the project
- Lightweight alternative to deep learning frameworks
- Can serve as a stepping stone before adding more complex neural architectures

**Implementation Considerations:**

- Create `mlp_model.py` in `src/vespatune/models/`
- Implement `MLPModel` class inheriting from `BaseModel`
- Implement `MLPPreprocessor` with proper feature scaling (StandardScaler required)
- Hyperparameters to tune:
  - `hidden_layer_sizes`: architecture, e.g., (100,), (100, 50), (200, 100, 50)
  - `activation`: activation function ('relu', 'tanh', 'logistic')
  - `solver`: optimizer ('adam', 'sgd', 'lbfgs')
  - `alpha`: L2 regularization (1e-5 to 1e-1)
  - `learning_rate_init`: initial learning rate (1e-4 to 1e-1)
  - `batch_size`: mini-batch size ('auto' or 32, 64, 128, 256)
  - `max_iter`: maximum iterations (100-1000)
  - `early_stopping`: use validation for early stopping (True)
- Preprocessing: Must include StandardScaler (set `searches_preprocessing = True`)
- ONNX export: Use `sklearn-onnx` converter
- Set `supports_categorical = False` (requires encoding and scaling)
- Set `supports_gpu = False`
- Requires validation set for early stopping

---

## Issue 3: Add Extra Trees (ExtraTreesClassifier/ExtraTreesRegressor) Support

**Title:** Add Extra Trees model support

**Labels:** enhancement, model

**Description:**

Add support for Extra Trees (Extremely Randomized Trees) ensemble method from scikit-learn.

**Rationale:**

- Similar to Random Forest but uses random thresholds for splits instead of optimal ones
- Often faster to train than Random Forest
- Can achieve comparable or better performance in some cases
- Reduces variance further than Random Forest
- Good for datasets with noisy features
- Provides diversity in ensemble methods available to users

**Implementation Considerations:**

- Create `extratrees_model.py` in `src/vespatune/models/`
- Very similar implementation to Random Forest
- Implement `ExtraTreesModel` class inheriting from `BaseModel`
- Implement `ExtraTreesPreprocessor` class
- Hyperparameters similar to Random Forest:
  - `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
  - `max_features`, `criterion`, `bootstrap`
- Set `supports_categorical = False`
- Set `supports_gpu = False`
- ONNX export: Use `sklearn-onnx` converter

---

## Issue 4: Add Support Vector Machine (SVM) Support

**Title:** Add Support Vector Machine (SVM) model support

**Labels:** enhancement, model

**Description:**

Add support for Support Vector Machines using scikit-learn's SVC and SVR (with linear, RBF, and polynomial kernels).

**Rationale:**

- Effective for small to medium-sized datasets
- Works well in high-dimensional spaces
- Versatile through different kernel functions
- Memory efficient (uses subset of training points)
- Good for cases where margin of separation is important
- Particularly effective for text classification and certain structured data problems

**Implementation Considerations:**

- Create `svm_model.py` in `src/vespatune/models/`
- Implement `SVMModel` class inheriting from `BaseModel`
- Implement `SVMPreprocessor` with mandatory StandardScaler
- Hyperparameters to tune:
  - `kernel`: kernel type ('linear', 'rbf', 'poly', 'sigmoid')
  - `C`: regularization parameter (1e-3 to 1e3)
  - `gamma`: kernel coefficient for RBF/poly/sigmoid ('scale', 'auto', or float)
  - `degree`: degree for polynomial kernel (2-5)
  - `coef0`: independent term in kernel function
- For large datasets, consider using `LinearSVC`/`LinearSVR` instead
- Preprocessing: MUST include StandardScaler (set `searches_preprocessing = True`)
- Set `supports_categorical = False`
- Set `supports_gpu = False`
- ONNX export: Use `sklearn-onnx` converter
- Note: Can be slow on large datasets (>10k samples), consider adding warning

---

## Issue 5: Add TabNet Support

**Title:** Add TabNet deep learning model support

**Labels:** enhancement, model, neural-network, deep-learning

**Description:**

Add support for TabNet, a modern deep learning architecture specifically designed for tabular data, using the `pytorch-tabnet` library.

**Rationale:**

- State-of-the-art deep learning architecture specifically designed for tabular data
- Provides interpretability through attention mechanism (feature importance per sample)
- Can handle both numerical and categorical features natively
- Self-supervised pre-training capabilities for unlabeled data
- Competitive with gradient boosting methods on many tabular datasets
- Growing popularity in the ML community for tabular data
- Provides GPU acceleration for large datasets

**Implementation Considerations:**

- Add `pytorch-tabnet` to dependencies (make it optional in `[tabnet]` extras)
- Create `tabnet_model.py` in `src/vespatune/models/`
- Implement `TabNetModel` class inheriting from `BaseModel`
- Implement `TabNetPreprocessor` class
- Hyperparameters to tune:
  - `n_d`, `n_a`: width of decision and attention layers (8-64)
  - `n_steps`: number of steps in the architecture (3-10)
  - `gamma`: coefficient for feature reuse (1.0-2.0)
  - `n_independent`, `n_shared`: number of independent/shared GLU layers
  - `lambda_sparse`: sparsity regularization (1e-6 to 1e-3)
  - `learning_rate`: (1e-4 to 1e-2)
  - `batch_size`: (256, 512, 1024, 2048)
- Set `supports_categorical = True` (native support)
- Set `supports_gpu = True`
- ONNX export: May require custom implementation or use PyTorch ONNX export
- Requires PyTorch as dependency

---

## Issue 6: Add Ridge and Lasso Regression Support

**Title:** Add Ridge and Lasso regression model support

**Labels:** enhancement, model

**Description:**

Add support for Ridge Regression (L2 regularization) and Lasso Regression (L1 regularization) from scikit-learn.

**Rationale:**

- Simple, interpretable linear models with regularization
- Complement existing Logistic Regression (which only supports classification)
- Ridge: Good when all features are potentially relevant (L2 doesn't zero out coefficients)
- Lasso: Performs feature selection by zeroing out coefficients (L1 regularization)
- Very fast to train, even on large datasets
- Useful as baseline models and for feature importance analysis
- Work well when linear relationships exist in the data

**Implementation Considerations:**

- Create `linear_regression_model.py` in `src/vespatune/models/`
- Implement `RidgeModel` and `LassoModel` classes inheriting from `BaseModel`
- Implement corresponding preprocessor classes with StandardScaler
- Can potentially extend existing `logreg_model.py` to include these
- Hyperparameters to tune:
  - `alpha`: regularization strength (1e-4 to 1e2)
  - For Lasso, also tune `selection`: ('cyclic', 'random')
- Set `supported_problem_types = ["single_column_regression", "multi_column_regression"]`
- Preprocessing: Must include StandardScaler and imputation
- Set `supports_categorical = False`
- Set `supports_gpu = False`
- Set `searches_preprocessing = True` (like LogReg)
- ONNX export: Use `sklearn-onnx` converter

---

## Issue 7: Add Elastic Net Support

**Title:** Add Elastic Net regression model support

**Labels:** enhancement, model

**Description:**

Add support for Elastic Net, which combines L1 and L2 regularization, from scikit-learn.

**Rationale:**

- Combines benefits of both Ridge and Lasso regression
- Useful when there are multiple correlated features
- Performs feature selection while handling multicollinearity
- More stable than Lasso for correlated features
- Natural extension after adding Ridge and Lasso
- Good for high-dimensional data with correlated features

**Implementation Considerations:**

- Add to `linear_regression_model.py` or create separate file
- Implement `ElasticNetModel` class inheriting from `BaseModel`
- Hyperparameters to tune:
  - `alpha`: overall regularization strength (1e-4 to 1e2)
  - `l1_ratio`: balance between L1 and L2 (0.1 to 0.9)
  - `selection`: ('cyclic', 'random')
- Set `supported_problem_types = ["single_column_regression", "multi_column_regression"]`
- Set `supports_categorical = False`
- Set `supports_gpu = False`
- Set `searches_preprocessing = True`
- ONNX export: Use `sklearn-onnx` converter

---

## Issue 8: Add Gradient Boosting (scikit-learn) Support

**Title:** Add scikit-learn Gradient Boosting model support

**Labels:** enhancement, model

**Description:**

Add support for scikit-learn's native GradientBoostingClassifier and GradientBoostingRegressor.

**Rationale:**

- Pure Python implementation without external dependencies (already have scikit-learn)
- No additional libraries needed beyond existing dependencies
- Good baseline gradient boosting method
- Well-documented and stable implementation
- Useful for comparison with XGBoost/LightGBM/CatBoost
- Some users prefer scikit-learn's implementations for simplicity

**Implementation Considerations:**

- Create `sklearn_gb_model.py` in `src/vespatune/models/`
- Implement `SklearnGBModel` class inheriting from `BaseModel`
- Hyperparameters to tune:
  - `n_estimators`: number of boosting stages (50-500)
  - `learning_rate`: learning rate (0.01-0.3)
  - `max_depth`: maximum depth of trees (3-10)
  - `min_samples_split`: minimum samples to split (2-20)
  - `min_samples_leaf`: minimum samples in leaf (1-10)
  - `subsample`: fraction of samples for fitting (0.5-1.0)
  - `max_features`: features to consider ('sqrt', 'log2', None)
- Set `supports_categorical = False`
- Set `supports_gpu = False`
- ONNX export: Use `sklearn-onnx` converter
- Note: Generally slower than XGBoost/LightGBM but no external dependency

---

## Issue 9: Add K-Nearest Neighbors (KNN) Support

**Title:** Add K-Nearest Neighbors (KNN) model support

**Labels:** enhancement, model

**Description:**

Add support for K-Nearest Neighbors using scikit-learn's KNeighborsClassifier and KNeighborsRegressor.

**Rationale:**

- Simple, intuitive algorithm useful for baseline comparisons
- Non-parametric method (no training phase)
- Works well for small datasets with clear clustering
- Useful for certain types of problems (recommendation, anomaly detection)
- Provides different approach compared to tree and linear models
- Fast prediction for small datasets

**Implementation Considerations:**

- Create `knn_model.py` in `src/vespatune/models/`
- Implement `KNNModel` class inheriting from `BaseModel`
- Hyperparameters to tune:
  - `n_neighbors`: number of neighbors (3-30)
  - `weights`: weight function ('uniform', 'distance')
  - `metric`: distance metric ('euclidean', 'manhattan', 'minkowski')
  - `p`: power parameter for Minkowski metric (1-3)
  - `algorithm`: algorithm to compute neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
- Preprocessing: Must include StandardScaler for distance-based methods
- Set `supports_categorical = False`
- Set `supports_gpu = False`
- Set `searches_preprocessing = True`
- ONNX export: Use `sklearn-onnx` converter
- Note: Can be memory-intensive and slow for large datasets (>10k samples)

---

## Issue 10: Add Naive Bayes Support

**Title:** Add Naive Bayes model support (GaussianNB, MultinomialNB)

**Labels:** enhancement, model

**Description:**

Add support for Naive Bayes classifiers from scikit-learn (GaussianNB for continuous features, MultinomialNB for count features).

**Rationale:**

- Fast training and prediction
- Works well for text classification and categorical data
- Requires minimal training data
- Provides probabilistic predictions naturally
- Good baseline for classification tasks
- Particularly effective when independence assumption holds reasonably well

**Implementation Considerations:**

- Create `naive_bayes_model.py` in `src/vespatune/models/`
- Implement `NaiveBayesModel` class inheriting from `BaseModel`
- Support both GaussianNB and MultinomialNB (choose based on data characteristics)
- Hyperparameters to tune (limited):
  - GaussianNB: `var_smoothing` (1e-10 to 1e-6)
  - MultinomialNB: `alpha` (smoothing parameter, 0.1-10.0)
- Set `supported_problem_types = ["binary_classification", "multi_class_classification"]`
- Set `supports_categorical = False`
- Set `supports_gpu = False`
- ONNX export: Use `sklearn-onnx` converter
- Note: Only for classification tasks

---

## Issue 11: Add NGBoost Support

**Title:** Add NGBoost (Natural Gradient Boosting) model support

**Labels:** enhancement, model, probabilistic

**Description:**

Add support for NGBoost, a probabilistic gradient boosting method that provides uncertainty estimates, using the `ngboost` library.

**Rationale:**

- Provides probabilistic predictions with uncertainty estimates
- Useful when understanding prediction uncertainty is important
- Based on natural gradient boosting framework
- Can use different base learners (decision trees, ridge regression)
- Particularly valuable for decision-making under uncertainty
- Competitive performance with traditional gradient boosting

**Implementation Considerations:**

- Add `ngboost` to dependencies (consider making it optional)
- Create `ngboost_model.py` in `src/vespatune/models/`
- Implement `NGBoostModel` class inheriting from `BaseModel`
- Hyperparameters to tune:
  - `Base`: base learner (DecisionTreeRegressor, RidgeRegressor)
  - `n_estimators`: number of boosting rounds (100-1000)
  - `learning_rate`: (0.001-0.1)
  - `minibatch_frac`: fraction for stochastic gradient (0.5-1.0)
  - Base learner hyperparameters (e.g., max_depth for trees)
- Set `supports_categorical = False`
- Set `supports_gpu = False`
- ONNX export: May require custom implementation
- Prediction output should include both point estimates and uncertainty (std/variance)

---

## Issue 12: Add HistGradientBoosting (scikit-learn) Support

**Title:** Add HistGradientBoostingClassifier/Regressor support

**Labels:** enhancement, model

**Description:**

Add support for scikit-learn's histogram-based gradient boosting (HistGradientBoostingClassifier and HistGradientBoostingRegressor).

**Rationale:**

- Fast, histogram-based implementation inspired by LightGBM
- Native support for missing values (no imputation needed)
- Native categorical feature support (since scikit-learn 1.0)
- Excellent for large datasets (>10k samples)
- Part of scikit-learn (no external dependency)
- Competitive with LightGBM in speed and performance
- Supports monotonic constraints

**Implementation Considerations:**

- Create `histgb_model.py` in `src/vespatune/models/`
- Implement `HistGBModel` class inheriting from `BaseModel`
- Hyperparameters to tune:
  - `learning_rate`: (0.01-0.3)
  - `max_iter`: number of boosting iterations (100-1000)
  - `max_depth`: maximum depth (3-31, or None)
  - `max_leaf_nodes`: maximum leaf nodes (10-255)
  - `min_samples_leaf`: minimum samples in leaf (1-100)
  - `l2_regularization`: L2 regularization (0-10)
  - `max_bins`: maximum number of bins (32-255)
- Set `supports_categorical = True` (native support for categorical features)
- Set `supports_gpu = False` (not yet in scikit-learn)
- Minimal preprocessing needed (handles missing values natively)
- ONNX export: Use `sklearn-onnx` converter
- Can mark certain features as categorical without encoding

---

## Implementation Priority Recommendations

Based on popularity, ease of implementation, and value-add:

### High Priority (Recommended to implement first):
1. **Random Forest** - Most popular ensemble method, easy to implement
2. **HistGradientBoosting** - Modern, fast, no extra dependencies, native categorical support
3. **Ridge/Lasso Regression** - Simple, complements LogReg, useful baselines

### Medium Priority:
4. **Extra Trees** - Easy (similar to Random Forest), provides variety
5. **Gradient Boosting (sklearn)** - No extra dependency, good baseline
6. **MLP (Neural Network)** - Adds neural network capability without heavy dependencies

### Lower Priority (More specialized):
7. **TabNet** - State-of-the-art but requires PyTorch dependency
8. **Elastic Net** - Nice to have after Ridge/Lasso
9. **SVM** - Useful for specific use cases, slower on large data
10. **KNN** - Simple baseline, limited scalability
11. **Naive Bayes** - Fast, but limited applicability
12. **NGBoost** - Specialized (probabilistic), extra dependency

---

## How to Use This Document

To create these issues on GitHub:

1. Go to the repository: https://github.com/vespaai-playground/vespatune/issues
2. Click "New Issue"
3. Copy the title, labels, and description from each issue above
4. Create one issue per model proposal
5. Consider adding the "good first issue" label for simpler implementations (Random Forest, Extra Trees, Ridge/Lasso)
6. Tag with "help wanted" if seeking community contributions

---

## Additional Considerations

When implementing any new model:

1. **Follow the existing pattern**: Use `BaseModel` abstract class
2. **Create preprocessor**: Implement model-specific `BasePreprocessor` subclass
3. **Register in `__init__.py`**: Add to `MODEL_REGISTRY` and `PREPROCESSOR_REGISTRY`
4. **Support ONNX export**: Implement `to_onnx()` method
5. **Add tests**: Create tests in `tests/` directory
6. **Update documentation**: Update README.md with new model info
7. **Handle edge cases**: Consider small datasets, missing values, categorical features
8. **Optimize hyperparameters**: Choose reasonable search spaces for Optuna
