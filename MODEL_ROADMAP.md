# VespaTune Model Roadmap

## Currently Supported Models

VespaTune currently supports 4 model types:

1. **XGBoost** - High-performance gradient boosting (default)
2. **LightGBM** - Fast gradient boosting with native categorical support
3. **CatBoost** - Gradient boosting optimized for categorical features
4. **Logistic Regression** - Linear classification model

## Proposed Model Additions

See [PROPOSED_MODEL_ISSUES.md](PROPOSED_MODEL_ISSUES.md) for detailed information about 12 proposed model additions.

### Quick Reference

| Priority | Model | Type | Key Benefits | Dependencies |
|----------|-------|------|--------------|--------------|
| **High** | Random Forest | Ensemble | Popular, robust, easy baseline | scikit-learn ✓ |
| **High** | HistGradientBoosting | Gradient Boosting | Fast, native categorical, no missing value issues | scikit-learn ✓ |
| **High** | Ridge/Lasso | Linear | Simple regression baselines, interpretable | scikit-learn ✓ |
| Medium | Extra Trees | Ensemble | Faster than RF, reduces variance | scikit-learn ✓ |
| Medium | Gradient Boosting (sklearn) | Gradient Boosting | Pure Python, no external deps | scikit-learn ✓ |
| Medium | MLP | Neural Network | Captures non-linear patterns | scikit-learn ✓ |
| Lower | TabNet | Deep Learning | SOTA for tabular, interpretable | **pytorch-tabnet** ✗ |
| Lower | Elastic Net | Linear | Combines L1+L2 regularization | scikit-learn ✓ |
| Lower | SVM | Kernel Methods | Good for small datasets | scikit-learn ✓ |
| Lower | KNN | Instance-based | Simple baseline | scikit-learn ✓ |
| Lower | Naive Bayes | Probabilistic | Fast, minimal training data | scikit-learn ✓ |
| Lower | NGBoost | Probabilistic | Uncertainty estimates | **ngboost** ✗ |

✓ = Already a dependency  
✗ = Requires new dependency

## How to Create GitHub Issues

**Important Note:** The issues are documented but not yet created in GitHub. To create them:

1. Visit: https://github.com/vespaai-playground/vespatune/issues/new
2. Use the information from `PROPOSED_MODEL_ISSUES.md`
3. For each issue:
   - Copy the **Title**
   - Add **Labels**: `enhancement`, `model` (plus any additional labels mentioned)
   - Copy the **Description** section
   - Paste the **Rationale** and **Implementation Considerations** into the issue body

Alternatively, you can use GitHub's CLI or API to bulk-create these issues:

```bash
# Example using GitHub CLI (gh)
# Install gh: https://cli.github.com/

# For each issue in PROPOSED_MODEL_ISSUES.md:
gh issue create \
  --repo vespaai-playground/vespatune \
  --title "Add Random Forest model support" \
  --label "enhancement,model" \
  --body "Content from PROPOSED_MODEL_ISSUES.md"
```

## Implementation Guidelines

When implementing a new model:

1. ✅ Follow the `BaseModel` abstract class pattern
2. ✅ Create a model-specific preprocessor inheriting from `BasePreprocessor`
3. ✅ Register both in `src/vespatune/models/__init__.py`
4. ✅ Implement ONNX export via `to_onnx()` method
5. ✅ Add comprehensive tests
6. ✅ Update README.md documentation
7. ✅ Optimize hyperparameter search spaces for Optuna
8. ✅ Handle categorical features appropriately
9. ✅ Support both classification and regression (where applicable)

## Contributing

See the main [README.md](README.md) for contribution guidelines. These model additions are great opportunities for community contributions!

For beginners, consider starting with:
- Random Forest (straightforward implementation)
- Extra Trees (similar to Random Forest)
- Ridge/Lasso Regression (extends existing LogReg pattern)

For advanced contributors:
- TabNet (requires deep learning expertise)
- NGBoost (probabilistic predictions)
- Custom model architectures
