# Customer Churn Prediction
Binary classification of telecom customer churn with Logistic Regression and Random Forest. Identified and corrected a distributional inconsistency in the original train/test split, improving test accuracy from 57.1% to 84.8%. Random Forest achieves 93.6% accuracy and AUC of 0.953 after the correction.

## Motivation

Customer churn is a classic binary classification problem, and a good one for practicing logistic regression: the coefficients translate directly into odds ratios, giving you interpretable answers about what actually drives churn.

## Design Decisions

**Why compare against Random Forest?**

To capture nonlinear patterns that logistic regression can't model, and to use variable importance as a way to cross-check which predictors actually matter most.

**How were features selected?**

Backward elimination using both AIC (k=2) and BIC (k=log(n)) to observe how different penalty strengths affect which predictors are retained.

## Key Results

Random Forest achieves 93.6% accuracy and AUC of 0.953 after correcting the train/test split. The strongest churn predictors from logistic regression: Contract Length (monthly odds ratio = 8.10) and Support Calls (odds ratio = 1.50).

| Model | Accuracy | Sensitivity | Specificity | F1 | AUC |
|-------|----------|-------------|-------------|------|-----|
| LR (original split) | 57.1% | 99.1% | 19.4% | 0.686 | 0.683 |
| LR AIC (re-split) | 84.8% | 84.8% | 84.7% | 0.861 | 0.908 |
| LR BIC (re-split) | 84.8% | 84.8% | 84.7% | 0.861 | 0.908 |
| **Random Forest (re-split)** | **93.6%** | **99.7%** | **85.9%** | **0.945** | **0.953** |

## Reflections & Next Steps

The most valuable takeaway: the original train/test split was flawed, and no hyperparameter tuning would have fixed 19.4% specificity. Always check data distribution before trusting any evaluation metric.

On the modeling side, Random Forest outperforms logistic regression on accuracy, but at the cost of interpretability. For a business question like churn, knowing *why* a customer is likely to leave is often more actionable than a slightly higher AUC.

Next steps:
- **Longitudinal data**: the current dataset is cross-sectional, capturing a snapshot of each customer. Tracking the same customers over time would allow early churn signals to be detected before they fully manifest.
- **Model interpretability**: SHAP values on the Random Forest would reveal whether it's capturing the same predictors as logistic regression or finding different patterns entirely.

## Repository

- `report/churn_report.pdf`: Final report
- `code/churn_analysis.ipynb`: Main analysis notebook
- `code/churn_analysis.R`: Clean R script version
- `code/config.r`: Configuration file (data paths)

## Tools

R · caret · pROC · randomForest · ggplot2 · corrplot · car

## References

muhammadshahidazeem. (2024). Customer Churn Dataset [Dataset]. Kaggle. https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset