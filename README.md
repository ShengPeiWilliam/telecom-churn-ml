# Customer Churn Prediction

Binary classification of telecom customer churn using a Kaggle dataset of over 500,000 records. Compares Logistic Regression with Random Forest after identifying and correcting a distributional inconsistency in the original train/test split.

## Motivation

The initial model looked fine on paper, 89.5% training accuracy. But test accuracy dropped to 57.1% with a specificity of only 19.4%. Rather than immediately tuning the model, the first step was asking why: distribution analysis revealed that the original train and test sets had fundamentally different feature distributions. This wasn't a modeling problem, it was a data problem. After a random re-split, the same Logistic Regression model jumped to 84.8% test accuracy without any changes to the model itself.

The lesson was clear before you tune anything, make sure the data partition is sound.

## Design Decisions

**Why start with Logistic Regression?**

For binary classification on structured data, logistic regression gives you interpretable coefficients with odds ratios and significance tests. You can say "monthly contract customers are 8.1x more likely to churn" in a way that directly informs business action. Starting simple also establishes a baseline that justifies whether more complex models add real value.

**Why compare against Random Forest?**

To quantify how much nonlinear patterns matter. Random Forest achieved 93.6% accuracy and AUC of 0.953 vs. Logistic Regression's 84.8% and 0.908. The gap confirms that interaction effects and nonlinearities exist in this data, but logistic regression still captures the core signal well enough for actionable insights.

**How were features selected?**

Backward elimination using both AIC (k=2) and BIC (k=log(n)). BIC removed `Tenure` with no impact on performance, suggesting it carried redundant information. VIF screening handled multicollinearity before model fitting.

## Key Results

| Model | Accuracy | Sensitivity | Specificity | F1 | AUC |
|-------|----------|-------------|-------------|------|-----|
| LR (original split) | 57.1% | 99.1% | 19.4% | 0.686 | 0.683 |
| LR AIC (re-split) | 84.8% | 84.8% | 84.7% | 0.861 | 0.908 |
| LR BIC (re-split) | 84.8% | 84.8% | 84.7% | 0.861 | 0.908 |
| **Random Forest (re-split)** | **93.6%** | **99.7%** | **85.9%** | **0.945** | **0.953** |

The strongest churn predictors from logistic regression: Contract Length (monthly odds ratio = 8.10, meaning month-to-month customers are 8x more likely to churn) and Support Calls (odds ratio = 1.50, each additional call increases churn odds by 50%).

## Reflections & Next Steps

The most important finding had nothing to do with modeling. The original dataset's train/test split was flawed, and no amount of hyperparameter tuning would have fixed 19.4% specificity. Catching this through distribution analysis before blaming the model saved significant effort and led to a much more honest evaluation.

That said, Random Forest's higher accuracy comes at the cost of interpretability. Logistic regression remains more practical for deriving actionable business insights. The Random Forest was also trained on a 50,000 subsample due to computational constraints, which may not fully represent the complete dataset. More fundamentally, the data is cross-sectional: without longitudinal records, it's not possible to track how individual customer behavior evolves or catch early warning signs of churn.

Next steps:
- **Class imbalance**: explore SMOTE or threshold tuning to improve sensitivity on the minority class.
- **Temporal validation**: a time-based split would better reflect real-world deployment where you predict future churn from historical data.
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