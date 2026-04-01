# Customer Churn Prediction
Binary classification of telecom customer churn using a Kaggle dataset of over 500,000 records. Identifies a distributional inconsistency in the original train/test split and demonstrates that correcting the data partition substantially improves model generalization. Compares Logistic Regression (AIC and BIC feature selection) with Random Forest, and derives actionable business insights through coefficient analysis and variable importance.

## Key Techniques
- Binary classification with distribution shift detection and correction
- Feature selection: backward elimination with AIC (k=2) and BIC (k=log(n))
- Model comparison: Logistic Regression vs. Random Forest
- Multicollinearity assessment via VIF
- Coefficient analysis with odds ratios and 95% confidence intervals
- 5-fold cross-validation with caret

## Tools
R &bull; caret &bull; pROC &bull; randomForest &bull; ggplot2 &bull; corrplot &bull; car

## Repository
- `report/churn_report.tex` &mdash; LaTeX source file
- `report/churn_report.pdf` &mdash; Final report
- `code/churn_analysis.ipynb` &mdash; Main analysis notebook
- `code/churn_analysis.R` &mdash; Clean R script version of the analysis  
- `code/config.r` &mdash; Configuration file (data paths)

## References

muhammadshahidazeem. (2024). Customer Churn Dataset [Dataset]. Kaggle.
https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset