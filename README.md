# Telco Customer Churn Prediction

Predicting which telecom customers are likely to leave, using the Telco Customer Churn dataset (7,043 customers, 21 columns). The project covers data cleaning, exploratory analysis, preprocessing, model comparison, hyperparameter tuning, threshold selection and feature importance.

Everything is in a single notebook: [main.ipynb](main.ipynb).

## Dataset

`Telco-Customer-Churn.csv` contains one row per customer with demographics, account information (tenure, contract, payment method, charges), subscribed services, and the target `Churn` (Yes/No).

About 26.5% of customers churned, so the classes are imbalanced. A model that always predicts "no churn" would score about 73.5% accuracy, which is why accuracy is not used as the main metric.

## Approach

1. **Cleaning:** dropped `customerID`; converted `TotalCharges` to numeric. The 11 blank values all belong to customers with `tenure = 0`, so they were filled with 0.
2. **Exploratory analysis:** target distribution, histograms, boxplots of numeric features by churn, count plots of categorical features by churn, and a correlation heatmap. `tenure` and `TotalCharges` are strongly correlated (0.83).
3. **Preprocessing:** binary columns encoded as 0/1, remaining categoricals one-hot encoded with `drop_first=True`, and `Churn` encoded as Yes = 1, No = 0.
4. **Split and scaling:** stratified 80/20 train/test split; `tenure`, `MonthlyCharges` and `TotalCharges` standardized with a scaler fitted on the training set only, to avoid data leakage.
5. **Models:** logistic regression (baseline and class-weighted), Random Forest, and Gradient Boosting.
6. **Tuning:** `RandomizedSearchCV` (20 candidates, 5-fold stratified CV, scored on ROC-AUC) on Gradient Boosting.
7. **Threshold selection:** the decision threshold was chosen on out-of-fold training predictions, so the test set was not used for tuning.
8. **Interpretation:** feature importances from the tuned model.

## Results

Test-set results for the churn class at the default 0.5 threshold:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.657 | 0.559 | 0.604 | 0.842 |
| Logistic Regression (balanced) | 0.505 | 0.783 | 0.614 | 0.842 |
| Random Forest | 0.543 | 0.730 | 0.623 | 0.843 |
| Gradient Boosting | 0.655 | 0.508 | 0.572 | 0.842 |

- All four models rank customers almost equally well (ROC-AUC about 0.84); they differ mainly in the precision/recall trade-off.
- **Tuned Gradient Boosting:** cross-validated ROC-AUC 0.849, test ROC-AUC 0.846.
- **Final model at threshold 0.33:** catches 267 of 374 churners (recall 0.71) with precision 0.54, at the cost of 227 false alarms among 1,035 loyal customers.
- **Most important features:** `tenure`, fiber-optic internet service, electronic-check payment, and contract type. Feature importance shows what the model relies on, not the direction of each effect.

The best threshold depends on the cost of a retention offer versus the value of a retained customer.

## Running the notebook

Requirements: Python 3 with `numpy`, `pandas`, `matplotlib`, `seaborn` and `scikit-learn`.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook main.ipynb
```

Run the cells from top to bottom. The CSV must be in the same folder as the notebook.

## Project structure

```
.
├── main.ipynb                  # full analysis and modeling
├── Telco-Customer-Churn.csv    # dataset
└── README.md
```
