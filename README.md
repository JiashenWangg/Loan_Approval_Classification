# Loan Default Classification (SBA Loans)

## Overview

This project builds an end-to-end **loan default classification pipeline** using a large-scale U.S. Small Business Administration (SBA) loan dataset (~900K observations). The objective is to predict whether a loan is **Paid in Full (PIF)** or **Charged Off (CHGOFF)**, with a primary focus on minimizing costly missed defaults under class imbalance.

The project emphasizes rigorous data cleaning, feature engineering, principled model comparison, and evaluation using metrics aligned with real-world credit-risk decision making.

---

## Problem Definition

* **Task**: Binary classification (default vs. non-default)
* **Positive class**: CHGOFF (loan default)
* **Key challenge**: Strong class imbalance and asymmetric error costs

  * Type II error (missed default) is significantly more costly than Type I error

---

## Data

* Source: U.S. SBA loan-level data
* Raw size: ~899K rows, 27 columns
* Final modeling dataset: ~857K rows, 17 features
* Target variable: `MIS_Status`

  * `0` = Paid in Full (PIF)
  * `1` = Charged Off (CHGOFF)

### Data Cleaning

* Removed identifiers and post-outcome variables to prevent data leakage
* Dropped rows with minimal missingness (<2%)
* Converted monetary fields to numeric and dates to datetime
* Applied business-rule constraints and outlier filtering

---

## Feature Engineering

Key engineered features include:

* **Disbursement Lag**: Time between loan approval and disbursement
* **SBA Guarantee Share**: SBA guaranteed portion of total loan amount
* **Franchise Indicator**: Binary indicator of franchise affiliation
* **Industry Aggregation**: NAICS codes grouped into higher-level sectors
* **High-cardinality handling**: Top-K encoding for Bank, State, and BankState

Final feature set:

* 7 numerical features
* 9 categorical features
* 1 binary target

---

## Modeling Approach

All models are trained using a unified **scikit-learn Pipeline** with consistent preprocessing for fair comparison.

### Models Evaluated

* **Logistic Regression** (interpretable baseline)
* **Random Forest** (bagging-based nonlinear model)
* **AdaBoost** (sequential boosting with weak learners)
* **XGBoost** (gradient-boosted trees with regularization)

Train–test split:

* 80% training / 20% testing
* Stratified by target class

---

## Evaluation Metrics

Model performance is evaluated using:

* **Recall (CHGOFF)** – primary metric (minimize missed defaults)
* Precision
* F1-score
* ROC-AUC
* Confusion matrix
* Precision–Recall and ROC curves

---

## Results Summary

| Model               | Accuracy  | Recall (Default) | F1 (Default) | ROC-AUC   |
| ------------------- | --------- | ---------------- | ------------ | --------- |
| Logistic Regression | ~0.86     | ~0.37            | ~0.49        | ~0.90     |
| Random Forest       | ~0.89     | ~0.42            | ~0.58        | ~0.92     |
| AdaBoost            | ~0.86     | ~0.40            | ~0.51        | ~0.90     |
| **XGBoost**         | **~0.93** | **~0.73**        | **~0.79**    | **~0.96** |

**XGBoost** provides the best trade-off between Type I and Type II errors, substantially reducing false negatives while maintaining strong overall accuracy.

---

## Model Interpretation

Feature importance analysis (gain-based) from XGBoost highlights:

* Loan term
* Disbursement timing
* Bank identity
* SBA guarantee share
* Urban vs. rural classification

These drivers align with financial intuition and known credit-risk factors.

---

## Key Takeaways

* Metric choice matters: optimizing recall for defaults materially changes model selection
* Tree-based boosting methods outperform linear and bagging models in complex credit data
* XGBoost is well-suited for production-grade credit-risk modeling under class imbalance


---

## Author

**Jiashen Wang**
