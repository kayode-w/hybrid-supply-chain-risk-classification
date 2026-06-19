# Supply Chain Risk Prediction: Hybrid ML Models

An investigation into whether blending multiple classification models together produces better predictions than any single model alone, applied to three supply chain risk problems using the DataCo Smart Supply Chain dataset.

## The Idea

The starting point was a simple question: if you take four different models, each with their own strengths, and combine their probability outputs using a weighted average, do you get something more useful? And does that approach hold up across three completely different classification problems on the same dataset?

The three problems tested:

- **Transaction Fraud** - is this order likely to be fraudulent?
- **Late Delivery Risk** - will this order arrive late?
- **Order Disruption** - will this order be cancelled, held, or flagged for payment review?

## What Was Built

Four base models were trained on each target:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Then six hybrid combinations were built for each target using weighted probability averaging. The weight given to each model in a combination is its F1 score on the positive class, so stronger models have more influence over the final prediction.

That gives 10 models per target and 30 evaluations in total.

## Key Findings

**Fraud:** The RF + XGB hybrid was the best performer at F1 0.70, narrowly beating XGBoost alone at 0.69. The only case where a hybrid clearly outperformed every base model.

**Late Delivery:** A plain Decision Tree won at F1 0.79. No ensemble or hybrid came close. Sometimes the simplest model is the right one.

**Disruption:** Everything struggled here. The best result was Decision Tree at F1 0.31. The dataset simply does not contain enough signal to predict cancellations and holds reliably. That is a data problem, not a modelling one.

## What This Shows

A single hybrid approach does not transfer equally well across different risk types. Each problem has its own characteristics. Fraud benefits from ensembling. Late delivery follows clean enough patterns that a single tree handles it well. Disruption needs richer data that this dataset does not provide.

## Dataset

DataCo Smart Supply Chain Dataset via mendeley.

## Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Jupyter Notebook
