# Predicting Supply Chain Risk with Hybrid Machine Learning

I wanted to find out whether combining machine learning models produces better risk
predictions than using one model on its own. The idea seemed reasonable going in.
Different algorithms make different kinds of mistakes, so blending them should
cancel some of that out.

Mostly, it didn't. This repository is the work behind that answer.

## The Question

The dissertation question was straightforward to state: how effectively can hybrid
machine learning techniques predict supply chain operational risk compared with the
base models they are built from?

The trap in a question like that is testing it once and calling it settled. A hybrid
that wins on one dataset and one target tells you very little. So I ran the same
approach across three separate risk problems on the DataCo Smart Supply Chain
dataset, 180,519 orders and 53 variables, to see whether any pattern held up.

## Three Targets, Not One

| Target | What it asks | Class balance |
|---|---|---|
| Transaction fraud | Is this order likely to be fraudulent? | Severely imbalanced |
| Late delivery risk | Will this order arrive late? | Near balanced |
| Order disruption | Will this order be cancelled, held, or flagged for payment review? | Moderately imbalanced |

I picked these three deliberately because they sit at different points on the
imbalance scale. Fraud is rare, late delivery is roughly a coin flip, disruption
sits in between. If hybrids only help when classes are balanced, or only when they
are skewed, running all three should expose that.

## How the Hybrids Work

Four base models on each target: Logistic Regression, Decision Tree, Random Forest
and XGBoost.

Then every pairwise combination of those four, giving six hybrids per target. Each
hybrid takes the predicted probabilities from both models and averages them, but
weighted rather than evenly. The weight is each model's F1 score on the positive
class, so the model that performs better on the minority class pulls the blended
probability further towards its own view.

Ten models per target, thirty evaluations in total.

Two details that mattered more than I expected when I started:

**SMOTE goes on the training set only.** Oversampling before the split leaks
synthetic minority examples into the test set and every score comes back inflated. I
caught this early, but it is an easy mistake to make.

**The default 0.50 threshold is wrong for imbalanced targets.** A model can rank
cases well and still score badly at 0.50 because it rarely pushes a probability that
high. I tuned the threshold per target rather than accepting the default, landing on
0.35 for fraud, 0.45 for late delivery and 0.35 for disruption. The
reasoning is found  in `threshold_tuning_notes.md`.

## What I Found

The summary is that hybrids gave selective gains rather than general ones.
They improved ROC-AUC and recall in places, which matters if businesses care more about
catching risky orders than about being right every time they flag one. But they did
not consistently beat the strongest base model, and on two of the three targets a
plain Decision Tree was the thing to beat.

Fraud was the one case where a hybrid came out ahead, and the margin was thin enough
that I would not want to claim much from it without repeating the split several
times. One test set is one sample.

## Where It Fell Short

**Order disruption never really worked.** I built that target by combining
cancellations, holds and payment review flags into a single label. That felt sensible
at the time, since all three are ways an order goes wrong. Looking at the results, I
think it was a mistake. A customer cancelling an order and a payment being held for
review are driven by completely different things, and asking one model to learn both
at once probably diluted whatever signal was there. Splitting it into three separate
targets is the first thing I would try with more time.

**The weighting scheme could be tighter.** Weighting by F1 score is intuitive, but it
is a fairly blunt instrument. A stacked model that learns the blend weights from the
data would be the natural next step.

**I only tested pairs.** Six pairwise hybrids per target was a manageable scope for a
dissertation, but three way and four way combinations were left on the table.

## Running It

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Run `supply_chain_dataset.ipynb` first for the cleaning and feature engineering, then
the prediction notebook for training and evaluation.

The dataset is not in this repository. Download it from the source below and put it
in `data/`.

## Data

Constante, F., Silva, F., & Pereira, A. (2019). *DataCo Smart Supply Chain for Big
Data Analysis*. Mendeley Data. https://doi.org/10.17632/8gx2fvg2k6.5

## Stack

Python, pandas, scikit-learn, XGBoost, imbalanced-learn, Jupyter.

## Context

Applied research project for the MSc in Business Analytics at Dublin Business School.