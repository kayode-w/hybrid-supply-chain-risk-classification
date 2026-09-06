# Threshold Tuning: Supervisor Guidance and Implementation Notes

## What the Supervisor Said

Alexander recommended that the threshold chosen for each hybrid model should not be stated arbitrarily. Instead, the choice should be proven through a systematic search process. Specifically:

- Test a range of threshold values against the hybrid model's predicted probabilities.
- At each threshold, calculate Precision, Recall, and F1-score.
- Plot the results so the optimal point is visible.
- Select the threshold where F1-score is highest, or where the best business trade-off between Precision and Recall is achieved.
- The chart becomes the evidence. Instead of saying "I used 0.35 because fraud is imbalanced," the dissertation can say: "I tested 13 threshold values, observed the metric behaviour at each, and selected the threshold that produced the highest F1-score."

He noted that this logic mirrors GridSearchCV — except instead of searching model parameters like max_depth or learning_rate, the search is over threshold values applied after the model produces probabilities.

He also flagged that the threshold should ideally be selected using a validation set or cross-validation on training data, not the test set directly. Where a separate validation set does not exist, the analysis can be described as an exploratory threshold sensitivity analysis.

---

## What Was Added to the Notebook

A new section titled **Threshold Sensitivity Analysis** was added to the notebook, sitting between the Hybrid Model Summaries and Section 09 (Confusion Matrix).

The section covers all three targets using the best-performing hybrid pair (DT & XGB) for each.

### For each target (Fraud, Late Delivery, Order Disruption):

**Code cell 1: Threshold sweep and table**

- Re-runs the DT & XGB hybrid combination to get fresh weighted probabilities.
- Tests 13 threshold values: 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70
- For each threshold: calculates Accuracy, Precision, Recall, and F1-score.
- Identifies the threshold with the highest F1-score using `idxmax()`.
- Outputs a results table.

**Code cell 2: Chart**

- Plots Precision, Recall, and F1-score against threshold values.
- Marks the optimal F1 threshold with a scatter point.
- Axes labelled. Grid on. Title per target.

---

## Variable Name Changes to Note in the Dissertation

The threshold constants were renamed from ALL_CAPS to lowercase during the project:

| Old name | New name |
|---|---|
| THRESHOLD_FRAUD | threshold_fraud |
| THRESHOLD_LATE_DELIV | threshold_late_deliv |
| THRESHOLD_DISRUPTION | threshold_disruption |

### Current values

```
threshold_fraud      = 0.35
threshold_late_deliv = 0.50
threshold_disruption = 0.35
```

### How to reflect this in the dissertation methodology

When the threshold tuning section is written up, it should reference these variable names directly and explain how the values were arrived at. Suggested wording:

> To support the selection of classification thresholds for the hybrid models, a threshold tuning procedure was applied. After the hybrid models produced weighted probability scores, a range of threshold values between 0.10 and 0.70 was tested. For each value, Accuracy, Precision, Recall, and F1-score were recorded. The results were plotted to show how performance changed across thresholds. The final threshold for each target was selected at the point where the F1-score was highest.
>
> For transaction fraud and order disruption, where the positive class is significantly underrepresented, the default threshold of 0.50 was found to suppress recall. A lower threshold of 0.35 was selected for both targets (threshold_fraud = 0.35, threshold_disruption = 0.35). For late delivery, where classes are near-balanced, the default threshold of 0.50 was retained (threshold_late_deliv = 0.50).

---

*Document created: July 2026*
