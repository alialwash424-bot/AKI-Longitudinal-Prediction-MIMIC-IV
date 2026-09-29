# Reproducibility Notes

## Analysis sequence

The repository contains code corresponding to:

1. AKI outcome and cohort construction
2. longitudinal feature engineering
3. patient-level data splitting
4. primary model fitting
5. ablation model fitting
6. held-out test evaluation
7. calibration analysis
8. patient-level bootstrap uncertainty estimation
9. aggregate result generation
10. exploratory subgroup performance analysis

## Missing-data handling

For exposure-trajectory predictors, absence of recorded exposure history was
represented as zero.

For non-exposure predictors requiring imputation, missing values were replaced
using means estimated from the training data.

Training-derived means and standard deviations were used for standardization.

## Final predictor specification

Each primary model used 252 predictors:

- 131 modifiable-exposure trajectory predictors
- 120 physiological vital-sign trajectory predictors
- 1 temporal context predictor

Predictor lists, groups, coefficients, training means, and training standard
deviations are provided in the model specification materials where applicable.

## Held-out evaluation

The test partition was separated at the patient level.

Patient-level test predictions themselves are intentionally not redistributed.

## Exploratory subgroup analysis

Sex, age-group, and sufficiently represented race/ethnicity groups were
evaluated descriptively in the held-out test set.

Patient-level bootstrap confidence intervals were used.

No formal between-group hypothesis tests were performed.
