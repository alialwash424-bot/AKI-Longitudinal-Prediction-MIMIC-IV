# Longitudinal Prediction of Acute Kidney Injury in MIMIC-IV

This repository contains reproducibility code, model specifications, and
aggregate results for the study:

**Longitudinal Physiological and Modifiable Exposure Trajectories for Early
Prediction of Acute Kidney Injury in Critically Ill Adults**

## Overview

The study evaluates longitudinal physiological and modifiable-exposure
trajectories for prediction of incident acute kidney injury at 6-, 12-, and
24-hour horizons in critically ill adults.

The primary prediction models use regularized logistic regression.

## Repository structure

- `code/cohort_and_outcomes/` — AKI outcome and cohort construction
- `code/feature_engineering/` — longitudinal trajectory features
- `code/modeling/` — primary and ablation model fitting
- `code/evaluation/` — held-out evaluation and calibration
- `code/reporting/` — figures, tables, and subgroup analyses
- `model_specification/` — predictors and fitted coefficients
- `results/` — aggregate performance results
- `documentation/` — data access and reproducibility information

## Data

Clinical source data were derived from MIMIC-IV.

Patient-level MIMIC-IV data are not redistributed in this repository.
Researchers must independently obtain authorized MIMIC-IV access through
PhysioNet and comply with applicable credentialing requirements and the
data-use agreement.

This repository intentionally excludes patient-level source data,
patient-level derived datasets, split manifests, and patient-level
prediction files.

## Evaluation

Primary model performance was evaluated in a held-out patient-level test
partition. Patient-level bootstrap resampling was used for confidence
interval estimation where specified.

## Scope

The subgroup analyses included here are exploratory predictive performance
analyses. They are not causal treatment-effect analyses and do not establish
statistically significant differences between demographic groups.

## Citation

Final publication and persistent repository citation information will be
added when available.
