# English Premier League — Match Result Prediction

A machine-learning study using 30 seasons of English Premier League data (1993–2022) to classify match outcomes (Home win / Draw / Away win), and — just as importantly — a worked example of how **target leakage** can produce misleadingly high accuracy.

> **Heads-up on the results:** The very high accuracies reported below are *not* a genuine predictive achievement. They are caused by leakage: the prediction target is derived directly from two of the input features. This README explains why, and what a leak-free version would look like. See [Important note on the results](#important-note-on-the-results).

---

## Overview

Predicting football results is a popular and intuitive entry point into classification. This project loads a historical Premier League match dataset, performs basic exploratory analysis, encodes categorical fields, and trains five standard scikit-learn classifiers to predict the **Full Time Result (FTR)** of each match.

The project also serves as a case study in a mistake that's easy to make and important to catch: feeding the model information it would never have at prediction time.

## Dataset

- **Source:** [Premier League Matches 1992–2022 (Kaggle)](https://www.kaggle.com/datasets/evangower/premier-league-matches-19922022)
- **Size:** 11,646 matches, 8 columns
- **Seasons covered:** 1993 to 2022

| Column | Description |
|---|---|
| `Season_End_Year` | Year the season ended |
| `Wk` | Match week |
| `Date` | Match date |
| `Home` | Home team |
| `HomeGoals` | Goals scored by the home team |
| `AwayGoals` | Goals scored by the away team |
| `Away` | Away team |
| `FTR` | Full Time Result — **H** (home win), **D** (draw), **A** (away win) |

Class balance of the target: H = 5,335, A = 3,301, D = 3,010 (home advantage is clearly visible).

## Methodology

1. **Load & inspect** — read the CSV, check shape, dtypes, and summary statistics.
2. **Type cleaning** — cast columns to appropriate types and parse `Date` as datetime.
3. **Encoding** — build a lookup table assigning each of the 50 teams a numeric `TeamID`, then map `Home` and `Away` to `Home_encoded` / `Away_encoded`. The target `FTR` is label-encoded to `FTR_encoded` (A=0, D=1, H=2).
4. **Train/test split** — `train_test_split` from scikit-learn.
5. **Modelling** — train and evaluate five classifiers, comparing accuracy and confusion matrices.

**Features used:** `Season_End_Year`, `Wk`, `HomeGoals`, `AwayGoals`, `Home_encoded`, `Away_encoded`
**Target:** `FTR_encoded`

## Models & results

| Model | Test accuracy |
|---|---|
| Decision Tree | ~0.9997 |
| Random Forest | ~0.9989 |
| Naive Bayes (Gaussian) | ~0.8950 |
| Logistic Regression | ~0.8832 |
| K-Nearest Neighbours | ~0.4960 |

These numbers should be read alongside the note below — they describe the model's ability to recover a rule it was effectively handed, not its ability to forecast unseen matches.

## Important note on the results

The target, `FTR`, is **defined by** `HomeGoals` and `AwayGoals`:

- `HomeGoals > AwayGoals` → **H**
- `HomeGoals < AwayGoals` → **A**
- `HomeGoals == AwayGoals` → **D**

Because `HomeGoals` and `AwayGoals` are included as input features, the model is given the final score when asked to predict the result. This is **target leakage** — the inputs contain the answer.

This explains the observed pattern of scores:

- **Tree-based models (Decision Tree, Random Forest)** can learn exact split thresholds, so they recover the rule almost perfectly (~100%).
- **Logistic Regression and Naive Bayes** approximate the same boundary and land around 88–89%.
- **KNN** drops to ~50% because the large, arbitrary team-ID values dominate the distance calculation and drown out the goal columns.

In a real forecasting setting, the final score is exactly what you're trying to predict — it isn't available before kickoff. So these accuracies do not reflect any real predictive skill.

## What a leak-free version would look like

To make this a genuine *pre-match* prediction task, the model must use only information known **before** the match is played. Concretely:

- **Remove** `HomeGoals` and `AwayGoals` from the feature set.
- **Engineer pre-match features**, for example:
  - recent form (points / goals over each team's last *n* matches),
  - rolling goals scored and conceded,
  - head-to-head history between the two teams,
  - home/away context, season stage, days of rest.
- **Respect time order** — split train/test chronologically (train on earlier seasons, test on later ones) rather than randomly, to avoid using the future to predict the past.
- **Expect modest accuracy** — football is noisy. Strong, honest models typically land in the ~50–55% range for the three-way H/D/A task, and beating the "always predict home win" baseline (~46% here) is already meaningful.

Reframed this way, a lower accuracy is a *better* result, because it reflects real predictive signal.

## Tech stack

- Python
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn


Make sure `eplmatches.csv` is in the same directory as the notebook.

## What I learned

- How to take a tabular dataset from raw CSV to a trained, evaluated classifier.
- How encoding choices (e.g. large arbitrary integer IDs) interact with different model families — particularly distance-based ones like KNN.
- How to recognise target leakage from the *shape* of the results, not just the data, and why a near-perfect score is a warning sign rather than a success.

## References

- Dataset: https://www.kaggle.com/datasets/evangower/premier-league-matches-19922022
- Inspiration / EDA reference: https://www.kaggle.com/code/ash316/eda-to-prediction-dietanic
