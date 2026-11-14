# Drift monitoring analysis

## Setup

- reference data: the training split (1176 rows)
- production data: the 294 row test split, after `simulate_drift()` in `src/monitor_drift.py` shifts it. The scenario is roughly a year after deployment: a pay review (+18% MonthlyIncome), everyone a few years older with more tenure, an office move that added commute distance for 60% of people, and a busy period with more overtime and business travel
- Evidently `DataDriftPreset` with the defaults for this data size: normed Wasserstein distance for numeric features and Jensen-Shannon distance for categorical ones, a feature counts as drifted at distance >= 0.1
- alert if more than 25% of features drifted

```bash
python -m src.train                      # so the script can also show the model impact
python -m src.monitor_drift
python -m src.monitor_drift --no-drift
```

## Results

| scenario | drifted features | share | exit code |
|---|---|---|---|
| clean holdout (`--no-drift`) | 2 / 30 | 6.7% | 0 |
| simulated production | 9 / 30 | 30.0% | 1 |

Drifted features in the simulated production data, sorted by score:

| feature | method | score | changed by the simulation? |
|---|---|---|---|
| DistanceFromHome | Wasserstein | 0.467 | yes |
| Age | Wasserstein | 0.339 | yes |
| YearsAtCompany | Wasserstein | 0.213 | yes |
| TotalWorkingYears | Wasserstein | 0.210 | yes |
| BusinessTravel | Jensen-Shannon | 0.204 | yes |
| TrainingTimesLastYear | Wasserstein | 0.191 | no |
| MonthlyIncome | Wasserstein | 0.186 | yes |
| OverTime | Jensen-Shannon | 0.125 | yes |
| YearsWithCurrManager | Wasserstein | 0.116 | no |

## 1. Which features drifted and why?

All seven features the simulation touched were flagged. DistanceFromHome is by far the biggest shift (most people got 4-10 extra miles), then Age / TotalWorkingYears / YearsAtCompany which all move together because the same people are a year or more older, then BusinessTravel and MonthlyIncome. OverTime is just over the line: the share of "Yes" went from 26% to about 46%, which is a big change for a binary column but Jensen-Shannon distance stays fairly small for two categories.

Two features drifted that the simulation did not touch: TrainingTimesLastYear and YearsWithCurrManager. They also show up in the `--no-drift` run, so they are not real drift. Both are small integer scales (0-6 and 0-17) compared on only 294 rows, so the train/test split itself is enough to push the normed Wasserstein distance over 0.1. Without them the share would be 7/30 = 23%, which is just under the alert threshold, so these noisy columns matter for whether the alert fires.

## 2. Would this drift affect model performance?

Yes. Because the production set is simulated from the labelled holdout we can actually measure it with the selected model (logistic regression, C=0.1):

| metric | clean holdout | simulated production |
|---|---|---|
| ROC AUC | 0.810 | 0.774 |
| F1 | 0.500 | 0.444 |
| precision | 0.395 | 0.313 |
| recall | 0.681 | 0.766 |
| accuracy | 0.782 | 0.694 |

OverTime and BusinessTravel are two of the strongest predictors in the model, so when a lot of employees switch to overtime and frequent travel their predicted risk goes up even though in the simulation nobody actually changed their mind about leaving. The model flags many more people: recall goes up, but precision drops by a fifth, accuracy drops 9 points and F1 ends up only just above the 0.40 minimum the pipeline requires. In practice HR would get a lot more false alarms.

One caveat: the simulation shifts the inputs but keeps the labels. In reality more overtime and longer commutes would probably also make more people leave, so some of those new "at risk" predictions could turn out to be right. We can't know that until real attrition labels come in, which takes months.

## 3. Recommended action

Investigate first, then retrain. Just continuing to monitor isn't enough: precision has already dropped a lot and F1 is sitting right at the deployment minimum.

1. Check that the shifts are real and not a data problem. Confirm with HR that there was a pay review, an office move and a busy period. Check that MonthlyIncome is still in the same currency and DistanceFromHome still in miles - a unit change upstream would look exactly like this.
2. In the meantime raise the decision threshold, or give HR a ranked list of risk scores instead of a yes/no flag, so the extra false positives do less damage.
3. Once the shifts are confirmed and a few months of labelled outcomes from the new period exist, retrain on recent data, run it through the same CI thresholds, and make the new training data the reference for drift monitoring.
4. Tune the monitor: use bigger production windows (a quarter instead of 294 rows) or per-feature thresholds for the small integer scales so TrainingTimesLastYear and friends stop triggering, and watch the features the model actually relies on (OverTime, MonthlyIncome, Age, tenure) more closely than the rest.
