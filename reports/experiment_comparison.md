# Experiment comparison

Ranked by val_roc_auc (5-fold stratified CV on the training split).

| runName             | model_type          | data_version                     |   val_roc_auc |   val_f1 |   val_recall |   val_precision | passed_thresholds   |
|:--------------------|:--------------------|:---------------------------------|--------------:|---------:|-------------:|----------------:|:--------------------|
| logreg_C0.1         | logistic_regression | 8bffa75e2a2c8608c3d8715d47cd3000 |        0.8304 |   0.4987 |       0.7368 |          0.3792 | True                |
| logreg_no_rate_cols | logistic_regression | 8bffa75e2a2c8608c3d8715d47cd3000 |        0.8298 |   0.4996 |       0.7368 |          0.3794 | True                |
| logreg_baseline     | logistic_regression | 8bffa75e2a2c8608c3d8715d47cd3000 |        0.8269 |   0.4981 |       0.7474 |          0.3748 | True                |
| gradient_boosting   | gradient_boosting   | 8bffa75e2a2c8608c3d8715d47cd3000 |        0.8213 |   0.4972 |       0.3737 |          0.7797 | False               |
| rf_500_deep         | random_forest       | 8bffa75e2a2c8608c3d8715d47cd3000 |        0.8097 |   0.2777 |       0.1684 |          0.8850 | False               |
| rf_200              | random_forest       | 8bffa75e2a2c8608c3d8715d47cd3000 |        0.7966 |   0.4968 |       0.4895 |          0.5179 | False               |

Selected run: **logreg_C0.1**

Test metrics of the selected run:

| metric | test |
|---|---|
| roc_auc | 0.8105 |
| f1 | 0.5000 |
| recall | 0.6809 |
| precision | 0.3951 |
| accuracy | 0.7823 |
