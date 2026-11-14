# Experiment comparison (sorted by roc_auc)

| runName             | model_type          | data_version                     |   roc_auc |     f1 |   recall |   precision | passed_thresholds   |
|:--------------------|:--------------------|:---------------------------------|----------:|-------:|---------:|------------:|:--------------------|
| logreg_no_rate_cols | logistic_regression | 8bffa75e2a2c8608c3d8715d47cd3000 |    0.8139 | 0.4923 |   0.6809 |      0.3855 | True                |
| gradient_boosting   | gradient_boosting   | 8bffa75e2a2c8608c3d8715d47cd3000 |    0.8135 | 0.3939 |   0.2766 |      0.6842 | False               |
| logreg_C0.1         | logistic_regression | 8bffa75e2a2c8608c3d8715d47cd3000 |    0.8105 | 0.5000 |   0.6809 |      0.3951 | True                |
| logreg_baseline     | logistic_regression | 8bffa75e2a2c8608c3d8715d47cd3000 |    0.8029 | 0.4615 |   0.6383 |      0.3614 | True                |
| logistic_regression | logistic_regression | 8bffa75e2a2c8608c3d8715d47cd3000 |    0.8029 | 0.4615 |   0.6383 |      0.3614 | True                |
| rf_500_deep         | random_forest       | 8bffa75e2a2c8608c3d8715d47cd3000 |    0.7933 | 0.1818 |   0.1064 |      0.6250 | False               |
| rf_200              | random_forest       | 8bffa75e2a2c8608c3d8715d47cd3000 |    0.7837 | 0.5192 |   0.5745 |      0.4737 | True                |

Best run by roc_auc: **logreg_no_rate_cols**

Best run that passed all thresholds: **logreg_no_rate_cols**
