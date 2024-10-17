from experiment import run_private_regression, RegressionMethod, FeatureSelectionMethod, run_nondp
from get_data import get_data_from_id
import numpy as np

all_task_ids = [
    361072, 361073, 361074, 361075, 361076, 361077, 361078, 361079,
    361085, 361087, 361088, 361089, 361091, 361092, 361093, 361094,
    361095, 361096, 361098, 361099, 361100, 361101, 361102, 361103, 361104
]

task_id = 361090

# regression generic info
num_trials = 10  # repeat experiments 10 times and report the quantiles R values to measure perf
train_frac = 0.9

# total privacy budget
total_epsilon = np.log(3)

# privacy budgets and params
feature_selection_epsilon = 0.05 * total_epsilon    # 5% of total epsilon
compute_n_epsilon = 0.05 * total_epsilon            # 5% of total epsilon
regression_epsilon = 0.9 * total_epsilon            # 90% of total epsilon
regression_delta = 1e-5                             #
compute_n_eta = 1e-4                                # failure probability for lower bound to choose # of models

# feature selection
number_of_features = 5  # or 10
X, Y = get_data_from_id(task_id)

print(f"Task ID: {task_id}, # features: {number_of_features}.\n============================================")
# BASELINE
use_lasso = True
results = run_nondp(use_lasso, X, Y, train_frac, num_trials)
print(f"LASSO NON-DP RESULTS:\n\tR^2 Quantiles: {results}")
use_lasso = False
results = run_nondp(use_lasso, X, Y, train_frac, num_trials)
print(f"NON-LASSO NON-DP RESULTS:\n\tR^2 Quantiles: {results}")

# DP METHODS
feature_selection_method = FeatureSelectionMethod.KENDALL
regression_method = RegressionMethod.BAS
method_name = f"{regression_method.name}-{feature_selection_method.name}"
print(f"Method: {method_name}. Running regression...")
results = run_private_regression(
    features=X,
    labels=Y,
    train_frac=train_frac,
    regression_method=regression_method,
    feature_selection_method=feature_selection_method,
    feature_selection_epsilon=feature_selection_epsilon,
    feature_selection_k=number_of_features,
    compute_n_epsilon=compute_n_epsilon,
    compute_n_eta=compute_n_eta,
    regression_epsilon=regression_epsilon,
    regression_delta=regression_delta,
    num_trials=num_trials
)
print(f"\tR^2 Quantiles: {results}")

# results = run_private_regression(
#     features=X,
#     labels=Y,
#     train_frac=0.9,
#     regression_method=RegressionMethod.BAS,  # Or RegressionMethod.TUKEY
#     feature_selection_method=FeatureSelectionMethod.KENDALL, # Or FeatureSelectionMethod.LASSO or FeatureSelectionMethod.NONE
#     feature_selection_epsilon=0.05 * np.log(3),
#     feature_selection_k=5,  # Or 10
#     compute_n_epsilon=0.05 * np.log(3),
#     compute_n_eta=1e-4,
#     regression_epsilon=0.9 * np.log(3),
#     regression_delta=1e-5,
#     num_trials=10,
# )
# print(f"R^2 quantiles: {results}")
#
