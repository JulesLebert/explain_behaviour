from pathlib import Path
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score
from sklearn.inspection import permutation_importance
import numpy as np

from explain_behaviour.models.behavioural_analysis_GBM import BehavioralAnalysisGBM
from explain_behaviour.helpers.plotting import plot_permutation_importance, set_font_axes

EXTS = ['png', 'pdf', 'svg']

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Configuration
config_name = 'whisker_classification_dlc'  # Change this to use different config files
load_dir = 'run_20250725_145707' # Set to a directory name to load previous results, e.g., 'run_20250318_171524'

results_dir = Path('/mnt/lsens-analysis/Jules_Lebert/results/context/behaviour_model/')
results_dir = results_dir / config_name
# Load previously saved results
run_name = load_dir
load_dir = results_dir / load_dir
config_path = load_dir / 'config.yaml'
config = load_config(config_path)
analyzer, results = BehavioralAnalysisGBM.load_results(load_dir)
print("\nLoaded Analysis Results:")

X_test = analyzer.X_test
y_test = analyzer.y_test

perm_results = permutation_importance(
        analyzer.model, X_test, y_test,
        n_repeats=10000,
        random_state=analyzer.random_state,
        scoring='balanced_accuracy',
        n_jobs=-1,
    )

# Compute 95% confidence intervals for each feature
importances = perm_results.importances  # shape: (n_features, n_repeats)
ci_lower = np.percentile(importances, 2.5, axis=1)
ci_upper = np.percentile(importances, 97.5, axis=1)

# Determine significance: CI does not include zero
significant = (ci_lower > 0) | (ci_upper < 0)

feature_names = X_test.columns if hasattr(X_test, 'columns') else np.arange(importances.shape[0])

# Print significant features
print("Significant features (95% CI does not include 0):")
for name, lower, upper, sig in zip(feature_names, ci_lower, ci_upper, significant):
    if sig:
        print(f"{name}: 95% CI = [{lower:.4f}, {upper:.4f}]")

# Plot only significant features, highlight them
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

sorted_idx = perm_results.importances_mean.argsort()
# Only plot significant features
sig_idx = sorted_idx[significant[sorted_idx]]
non_sig_idx = sorted_idx[~significant[sorted_idx]]

# Plot non-significant in gray
ax.barh([feature_names[i] for i in non_sig_idx], perm_results.importances_mean[non_sig_idx],
        xerr=[perm_results.importances_mean[non_sig_idx] - ci_lower[non_sig_idx],
              ci_upper[non_sig_idx] - perm_results.importances_mean[non_sig_idx]],
        color='lightgray', edgecolor='k', label='Not significant', alpha=0.7)
# Plot significant in color
ax.barh([feature_names[i] for i in sig_idx], perm_results.importances_mean[sig_idx],
        xerr=[perm_results.importances_mean[sig_idx] - ci_lower[sig_idx],
              ci_upper[sig_idx] - perm_results.importances_mean[sig_idx]],
        color='tab:blue', edgecolor='k', label='Significant', alpha=0.9)

ax.set_xlabel('Permutation Importance')
ax.set_title('Permutation Importances', fontsize=10)
ax.legend()
fig.tight_layout()
sns.despine(fig=fig)

save_path=results_dir / run_name / 'figures'
save_path.mkdir(exist_ok=True)
save_fig_name = save_path / 'permutation_importance'
for ext in EXTS:
    fig.savefig(save_fig_name.with_suffix(f'.{ext}'), dpi=300)
