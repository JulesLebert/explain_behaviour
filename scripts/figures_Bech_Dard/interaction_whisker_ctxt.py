from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from explain_behaviour.models.behavioural_analysis_GBM import BehavioralAnalysisGBM
# from explain_behaviour.helpers import plotting

import seaborn as sns
import matplotlib.pyplot as plt

EXTS = ['png', 'pdf', 'svg']

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Configuration
config_name = 'whisker_classification_dlc'  # Change this to use different config files
load_dir = 'run_20250725_145707' # Set to a directory name to load previous results, e.g., 'run_20250318_171524'

# results_dir = Path(__file__).parent.parent / 'results' / config_name
results_dir = Path('/mnt/lsens-analysis/Jules_Lebert/results/context/behaviour_model') / config_name
results_dir.mkdir(parents=True, exist_ok=True)

# Load previously saved results
run_name = load_dir
load_dir = results_dir / load_dir
config_path = load_dir / 'config.yaml'
config = load_config(config_path)
analyzer, results = BehavioralAnalysisGBM.load_results(load_dir)
print("\nLoaded Analysis Results:")

X_disp = pd.concat([analyzer.analysis_results['X_train'], analyzer.analysis_results['X_test']])

shap_values = analyzer.analysis_results['shap_values']
# Ensure shap_values is 2D
if len(shap_values.shape) == 1:
    shap_values = shap_values.reshape(-1, 1)

# Create column names for shap_values: shap_{col}
shap_col_names = [f"shap_{col}" for col in X_disp.columns]
shap_df = pd.DataFrame(shap_values, columns=shap_col_names, index=X_disp.index)

# Merge X_disp and shap_df over columns
X_disp = pd.concat([X_disp, shap_df], axis=1)
X_disp = X_disp.loc[(X_disp['Whisker Speed'] <= 20) & (X_disp['Whisker Speed'] > 0.1)]

# sns.histplot(
#     X_disp['Whisker Angle std'], 
#     stat='proportion',
#     bins=np.logspace(0.1,1.5, 50),
#     )
# sns.ecdfplot(data=X_disp, x='Whisker Angle std')

interaction_features = ['Whisker Speed', 'Context']

type_a = 'continuous'
X_disp['Context'] = X_disp['Context'].astype(bool)

feature_a, feature_b = interaction_features
feature_a_index = X_disp.columns.get_loc(feature_a)
feature_b_index = X_disp.columns.get_loc(feature_b)

fig, ax = plt.subplots(1, 1, figsize=(5,3), dpi=300)
# Create custom palette mapping 0 to dark magenta and 1 to green
custom_palette = {0: 'darkmagenta', 1: 'green'}  # Dark magenta and green

plots_kwargs = {
    'edgecolor' : None,
    'linewidth' : 1,
    'alpha' : 0.3,
    'size' : 4,
}

num_bins = 50  # You can adjust the number of bins
# Use log-spaced bins for 'Whisker Angle std'
min_val = X_disp['Whisker Speed'].min()
max_val = X_disp['Whisker Speed'].max()
# Ensure min_val > 0 for logspace
min_val = max(min_val, 1e-3)
log_bins = np.logspace(np.log10(min_val), np.log10(max_val), num_bins + 1)
X_disp['Whisker_bin'] = pd.cut(X_disp['Whisker Speed'], bins=log_bins, include_lowest=True)
# X_disp['bin_center'] = X_disp['Whisker_bin'].apply(lambda x: x.mid)
# For log bins, use geometric mean of bin edges as center
X_disp['bin_center'] = X_disp['Whisker_bin'].apply(lambda x: np.sqrt(x.left * x.right) if pd.notnull(x) else np.nan)

# grouped = X_disp.groupby(['Whisker_bin', 'Context'])['shap_Whisker Angle std'].agg(['mean', 'std', 'count', 'sem']).reset_index()
# grouped = grouped.rename(columns={'mean': 'shap_Whisker Angle std'})

sns.lineplot(
    data=X_disp,
    x='bin_center',
    y='shap_Whisker Speed',
    hue='Context',
    palette=custom_palette,
    ax=ax,
)
ax.set_xscale('log')
ax.set_xlabel('Whisker speed (deg/s)')
ax.set_ylabel('Shap value shisker speed')
ax.axhline(y=0, linestyle='--', c='k')
# plotting.set_font_axes(ax=ax, add_size)
sns.despine(fig=fig, offset=10)


save_path=results_dir / run_name / 'figures'
save_path.mkdir(exist_ok=True)
save_file = save_path / f'interaction_{interaction_features}'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

for ext in EXTS:
    fig.savefig(save_file.with_suffix(f'.{ext}'), dpi=300)

X = pd.concat([analyzer.analysis_results['X_train'], analyzer.analysis_results['X_test']])

