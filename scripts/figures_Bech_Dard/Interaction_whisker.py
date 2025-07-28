from pathlib import Path
import pandas as pd
import yaml
from explain_behaviour.models.behavioural_analysis_GBM import BehavioralAnalysisGBM
from explain_behaviour.preprocessing.trial_features import prepare_behavior_features

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

interaction_features = ['Whisker trial in block', 'Context']
type_a = 'discrete'


"""Plot interaction only results."""
X_disp = pd.concat([analyzer.analysis_results['X_train'], analyzer.analysis_results['X_test']])
# X_disp = X_disp.sort_index()
shap_values = analyzer.analysis_results['shap_values']

# Remap whisker trial values to negative numbers
mapping = {8: -1, 7: -2, 6: -3, 5: -4}
X_disp['Whisker trial in block'] = X_disp['Whisker trial in block'].astype(int)
X_disp['Whisker trial in block'] = X_disp['Whisker trial in block'].map(mapping).fillna(X_disp['Whisker trial in block'])
# Convert 'Context' to boolean (0 or 1)
X_disp['Context'] = X_disp['Context'].astype(bool)

# Define a function to compute 'context_to_plot'
def get_context_to_plot(row):
    if row['Whisker trial in block'] > 0:
        return row['Context']
    else:
        return not row['Context']

X_disp['Context'] = X_disp.apply(get_context_to_plot, axis=1)


# Ensure shap_values is 2D
if len(shap_values.shape) == 1:
    shap_values = shap_values.reshape(-1, 1)

feature_a, feature_b = interaction_features
# feature_a = 'Whisker trial in block'
# feature_b = 'context_to_plot'
feature_a_index = X_disp.columns.get_loc(feature_a)
feature_b_index = X_disp.columns.get_loc(feature_b)

fig, ax = plt.subplots(1, 1, figsize=(5,3), dpi=300)
# Create custom palette mapping 0 to dark magenta and 1 to green
custom_palette = {0: 'darkmagenta', 1: 'green'}  # Dark magenta and green

plots_kwargs_strip = {
    'edgecolor' : None,
    'linewidth' : 0.1,
    'alpha' : 0.3,
    'size' : 4,
}

plots_kwargs_point = {
    'linewidth' : 0,
    'alpha' : 0.8,
    'markersize' : 10,
}

if type_a == 'discrete':
    sns.violinplot(
        x=X_disp[feature_a],
        y=shap_values[:, feature_a_index],
        hue=X_disp[feature_b],
        palette=custom_palette,
        ax=ax,
        fill=False,
        inner="quart",
        linewidth=1,
        cut=0,
        density_norm='width',
        # legend=True,
        # dodge=True,
        # **plots_kwargs_strip,
    )
    # ax.axhline(y=0, linestyle='--', lw=1, color='grey')
    # sns.pointplot(
    #     x=X_disp[feature_a],
    #     y=shap_values[:, feature_a_index],
    #     hue=X_disp[feature_b],
    #     palette=custom_palette,
    #     legend=False,
    #     dodge=True,
    #     ax=ax,
    #     **plots_kwargs_point,
    # )
    # Convert x-axis tick labels from float to int
    # ax.set_xticklabels([int(float(x.get_text())) for x in ax.get_xticklabels()])
    # ax.legend(labels=['R-', 'R+'])
elif type_a == 'continuous':
    # Create scatter plot
    sns.scatterplot(
        x=X_disp[feature_a],
        y=shap_values[:, feature_a_index],
        hue=X_disp[feature_b],
        palette=custom_palette,
        ax=ax,
        legend=False  # Don't create automatic legend
    )
else:
    raise NotImplementedError
# Create custom legend with correct colors
handles = [plt.scatter([], [], c=color, label=label) 
            for label, color in zip(['to R-', 'to R+'], custom_palette.values())]
ax.legend(handles=handles)
ax.set_xlabel(feature_a)
ax.set_ylabel(f'SHAP value of \n{feature_a.lower()}')
ax.axvline(x=3.5, c='k', linestyle='--')
sns.despine(fig, offset=10)
fig.tight_layout()

save_path=results_dir / run_name / 'figures' / f'interaction_{interaction_features}',
save_path = save_path[0]

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

for ext in EXTS:
    fig.savefig(save_path.with_suffix(f'.{ext}'), dpi=300)
