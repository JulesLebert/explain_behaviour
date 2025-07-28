from pathlib import Path
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score
import numpy as np

from explain_behaviour.models.behavioural_analysis_GBM import BehavioralAnalysisGBM
from explain_behaviour.preprocessing.trial_features import prepare_behavior_features

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

df_test = pd.read_csv(load_dir / 'df_test.csv')
# df_test = df_test.drop(columns=['Unnamed: 0'])
# if 'dlc_features' in config:
#     config['whisker_features'] = config['whisker_features'] + config['dlc_features']
#     config['feature_labels'] = config['feature_labels'] + config['dlc_features_labels']

# Select features
df_test = df_test[config['whisker_features']]
df_test = df_test.rename(columns=dict(zip(df_test.columns, config['feature_labels'])))

df_use = df_test.copy()
categorical_cols = config['categorical_columns']
outcome_col = config['outcome_column']
cat_mapping = results['cat_mappings']


for col in cat_mapping:
    if col in df_use.columns:
        df_use[col] = df_use[col].map(cat_mapping[col])

# Split features and target
X = df_use.drop(columns=[outcome_col])
y = df_use[outcome_col]
feature_names = X.columns.tolist()

y_pred = analyzer.model.predict(X)

X['y_true'] = y
X['y_pred'] = y_pred
bal_acc = balanced_accuracy_score(y, y_pred)
print(f'Balanced accuracy: {bal_acc}')

# Step 1: Assign block index
# Detect when context changes
X['context_change'] = (X['Context'] != X['Context'].shift()).cumsum()
# Count trials within each context block
X['trial_in_block'] = X.groupby('context_change').cumcount()
# Each block is 8 trials, so increment block index every 8 trials
X['block_index'] = X['context_change'] + (X['trial_in_block'] // 8)  # 1000 is arbitrary to separate blocks
# Now, block_index is unique for each 8-trial block of the same context

# Step 2: Group by block_index and calculate means
block_means = X.groupby('block_index').agg({
       'y_true': 'mean',
       'y_pred': 'mean',
       'Context': 'first'  
   }).reset_index()
block_means['y_true'] = block_means['y_true'] * 100
block_means['y_pred'] = block_means['y_pred'] * 100
block_means['trial'] = block_means['block_index'] * 20 - 10

custom_palette = {0: 'darkmagenta', 1: 'green'} 
colors_lines = {'true': 'orange', 'pred': 'saddlebrown'}

# Step 3: Plot
fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
line_kwargs = {
    'marker' : 'o',
    # 'markersize' : 2,
    'linewidth' : 2,
    'alpha' : 1,
}

for cond in ['true', 'pred']:
    sns.lineplot(
        x='trial',
        y=f'y_{cond}',
        data=block_means,
        color=colors_lines[cond],
        markeredgecolor=colors_lines[cond],
        ax=ax,
        label=cond,
        **line_kwargs,
    )
    # sns.lineplot(
    #     x='trial',
    #     y='y_pred',
    #     data=block_means,
    #     color=colors_lines['pred'],
    #     markeredgecolor=colors_lines['pred'],
    #     ax=ax,
    #     **line_kwargs,
    # )

ax.set_xlabel('Trial')
ax.set_ylabel('Lick probability (%)')

# Color background according to context
prev_context = None
start_idx = 0
alpha = 0.5
for i, context in enumerate(block_means['Context']):
    if prev_context is None:
        prev_context = context
        start_idx = i
    elif context != prev_context:
        # Shade the previous region
        start_x = block_means['trial'].iloc[start_idx] - 10
        end_x = block_means['trial'].iloc[i-1] + (block_means['trial'].iloc[1] - block_means['trial'].iloc[0]) - 10
        ax.axvspan(start_x, end_x, color=custom_palette[prev_context], alpha=alpha, zorder=1)
        prev_context = context
        start_idx = i
# Shade the last region
start_x = block_means['trial'].iloc[start_idx] - 10
end_x = block_means['trial'].iloc[-1] + (block_means['trial'].iloc[1] - block_means['trial'].iloc[0]) - 10
ax.axvspan(start_x, end_x, color=custom_palette[prev_context], alpha=alpha, zorder=1)

# plt.title('Mean y_true and y_pred per 8-trial Context Block')
ax.legend()
sns.despine(fig=fig, trim=True, offset=0)
fig.tight_layout()
# plt.show()

save_dir = load_dir / 'figures'
save_dir.mkdir(exist_ok=True)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

for ext in EXTS:
    fig.savefig(save_dir / f'single_session_prediction_bal_acc_{np.round(bal_acc, decimals=3)}.{ext}')