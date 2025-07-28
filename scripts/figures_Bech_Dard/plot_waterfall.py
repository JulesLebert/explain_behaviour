from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from explain_behaviour.models.behavioural_analysis_GBM import BehavioralAnalysisGBM
import xgboost as xgb
import shap
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
save_dir = load_dir / 'figures'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

X = analyzer.X_train
y = analyzer.y_train
np.random.seed(42)
trials = np.random.randint(0, X.shape[0], 10)
for trial in trials:
    shap.waterfall_plot(
        # analyzer.explainer(X)[trial],
        analyzer.explainer(X.iloc[[trial]])[0],
        show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

    for ext in EXTS:
        fig.savefig(save_dir / f'shap_waterfall_{trial}.{ext}', 
                    dpi=300, bbox_inches='tight')