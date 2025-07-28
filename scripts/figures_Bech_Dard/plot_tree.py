from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from explain_behaviour.models.behavioural_analysis_GBM import BehavioralAnalysisGBM
import xgboost as xgb
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
results_dir = Path('/mnt/lsens-analysis/Jules_Lebert/results/context/behaviour_model/') / config_name
results_dir.mkdir(parents=True, exist_ok=True)

# Load previously saved results
run_name = load_dir
load_dir = results_dir / load_dir
config_path = load_dir / 'config.yaml'
config = load_config(config_path)
analyzer, results = BehavioralAnalysisGBM.load_results(load_dir)
print("\nLoaded Analysis Results:")

# tree_idx = 0
save_dir = results_dir / run_name / 'figures' 
save_dir.mkdir(exist_ok=True)
for tree_idx in range(10):
    save_path = save_dir / f'tree_example_{tree_idx}'

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

    for ext in EXTS:
        if ext.lower() == 'svg':
            # Generate true vector SVG using graphviz directly
            g = xgb.to_graphviz(analyzer.model, tree_idx=tree_idx)
            svg_content = g.pipe(format='svg', encoding='utf-8')
            with open(save_path.with_suffix(f'.{ext}'), 'w', encoding='utf-8') as f:
                f.write(svg_content)
        else:
            # Use matplotlib for other formats
            ax = xgb.plot_tree(analyzer.model, tree_idx=tree_idx)
            fig = plt.gcf()
            fig.savefig(save_path.with_suffix(f'.{ext}'), dpi=300)
            plt.close(fig)
