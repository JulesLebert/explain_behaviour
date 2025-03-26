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

def plot_interaction_only(
    analyzer,
    interaction_features,
    save_path,
):
    """Plot interaction only results."""
    X_disp = pd.concat([analyzer.analysis_results['X_train'], analyzer.analysis_results['X_test']])
    shap_values = analyzer.analysis_results['shap_values']

    feature_a, feature_b = interaction_features
    feature_a_index = X_disp.columns.get_loc(feature_a)
    feature_b_index = X_disp.columns.get_loc(feature_b)

    fig, ax = plt.subplots(1, 1, figsize=(6,4), dpi=300)
    # Create custom palette mapping 0 to dark magenta and 1 to green
    custom_palette = {0: 'darkmagenta', 1: 'green'}  # Dark magenta and green
    
    plots_kwargs_strip = {
        'edgecolor' : None,
        'linewidth' : 0.1,
        'alpha' : 0.3,
        'size' : 4,
    }

    plots_kwargs_point = {
        'linewidth' : 0.5,
        'alpha' : 0.8,
        'markersize' : 10,
    }

    sns.stripplot(
        x=X_disp[feature_a],
        y=shap_values[:, feature_a_index],
        hue=X_disp[feature_b],
        palette=custom_palette,
        ax=ax,
        legend=True,
        dodge=True,
        **plots_kwargs_strip,
    )
    sns.pointplot(
        x=X_disp[feature_a],
        y=shap_values[:, feature_a_index],
        hue=X_disp[feature_b],
        palette=custom_palette,
        legend=False,
        dodge=True,
        ax=ax,
        **plots_kwargs_point,
    )
    ax.set_xlabel(feature_a)
    ax.set_ylabel(f'SHAP value of \n{feature_a.lower()}')
    # # Update legend labels
    # handles = ax.get_legend().get_patches()
    ax.legend(['R-', 'R+'])
    sns.despine(fig, trim=True)
    fig.tight_layout()
    for ext in EXTS:
        fig.savefig(save_path.with_suffix(f'.{ext}'), dpi=300)
    return fig, ax

def main():
    # Configuration
    config_name = 'whisker_classification'  # Change this to use different config files
    load_dir = None # Set to a directory name to load previous results, e.g., 'run_20250318_171524'
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'configs' / 'analysis' / f'{config_name}.yaml'
    config = load_config(config_path)
    
    # Set up results directory
    results_dir = Path(__file__).parent.parent / 'results' / config_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if load_dir:
        # Load previously saved results
        run_name = load_dir
        load_dir = results_dir / load_dir
        analyzer, results = BehavioralAnalysisGBM.load_results(load_dir)
        print("\nLoaded Analysis Results:")
    else:
        run_name = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        # Load data and run analysis
        df_path = Path(__file__).parent.parent / 'data' / 'expert_data.csv'
        df = pd.read_csv(df_path)
        
        # Prepare features for analysis
        df_use = prepare_behavior_features(df)

        if config['mode'] == 'classification':
                # Filter for whisker trials
                df_use = df_use.loc[df_use.trial_type == 'whisker_trial']

        if config['mode'] == 'regression':
            df_use = df_use.dropna(subset=['reaction_time'])
            df_use = df_use.loc[df_use.trial_type.isin(['whisker_trial', 'auditory_trial'])]
            df_use = df_use.loc[df_use.trial_outcome == 'Hit']

        # Select features
        df_use = df_use[config['whisker_features']]
        df_use = df_use.rename(columns=dict(zip(df_use.columns, config['feature_labels'])))
        
        # Initialize analyzer
        analyzer = BehavioralAnalysisGBM(
            save_path=results_dir / run_name,
            mode=config['mode'],
        )
        
        # Run analysis
        results = analyzer.run_analysis(
            df=df_use,
            outcome_col=config['outcome_column'],
            categorical_cols=config['categorical_columns'],
            interaction_features=config['interaction_features'],
            **config['analysis_params']
        )
        
        # Generate visualizations
        analyzer.plot_results()
        
        # if save_results:
            # Save results
        save_dir = results_dir / run_name
        analyzer.save_results(save_dir)
        # Save config as YAML
        config_file = save_dir / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        print(f"\nSaved results to {save_dir}")
    
    # Print results
    if config['mode'] == 'classification':
        print("\nAnalysis Results:")
        print(f"Balanced Accuracy: {results['metrics']['balanced_accuracy']:.3f}")
        print(f"F1 Score: {results['metrics']['f1_score']:.3f}")
        print("\nTop Features by Importance:")
        print(results['feature_importance'].head())

    # Plot interaction only results
    fig, ax = plot_interaction_only(
        analyzer,
        interaction_features=config['interaction_features'][1],
        save_path=results_dir / run_name / 'analysis' / 'interaction_whisker_trial_context.png'
    )
    fig.show()

if __name__ == '__main__':
    main() 