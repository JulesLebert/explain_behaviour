from pathlib import Path
import pandas as pd
import yaml
from explain_behaviour.models.binary_classification import BehavioralAnalysisGBM
from explain_behaviour.preprocessing.trial_features import prepare_whisker_features

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Configuration
    config_name = 'whisker'  # Change this to use different config files
    load_dir = None  # Set to a directory name to load previous results, e.g., 'run_20240315_143022'
    save_results = True  # Set to True to save results after running analysis
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'configs' / 'analysis' / f'{config_name}.yaml'
    config = load_config(config_path)
    
    # Set up results directory
    results_dir = Path(__file__).parent.parent / 'results' / config_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if load_dir:
        # Load previously saved results
        load_dir = results_dir / load_dir
        analyzer, results = BehavioralAnalysisGBM.load_results(load_dir)
        print("\nLoaded Analysis Results:")
    else:
        # Load data and run analysis
        df_path = Path(__file__).parent.parent / 'data' / 'expert_data.csv'
        df = pd.read_csv(df_path)
        
        # Prepare features for analysis
        df_use = prepare_whisker_features(df)
        df_use = df_use[config['whisker_features']]
        
        # Initialize analyzer
        analyzer = BehavioralAnalysisGBM(
            save_path=results_dir,
            mode='classification',
        )
        
        # Run analysis
        results = analyzer.run_analysis(
            df=df_use,
            outcome_col='lick_flag',
            categorical_cols=config['categorical_columns'],
            interaction_features=config['interaction_features'],
            **config['analysis_params']
        )
        
        # Generate visualizations
        analyzer.plot_results()
        
        if save_results:
            # Save results
            save_dir = results_dir / f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            analyzer.save_results(save_dir)
            print(f"\nSaved results to {save_dir}")
    
    # Print results
    print("\nAnalysis Results:")
    print(f"Balanced Accuracy: {results['metrics']['balanced_accuracy']:.3f}")
    print(f"F1 Score: {results['metrics']['f1_score']:.3f}")
    print("\nTop Features by Importance:")
    print(results['feature_importance'].head())

if __name__ == '__main__':
    main() 