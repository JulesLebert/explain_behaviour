from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, confusion_matrix, 
    log_loss, mean_squared_error, mean_absolute_error, r2_score
)
from imblearn.over_sampling import SMOTE
from typing import Dict, Tuple, Optional, List, Union
import optuna
import joblib

import explain_behaviour.helpers.plotting as plottings

EXTS = ['png', 'pdf', 'svg']

class BehavioralAnalysisGBM:
    """
    A class for analyzing behavioral data using XGBoost classification.
    """
    def __init__(
        self,
        save_path: Optional[Path] = None,
        random_state: int = 123,
        mode: str = 'classification'  # 'classification' or 'regression'
    ):
        self.save_path = save_path or Path.cwd() / 'results'
        self.random_state = random_state
        self.mode = mode
        self.model = None
        self.shap_values = None
        self.feature_names = None
        self.explainer = None
        self.metrics = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        outcome_col: str = 'Outcome',
        categorical_cols: Optional[List[str]] = None,
        test_size: float = 0.33
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
        """
        Prepare data for analysis by handling categorical variables and splitting into train/test sets.
        """
        # Create copy to avoid modifying original
        df_use = df.copy()
        
        # Handle categorical columns
        cat_mappings = {}
        if categorical_cols:
            for col in categorical_cols:
                if col in df_use.columns:
                    mapping = {val: idx for idx, val in enumerate(df_use[col].unique())}
                    df_use[col] = df_use[col].map(mapping)
                    cat_mappings[col] = mapping

        # Split features and target
        X = df_use.drop(columns=[outcome_col])
        y = df_use[outcome_col]
        self.feature_names = X.columns.tolist()

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if self.mode == 'classification' else None,
        )

        self.X = X
        self.X_train = X_train
        self.X_test = X_test
        self.y = y
        self.y_train = y_train
        self.y_test = y_test

        return X_train, X_test, y_train, y_test, cat_mappings

    def load_parameters(self, param_path: Optional[Path] = None, filename: str = 'best_params.npy') -> Dict:
        """
        Load previously saved hyperparameters.
        
        Args:
            param_path: Path to the saved parameters. If None, will look in default location.
            
        Returns:
            Dict of parameters
        """
        if param_path is None:
            param_path = self.save_path / 'optuna_studies' / filename
            
        if not param_path.exists():
            raise FileNotFoundError(f"No saved parameters found at {param_path}")
            
        try:
            params = np.load(param_path, allow_pickle=True).item()
            print(f"Loaded parameters from {param_path}")
            return params
        except Exception as e:
            raise Exception(f"Error loading parameters: {str(e)}")

    def save_parameters(self, params: Dict, filename: str = 'best_params.npy'):
        """
        Save parameters to file.
        """
        save_dir = self.save_path / '_optuna_studies'
        save_dir.mkdir(exist_ok=True, parents=True)
        param_path = save_dir / filename
        np.save(param_path, params)
        print(f"Saved parameters to {param_path}")

    def train_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        params: Optional[Dict] = None,
        use_smote: bool = False
    ) -> Dict:
        """
        Train the XGBoost model and return performance metrics.
        """
        if use_smote and self.mode == 'regression':
            raise ValueError("SMOTE is not supported for regression tasks")

        if use_smote:
            smote = SMOTE(random_state=self.random_state)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        else:
            X_train_smote, y_train_smote = X_train, y_train

        default_params = {
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        if self.mode == 'classification':
            default_params['objective'] = 'binary:logistic'
            model_class = xgb.XGBClassifier
            default_params['eval_metric'] = 'logloss'
        else:
            default_params['objective'] = 'reg:squarederror'
            model_class = xgb.XGBRegressor
            default_params['eval_metric'] = 'rmse'
        
        if params:
            default_params.update(params)

        # Add early stopping parameters
        default_params['early_stopping_rounds'] = 100

        self.model = model_class(**default_params)
        
        # Create evaluation set
        eval_set = [(X_test, y_test)]
        self.model.fit(
            X_train_smote,
            y_train_smote,
            eval_set=eval_set,
            verbose=False
        )

        # Calculate metrics based on mode
        y_pred = self.model.predict(X_test)
        if self.mode == 'classification':
            metrics = {
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'balanced_accuracy_train': balanced_accuracy_score(y_train, self.model.predict(X_train)),
                'f1_score': f1_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        else:
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }

        return metrics

    def calculate_feature_importance(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[np.ndarray, Dict]:
        """
        Calculate permutation importance.
        """
       
        perm_result = permutation_importance(
            self.model, X_test, y_test,
            n_repeats=100,
            random_state=self.random_state,
            scoring='neg_mean_squared_error' if self.mode == 'regression' else None
        )
        
        return perm_result

    def plot_results(
        self,
        interaction_features: Optional[List[Tuple[str, str]]] = None,
        file_prefix: Optional[str] = None,
        save_prefix: Optional[str] = None,
    ):
        """Generate and save visualization plots."""
        if not hasattr(self, 'analysis_results'):
            raise ValueError("No analysis results found. Run run_analysis() first.")

        # Use stored values if not provided
        interaction_features = interaction_features or self.analysis_results['interaction_features']
        file_prefix = file_prefix or self.analysis_results['file_prefix']
        save_prefix = save_prefix or self.analysis_results['save_prefix']

        # Create save directory
        save_dir = self.save_path / save_prefix
        save_dir.mkdir(exist_ok=True, parents=True)

        # Extract data from results
        X = pd.concat([self.analysis_results['X_train'], self.analysis_results['X_test']])
        X_train = self.analysis_results['X_train']
        X_test = self.analysis_results['X_test']
        shap_values = self.analysis_results['shap_values']
        perm_result = self.analysis_results['perm_result']
        cat_mappings = self.analysis_results['cat_mappings']
        feature_names = self.analysis_results['feature_names']

        # Set default interaction features if none provided
        if interaction_features is None:
            top_features = pd.Series(
                np.abs(shap_values).mean(0),
                index=feature_names
            ).nlargest(2).index
            interaction_features = [(top_features[0], top_features[1])]

        # Generate plots
        self._plot_shap_waterfall(X, save_dir, file_prefix)
        self._plot_feature_importance(
            X, X_train, X_test, shap_values, perm_result,
            save_dir, file_prefix
        )
        # self._plot_interactions(
        #     X, shap_values, interaction_features,
        #     save_dir, file_prefix, cat_mappings
        # )
        

    def _plot_shap_waterfall(self, X, save_dir, file_prefix):
        """Plot SHAP waterfall plot."""
        np.random.seed(42)
        trial = np.random.randint(0, X.shape[0])
        shap.waterfall_plot(
            self.explainer(X)[trial],
            show=False,
        )
        fig = plt.gcf()
        fig.tight_layout()
        for ext in EXTS:
            fig.savefig(save_dir / f'{file_prefix}_shap_waterfall.{ext}', 
                       dpi=300, bbox_inches='tight')

    def _plot_feature_importance(self, X, X_train, X_test, shap_values, 
                               perm_result, save_dir, file_prefix):
        """Plot feature importance plots."""
        cmap = "flare"
        fig, ax_dict = plottings.full_shap_plot(
            xg_reg=self.model,
            shap_values=shap_values,
            X=X,
            X_train=X_train,
            X_test=X_test,
            perm_result=perm_result,
            cmapcustom=cmap,
        )

        # Set appropriate title based on mode
        metric_name = 'R²' if self.mode == 'regression' else 'Balanced Accuracy'
        metric_value = self.metrics['r2'] if self.mode == 'regression' else self.metrics['balanced_accuracy']
        fig.suptitle(f'Test {metric_name}: {metric_value:.3f}')
        fig.tight_layout()
        for ext in EXTS:
            fig.savefig(save_dir / f'{file_prefix}_feature_importance.{ext}', 
                       dpi=300, bbox_inches='tight')

    def _plot_interactions(self, X, shap_values, interaction_features,
                          save_dir, file_prefix, cat_mappings):
        """Plot interaction plots."""
        shap_interaction_values = self.explainer.shap_interaction_values(X)
        X_disp = X.copy()
        cmap = "flare"

        for (feature_a, feature_b) in interaction_features:
            fig, axes = plt.subplots(1, 3, figsize=(15,4))
            plottings.plot_interaction_single(
                X,
                X_disp,
                shap_values, 
                shap_interaction_values, 
                feature_a,
                feature_b,
                axes,
                cmap,
            )       
            sns.despine(fig, trim=True)  
            fig.tight_layout()
            for ext in EXTS:
                fig.savefig(save_dir / f'{file_prefix}_{feature_a}_{feature_b}_shap_interaction.{ext}', 
                           dpi=300)

    def optimize_parameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_smote: bool = False,
        n_trials: int = 1000,
        save_study: bool = True,
        file_prefix: str = '',
    ) -> Dict:
        """
        Optimize model hyperparameters using Optuna.
        """
        def objective(trial, X_train, X_valid, y_train, y_valid):
            param_grid = {
                "subsample": trial.suggest_float("subsample", 0.1, 1),
                "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.5, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10, log=True),
                "gamma": trial.suggest_float("gamma", 0, 20),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            }

            if self.mode == 'classification':
                param_grid["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1, 10)
                param_grid["eval_metric"] = "logloss"
            else:
                param_grid["eval_metric"] = "rmse"

            # Add early stopping parameter
            param_grid["early_stopping_rounds"] = 100

            model = xgb.XGBClassifier(**param_grid) if self.mode == 'classification' else xgb.XGBRegressor(**param_grid)
            model.set_params(random_state=self.random_state, verbosity=0)

            # Create evaluation set
            eval_set = [(X_valid, y_valid)]
            
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False
            )

            preds = model.predict_proba(X_valid)[:, 1] if self.mode == 'classification' else model.predict(X_valid)
            return log_loss(y_valid, preds) if self.mode == 'classification' else mean_squared_error(y_valid, preds)

        # Create study
        study = optuna.create_study(direction="minimize", study_name="XGBoost " + self.mode.capitalize())

        # Split data for optimization
        X_train_opt, X_valid, y_train_opt, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state
        )

        # Apply SMOTE if requested (only for classification)
        if use_smote and self.mode == 'classification':
            smote = SMOTE(random_state=self.random_state)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_opt, y_train_opt)
        else:
            X_train_smote, y_train_smote = X_train_opt, y_train_opt

        # Create optimization function
        func = lambda trial: objective(trial, X_train_smote, X_valid, y_train_smote, y_valid)

        # Run optimization
        study.optimize(func, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)

        # Print results
        print("Number of finished trials: ", len(study.trials))
        print(f"Best value of - {'logloss' if self.mode == 'classification' else 'mse'}: {study.best_value:.5f}")
        print(f"Best params:")
        for key, value in study.best_params.items():
            print(f"\t{key}: {value}")

        # Save study if requested
        if save_study and self.save_path:
            study_path = self.save_path / 'optuna_studies'
            study_path.mkdir(exist_ok=True, parents=True)
            joblib.dump(study, study_path / f'{file_prefix}_{self.mode}_optuna_study.pkl')

        return study.best_params

    def run_analysis(
        self,
        df: pd.DataFrame,
        outcome_col: str = 'Outcome',
        categorical_cols: Optional[List[str]] = None,
        params: Optional[Dict] = None,
        param_path: Optional[Path] = None,
        use_smote: bool = False,
        optimize: bool = False,
        n_trials: int = 1000,
        interaction_features: Optional[List[Tuple[str, str]]] = None,
        save_prefix: str = 'analysis',
        file_prefix: str = '',
    ) -> Dict:
        """
        Run the complete analysis pipeline.
        
        Args:
            df: Input DataFrame
            outcome_col: Name of the outcome column
            categorical_cols: List of categorical column names
            params: Dictionary of model parameters (optional)
            param_path: Path to saved parameters (optional)
            use_smote: Whether to use SMOTE for imbalanced data
            optimize: Whether to run parameter optimization
            n_trials: Number of optimization trials
            interaction_features: List of feature pairs to analyze interactions
            save_prefix: Prefix for saved files
        """
        # Prepare data
        X_train, X_test, y_train, y_test, cat_mappings = self.prepare_data(
            df,
            outcome_col,
            categorical_cols
        )
        
        # Handle parameters
        param_file = f'{file_prefix}_{self.mode}_best_params.npy'
        if optimize:
            print("Optimizing hyperparameters...")
            params = self.optimize_parameters(
                X_train,
                y_train,
                use_smote=use_smote,
                n_trials=n_trials,
                file_prefix=file_prefix,
            )
            # Save the optimized parameters
            self.save_parameters(params, filename=param_file)
            print("Optimization completed.")
        elif params is None:
            # Try to load parameters if not provided and not optimizing
            try:
                params = self.load_parameters(param_path, filename=param_file)
            except FileNotFoundError:
                print("No saved hyperparameters found. Using default parameters.")
                params = {}

        # Train model
        self.metrics = self.train_model(X_train, X_test, y_train, y_test, params, use_smote)

        # Calculate feature importance
        X = pd.concat([X_train, X_test])

        # Use a subset of training data as background
        background_data = X_train.sample(min(100, len(X_train)), random_state=self.random_state)
        print(f"Starting SHAP explainer")
        
        # Ensure all data is float64
        background_data = background_data.astype(float)
        X = X.astype(float)
        
        # Create explainer with model output set to 'raw' for both classification and regression
        self.explainer = shap.TreeExplainer(
            self.model,
            data=background_data,
            feature_perturbation='interventional',
            model_output='raw'  # Always use 'raw' for both classification and regression
        )
        
        # Get SHAP values for all data points
        if self.mode == 'classification':
            # For classification, get class probabilities
            shap_values = self.explainer.shap_values(X)#[1] # Get values for positive class
            # Ensure shap_values is 2D with correct number of features
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(-1, X.shape[1])
        else:
            # For regression, get raw predictions
            shap_values = self.explainer.shap_values(X)
            # Ensure shap_values is 2D with correct number of features
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(-1, X.shape[1])
        print(f"SHAP explainer completed")
        self.shap_values = shap_values

        # Calculate feature importance
        perm_result = self.calculate_feature_importance(X_test, y_test)

        # Calculate feature importance scores
        feature_importance = pd.Series(
            np.abs(self.shap_values).mean(0),
            index=self.feature_names
        ).sort_values(ascending=False)

        # Store results for later use
        self.analysis_results = {
            'metrics': self.metrics,
            'feature_importance': feature_importance,
            'cat_mappings': cat_mappings,
            'model': self.model,
            'explainer': self.explainer,
            'shap_values': self.shap_values,
            'perm_result': perm_result,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'interaction_features': interaction_features,
            'file_prefix': file_prefix,
            'save_prefix': save_prefix,
        }

        return self.analysis_results

    def save_results(self, save_dir: Path, file_prefix: str = ''):
        """
        Save analysis results, including the analyzer object and figures.
        
        Args:
            save_dir: Directory to save results
            file_prefix: Prefix for saved files
        """
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save analyzer object
        analyzer_path = save_dir / f'{file_prefix}_analyzer.pkl'
        joblib.dump(self, analyzer_path)
        
        # Save analysis results
        results_path = save_dir / f'{file_prefix}_results.pkl'
        joblib.dump(self.analysis_results, results_path)
        
        # # Save figures
        # figures_dir = save_dir / 'figures'
        # figures_dir.mkdir(exist_ok=True)
        
        # # Save feature importance plot
        # if hasattr(self, 'analysis_results') and 'feature_importance' in self.analysis_results:
        #     plt.figure(figsize=(10, 6))
        #     self._plot_feature_importance(
        #         self.analysis_results['X_train'],
        #         self.analysis_results['X_test'],
        #         self.analysis_results['shap_values'],
        #         self.analysis_results['perm_result'],
        #         figures_dir,
        #         file_prefix
        #     )
        #     plt.close()
        
        print(f"Saved analysis results to {save_dir}")

    @classmethod
    def load_results(cls, load_dir: Path, file_prefix: str = '') -> Tuple['BehavioralAnalysisGBM', Dict]:
        """
        Load previously saved analysis results.
        
        Args:
            load_dir: Directory containing saved results
            file_prefix: Prefix of saved files
            
        Returns:
            Tuple of (analyzer object, analysis results)
        """
        load_dir = Path(load_dir)
        
        # Load analyzer object
        analyzer_path = load_dir / f'{file_prefix}_analyzer.pkl'
        if not analyzer_path.exists():
            raise FileNotFoundError(f"No saved analyzer found at {analyzer_path}")
        analyzer = joblib.load(analyzer_path)
        
        # Load analysis results
        results_path = load_dir / f'{file_prefix}_results.pkl'
        if not results_path.exists():
            raise FileNotFoundError(f"No saved results found at {results_path}")
        results = joblib.load(results_path)
        
        # Restore results to analyzer
        analyzer.analysis_results = results
        
        print(f"Loaded analysis results from {load_dir}")
        return analyzer, results

def main():
    df = pd.read_csv('/Users/lebert/home/code/context/explain_behaviour/data/expert_table.csv')

    # df_use, cat_mappings = extract_data_for_classification(df, positive_label='Miss', negative_label='Hit')
    # Initialize the analyzer
    analyzer = BehavioralAnalysisGBM(save_path=Path('results'))

    # Example usage with your data
    results = analyzer.run_analysis(
        df=df,
        outcome_col='Outcome',
        optimize=True,
        categorical_cols=['Ferret', 'Other_Categories'],
        interaction_features=[('Ferret', 'SNR'), ('Trial duration', 'SNR')],
        use_smote=True
    )

    # Access results
    print(f"Balanced Accuracy: {results['metrics']['balanced_accuracy']:.2f}")
    print("\nTop Features:")
    print(results['feature_importance'].head())

    # Generate visualizations
    analyzer.plot_results()

# # Initialize the analyzer
# analyzer = BehavioralAnalysis(save_path=Path('results'))

# # 1. Run with optimization (will save parameters)
# results = analyzer.run_analysis(
#     df=your_dataframe,
#     optimize=True,
#     n_trials=1000
# )

# # 2. Run with previously saved parameters (automatically loads them)
# results = analyzer.run_analysis(
#     df=your_dataframe,
#     optimize=False  # Will automatically load saved parameters
# )

# # 3. Run with parameters from a specific file
# results = analyzer.run_analysis(
#     df=your_dataframe,
#     param_path=Path('path/to/specific/params.npy')
# )

# # 4. Run with manually provided parameters
# results = analyzer.run_analysis(
#     df=your_dataframe,
#     params=your_params_dict
# )

if __name__ == '__main__':
    main()