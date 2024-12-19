from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, log_loss
from imblearn.over_sampling import SMOTE
from typing import Dict, Tuple, Optional, List, Union
import optuna
import joblib

import explain_behaviour.helpers.plotting as plottings

class BehavioralAnalysisGBM:
    """
    A class for analyzing behavioral data using LightGBM classification.
    """
    def __init__(
        self,
        save_path: Optional[Path] = None,
        random_state: int = 123
    ):
        self.save_path = save_path or Path.cwd() / 'results'
        self.random_state = random_state
        self.model = None
        self.shap_values = None
        self.feature_names = None

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
            stratify=y
        )

        self.X = X
        self.X_train = X_train
        self.X_test = X_test
        self.y = y
        self.y_train = y_train
        self.y_test = y_test

        return X_train, X_test, y_train, y_test, cat_mappings

    def load_parameters(self, param_path: Optional[Path] = None) -> Dict:
        """
        Load previously saved hyperparameters.
        
        Args:
            param_path: Path to the saved parameters. If None, will look in default location.
            
        Returns:
            Dict of parameters
        """
        if param_path is None:
            param_path = self.save_path / 'optuna_studies' / 'best_params.npy'
            
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
        save_dir = self.save_path / 'optuna_studies'
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
        Train the LightGBM model and return performance metrics.
        """
        if use_smote:
            smote = SMOTE(random_state=self.random_state)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        else:
            X_train_smote, y_train_smote = X_train, y_train

        default_params = {
            'objective': 'binary',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': -1
        }
        
        if params:
            default_params.update(params)

        self.model = lgb.LGBMClassifier(**default_params)
        
        self.model.fit(
            X_train_smote,
            y_train_smote,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=300)]
        )

        # Calculate metrics
        y_pred = self.model.predict(X_test)
        metrics = {
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'balanced_accuracy_train': balanced_accuracy_score(y_train, self.model.predict(X_train)),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        return metrics

    def calculate_feature_importance(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[np.ndarray, Dict]:
        """
        Calculate SHAP values and permutation importance.
        """
       
        perm_result = permutation_importance(
            self.model, X_test, y_test,
            n_repeats=100,
            random_state=self.random_state
        )
        
        return perm_result

    def plot_results(
        self,
        X: pd.DataFrame,
        X_train,
        X_test,
        perm_result: Dict,
        interaction_features: List[Tuple[str, str]],
        file_prefix: str,
        save_prefix: str = 'analysis',
        cat_mappings: Optional[Dict] = None
    ):
        """
        Generate and save visualization plots.
        """
        # Create save directory
        save_dir = self.save_path / save_prefix
        save_dir.mkdir(exist_ok=True, parents=True)

        cmap = sns.color_palette("flare", as_cmap=True)


        fig, ax_dict = plottings.full_shap_plot(
            xg_reg=self.model,
            shap_values=self.shap_values,
            # shap_values2=shap_values2,
            X=X,
            X_train=X_train,
            X_test=X_test,
            perm_result=perm_result,
            cmapcustom=cmap,
            # cmapsummary=cmapsummary,
        )

        fig.suptitle(f'Test balanced accuracy: {self.metrics["balanced_accuracy"]}')
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(save_dir / f'{file_prefix}_feature_importance.{ext}', dpi=300, bbox_inches='tight')

        shap_interaction_values = self.explainer.shap_interaction_values(X)
        self.shap_interaction_values = shap_interaction_values

        X_disp = X.copy()

        # if cat_mappings is not None:
        #     for cat, d in cat_mappings.items():
        #         X_disp[cat] = X_disp[cat].map(d) 

        for (feature_a, feature_b) in interaction_features:
            fig, axes = plt.subplots(
                1, 3,
                figsize=(15,4),
                )
            plottings.plot_interaction_single(
                X,
                X_disp,
                self.shap_values, 
                shap_interaction_values, 
                feature_a,
                feature_b,
                axes,
                cmap,
                )       
            sns.despine(fig, trim=True)  
            fig.tight_layout()
            for ext in ['png', 'pdf']:
                fig.savefig(save_dir / f'{file_prefix}_{feature_a}_{feature_b}_shap_interaction.{ext}', dpi=300,)      

        # fig, axes = plottings.plot_interactions_full(
        #     model=self.model, 
        #     X=X, 
        #     shap_values=self.shap_values,
        #     shap_interaction_values=shap_interaction_values,
        #     interactions=interaction_features,
        #     cat_mappings=cat_mappings,
        #     cmap=cmap,
        # )
        # fig.tight_layout()
        # for ext in ['png', 'pdf']:
        #     fig.savefig(save_dir / f'{file_prefix}_shap_interaction.{ext}', dpi=300,)


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
                "num_leaves": trial.suggest_int("num_leaves", 20, 500),
                "max_depth": trial.suggest_int("max_depth", -1, 20),
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 200),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 0, 20),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 20),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 300, log=True),
                "max_bin": trial.suggest_int("max_bin", 100, 1000),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 600),
                "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.001, 50, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            }

            model = lgb.LGBMClassifier(
                objective="binary",
                random_state=self.random_state,
                verbosity=-1,
                **param_grid,
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="binary_logloss",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                ],
            )

            preds = model.predict_proba(X_valid)[:, 1]
            return log_loss(y_valid, preds)

        # Create study
        study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")

        # Split data for optimization
        X_train_opt, X_valid, y_train_opt, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state
        )

        # Apply SMOTE if requested
        if use_smote:
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
        print(f"Best value of - logloss: {study.best_value:.5f}")
        print(f"Best params:")
        for key, value in study.best_params.items():
            print(f"\t{key}: {value}")

        # Save study if requested
        if save_study and self.save_path:
            study_path = self.save_path / f'{file_prefix}_optuna_studies'
            study_path.mkdir(exist_ok=True, parents=True)
            joblib.dump(study, study_path / f'{file_prefix}_optuna_study.pkl')

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
        if optimize:
            print("Optimizing hyperparameters...")
            params = self.optimize_parameters(
                X_train,
                y_train,
                use_smote=use_smote,
                n_trials=n_trials
            )
            # Save the optimized parameters
            self.save_parameters(params)
            print("Optimization completed.")
        elif params is None:
            # Try to load parameters if not provided and not optimizing
            try:
                params = self.load_parameters(param_path)
            except FileNotFoundError:
                print("No saved hyperparameters found. Using default parameters.")
                params = {}

        # Train model
        self.metrics = self.train_model(X_train, X_test, y_train, y_test, params, use_smote)

        # Calculate feature importance
        X = pd.concat([X_train, X_test])

        self.explainer = shap.TreeExplainer(self.model)
        shap_values = self.explainer.shap_values(X)
        self.shap_values = shap_values

        perm_result = self.calculate_feature_importance(X_test, y_test)

        # Generate plots
        if interaction_features is None:
            # Default to top 2 most important features for interaction
            top_features = pd.Series(
                np.abs(shap_values).mean(0),
                index=self.feature_names
            ).nlargest(2).index
            interaction_features = [(top_features[0], top_features[1])]

        self.plot_results(
            X,
            X_train=X_train,
            X_test=X_test,
            perm_result=perm_result,
            interaction_features=interaction_features,
            file_prefix=file_prefix,
            save_prefix=save_prefix,
            cat_mappings=cat_mappings,
        )

        return {
            'metrics': self.metrics,
            'feature_importance': pd.Series(
                np.abs(shap_values).mean(0),
                index=self.feature_names
            ).sort_values(ascending=False),
            'cat_mappings': cat_mappings,
        }
    

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