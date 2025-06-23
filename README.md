# Behavioral Analysis with XGBoost

This project provides a framework for analyzing behavioral data using XGBoost models with SHAP (SHapley Additive exPlanations) for interpretability. It's particularly designed for analyzing behavioral experiments with features like trial context, timing, and outcomes.

## Features

- **Flexible Analysis Modes**:
  - Classification mode for binary outcomes
  - Regression mode for continuous outcomes

- **Advanced Feature Engineering**:
  - Automatic feature creation from trial data
  - Handling of categorical variables
  - Support for trial sequences and context blocks

- **Model Training and Optimization**:
  - XGBoost model implementation
  - Hyperparameter optimization using Optuna
  - Support for SMOTE to handle class imbalance
  - Early stopping to prevent overfitting

- **Interpretability Tools**:
  - SHAP values for feature importance
  - Feature interaction analysis
  - Permutation importance
  - Visualization tools for model interpretation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JulesLebert/explain_behaviour.git
cd explain_behaviour
```

2. Install dependencies:
```bash
pip install -e .
```

## Usage

### Basic Usage

```python
from explain_behaviour.models.behavioural_analysis_GBM import BehavioralAnalysisGBM
from pathlib import Path

# Initialize analyzer
analyzer = BehavioralAnalysisGBM(
    save_path=Path('results'),
    mode='classification'  # or 'regression'
)

# Run analysis
results = analyzer.run_analysis(
    df=your_dataframe,
    outcome_col='Outcome',
    categorical_cols=['Context', 'Subject'],
    optimize=True,
    use_smote=True
)

# Generate visualizations
analyzer.plot_results()
```

### Configuration

The project uses YAML configuration files for analysis parameters. Example configuration:

```yaml
mode: classification

analysis_params:
  optimize: true
  use_smote: false
  n_trials: 200
  file_prefix: whisker_all_contexts

whisker_features:
  - context
  - subject
  - cum_water
  - prev_trial_correct
  - trial_in_block
  - whisker_trial_in_block
  # ... more features

feature_labels:
  - Context
  - Mouse ID
  - Cumulative water
  # ... corresponding labels

outcome_column: Lick flag

categorical_columns:
  - Context
  - Mouse ID
  - Previous trial correct

interaction_features:
  - [Whisker trial in block, Context]
  - [Trial in block, Context]
```

## Project Structure

```
explain_behaviour/
├── configs/
│   └── analysis/
│       └── whisker_classification.yaml
├── explain_behaviour/
│   ├── models/
│   │   └── behavioural_analysis_GBM.py
│   ├── preprocessing/
│   │   └── trial_features.py
│   └── helpers/
│       └── plotting.py
├── scripts/
│   └── analyze_behavior.py
└── data/
    └── expert_data.csv
```

## Features Generated

The preprocessing module generates various features including:

- Basic trial information (context, subject, session)
- Previous trial outcomes and context
- Time-based features (time since last stimulus)
- Cumulative water received
- Trial position within blocks
- Whisker trial specific features
- Reaction times

## Outputs

The analysis generates:

1. **Model Performance Metrics**:
   - Classification: Balanced accuracy, F1 score
   - Regression: MSE, RMSE, MAE, R²

2. **Feature Importance**:
   - SHAP values
   - Permutation importance
   - Feature interaction plots

3. **Visualizations**:
   - SHAP summary plots
   - Feature importance plots
   - Interaction plots
   - Waterfall plots
