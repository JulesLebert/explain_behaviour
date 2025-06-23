"""
Configuration parameters for behavioral analysis.
"""

# Categorical columns for analysis
CATEGORICAL_COLUMNS = [
    'context',
    'subject',
    'prev_trial_correct',
    'prev_whisker_rewarded',
    'last_stimulus_type',
]

# Feature interactions to analyze
INTERACTION_FEATURES = [
    ('context', 'whisker_trial_in_block'),
    # ('last_stimulus_type', 'context'),
    # ('time_since_last_stimulus', 'context'),
]

# Analysis parameters
ANALYSIS_PARAMS = {
    'optimize': False,  # Whether to run hyperparameter optimization
    'use_smote': False,  # Whether to use SMOTE for handling class imbalance
    'n_trials': 500,  # Number of optimization trials
    'file_prefix': 'whisker_all_contexts'
}

# Feature columns for whisker analysis
WHISKER_FEATURES = [
    'context',
    'subject',
    'cum_water',
    'prev_trial_correct',
    # 'trial_number',
    # 'context_block_index',
    'trial_in_block',
    'whisker_trial_in_block',
    'prev_trial_water',
    'last_stimulus_type',
    'prev_whisker_rewarded',
    'time_since_last_stimulus',
    'lick_flag'
] 