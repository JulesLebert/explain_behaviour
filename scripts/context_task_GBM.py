from pathlib import Path
import pandas as pd
import numpy as np
import shap

from explain_behaviour.models.binary_classification import BehavioralAnalysisGBM

def create_trial_features(df):
    """
    Create a feature-rich DataFrame for model training based on trial data.
    """
    # Create new DataFrame with required features
    features = pd.DataFrame()
    
    # Basic information
    features['context'] = df['context']
    features['subject'] = df['subject']
    features['session'] = df['session_name']
    features['trial_type'] = df['trial_type']
    features['lick_flag'] = df['lick_flag']
    
    # Previous trial context (shift by 1, fill first trial with -1)
    features['prev_context'] = df.groupby('session_name')['context'].shift(1).fillna(-1)
    
    # Calculate cumulative water received (50µL per hit)
    water_per_hit = 5  # microliters
    hit_mask = df['trial_outcome'] == 'Hit'
    features['cum_water'] = (hit_mask.groupby(df['session_name']).cumsum() * water_per_hit).shift(1).fillna(0)
    
    # Previous trial correctness (Hit or CR counts as correct)
    correct_trials = df['trial_outcome'].isin(['Hit', 'CR'])
    features['prev_trial_correct'] = correct_trials.groupby(df['session_name']).shift(1).fillna(False)
    
    # Trial numbers
    features['trial_number'] = df.groupby('session_name').cumcount() + 1
    
    # Create context block index first
    # This identifies when context changes within a session
    features['context_block_index'] = (features.groupby('session')['context']
                                     .transform(lambda x: (x != x.shift()).cumsum()))
    
    # Now we can use context_block_index for trial counting
    features['trial_in_block'] = features.groupby(['session', 'context_block_index']).cumcount() + 1
    
    # Index of whisker trials in context block
    whisker_mask = df['trial_type'] == 'whisker_trial'
    # First create a temporary dataframe for whisker trials
    whisker_trials = features[whisker_mask].copy()
    whisker_trials['whisker_trial_in_block'] = whisker_trials.groupby(['session', 'context_block_index']).cumcount() + 1
    
    # Initialize whisker_trial_in_block column with NaN
    features['whisker_trial_in_block'] = np.nan
    # Update only whisker trials with their counts
    features.loc[whisker_mask, 'whisker_trial_in_block'] = whisker_trials['whisker_trial_in_block']
    
    # Whether mouse received water in previous trial
    features['prev_trial_water'] = hit_mask.groupby(df['session_name']).shift(1).fillna(False)
    
    return features

def main():
    df_path = Path('/Users/lebert/home/code/context/explain_behaviour/data/expert_table.csv')

    df = pd.read_csv(df_path)
    # Outcome:
    # First, let's create trial outcome labels
    df['trial_outcome'] = 'Unknown'

    # Label auditory trials
    auditory_mask = df['trial_type'] == 'auditory_trial'
    df.loc[auditory_mask & (df['lick_flag'] == 1), 'trial_outcome'] = 'Hit'
    df.loc[auditory_mask & (df['lick_flag'] == 0), 'trial_outcome'] = 'Miss'

    # Label whisker trials
    whisker_mask = df['trial_type'] == 'whisker_trial'
    # Context 1: normal whisker trials
    df.loc[whisker_mask & (df['context'] == 1) & (df['lick_flag'] == 1), 'trial_outcome'] = 'Hit'
    df.loc[whisker_mask & (df['context'] == 1) & (df['lick_flag'] == 0), 'trial_outcome'] = 'Miss'
    # Context 0: catch trials
    df.loc[whisker_mask & (df['context'] == 0) & (df['lick_flag'] == 0), 'trial_outcome'] = 'CR'
    df.loc[whisker_mask & (df['context'] == 0) & (df['lick_flag'] == 1), 'trial_outcome'] = 'FA'

    no_stim_mask = df['trial_type'] == 'no_stim_trial'
    df.loc[no_stim_mask & (df['lick_flag'] == 0), 'trial_outcome'] = 'CR'
    df.loc[no_stim_mask & (df['lick_flag'] == 1), 'trial_outcome'] = 'FA'

    features_df = create_trial_features(df)
    whisker_df = features_df.loc[features_df.trial_type == 'whisker_trial']

    df_use = whisker_df[[
        'context',
        'subject',
        # 'prev_context',
        'cum_water',
        'prev_trial_correct',
        'trial_number',
        'context_block_index',
        'trial_in_block',
        'whisker_trial_in_block',
        'prev_trial_water',
        'lick_flag'
    ]]

    categories = [
        'context',
        'subject',
        # 'prev_context',
        'prev_trial_correct',
        'prev_trial_water'
    ]

    analyzer = BehavioralAnalysisGBM(
        save_path=Path('/Users/lebert/home/code/context/explain_behaviour/results'),
        )

    interactions = [
        ('context', 'whisker_trial_in_block'),
        # ('whisker_trial_in_block', 'trial_in_block'),
        # ('context', 'trial_in_block'),
        # ('context', 'prev_trial_water'),
        ]

    results = analyzer.run_analysis(
        df=df_use,
        outcome_col='lick_flag',
        optimize=True,
        categorical_cols=categories,
        interaction_features = interactions,
        # interaction_features=[('Ferret', 'SNR'), ('Trial duration', 'SNR')],
        use_smote=False,
        n_trials=100,
        file_prefix='whisker_all_contexts'
    )

if __name__ == "__main__":
    main()