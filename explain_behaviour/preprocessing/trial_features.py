from pathlib import Path
import pandas as pd
import numpy as np

def create_trial_features(df, config=None):
    """
    Create a feature-rich DataFrame for model training based on trial data.
    
    Args:
        df (pd.DataFrame): Raw trial data with columns for context, session_name, trial_type, etc.
        
    Returns:
        pd.DataFrame: Processed features for analysis
    """
    # Create new DataFrame with required features
    features = pd.DataFrame()

    if "BaselineMotionEnergy" in config['whisker_features']:
        df = add_baseline_motion_energy(df)
        features['BaselineMotionEnergy'] = df['BaselineMotionEnergy']
        features = features.loc[features.BaselineMotionEnergy < 100000]
    # Basic information
    features['context'] = df['context']
    features['subject'] = df['subject']
    features['session'] = df['session_name']
    features['trial_type'] = df['trial_type']
    features['lick_flag'] = df['lick_flag']
    features['trial_outcome'] = df['trial_outcome']
    features['time_in_context'] = df['time_in_context']
    
    # Previous trial context (shift by 1, fill first trial with -1)
    features['prev_context'] = df.groupby('session_name')['context'].shift(1).fillna(-1)
    
    # Last stimulus type (auditory or whisker, excluding no_stim_trial)
    # First create a mask for stimulus trials
    stim_mask = df['trial_type'].isin(['auditory_trial', 'whisker_trial'])
    # Get the last stimulus type, excluding no_stim_trial
    features['last_stimulus_type'] = (
        df.loc[stim_mask, 'trial_type']
        .groupby(df['session_name'])
        .shift(1)
        # .fillna('unknown')
    )
    
    # Whether previous whisker trial was rewarded (Hit)
    # First create a mask for whisker trials
    whisker_mask = df['trial_type'] == 'whisker_trial'
    # Create a mask for rewarded whisker trials (Hit)
    whisker_hit_mask = whisker_mask & (df['trial_outcome'] == 'Hit')
    # Shift the mask to get previous trial's reward status
    features['prev_whisker_rewarded'] = whisker_hit_mask.groupby(df['session_name']).shift(1).fillna(False)
    
    # Time since last stimulus trial
    # First identify stimulus trials (auditory or whisker)
    stim_trials = df['trial_type'].isin(['auditory_trial', 'whisker_trial'])
    # Calculate time since last stimulus using start_time, excluding current trial
    features['time_since_last_stimulus'] = (
        df.groupby('session_name')
        .apply(lambda x: x['start_time'] - x['start_time'].where(stim_trials).shift(1).ffill())
        .reset_index(level=0, drop=True)
    )
    
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
    # First create a temporary dataframe for whisker trials
    whisker_trials = features[whisker_mask].copy()
    whisker_trials['whisker_trial_in_block'] = whisker_trials.groupby(['session', 'context_block_index']).cumcount() + 1
    
    # Initialize whisker_trial_in_block column with NaN
    features['whisker_trial_in_block'] = np.nan
    # Update only whisker trials with their counts
    features.loc[whisker_mask, 'whisker_trial_in_block'] = whisker_trials['whisker_trial_in_block']
    
    # Whether mouse received water in previous trial
    features['prev_trial_water'] = hit_mask.groupby(df['session_name']).shift(1).fillna(False)

    # Reaction time
    features['reaction_time'] = df['lick_time'] - df['start_time']
    
    return features

def add_baseline_motion_energy(
        df, 
        baseline_energy_path='/Volumes/Petersen-Lab/z_LSENS/Share/Pol_Bech/Bech_Dard et al 2025/Figure_data/facemap_plots/Facemap movement analysis_2025_06_16.19-22-40/baseline_energy.csv',
        ):
    baseline_energy_path = Path(baseline_energy_path)
    df_energy = pd.read_csv(baseline_energy_path)

    df['id'] = df.groupby('session_name').cumcount()

    df_merged = pd.merge(
        df,
        df_energy,
        left_on=['session_name', 'id'],
        right_on=['session_id', 'id'],
        how='inner',
        suffixes=('', '_y')  # Keep original column names for left df, add _y suffix for right df
    )
    
    # Drop the duplicate columns (those ending with _y)
    columns_to_drop = [col for col in df_merged.columns if col.endswith('_y')]
    df_merged = df_merged.drop(columns=columns_to_drop)
    
    # Drop the session_id column since we already have session_name
    if 'session_id' in df_merged.columns:
        df_merged = df_merged.drop(columns=['session_id'])
    

    return df_merged

def label_trial_outcomes(df):
    """
    Label trial outcomes based on trial type, context, and response.
    
    Args:
        df (pd.DataFrame): Raw trial data
        
    Returns:
        pd.DataFrame: DataFrame with added trial_outcome column
    """
    df = df.copy()
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
    
    # Label no stimulus trials
    no_stim_mask = df['trial_type'] == 'no_stim_trial'
    df.loc[no_stim_mask & (df['lick_flag'] == 0), 'trial_outcome'] = 'CR'
    df.loc[no_stim_mask & (df['lick_flag'] == 1), 'trial_outcome'] = 'FA'
    
    return df

def prepare_behavior_features(df, config = None):
    """
    Prepare features specifically for whisker trial analysis.
    
    Args:
        df (pd.DataFrame): Raw trial data
        
    Returns:
        pd.DataFrame: Processed features for whisker analysis
    """
    # Label outcomes
    df = label_trial_outcomes(df)
    
    # Create features
    features_df = create_trial_features(df, config)
        
    df_use = features_df
    # # Select features for analysis
    # df_use = whisker_df[[
    #     'context',
    #     'subject',
    #     'cum_water',
    #     'prev_trial_correct',
    #     'trial_number',
    #     'context_block_index',
    #     'trial_in_block',
    #     'whisker_trial_in_block',
    #     'prev_trial_water',
    #     'lick_flag'
    # ]]
    
    return df_use 