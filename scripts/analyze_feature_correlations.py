from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, LabelEncoder

from explain_behaviour.preprocessing.trial_features import prepare_behavior_features
from explain_behaviour.helpers.helpers import load_config

def prepare_features_for_correlation(X: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for correlation analysis by handling categorical and boolean columns."""
    X_prep = X.copy()
    
    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include=['bool']).columns
    X_prep[bool_cols] = X_prep[bool_cols].astype(int)
    
    # Convert categorical columns to numeric using LabelEncoder
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X_prep[col] = le.fit_transform(X_prep[col])
    
    return X_prep

def calculate_vif(X: pd.DataFrame) -> pd.Series:
    """Calculate Variance Inflation Factor for each feature."""
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Handle NaN values
    if X.isna().any().any():
        print("\nWarning: Found NaN values. Dropping rows with NaN values.")
        X = X.dropna()
    
    # Scale the features
    X_scaled = StandardScaler().fit_transform(X)
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    
    return vif_data.sort_values('VIF', ascending=False)

def plot_correlation_matrix(corr_matrix: pd.DataFrame, save_path: Path, title: str = "Feature Correlation Matrix"):
    """Plot correlation matrix with annotations."""
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True,
                cmap='RdBu_r',
                center=0,
                fmt='.2f',
                square=True)
    plt.title(title)
    plt.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(save_path.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_clustered_correlation(corr_matrix: pd.DataFrame, save_path: Path):
    """Plot clustered correlation matrix."""
    # Convert correlation matrix to distance matrix
    distance_matrix = 1 - np.abs(corr_matrix)
    condensed_dist = squareform(distance_matrix)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='ward')
    
    # Get cluster assignments
    n_clusters = 3  # Adjust this number based on your needs
    cluster_assignments = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Reorder correlation matrix based on clusters
    cluster_order = np.argsort(cluster_assignments)
    corr_clustered = corr_matrix.iloc[cluster_order, cluster_order]
    
    # Plot clustered correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_clustered, dtype=bool))
    sns.heatmap(corr_clustered,
                mask=mask,
                annot=True,
                cmap='RdBu_r',
                center=0,
                fmt='.2f',
                square=True)
    plt.title("Clustered Feature Correlation Matrix")
    plt.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(save_path.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print feature clusters
    print("\nFeature Clusters:")
    for i in range(1, n_clusters + 1):
        cluster_features = corr_matrix.columns[cluster_assignments == i].tolist()
        print(f"\nCluster {i}:")
        print(", ".join(cluster_features))

def plot_vif(vif_data: pd.DataFrame, save_path: Path):
    """Plot VIF values as a bar plot."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=vif_data, x='VIF', y='Feature')
    plt.title("Variance Inflation Factors (VIF)")
    plt.xlabel("VIF")
    plt.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(save_path.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Configuration
    config_name = 'whisker_classification'
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'configs' / 'analysis' / f'{config_name}.yaml'
    config = load_config(config_path)
    
    # Load and prepare data
    df_path = Path(__file__).parent.parent / 'data' / 'expert_data.csv'
    df = pd.read_csv(df_path)
    df_use = prepare_behavior_features(df)
    
    # Filter data based on mode
    if config['mode'] == 'classification':
        df_use = df_use.loc[df_use.trial_type == 'whisker_trial']
    elif config['mode'] == 'regression':
        df_use = df_use.dropna(subset=['reaction_time'])
        df_use = df_use.loc[df_use.trial_type.isin(['whisker_trial', 'auditory_trial'])]
        df_use = df_use.loc[df_use.trial_outcome == 'Hit']
    
    # Select features and rename columns
    X = df_use[config['whisker_features']].copy()
    X = X.rename(columns=dict(zip(X.columns, config['feature_labels'])))
    
    # Print data types before preparation
    print("\nData types before preparation:")
    print(X.dtypes)
    
    # Prepare features for correlation analysis
    X_prep = prepare_features_for_correlation(X)
    
    # Print data types after preparation
    print("\nData types after preparation:")
    print(X_prep.dtypes)
    
    # Create save directory
    save_dir = Path(__file__).parent.parent / 'results' / config_name / 'correlation_analysis'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate and plot correlation matrix
    corr_matrix = X_prep.corr()
    plot_correlation_matrix(corr_matrix, save_dir / 'correlation_matrix')
    
    # Plot clustered correlation matrix
    plot_clustered_correlation(corr_matrix, save_dir / 'clustered_correlation')
    
    # Calculate and plot VIF
    vif_data = calculate_vif(X_prep)
    plot_vif(vif_data, save_dir / 'vif_plot')
    
    # Print high VIF features
    print("\nFeatures with high VIF (>5):")
    print(vif_data[vif_data['VIF'] > 5])
    
    # Save results to CSV
    corr_matrix.to_csv(save_dir / 'correlation_matrix.csv')
    vif_data.to_csv(save_dir / 'vif_values.csv')

if __name__ == '__main__':
    main() 