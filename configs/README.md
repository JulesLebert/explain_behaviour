# Configuration Files

This directory contains configuration files for different types of analyses in the project.

## Structure

- `analysis/`: Contains analysis-specific configurations
  - `whisker.yaml`: Configuration for whisker-specific behavioral analysis
  - `auditory.yaml`: Configuration for auditory-specific behavioral analysis
  - `combined.yaml`: Configuration for combined behavioral analysis

## Usage

To use a configuration file in your scripts:

```python
from pathlib import Path
import yaml

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Example usage
config_path = Path('configs/analysis/whisker.yaml')
config = load_config(config_path)
```

## Configuration Types

### Analysis Configurations
Each analysis configuration file contains:
- Categorical columns for analysis
- Feature interactions to analyze
- Analysis parameters (optimization settings, etc.)
- Feature columns specific to the analysis type

## Adding New Configurations

When adding a new configuration:
1. Create a new YAML file in the appropriate subdirectory
2. Follow the existing structure and naming conventions
3. Update this README with the new configuration details
4. Add comments to explain any new parameters or features 