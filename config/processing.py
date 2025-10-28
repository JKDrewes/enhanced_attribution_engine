"""
Data processing configuration settings.
Controls sample sizes and other processing parameters.
"""
from typing import Optional, Union
from pathlib import Path
import json
import pandas as pd


class ProcessingConfig:
    def __init__(self):
        self._sample_size: Optional[int] = None
        self.config_file = Path(__file__).parent / "processing_config.json"
        self.load_config()

    @property
    def sample_size(self) -> Optional[int]:
        """Get the current sample size. None means process all records."""
        return self._sample_size

    @sample_size.setter
    def sample_size(self, value: Optional[Union[int, str]]) -> None:
        """Set the sample size for data processing.
        
        Args:
            value: Integer for specific sample size, "NONE" or None for full processing
        """
        if isinstance(value, str) and value.upper() == "NONE":
            value = None
        elif value is not None:
            value = int(value)
            if value <= 0:
                raise ValueError("Sample size must be positive")
        self._sample_size = value
        self.save_config()

    def load_config(self) -> None:
        """Load configuration from JSON file."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self._sample_size = config.get('sample_size')

    def save_config(self) -> None:
        """Save current configuration to JSON file."""
        config = {
            'sample_size': self._sample_size
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def apply_sample(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Apply the current sample size to a dataframe.

        Args:
            df: pandas DataFrame to sample

        Returns:
            DataFrame with sampling applied (or original if sample_size is None)
        """
        if self.sample_size is not None and len(df) > self.sample_size:
            return df.sample(n=self.sample_size, random_state=42)
        return df


# Global instance for easy import
config = ProcessingConfig()

# Command-line helper
def set_sample_size(size: Optional[Union[int, str]] = None) -> None:
    """Set the sample size from command line or scripts.
    
    Args:
        size: Integer for specific size, "NONE" for full processing
    """
    config.sample_size = size
    current = "FULL" if config.sample_size is None else str(config.sample_size)
    print(f"Sample size set to: {current}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Set data processing sample size")
    parser.add_argument(
        "size",
        nargs="?",
        default="NONE",
        help="Sample size (integer) or NONE for full processing"
    )
    args = parser.parse_args()
    set_sample_size(args.size)