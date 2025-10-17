"""
Configuration Management
"""

import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class Config:
    """
    Configuration manager for experiments.
    
    Example:
        >>> config = Config.from_yaml('config.yaml')
        >>> print(config.model.hidden_dim)
        >>> config.set('model.dropout', 0.3)
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Key in dot notation (e.g., 'model.hidden_dim')
            default: Default value if key not found
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def __getattr__(self, name: str):
        """Allow attribute-style access."""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        value = self._config.get(name)
        if isinstance(value, dict):
            return Config(value)
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config.copy()
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create from dictionary."""
        return cls(config_dict)
    
    def save(self, filepath: str):
        """Save configuration to YAML file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(filepath: str) -> Config:
    """Load configuration from YAML file."""
    return Config.from_yaml(filepath)
