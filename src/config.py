from typing import Dict, Any, Optional
import yaml
import os

class ConfigManager:
    _instance = None
    
    def __new__(cls, config_path: str = "config.yaml"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._config = {}
            cls._instance._config_path = config_path
            cls._instance._local_config_path = "config-local.yaml"
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load config with priority: config-local.yaml > config.yaml > defaults"""
        # Start with default config
        self._config = self._get_default_config()
        
        # Load base config.yaml if exists
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    base_config = yaml.safe_load(f) or {}
                    self._config = self._merge_configs(self._config, base_config)
            except Exception as e:
                print(f"Warning: Failed to load base config file ({self._config_path}): {e}")
        
        # Override with config-local.yaml if exists (for sensitive data)
        if os.path.exists(self._local_config_path):
            try:
                with open(self._local_config_path, 'r', encoding='utf-8') as f:
                    local_config = yaml.safe_load(f) or {}
                    self._config = self._merge_configs(self._config, local_config)
                    print(f"Loaded local config from {self._local_config_path}")
            except Exception as e:
                print(f"Warning: Failed to load local config file ({self._local_config_path}): {e}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two config dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "api": {
                "key": "",
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com",
                "timeout": 60
            },
            "data": {
                "default_path": "example_data/example_expression.csv",
                "diff_expr_path": "example_data/differential_expression.csv",
                "time_series_path": "example_data/time_series.csv"
            },
            "visualization": {
                "default_dpi": 300,
                "default_style": "seaborn",
                "default_palette": "viridis",
                "output_dir": "output"
            },
            "agent": {
                "max_iterations": 10,
                "verbose": True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_api_key(self) -> str:
        return self.get("api.key", "")
    
    def get_model(self) -> str:
        return self.get("api.model", "deepseek-chat")
    
    def get_default_data_path(self) -> str:
        return self.get("data.default_path", "")
    
    def get_diff_expr_path(self) -> str:
        return self.get("data.diff_expr_path", "")
    
    def get_time_series_path(self) -> str:
        return self.get("data.time_series_path", "")
    
    def get_default_dpi(self) -> int:
        return self.get("visualization.default_dpi", 300)
    
    def get_output_dir(self) -> str:
        return self.get("visualization.output_dir", "output")
    
    def get_max_iterations(self) -> int:
        return self.get("agent.max_iterations", 10)
    
    def get_verbose(self) -> bool:
        return self.get("agent.verbose", True)
    
    def update_config(self, updates: Dict[str, Any]):
        """Update config with new values (only updates config.yaml, not config-local.yaml)"""
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self._config = update_dict(self._config, updates)
        
        # Read existing config.yaml to preserve comments and structure
        base_config = {}
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    base_config = yaml.safe_load(f) or {}
            except:
                pass
        
        # Merge updates into base config (but exclude api.key if local config exists)
        if os.path.exists(self._local_config_path):
            # If local config exists, don't save api.key to config.yaml
            updates_copy = updates.copy()
            if 'api' in updates_copy and isinstance(updates_copy['api'], dict):
                updates_copy['api'] = {k: v for k, v in updates_copy['api'].items() if k != 'key'}
                if not updates_copy['api']:
                    del updates_copy['api']
            base_config = self._merge_configs(base_config, updates_copy)
        else:
            base_config = self._merge_configs(base_config, updates)
        
        # Save only to config.yaml
        with open(self._config_path, 'w', encoding='utf-8') as f:
            yaml.dump(base_config, f, default_flow_style=False, sort_keys=False)
    
    def reload(self):
        """Reload config from file"""
        self._load_config()

# Global config instance
config = ConfigManager()
