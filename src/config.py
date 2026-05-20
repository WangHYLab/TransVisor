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
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Failed to load config file: {e}")
        else:
            self._config = self._get_default_config()
    
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
        """Update config with new values"""
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self._config = update_dict(self._config, updates)
        
        with open(self._config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    def reload(self):
        """Reload config from file"""
        self._load_config()

# Global config instance
config = ConfigManager()
