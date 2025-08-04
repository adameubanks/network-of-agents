"""
Configuration manager for the network of agents simulation.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """
    Manages configuration for the simulation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (JSON)
        """
        self.config_path = config_path or "config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_simulation_params(self) -> Dict[str, Any]:
        """
        Get simulation parameters.
        
        Returns:
            Dictionary of simulation parameters
        """
        return {
            'n_agents': int(self.get('simulation.n_agents', 30)),
            'epsilon': float(self.get('simulation.epsilon', 0.001)),
            'theta': int(self.get('simulation.theta', 7)),
            'num_timesteps': int(self.get('simulation.num_timesteps', 300)),
            'initial_connection_probability': float(self.get('simulation.initial_connection_probability', 0.05)),
            'random_seed': self.get('simulation.random_seed'),
            'initial_opinion_diversity': float(self.get('simulation.initial_opinion_diversity', 0.8))
        }
    
    def get_llm_params(self) -> Dict[str, Any]:
        """
        Get LLM parameters.
        
        Returns:
            Dictionary of LLM parameters
        """
        return {
            'model': str(self.get('llm.model', 'gpt-4')),
            'max_tokens': int(self.get('llm.max_tokens', 1000)),
            'temperature': float(self.get('llm.temperature', 0.7)),
            'api_key_env': str(self.get('llm.api_key_env', 'OPENAI_API_KEY'))
        }
    
    def get_topics(self) -> list:
        """
        Get topics list.
        
        Returns:
            List of topics
        """
        return self.get('topics', ['Climate Change', 'Economic Policy', 'Social Justice'])
    
    def get_bias_testing_params(self) -> Dict[str, Any]:
        """
        Get bias testing parameters.
        
        Returns:
            Dictionary of bias testing parameters
        """
        return {
            'enabled': bool(self.get('bias_testing.enabled', True)),
            'topic_pairs': self.get('bias_testing.topic_pairs', []),
            'use_consistent_seeds': bool(self.get('bias_testing.use_consistent_seeds', True)),
            'seed_base': int(self.get('bias_testing.seed_base', 42))
        }
    
    def get_visualization_params(self) -> Dict[str, Any]:
        """
        Get visualization parameters.
        
        Returns:
            Dictionary of visualization parameters
        """
        return {
            'plot_style': str(self.get('visualization.plot_style', 'seaborn')),
            'figure_size': self.get('visualization.figure_size', [12, 8]),
            'dpi': int(self.get('visualization.dpi', 300))
        }
    
    def get_output_directory(self) -> str:
        """
        Get output directory.
        
        Returns:
            Output directory path
        """
        return str(self.get('output_directory', 'simulation_results'))
    
    def validate_config(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid
        """
        required_keys = ['n_agents', 'epsilon', 'theta', 'num_timesteps']
        
        sim_params = self.get_simulation_params()
        
        for key in required_keys:
            if key not in sim_params:
                print(f"Error: Missing required parameter '{key}'")
                return False
        
        if sim_params['n_agents'] <= 0:
            print("Error: n_agents must be positive")
            return False
        
        if sim_params['epsilon'] <= 0:
            print("Error: epsilon must be positive")
            return False
        
        if sim_params['theta'] <= 0:
            print("Error: theta must be positive")
            return False
        
        if sim_params['num_timesteps'] <= 0:
            print("Error: num_timesteps must be positive")
            return False
        
        return True 