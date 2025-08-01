"""
Configuration manager for loading and validating simulation configurations.
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path


class ConfigManager:
    """
    Manages configuration loading and validation for the simulation.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs = {}
        self.load_all_configs()
    
    def load_all_configs(self):
        """Load all configuration files from the config directory."""
        if not self.config_dir.exists():
            print(f"Warning: Config directory {self.config_dir} does not exist")
            return
        
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            self.configs[config_name] = self.load_config(config_file)
    
    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config {config_path}: {e}")
            return {}
    
    def get_config(self, config_name: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get a specific configuration.
        
        Args:
            config_name: Name of the configuration
            default: Default value to return if config not found
            
        Returns:
            Configuration dictionary
        """
        if default is None:
            default = {}
        return self.configs.get(config_name, default)
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """
        Get simulation configuration.
        
        Returns:
            Simulation configuration dictionary
        """
        return self.get_config("default_config").get("simulation", {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration.
        
        Returns:
            LLM configuration dictionary
        """
        return self.get_config("default_config").get("llm", {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """
        Get visualization configuration.
        
        Returns:
            Visualization configuration dictionary
        """
        return self.get_config("default_config").get("visualization", {})
    
    def get_storage_config(self) -> Dict[str, Any]:
        """
        Get storage configuration.
        
        Returns:
            Storage configuration dictionary
        """
        return self.get_config("default_config").get("storage", {})
    
    def get_bias_testing_config(self) -> Dict[str, Any]:
        """
        Get bias testing configuration.
        
        Returns:
            Bias testing configuration dictionary
        """
        return self.get_config("default_config").get("bias_testing", {})
    
    def get_topics_config(self) -> Dict[str, Any]:
        """
        Get topics configuration.
        
        Returns:
            Topics configuration dictionary
        """
        return self.get_config("topics_config", {})
    
    def get_topic_pairs(self, pair_type: str = "language_bias") -> List[List[str]]:
        """
        Get topic pairs for bias testing.
        
        Args:
            pair_type: Type of topic pairs ("language_bias" or "framing_bias")
            
        Returns:
            List of topic pairs
        """
        topics_config = self.get_topics_config()
        topic_pairs = topics_config.get("topic_pairs", {})
        return topic_pairs.get(pair_type, [])
    
    def get_political_topics(self) -> List[str]:
        """
        Get political topics.
        
        Returns:
            List of political topics
        """
        topics_config = self.get_topics_config()
        return topics_config.get("political_topics", [])
    
    def get_social_topics(self) -> List[str]:
        """
        Get social topics.
        
        Returns:
            List of social topics
        """
        topics_config = self.get_topics_config()
        return topics_config.get("social_topics", [])
    
    def get_economic_topics(self) -> List[str]:
        """
        Get economic topics.
        
        Returns:
            List of economic topics
        """
        topics_config = self.get_topics_config()
        return topics_config.get("economic_topics", [])
    
    def get_technology_topics(self) -> List[str]:
        """
        Get technology topics.
        
        Returns:
            List of technology topics
        """
        topics_config = self.get_topics_config()
        return topics_config.get("technology_topics", [])
    
    def get_neutral_topics(self) -> List[str]:
        """
        Get neutral topics.
        
        Returns:
            List of neutral topics
        """
        topics_config = self.get_topics_config()
        return topics_config.get("neutral_topics", [])
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_simulation_keys = ["n_agents", "n_topics", "epsilon", "theta", "num_timesteps"]
        
        simulation_config = config.get("simulation", {})
        for key in required_simulation_keys:
            if key not in simulation_config:
                print(f"Missing required simulation parameter: {key}")
                return False
        
        # Validate parameter ranges
        if simulation_config["n_agents"] <= 0:
            print("n_agents must be positive")
            return False
        
        if simulation_config["n_topics"] <= 0:
            print("n_topics must be positive")
            return False
        
        if simulation_config["epsilon"] <= 0:
            print("epsilon must be positive")
            return False
        
        if simulation_config["theta"] <= 0:
            print("theta must be positive")
            return False
        
        if simulation_config["num_timesteps"] <= 0:
            print("num_timesteps must be positive")
            return False
        
        return True
    
    def create_simulation_config(self, 
                                n_agents: int = 50,
                                n_topics: int = 3,
                                topics: Optional[List[str]] = None,
                                **kwargs) -> Dict[str, Any]:
        """
        Create a simulation configuration.
        
        Args:
            n_agents: Number of agents
            n_topics: Number of topics
            topics: List of topics
            **kwargs: Additional configuration parameters
            
        Returns:
            Simulation configuration dictionary
        """
        # Get default configuration
        default_config = self.get_simulation_config()
        
        # Create custom configuration
        config = {
            "simulation": {
                "n_agents": n_agents,
                "n_topics": n_topics,
                "epsilon": kwargs.get("epsilon", default_config.get("epsilon", 1e-6)),
                "theta": kwargs.get("theta", default_config.get("theta", 7)),
                "num_timesteps": kwargs.get("num_timesteps", default_config.get("num_timesteps", 180)),
                "initial_connection_probability": kwargs.get("initial_connection_probability", 
                                                           default_config.get("initial_connection_probability", 0.2))
            },
            "llm": self.get_llm_config(),
            "visualization": self.get_visualization_config(),
            "storage": self.get_storage_config(),
            "bias_testing": self.get_bias_testing_config()
        }
        
        # Add topics if provided
        if topics:
            config["topics"] = topics
        
        return config
    
    def save_config(self, config: Dict[str, Any], filename: str):
        """
        Save a configuration to a file.
        
        Args:
            config: Configuration to save
            filename: Name of the file to save to
        """
        config_path = self.config_dir / filename
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Error saving config {config_path}: {e}")
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all loaded configurations.
        
        Returns:
            Dictionary of all configurations
        """
        return self.configs.copy() 