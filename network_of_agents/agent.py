"""
Agent class for post generation and interpretation.
"""

import numpy as np
from typing import List, Optional
from .llm_client import LLMClient


class Agent:
    """
    Represents an individual agent in the social network.
    """
    
    def __init__(self, agent_id: int, initial_opinion: float = 0.5):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            initial_opinion: Initial opinion value (0-1)
        """
        self.agent_id = agent_id
        self.current_opinion = np.clip(initial_opinion, 0.0, 1.0)
    
    def generate_post(self, llm_client: LLMClient, topic: str) -> str:
        """
        Generate a post about the given topic.
        
        Args:
            llm_client: LLM client for post generation
            topic: Topic to generate post about
            
        Returns:
            Generated post text
        """
        return llm_client.generate_post(topic, self.agent_id)
    
    def interpret_posts(self, llm_client: LLMClient, posts: List[str], topic: str) -> List[float]:
        """
        Interpret posts to understand others' opinions.
        
        Args:
            llm_client: LLM client for post interpretation
            posts: List of posts to interpret
            topic: Topic the posts are about
            
        Returns:
            List of interpreted opinion values (0-1)
        """
        return llm_client.interpret_posts(posts, topic)
    
    def update_opinion(self, new_opinion: float):
        """
        Update the agent's opinion.
        
        Args:
            new_opinion: New opinion value (0-1)
        """
        self.current_opinion = np.clip(new_opinion, 0.0, 1.0)
    
    def get_opinion(self) -> float:
        """
        Get current opinion value.
        
        Returns:
            Current opinion value (0-1)
        """
        return self.current_opinion 