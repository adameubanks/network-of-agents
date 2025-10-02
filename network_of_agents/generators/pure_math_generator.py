"""
Pure mathematical opinion generator implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from ..core.base_models import OpinionGenerator

class PureMathGenerator(OpinionGenerator):
    """
    Pure mathematical opinion generator.
    
    Generates opinions and posts using direct mathematical computation
    without any LLM involvement.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
    
    def generate_opinion(self, 
                        agent_id: int,
                        current_opinion: float,
                        topic: Dict[str, Any],
                        neighbor_posts: List[str] = None) -> float:
        """
        Generate an opinion using pure mathematical computation.
        
        For pure math, we just return the current opinion unchanged
        since the opinion dynamics model handles the updates.
        
        Args:
            agent_id: ID of the agent
            current_opinion: Current opinion value
            topic: Topic information
            neighbor_posts: Posts from neighboring agents (ignored)
            
        Returns:
            Opinion value (unchanged)
        """
        return current_opinion
    
    def generate_post(self, 
                     agent_id: int,
                     current_opinion: float,
                     topic: Dict[str, Any],
                     neighbor_posts: List[str] = None) -> str:
        """
        Generate a post expressing the agent's opinion.
        
        Args:
            agent_id: ID of the agent
            current_opinion: Current opinion value
            topic: Topic information
            neighbor_posts: Posts from neighboring agents (ignored)
            
        Returns:
            Generated post text
        """
        # Simple text generation based on opinion value
        if current_opinion > 0.5:
            stance = "strongly agree"
        elif current_opinion > 0.1:
            stance = "agree"
        elif current_opinion > -0.1:
            stance = "neutral"
        elif current_opinion > -0.5:
            stance = "disagree"
        else:
            stance = "strongly disagree"
        
        # Choose statement based on opinion
        if current_opinion > 0:
            statement = topic["statement_a"]
        else:
            statement = topic["statement_b"]
        
        post = f"Agent {agent_id}: I {stance} that {statement.lower()}"
        
        return post

