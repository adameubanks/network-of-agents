"""
Agent class for post generation and interpretation.
"""

import numpy as np
import random
from typing import List, Optional


class Agent:
    """
    Represents an individual agent in the social network.
    """
    
    def __init__(self, agent_id: int, random_seed: Optional[int] = None):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            random_seed: Random seed for reproducible opinion generation
        """
        self.agent_id = agent_id
        
        # Initialize with random opinion
        self.current_opinion = np.random.uniform(-1, 1)
    
    def generate_post_prompt(self, topic) -> str:
        """
        Generate agent-specific prompt for post generation.
        
        Args:
            topic: Tuple of (a, b) to generate post about
            
        Returns:
            Agent-specific prompt
        """
        a, b = str(topic[0]).strip(), str(topic[1]).strip()
        
        return f"""
Write a short, social-media style post (1-3 sentences, <320 characters) in first person
about {a} vs {b}. Your current opinion: {self.current_opinion:.3f} (-1=agrees with {a}, 1=agrees with {b}).
If you see other agents' posts, you may respond to 1-2 of them by name (e.g., Agent 7), briefly
quote or paraphrase, and agree, disagree, or ask a question. Prose only, no numeric score.
"""

    def update_opinion(self, new_opinion: float):
        """
        Update the agent's opinion.
        
        Args:
            new_opinion: New opinion value (-1 to 1)
        """
        self.current_opinion = new_opinion
    
    def get_opinion(self) -> float:
        """
        Get current opinion value.
        
        Returns:
            Current opinion value (-1 to 1)
        """
        return self.current_opinion 

    def interpret_post_prompt(self, post: str, topic) -> str:
        """
        Generate agent-specific interpretation prompt for a single post.
        
        Args:
            post: Single post to interpret
            topic: Tuple of (a, b) the post is about
            
        Returns:
            Agent-specific interpretation prompt
        """
        a, b = str(topic[0]).strip(), str(topic[1]).strip()
        
        return f"""
Rate the post on the axis {a} (-1) ↔ {b} (1).
-1.000 = agrees with this statement: {a} over this statement: {b}
 1.000 = agrees with this statement: {b} over this statement: {a}

Post: "{post}"
Respond with ONLY one number in [-1.000, 1.000] on its own line. Use 0.000 if neutral. Make sure to include at least two points of precision.
""" 

 