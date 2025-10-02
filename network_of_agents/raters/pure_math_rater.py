"""
Pure mathematical opinion rater implementation.
"""

import numpy as np
from typing import Dict, Any
from ..core.base_models import OpinionRater

class PureMathRater(OpinionRater):
    """
    Pure mathematical opinion rater.
    
    Rates posts based on opinion similarity without using LLMs.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
    
    def rate_post(self, 
                 rater_agent_id: int,
                 rater_opinion: float,
                 poster_agent_id: int,
                 post: str,
                 topic: Dict[str, Any]) -> float:
        """
        Rate a post using pure mathematical computation.
        
        Args:
            rater_agent_id: ID of the agent doing the rating
            rater_opinion: Current opinion of the rater
            poster_agent_id: ID of the agent who wrote the post
            post: Post text to rate
            topic: Topic information
            
        Returns:
            Rating value in [-1, 1] range
        """
        # Simple similarity-based rating
        # For pure math, we assume the poster's opinion is similar to the rater's
        # This is a simplified approach - in practice, you might extract opinion from post
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.1)
        # For pure math, assume perfect similarity (rater rates their own opinion highly)
        similarity = 1.0
        rating = 2 * similarity - 1 + noise
        
        # Clamp to [-1, 1] range
        rating = np.clip(rating, -1.0, 1.0)
        
        return float(rating)

