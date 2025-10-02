"""
LLM-based opinion rater implementation.
"""

import os
import openai
import re
from typing import Dict, Any
from ..core.base_models import OpinionRater

class LLMRater(OpinionRater):
    """
    LLM-based opinion rater.
    
    Uses a language model to rate posts and extract opinion values.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.model_name = parameters["model_name"]
        self.api_key_env = parameters["api_key_env"]
        self.max_tokens = parameters.get("max_tokens", 50)
        self.temperature = parameters.get("temperature", 0.3)
        
        # Initialize OpenAI client
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {self.api_key_env}")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    def rate_post(self, 
                 rater_agent_id: int,
                 rater_opinion: float,
                 poster_agent_id: int,
                 post: str,
                 topic: Dict[str, Any]) -> float:
        """
        Rate a post using LLM.
        
        Args:
            rater_agent_id: ID of the agent doing the rating
            rater_opinion: Current opinion of the rater
            poster_agent_id: ID of the agent who wrote the post
            post: Post text to rate
            topic: Topic information
            
        Returns:
            Rating value in [-1, 1] range
        """
        # Create prompt for rating
        prompt = f"""You are rating a social media post about: {topic['name']}

The topic is: {topic['statement_a']} vs {topic['statement_b']}

Post to rate: "{post}"

Rate this post on a scale from -1 to +1 where:
-1 = strongly disagrees with statement A (supports statement B)
0 = neutral or unclear
+1 = strongly agrees with statement A (disagrees with statement B)

Respond with only a number between -1 and 1."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a social media user rating posts. Respond with only a number between -1 and 1."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            rating_text = response.choices[0].message.content.strip()
            
            # Extract number from response
            rating_match = re.search(r'-?\d+\.?\d*', rating_text)
            if rating_match:
                rating = float(rating_match.group())
                # Clamp to [-1, 1] range
                rating = max(-1.0, min(1.0, rating))
                return rating
            else:
                # Fallback to neutral rating
                return 0.0
                
        except Exception as e:
            # Fallback to simple similarity-based rating
            similarity = 1 - abs(rater_opinion - 0.0)  # Assume neutral poster
            rating = 2 * similarity - 1
            return float(rating)

