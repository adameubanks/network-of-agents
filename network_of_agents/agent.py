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
        
        # Initialize with normal distribution as per experimental design
        # Normal distribution (μ=0.0, σ=0.3) clipped to [-1, 1]
        if random_seed is not None:
            np.random.seed(random_seed + agent_id)
        self.current_opinion = np.clip(np.random.normal(0.0, 0.3), -1.0, 1.0)
    
    def generate_post_prompt(self, topic, neighbor_posts: Optional[List[str]] = None) -> str:
        """Generate agent-specific prompt for post generation."""
        a, b = str(topic[0]).strip(), str(topic[1]).strip()
        
        prompt = f"""Write a short, social-media style post (1-3 sentences, <320 characters) in first person
about {a} vs {b}. Your current opinion: {self.current_opinion:.3f} (-1=agrees with {a}, 1=agrees with {b})."""

        if neighbor_posts and len(neighbor_posts) > 0:
            prompt += f"""

Here are posts from your connected neighbors that you can respond to:
"""
            for i, post in enumerate(neighbor_posts[:5]):
                prompt += f"{post}\n"
            
            prompt += """
You may respond to 1-2 of them by name (e.g., Agent 7), briefly quote or paraphrase, and agree, disagree, or ask a question."""
        else:
            prompt += """
If you see other agents' posts, you may respond to 1-2 of them by name (e.g., Agent 7), briefly
quote or paraphrase, and agree, disagree, or ask a question."""

        prompt += """ Prose only, no numeric score.

IMPORTANT: You must generate actual text content. Do not just reason about it - write the actual post.
"""
        
        return prompt

    def update_opinion(self, new_opinion: float):
        """
        Update the agent's opinion.
        
        Args:
            new_opinion: New opinion value (-1 to 1)
        """
        # Enforce bounds to keep opinions in [-1, 1] range
        self.current_opinion = np.clip(new_opinion, -1.0, 1.0)
    
    def get_opinion(self) -> float:
        """
        Get current opinion value.
        
        Returns:
            Current opinion value (-1 to 1)
        """
        return self.current_opinion 

    def interpret_post_prompt(self, post: str, topic) -> str:
        """Generate agent-specific interpretation prompt for a single post."""
        a, b = str(topic[0]).strip(), str(topic[1]).strip()
        
        return f"""
Analyze this post and determine where it falls on the opinion spectrum between "{a}" and "{b}".

-1.000 = The post clearly supports/advocates for "{a}" over "{b}"
 0.000 = The post is neutral or balanced between "{a}" and "{b}"  
 1.000 = The post clearly supports/advocates for "{b}" over "{a}"

Focus on the post's POSITION on the topic, not how well-written or agreeable it is.

Post: "{post}"

Respond with ONLY one number in [-1.000, 1.000] on its own line. Use 0.000 if neutral. Make sure to include at least two points of precision.
""" 

 