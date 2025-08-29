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
    
    def generate_post_prompt(self, topic: str) -> str:
        """
        Generate agent-specific prompt for post generation.
        
        Args:
            topic: Topic to generate post about
            
        Returns:
            Agent-specific prompt
        """
        # Support either a single topic or a vs-topic formatted as "Topic A vs Topic B"
        s = topic.strip(); low = s.lower(); sep = " vs "
        i = low.find(sep)
        if i != -1:
            a = s[:i].strip(); b = s[i + len(sep):].strip()
            prompt = f"""
Write a short, social-media style post (1-3 sentences, ≤320 characters) in first person
about {a} vs {b}. Your current opinion: {self.current_opinion:.3f} (-1=favors {a}, 1=favors {b}).
If you see other agents' posts, you may respond to 1-2 of them by name (e.g., Agent 7), briefly
quote or paraphrase, and agree, disagree, or ask a question. Prose only, no numeric score.
"""
            return prompt

        prompt = f"""
Write a short, social-media style post (1-3 sentences, ≤320 characters) in first person
about {topic}. Your current opinion: {self.current_opinion:.3f} (-1=oppose, 1=support).
If you see other agents' posts, you may respond to 1-2 of them by name (e.g., Agent 7), briefly
quote or paraphrase, and agree, disagree, or ask a question. Prose only, no numeric score.
"""
        return prompt

    
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

    def interpret_post_prompt(self, post: str, topic: str) -> str:
        """
        Generate agent-specific interpretation prompt for a single post.
        
        Args:
            post: Single post to interpret
            topic: Topic the post is about
            
        Returns:
            Agent-specific interpretation prompt
        """
        # Support either a single topic or a vs-topic formatted as "Topic A vs Topic B"
        s = topic.strip(); low = s.lower(); sep = " vs "
        i = low.find(sep)
        if i != -1:
            a = s[:i].strip(); b = s[i + len(sep):].strip()
            prompt = f"""
Rate the post on the axis {a} (-1) ↔ {b} (1).
-1.000 = strongly favors {a} over {b}
 1.000 = strongly favors {b} over {a}

Post: "{post}"
Respond with ONLY one number in [-1.000, 1.000] on its own line. Use 0.000 if neutral.
"""
            return prompt
        prompt = f"""
Rate the post about "{topic}" on a -1 to 1 scale.
-1.000 = strongly oppose, 1.000 = strongly support

Post: "{post}"
Respond with ONLY one number in [-1.000, 1.000] on its own line. Use 0.000 if neutral.
"""
        return prompt 

 