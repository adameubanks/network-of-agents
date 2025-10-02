"""
LLM-based opinion generator implementation.
"""

import os
import openai
from typing import Dict, Any, List, Optional
from ..core.base_models import OpinionGenerator

class LLMGenerator(OpinionGenerator):
    """
    LLM-based opinion generator.
    
    Uses a language model to generate opinions and posts based on
    the topic and neighbor context.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.model_name = parameters["model_name"]
        self.api_key_env = parameters["api_key_env"]
        self.max_tokens = parameters.get("max_tokens", 100)
        self.temperature = parameters.get("temperature", 0.7)
        
        # Initialize OpenAI client
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {self.api_key_env}")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_opinion(self, 
                        agent_id: int,
                        current_opinion: float,
                        topic: Dict[str, Any],
                        neighbor_posts: List[str] = None) -> float:
        """
        Generate an opinion using LLM.
        
        For LLM-based opinion dynamics, we generate a post and then
        extract the opinion from it. The actual opinion update happens
        through the rating mechanism.
        
        Args:
            agent_id: ID of the agent
            current_opinion: Current opinion value
            topic: Topic information
            neighbor_posts: Posts from neighboring agents
            
        Returns:
            Opinion value (unchanged for now, will be updated through rating)
        """
        return current_opinion
    
    def generate_post(self, 
                     agent_id: int,
                     current_opinion: float,
                     topic: Dict[str, Any],
                     neighbor_posts: List[str] = None) -> str:
        """
        Generate a post expressing the agent's opinion using LLM.
        
        Args:
            agent_id: ID of the agent
            current_opinion: Current opinion value
            topic: Topic information
            neighbor_posts: Posts from neighboring agents
            
        Returns:
            Generated post text
        """
        # Build context from neighbor posts
        context = ""
        if neighbor_posts:
            context = "Here are some recent posts from your neighbors:\n"
            for i, post in enumerate(neighbor_posts[:6]):  # Limit to 6 posts
                context += f"{i+1}. {post[:220]}\n"  # Truncate long posts
            context += "\n"
        
        # Create prompt
        prompt = f"""You are participating in a social media discussion about: {topic['name']}

{context}The topic is: {topic['statement_a']} vs {topic['statement_b']}

Your current opinion is {current_opinion:.2f} (where -1 = strongly disagree, 0 = neutral, +1 = strongly agree).

Write a short social media post (1-3 sentences, max 320 characters) expressing your opinion on this topic. Be authentic and natural in your language."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a social media user participating in online discussions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            post = response.choices[0].message.content.strip()
            return post
            
        except Exception as e:
            # Fallback to simple text generation
            return f"Agent {agent_id}: I have an opinion of {current_opinion:.2f} on this topic."

