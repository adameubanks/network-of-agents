"""
Agent class for post generation and interpretation.
"""

import numpy as np
import random
from typing import List, Optional
from .llm_client import LLMClient


class Agent:
    """
    Represents an individual agent in the social network with unique personality.
    """
    
    def __init__(self, agent_id: int):
        """
        Initialize an agent with personality traits.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        self.agent_id = agent_id
        
        # Initialize with random opinion
        self.current_opinion = np.random.random()

        # Set random seed for consistent personality (but don't affect numpy)
        import random as python_random
        python_random.seed(agent_id)
        
        # Generate personality traits
        self.personality = self._generate_personality()
    
    def _generate_personality(self) -> dict:
        """
        Generate MBTI personality traits for the agent.
        
        Returns:
            Dictionary containing personality traits
        """
        # MBTI types
        mbti_types = [
            "INTJ", "INTP", "ENTJ", "ENTP",
            "INFJ", "INFP", "ENFJ", "ENFP",
            "ISTJ", "ISFJ", "ESTJ", "ESFJ",
            "ISTP", "ISFP", "ESTP", "ESFP"
        ]
        mbti_type = random.choice(mbti_types)
        
        return {
            "mbti_type": mbti_type
        }
    
    def generate_post_prompt(self, topic: str) -> str:
        """
        Generate agent-specific prompt for post generation.
        
        Args:
            topic: Topic to generate post about
            
        Returns:
            Agent-specific prompt
        """
        mbti_type = self.personality["mbti_type"]
        
        # MBTI-specific instructions
        mbti_instructions = {
            "INTJ": "Be strategic and analytical. Focus on long-term implications and logical reasoning.",
            "INTP": "Be curious and theoretical. Question assumptions and explore complex ideas.",
            "ENTJ": "Be decisive and direct. Take charge and express strong, confident opinions.",
            "ENTP": "Be innovative and argumentative. Challenge conventional thinking with wit.",
            "INFJ": "Be idealistic and empathetic. Focus on human values and deeper meaning.",
            "INFP": "Be authentic and values-driven. Express personal feelings and moral convictions.",
            "ENFJ": "Be inspiring and supportive. Encourage others and focus on harmony.",
            "ENFP": "Be enthusiastic and creative. Express excitement and explore possibilities.",
            "ISTJ": "Be practical and responsible. Focus on facts, order, and traditional values.",
            "ISFJ": "Be caring and loyal. Emphasize harmony, cooperation, and practical support.",
            "ESTJ": "Be organized and decisive. Value efficiency, structure, and clear leadership.",
            "ESFJ": "Be warm and sociable. Focus on helping others and maintaining social harmony.",
            "ISTP": "Be flexible and pragmatic. Adapt to situations with practical problem-solving.",
            "ISFP": "Be gentle and artistic. Express personal values through quiet, creative means.",
            "ESTP": "Be energetic and action-oriented. Focus on immediate results and practical solutions.",
            "ESFP": "Be spontaneous and friendly. Bring energy and fun to social interactions."
        }
        
        prompt = f"""
Generate a 1-2 sentence social media post about {topic}.
Your MBTI type ({mbti_type}): {mbti_instructions[mbti_type]}
Your current opinion intensity: {self.current_opinion:.2f} (0=strongly oppose, 1=strongly support)

Examples of opinion levels for any topic:
- Opinion 0.0: "I strongly oppose this and believe it's wrong."
- Opinion 0.2: "I'm mostly against this but see some valid points."
- Opinion 0.5: "I'm neutral about this - I don't have strong feelings either way."
- Opinion 0.8: "I mostly support this but have some concerns."
- Opinion 1.0: "I fully support this and believe it's right."

Your opinion is {self.current_opinion:.2f}, so your post should reflect this level of support/opposition for {topic}.
Keep it under 200 characters.
Make the post reflect your unique personality and current opinion level.
"""
        return prompt

    def generate_post(self, llm_client: LLMClient, topic: str) -> str:
        """
        Generate a post about the given topic using agent-specific prompting.
        
        Args:
            llm_client: LLM client for post generation
            topic: Topic to generate post about
            
        Returns:
            Generated post text
        """
        prompt = self.generate_post_prompt(topic)
        return llm_client._generate_single_text(prompt, max_tokens=100, temperature=llm_client.generation_temperature)
    
    def interpret_posts(self, llm_client: LLMClient, posts: List[str], topic: str) -> List[float]:
        """
        Interpret posts using agent-specific prompting with individual calls.
        
        Args:
            llm_client: LLM client for post interpretation
            posts: List of posts to interpret
            topic: Topic the posts are about
            
        Returns:
            List of interpreted opinion values (0-1)
        """
        interpretations = []
        for post in posts:
            opinion = self.interpret_single_post(llm_client, post, topic)
            interpretations.append(opinion)
        return interpretations
    
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

    def interpret_single_post_prompt(self, post: str, topic: str) -> str:
        """
        Generate agent-specific interpretation prompt for a single post.
        
        Args:
            post: Single post to interpret
            topic: Topic the post is about
            
        Returns:
            Agent-specific interpretation prompt
        """
        mbti_type = self.personality["mbti_type"]
        
        # MBTI-specific interpretation biases
        mbti_biases = {
            "INTJ": "Analyze posts strategically. Look for logical consistency and long-term implications.",
            "INTP": "Question assumptions in posts. Consider alternative perspectives and theoretical frameworks.",
            "ENTJ": "Evaluate posts for effectiveness and decisiveness. Look for strong, confident positions.",
            "ENTP": "Challenge conventional thinking in posts. Consider innovative and argumentative perspectives.",
            "INFJ": "Interpret posts for deeper meaning and human values. Consider emotional and moral implications.",
            "INFP": "Evaluate posts for authenticity and personal values. Consider emotional truth and moral convictions.",
            "ENFJ": "Look for inspiring and supportive elements in posts. Consider harmony and encouragement.",
            "ENFP": "Seek enthusiasm and creativity in posts. Look for excitement and possibility exploration.",
            "ISTJ": "Focus on facts, order, and traditional values in posts. Look for practical and responsible content.",
            "ISFJ": "Consider caring and loyal elements in posts. Look for harmony and practical support.",
            "ESTJ": "Evaluate posts for organization and efficiency. Look for clear structure and leadership.",
            "ESFJ": "Consider warm and sociable elements in posts. Look for social harmony and helpfulness.",
            "ISTP": "Look for flexible and pragmatic approaches in posts. Consider practical problem-solving.",
            "ISFP": "Seek gentle and artistic elements in posts. Look for personal values and quiet creativity.",
            "ESTP": "Focus on energetic and action-oriented content in posts. Look for immediate practical solutions.",
            "ESFP": "Look for spontaneous and friendly elements in posts. Consider energy and social fun."
        }
        
        prompt = f"""
Analyze the following post about {topic} and provide a single opinion value.
Your MBTI type ({mbti_type}): {mbti_biases[mbti_type]}

Examples of opinion ratings for {topic}:
- 0.0: "I strongly oppose this and believe it's wrong."
- 0.2: "I'm mostly against this but see some valid points."
- 0.5: "I'm neutral about this - I don't have strong feelings either way."
- 0.8: "I mostly support this but have some concerns."
- 1.0: "I fully support this and believe it's right."

Post to analyze: "{post}"

Provide a single number between 0 and 1:
0 = strongly oppose/disagree
1 = strongly support/agree

IMPORTANT: Respond with ONLY a single number between 0 and 1.
"""
        return prompt
    
    def interpret_single_post(self, llm_client: LLMClient, post: str, topic: str) -> float:
        """
        Interpret a single post using agent-specific prompting.
        
        Args:
            llm_client: LLM client for post interpretation
            post: Single post to interpret
            topic: Topic the post is about
            
        Returns:
            Single interpreted opinion value (0-1)
        """
        prompt = self.interpret_single_post_prompt(post, topic)
        response = llm_client._generate_single_text(prompt, max_tokens=20, temperature=llm_client.rating_temperature)
        opinion = llm_client._parse_opinion_response(response)
        return opinion 