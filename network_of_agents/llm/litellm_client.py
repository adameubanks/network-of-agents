"""
LiteLLM client for API integration with various LLM providers.
"""

import os
from typing import Dict, List, Optional, Any
import litellm
from dotenv import load_dotenv

load_dotenv()


class LiteLLMClient:
    """
    Client for interacting with LLM APIs using LiteLLM.
    """
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize the LiteLLM client.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the LLM provider (if not provided, will use environment variable)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key must be provided or set in environment variables")
        
        # Set the API key for LiteLLM
        litellm.api_key = self.api_key
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""
    
    def generate_opinion_vector(self, topics: List[str], persona: str) -> List[float]:
        """
        Generate an opinion vector for a given persona and topics.
        
        Args:
            topics: List of topics to generate opinions for
            persona: Description of the agent persona
            
        Returns:
            List of opinion values (0-1) for each topic
        """
        prompt = f"""
        You are an AI agent with the following persona: {persona}
        
        For each of the following topics, provide a single number between 0 and 1 representing your level of support:
        0 = strongly oppose/disagree
        1 = strongly support/agree
        
        Topics: {', '.join(topics)}
        
        Respond with only a comma-separated list of numbers, one for each topic, in order.
        Example: 0.8, 0.2, 0.9
        """
        
        response = self.generate_text(prompt, max_tokens=200, temperature=0.3)
        
        try:
            # Parse the response to extract numbers
            opinions = [float(x.strip()) for x in response.split(',')]
            
            # Ensure we have the right number of opinions
            if len(opinions) != len(topics):
                raise ValueError(f"Expected {len(topics)} opinions, got {len(opinions)}")
            
            # Clamp values to [0, 1]
            opinions = [max(0.0, min(1.0, op)) for op in opinions]
            
            return opinions
        except Exception as e:
            print(f"Error parsing opinion vector: {e}")
            # Return random opinions as fallback
            import random
            return [random.random() for _ in topics]
    
    def analyze_converged_opinions(self, topics: List[str], opinion_vector: List[float]) -> str:
        """
        Generate content that reflects the converged opinion vector.
        
        Args:
            topics: List of topics
            opinion_vector: Opinion values for each topic
            
        Returns:
            Generated content reflecting the opinions
        """
        prompt = f"""
        Based on the following opinion values for these topics, generate a short social media post (2-3 sentences) that reflects these opinions:
        
        Topics and opinions:
        {', '.join([f'{topic}: {opinion:.2f}' for topic, opinion in zip(topics, opinion_vector)])}
        
        The post should be natural and reflect the strength of opinions (higher values = stronger support).
        """
        
        return self.generate_text(prompt, max_tokens=150, temperature=0.7)
    
    def detect_bias_patterns(self, topic_pairs: List[tuple], convergence_data: Dict) -> Dict[str, Any]:
        """
        Analyze bias patterns in convergence data.
        
        Args:
            topic_pairs: List of topic pairs to compare (e.g., [("queers", "gays"), ...])
            convergence_data: Dictionary containing convergence metrics for each topic
            
        Returns:
            Dictionary containing bias analysis results
        """
        prompt = f"""
        Analyze the following convergence data for potential bias patterns:
        
        Topic pairs: {topic_pairs}
        Convergence data: {convergence_data}
        
        Look for:
        1. Differences in convergence speed between topic pairs
        2. Differences in final convergence points
        3. Potential bias indicators
        
        Provide a brief analysis in 2-3 sentences.
        """
        
        analysis = self.generate_text(prompt, max_tokens=300, temperature=0.5)
        
        return {
            "analysis": analysis,
            "topic_pairs": topic_pairs,
            "convergence_data": convergence_data
        } 