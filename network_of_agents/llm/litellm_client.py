"""
Simplified LiteLLM client for opinion interpretation and generation.
"""

import os
from typing import Dict, List, Optional, Any
import litellm
from dotenv import load_dotenv
import re

load_dotenv()


class LiteLLMClient:
    """
    Simplified client for LLM-based opinion interpretation and generation.
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
    
    def interpret_text_to_opinions(self, text: str, topics: List[str]) -> List[float]:
        """
        Interpret text content to extract opinion values for given topics.
        
        Args:
            text: Text content to analyze
            topics: List of topics to extract opinions for
            
        Returns:
            List of opinion values (0-1) for each topic
        """
        prompt = f"""
Analyze the following text and provide opinion values for each topic.
For each topic, provide a single number between 0 and 1:
0 = strongly oppose/disagree
1 = strongly support/agree

Text to analyze: "{text}"

Topics: {', '.join(topics)}

IMPORTANT: Respond with ONLY a comma-separated list of numbers, one for each topic, in order.
Do not include any explanations, text, or other content.
Example format: 0.8, 0.2, 0.9
"""
        
        response = self._generate_text(prompt, max_tokens=200, temperature=0.3)
        return self._parse_opinion_response(response, len(topics))
    
    def generate_text_from_opinions(self, topics: List[str], opinion_vector: List[float]) -> str:
        """
        Generate text content that reflects the given opinion vector.
        
        Args:
            topics: List of topics
            opinion_vector: Opinion values (0-1) for each topic
            
        Returns:
            Generated text reflecting the opinions
        """
        # Create a description of the opinions
        opinion_desc = []
        for i, (topic, opinion) in enumerate(zip(topics, opinion_vector)):
            if opinion > 0.7:
                stance = "strongly support"
            elif opinion > 0.5:
                stance = "support"
            elif opinion > 0.3:
                stance = "somewhat oppose"
            else:
                stance = "strongly oppose"
            opinion_desc.append(f"{topic}: {stance} ({opinion:.2f})")
        
        prompt = f"""
Generate a short social media post that reflects the following opinions:

{', '.join(opinion_desc)}

The post should be natural, conversational, and reflect these opinions without being overly explicit.
Keep it under 200 words.
"""
        
        return self._generate_text(prompt, max_tokens=300, temperature=0.7)
    
    def _generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        response = litellm.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def _parse_opinion_response(self, response: str, expected_count: int) -> List[float]:
        """
        Parse LLM response to extract opinion values.
        
        Args:
            response: Raw LLM response
            expected_count: Expected number of opinion values
            
        Returns:
            List of opinion values
        """
        # Clean the response - remove any non-numeric content
        response = response.strip()
        
        # Try to extract just the numeric part
        numeric_match = re.search(r'[\d.,\s]+', response)
        if numeric_match:
            response = numeric_match.group()
        
        # Parse the response to extract numbers
        try:
            opinions = [float(x.strip()) for x in response.split(',')]
        except ValueError as e:
            raise ValueError(f"Failed to parse LLM response '{response}': {e}")
        
        # Ensure we have the right number of opinions
        if len(opinions) != expected_count:
            raise ValueError(f"Expected {expected_count} opinions, got {len(opinions)} from response '{response}'")
        
        # Clamp values to [0, 1]
        opinions = [max(0.0, min(1.0, op)) for op in opinions]
        
        return opinions
    
    def measure_round_trip_loss(self, topics: List[str], original_opinions: List[float]) -> Dict[str, Any]:
        """
        Measure the loss between original opinions and round-trip interpreted opinions.
        
        Args:
            topics: List of topics
            original_opinions: Original opinion values (0-1) for each topic
            
        Returns:
            Dictionary containing loss metrics and round-trip data
        """
        # Generate text from original opinions
        generated_text = self.generate_text_from_opinions(topics, original_opinions)
        
        # Interpret the generated text back to opinions
        interpreted_opinions = self.interpret_text_to_opinions(generated_text, topics)
        
        # Calculate losses
        losses = []
        for i, (orig, interp) in enumerate(zip(original_opinions, interpreted_opinions)):
            loss = abs(orig - interp)
            losses.append(loss)
        
        # Calculate metrics
        total_loss = sum(losses)
        average_loss = total_loss / len(losses) if losses else 0.0
        max_loss = max(losses) if losses else 0.0
        min_loss = min(losses) if losses else 0.0
        
        return {
            'original_opinions': original_opinions,
            'generated_text': generated_text,
            'interpreted_opinions': interpreted_opinions,
            'losses': losses,
            'total_loss': total_loss,
            'average_loss': average_loss,
            'max_loss': max_loss,
            'min_loss': min_loss,
            'topics': topics
        } 