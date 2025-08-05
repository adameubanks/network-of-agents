"""
Streamlined LLM client for post generation and interpretation.
"""

import os
from typing import List, Optional, Dict
import litellm
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    Streamlined client for LLM-based post generation and interpretation.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini", 
                 batch_config: Optional[Dict] = None, generation_temperature: float = 0.9, 
                 rating_temperature: float = 0.1):
        """
        Initialize LLM client.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the LLM model to use
            batch_config: Batch processing configuration
            generation_temperature: Sampling temperature for post generation (default: 0.9)
            rating_temperature: Sampling temperature for opinion rating (default: 0.1)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("No API key provided or found in environment")
        
        # Set up litellm
        litellm.set_verbose = False
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        # Model configuration
        self.model_name = model_name
        
        # Temperature configuration - these should be set from config.json
        self.generation_temperature = generation_temperature
        self.rating_temperature = rating_temperature
        
        # Batch processing configuration
        self.max_workers = batch_config.get('max_workers', 50) if batch_config else 50
        self.timeout = batch_config.get('timeout', 600) if batch_config else 600
    
    def generate_posts_for_agents(self, topic: str, agents: List) -> List[str]:
        """
        Generate posts for multiple agents using their specific personalities.
        
        Args:
            topic: Topic to generate posts about
            agents: List of agent objects with personalities
            
        Returns:
            List of generated post texts
        """
        prompts = []
        for agent in agents:
            prompt = agent.generate_post_prompt(topic)
            prompts.append(prompt)
        
        responses = self._call_llm(prompts, max_tokens=100, temperature=self.generation_temperature)
        return [response.strip() for response in responses]
    
    def interpret_posts_for_agents(self, posts: List[str], topic: str, agents: List) -> List[List[float]]:
        """
        Interpret posts using agent-specific personalities with individual calls.
        
        Args:
            posts: List of posts to analyze
            topic: Topic to extract opinions for
            agents: List of agent objects with personalities
            
        Returns:
            List of opinion values (0-1) for each agent's interpretation
        """
        all_interpretations = []
        
        for agent in agents:
            interpretations = []
            for post in posts:
                opinion = agent.interpret_single_post(self, post, topic)
                interpretations.append(opinion)
            all_interpretations.append(interpretations)
        
        return all_interpretations
    
    def interpret_single_post(self, agent, post: str, topic: str) -> float:
        """
        Interpret a single post using agent-specific personality.
        
        Args:
            agent: Agent object with personality
            post: Single post to interpret
            topic: Topic the post is about
            
        Returns:
            Single opinion value (0-1)
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = agent.interpret_single_post_prompt(post, topic)
                response = self._generate_single_text(prompt, max_tokens=20)
                opinion = self._parse_opinion_response(response)
                return opinion
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Warning: Failed to interpret post after {max_retries} attempts. Using fallback value 0.5. Error: {e}")
                    return 0.5  # Fallback to neutral opinion
                else:
                    print(f"Attempt {attempt + 1} failed, retrying... Error: {e}")
                    continue
    
    def _generate_single_text(self, prompt: str, max_tokens: int = 1000, temperature: float = None) -> str:
        """
        Generate single text using the LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
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
            raise Exception(f"LLM generation failed: {e}")
    
    def _parse_multiple_opinion_response(self, response: str, expected_count: int) -> List[float]:
        """
        DEPRECATED: Parse the LLM response to extract multiple opinion values.
        This method is deprecated in favor of individual opinion calls.
        
        Args:
            response: LLM response text
            expected_count: Expected number of opinion values
            
        Returns:
            List of opinion values
        """
        # Clean the response
        response = response.strip()
        response = response.replace('\n', ' ').replace('  ', ' ')
        
        # Extract numbers
        import re
        numbers = re.findall(r'0\.\d+|1\.0|0|1', response)
        
        # Handle cases where we don't get exactly the expected number
        if len(numbers) < expected_count:
            raise ValueError(f"Expected {expected_count} numbers, got {len(numbers)}")
        elif len(numbers) > expected_count:
            print(f"Warning: Expected {expected_count} numbers, got {len(numbers)}. Truncating.")
            numbers = numbers[:expected_count]
        
        opinions = [float(num) for num in numbers]
        
        # Ensure values are in [0, 1] range
        opinions = [max(0.0, min(1.0, op)) for op in opinions]
        
        return opinions
    
    def _call_llm(self, prompts: List[str], max_tokens: int = 1000, temperature: float = None) -> List[str]:
        """
        Call LLM with batch processing.
        
        Args:
            prompts: List of prompts to send
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated responses
        """
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        
        responses = litellm.batch_completion(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            max_workers=min(len(messages), self.max_workers),
            timeout=self.timeout
        )
        
        results = []
        for response in responses:
            if hasattr(response, 'choices') and len(response.choices) > 0:
                results.append(response.choices[0].message.content)
            else:
                raise Exception("LLM response missing choices")
        
        return results
    
    def _parse_opinion_response(self, response: str) -> float:
        """
        Parse the LLM response to extract a single opinion value.
        
        Args:
            response: LLM response text
            
        Returns:
            Opinion value (0-1)
        """
        # Clean the response
        response = response.strip()
        response = response.replace('\n', '').replace(' ', '')
        
        # Extract number
        import re
        numbers = re.findall(r'0\.\d+|1\.0|0|1', response)
        
        if len(numbers) == 0:
            # Try a more flexible pattern
            numbers = re.findall(r'\d+\.?\d*', response)
            if len(numbers) == 0:
                raise ValueError(f"No number found in response: '{response}'")
            else:
                # Convert to float and normalize to [0,1]
                try:
                    value = float(numbers[0])
                    if value > 1:
                        value = value / 100  # Assume it's a percentage
                    return max(0.0, min(1.0, value))
                except ValueError:
                    raise ValueError(f"Could not parse number from response: '{response}'")
        
        opinion = float(numbers[0])
        
        # Ensure value is in [0, 1] range
        opinion = max(0.0, min(1.0, opinion))
        
        return opinion 