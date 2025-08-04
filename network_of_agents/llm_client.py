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
    
    def __init__(self, api_key: Optional[str] = None, batch_config: Optional[Dict] = None):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key for the LLM provider (if not provided, will use environment variable)
            batch_config: Configuration for batch processing
        """
        self.model_name = "gpt-4o-mini"  # Hardcoded model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key must be provided or set in environment variables")
        
        # Set the API key for LiteLLM
        litellm.api_key = self.api_key
        
        # Set batch configuration
        self.batch_config = batch_config or {}
        self.max_workers = self.batch_config.get('max_workers', 50)
        self.timeout = self.batch_config.get('timeout', 600)
        self.enable_fallback = self.batch_config.get('enable_fallback', True)
    
    def generate_post(self, topic: str, agent_id: int) -> str:
        """
        Generate a post about the given topic.
        
        Args:
            topic: Topic to generate post about
            agent_id: ID of the agent generating the post
            
        Returns:
            Generated post text (1-2 sentences)
        """
        prompt = f"""
Generate a 1-2 sentence social media post about {topic}.
The post should be natural and conversational, like something you'd see on social media.
Keep it under 200 characters.
"""
        
        response = self._generate_text(prompt, max_tokens=100, temperature=0.7)
        return response.strip()
    
    def generate_posts_batch(self, topic: str, agent_ids: List[int]) -> List[str]:
        """
        Generate posts for multiple agents in batch.
        
        Args:
            topic: Topic to generate posts about
            agent_ids: List of agent IDs to generate posts for
            
        Returns:
            List of generated post texts
        """
        messages = []
        for agent_id in agent_ids:
            prompt = f"""
Generate a 1-2 sentence social media post about {topic}.
The post should be natural and conversational, like something you'd see on social media.
Keep it under 200 characters.
"""
            messages.append([{"role": "user", "content": prompt}])
        
        try:
            responses = litellm.batch_completion(
                model=self.model_name,
                messages=messages,
                max_tokens=100,
                temperature=0.7,
                max_workers=min(len(messages), self.max_workers),
                timeout=self.timeout
            )
            
            posts = []
            for response in responses:
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    posts.append(response.choices[0].message.content.strip())
                else:
                    # Fallback for failed responses
                    posts.append(f"Default post about {topic}")
            
            return posts
            
        except Exception as e:
            # Fallback to sequential processing
            if self.enable_fallback:
                print(f"Batch generation failed, falling back to sequential: {e}")
                return [self.generate_post(topic, agent_id) for agent_id in agent_ids]
            else:
                raise e
    
    def interpret_single_post(self, post: str, topic: str) -> float:
        """
        Interpret a single post to extract opinion value for the given topic.
        
        Args:
            post: Post to analyze
            topic: Topic to extract opinion for
            
        Returns:
            Opinion value (0-1) for the post
        """
        prompt = f"""
Analyze the following post about {topic} and provide an opinion value.
Provide a single number between 0 and 1:
0 = strongly oppose/disagree
1 = strongly support/agree

Post: "{post}"

IMPORTANT: Respond with ONLY a single number between 0 and 1.
Do not include any explanations, text, or other content.
Example format: 0.8
"""
        
        response = self._generate_text(prompt, max_tokens=50, temperature=0.3)
        return self._parse_single_opinion_response(response)
    
    def interpret_posts_batch(self, posts: List[str], topic: str) -> List[float]:
        """
        Interpret multiple posts in batch to extract opinion values.
        
        Args:
            posts: List of posts to analyze
            topic: Topic to extract opinions for
            
        Returns:
            List of opinion values (0-1) for each post
        """
        messages = []
        for post in posts:
            prompt = f"""
Analyze the following post about {topic} and provide an opinion value.
Provide a single number between 0 and 1:
0 = strongly oppose/disagree
1 = strongly support/agree

Post: "{post}"

IMPORTANT: Respond with ONLY a single number between 0 and 1.
Do not include any explanations, text, or other content.
Example format: 0.8
"""
            messages.append([{"role": "user", "content": prompt}])
        
        try:
            responses = litellm.batch_completion(
                model=self.model_name,
                messages=messages,
                max_tokens=50,
                temperature=0.3,
                max_workers=min(len(messages), self.max_workers),
                timeout=self.timeout
            )
            
            opinions = []
            for response in responses:
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    opinion = self._parse_single_opinion_response(response.choices[0].message.content)
                    opinions.append(opinion)
                else:
                    # Fallback for failed responses
                    import random
                    opinions.append(random.random())
            
            return opinions
            
        except Exception as e:
            # Fallback to sequential processing
            if self.enable_fallback:
                print(f"Batch interpretation failed, falling back to sequential: {e}")
                return [self.interpret_single_post(post, topic) for post in posts]
            else:
                raise e
    
    def interpret_posts(self, posts: List[str], topic: str) -> List[float]:
        """
        Interpret posts to extract opinion values for the given topic.
        
        Args:
            posts: List of posts to analyze
            topic: Topic to extract opinions for
            
        Returns:
            List of opinion values (0-1) for each post
        """
        opinions = []
        for post in posts:
            opinion = self.interpret_single_post(post, topic)
            opinions.append(opinion)
        return opinions
    
    def _generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate text using the LLM.
        
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
    
    def _parse_single_opinion_response(self, response: str) -> float:
        """
        Parse the LLM response to extract a single opinion value.
        
        Args:
            response: LLM response text
            
        Returns:
            Opinion value (0-1)
        """
        try:
            # Clean the response
            response = response.strip()
            response = response.replace('\n', '').replace(' ', '')
            
            # Extract number
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', response)
            
            if len(numbers) == 0:
                raise ValueError("No number found in response")
            
            opinion = float(numbers[0])
            
            # Ensure value is in [0, 1] range
            opinion = max(0.0, min(1.0, opinion))
            
            return opinion
            
        except Exception as e:
            # If parsing fails, return random value
            import random
            return random.random()
    
    def _parse_opinion_response(self, response: str, expected_count: int) -> List[float]:
        """
        Parse the LLM response to extract opinion values.
        
        Args:
            response: LLM response text
            expected_count: Expected number of opinion values
            
        Returns:
            List of opinion values
        """
        try:
            # Clean the response
            response = response.strip()
            response = response.replace('\n', '').replace(' ', '')
            
            # Extract numbers
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', response)
            
            if len(numbers) != expected_count:
                raise ValueError(f"Expected {expected_count} numbers, got {len(numbers)}")
            
            opinions = [float(num) for num in numbers]
            
            # Ensure values are in [0, 1] range
            opinions = [max(0.0, min(1.0, op)) for op in opinions]
            
            return opinions
            
        except Exception as e:
            # If parsing fails, return random values
            import random
            return [random.random() for _ in range(expected_count)] 