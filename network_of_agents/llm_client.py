"""
Streamlined LLM client for post generation and interpretation.
"""

import os
from typing import List, Optional, Dict
import numpy as np
import litellm
from dotenv import load_dotenv
import logging
import time

load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Streamlined client for LLM-based post generation and interpretation.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = None):
        """
        Initialize LLM client.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the LLM model to use
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("No API key provided or found in environment")
        
        # Set up litellm
        litellm.set_verbose = False
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        # Model configuration
        self.model_name = model_name
        
        # Batch processing configuration with defaults
        self.max_workers = 50
        self.timeout = 600
    
    def generate_posts_for_agents(self, topic: str, agents: List) -> List[str]:
        """
        Generate posts for multiple agents using their specific personalities.
        
        Args:
            topic: Topic to generate posts about
            agents: List of agent objects with personalities
            
        Returns:
            List of generated post texts
        """
        logger.info(f"Generating posts for {len(agents)} agents on topic: {topic}")
        start_time = time.time()
        
        prompts = []
        for agent in agents:
            prompt = agent.generate_post_prompt(topic)
            prompts.append(prompt)
        
        logger.info(f"Generated {len(prompts)} prompts, calling LLM API...")
        
        try:
            responses = self._call_llm(prompts, max_tokens=400)
            logger.info(f"LLM API call successful. Received {len(responses)} responses")
            
            # Validate response count
            if len(responses) != len(agents):
                logger.error(f"RESPONSE COUNT MISMATCH: Expected {len(agents)} responses, got {len(responses)}")
                raise ValueError(f"Expected {len(agents)} responses, got {len(responses)}")
            
            # Clean and validate responses
            cleaned_responses = []
            for i, response in enumerate(responses):
                if response and response.strip():
                    cleaned_responses.append(response.strip())
                else:
                    logger.warning(f"Empty response for agent {i}, using fallback")
                    cleaned_responses.append(f"Agent {i} has no opinion on {topic}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Post generation completed in {elapsed_time:.2f}s. Generated {len(cleaned_responses)} posts")
            
            return cleaned_responses
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Post generation failed after {elapsed_time:.2f}s: {str(e)}")
            raise
    
    def interpret_neighbor_posts(self, posts: List[str], topic: str, agents: List, adjacency_matrix: np.ndarray) -> tuple[List[float], List[List[tuple[int, float]]]]:
        """
        Interpret connected neighbors' posts for each agent.
        
        Args:
            posts: List of posts to analyze
            topic: Topic to extract opinions for
            agents: List of agent objects with personalities
            adjacency_matrix: Current network connections
            
        Returns:
            Tuple of (neighbor_opinions, individual_ratings) where:
            - neighbor_opinions: List of average opinion values for each agent
            - individual_ratings: List of lists, each containing (neighbor_id, rating) tuples for that agent
        """
        logger.info(f"Interpreting neighbor posts for {len(agents)} agents on topic: {topic}")
        start_time = time.time()
        
        # Validate input data
        if len(posts) != len(agents):
            logger.error(f"INPUT VALIDATION FAILED: posts length ({len(posts)}) != agents length ({len(agents)})")
            raise ValueError(f"Posts length ({len(posts)}) must equal agents length ({len(agents)})")
        
        neighbor_opinions = []
        individual_ratings = []
        
        for i, agent in enumerate(agents):
            # Find connected neighbors for this agent
            connections = adjacency_matrix[i, :]
            neighbor_indices = np.where(connections == 1)[0]
            
            if len(neighbor_indices) == 0:
                # No connections - use agent's own opinion
                logger.debug(f"Agent {i} has no connections, using own opinion: {agent.get_opinion()}")
                neighbor_opinions.append(agent.get_opinion())
                individual_ratings.append([])
            else:
                # Rate each connected neighbor's post individually
                logger.debug(f"Agent {i} has {len(neighbor_indices)} connections, rating neighbor posts")
                agent_ratings = []
                for neighbor_idx in neighbor_indices:
                    try:
                        neighbor_post = posts[neighbor_idx]
                        prompt = agent.interpret_post_prompt(neighbor_post, topic)
                        response = self._call_llm([prompt], max_tokens=100)
                        rating = self._parse_opinion_response(response[0])
                        agent_ratings.append((neighbor_idx, rating))
                        logger.debug(f"Agent {i} rated neighbor {neighbor_idx} post: {rating}")
                    except Exception as e:
                        logger.warning(f"Failed to rate neighbor {neighbor_idx} post for agent {i}: {str(e)}")
                        # Use neutral rating as fallback
                        agent_ratings.append((neighbor_idx, 0.0))
                
                # Average the ratings of all connected neighbors' posts
                if agent_ratings:
                    avg_rating = sum(rating for _, rating in agent_ratings) / len(agent_ratings)
                    neighbor_opinions.append(avg_rating)
                    individual_ratings.append(agent_ratings)
                    logger.debug(f"Agent {i} average neighbor rating: {avg_rating}")
                else:
                    logger.warning(f"Agent {i} has no valid ratings, using neutral opinion")
                    neighbor_opinions.append(0.0)
                    individual_ratings.append([])
        
        elapsed_time = time.time() - start_time
        logger.info(f"Interpretation completed in {elapsed_time:.2f}s. Generated {len(neighbor_opinions)} opinions and {len(individual_ratings)} rating sets")
        
        # Final validation
        if len(neighbor_opinions) != len(agents) or len(individual_ratings) != len(agents):
            logger.error(f"OUTPUT VALIDATION FAILED: opinions={len(neighbor_opinions)}, ratings={len(individual_ratings)}, expected={len(agents)}")
            raise ValueError(f"Output validation failed: opinions={len(neighbor_opinions)}, ratings={len(individual_ratings)}, expected={len(agents)}")
        
        return neighbor_opinions, individual_ratings
    

    
    def _generate_single_text(self, prompt: str, max_tokens: int = 1000, temperature: float = None) -> str:
        """
        Generate single text using the LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (uses class default if None)
            
        Returns:
            Generated text
        """
        response = litellm.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    

    
    def _call_llm(self, prompts: List[str], max_tokens: int = 1000) -> List[str]:
        """
        Call LLM with batch processing.
        
        Args:
            prompts: List of prompts to send
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of generated responses
        """
        logger.info(f"Making batch LLM call with {len(prompts)} prompts, max_tokens={max_tokens}")
        start_time = time.time()
        
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        
        try:
            logger.debug(f"Calling litellm.batch_completion with {len(messages)} message batches")
            responses = litellm.batch_completion(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                max_workers=min(len(messages), self.max_workers),
                timeout=self.timeout
            )
            logger.info(f"Batch API call successful, received {len(responses)} responses")
            
            results = []
            for i, response in enumerate(responses):
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    results.append(response.choices[0].message.content)
                else:
                    logger.error(f"Response {i} missing choices: {response}")
                    raise Exception("LLM response missing choices")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Batch LLM call completed in {elapsed_time:.2f}s. Processed {len(results)} responses")
            
            # Validate response count
            if len(results) != len(prompts):
                logger.error(f"RESPONSE COUNT MISMATCH: Expected {len(prompts)} responses, got {len(results)}")
                raise ValueError(f"Expected {len(prompts)} responses, got {len(results)}")
            
            return results
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Batch LLM call failed after {elapsed_time:.2f}s: {str(e)}")
            raise
    
    def _parse_opinion_response(self, response: str) -> float:
        """
        Parse the LLM response to extract a single opinion value.
        
        Args:
            response: LLM response text
            
        Returns:
            Opinion value (-1 to 1)
        """
        # Clean the response
        response = response.strip()
        response = response.replace('\n', '').replace(' ', '')
        
        # Extract number with improved regex to allow negative decimals and more granular values
        import re
        # Updated pattern to match: -1.0, -0.234, 0.789, 1.0, etc.
        numbers = re.findall(r'-?\d+\.?\d*', response)
        
        if len(numbers) == 0:
            raise ValueError(f"No number found in response: '{response}'")
        
        try:
            opinion = float(numbers[0])
            # Ensure value is in [-1, 1] range
            opinion = max(-1.0, min(1.0, opinion))
            return opinion
        except ValueError:
            logger.error(f"Could not parse number from response: '{response}'")
            raise ValueError(f"Could not parse number from response: '{response}'")

    def _call_llm_with_retry(self, prompts: List[str], max_tokens: int = 1000, max_retries: int = 3) -> List[str]:
        """
        Call LLM with retry logic for improved reliability.
        
        Args:
            prompts: List of prompts to send
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of generated responses
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"LLM call attempt {attempt + 1}/{max_retries}")
                return self._call_llm(prompts, max_tokens)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"LLM call failed after {max_retries} attempts: {str(e)}")
                    raise
                else:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"LLM call attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
    
    def generate_posts_for_agents_with_retry(self, topic: str, agents: List, neighbor_posts_per_agent: Optional[List[List[str]]] = None) -> List[str]:
        """Generate posts with retry logic, optionally including prior neighbor posts."""
        prompts: List[str] = []
        for i, agent in enumerate(agents):
            base_prompt = agent.generate_post_prompt(topic)
            if neighbor_posts_per_agent is not None and i < len(neighbor_posts_per_agent):
                neighbor_texts = neighbor_posts_per_agent[i]
                if neighbor_texts:
                    # Append raw neighbor texts separated by newlines, no labels/metadata
                    combined = "\n".join(neighbor_texts)
                    base_prompt = f"{base_prompt}\nWhen generating your statement, respond to these posts from your social network. These are posts made from people you are connected to:\n{combined}"
            prompts.append(base_prompt)
        return self._call_llm_with_retry(prompts, max_tokens=400)
    
    def interpret_neighbor_posts_with_retry(self, posts: List[str], topic: str, agents: List, adjacency_matrix: np.ndarray) -> tuple[List[float], List[List[tuple[int, float]]]]:
        """Interpret posts with retry logic."""
        # This method is more complex, so we'll implement retry at the individual rating level
        return self.interpret_neighbor_posts(posts, topic, agents, adjacency_matrix) 