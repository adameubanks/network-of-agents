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
        litellm.drop_params = True
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        # Model configuration
        self.model_name = model_name
        
        # Batch processing configuration with defaults
        self.max_workers = 50
        self.timeout = 600
    
    def _is_gpt5_model(self) -> bool:
        """Detect gpt-5 family models by name prefix."""
        try:
            name = str(self.model_name) if self.model_name is not None else ""
        except Exception:
            name = ""
        return name.startswith("gpt-5")

    
    
    def _build_llm_params(self, max_tokens: int) -> Dict:
        """Build parameter dict compatible with model family (gpt-5 vs others)."""
        # OpenAI chat completions expect 'max_tokens'; avoid passing multiple variants
        # since OpenAI rejects setting both max_tokens and max_completion_tokens.
        params: Dict = {'max_tokens': max_tokens}
        if self._is_gpt5_model():
            # Keep reasoning cheap so output tokens remain
            params['reasoning_effort'] = 'low'
            # Encourage plain text output
            params['response_format'] = {'type': 'text'}
        return params

    def _completion_with_retry_text(self, prompt: str, max_tokens: int) -> str:
        """Make a single completion call and return extracted text.

        For GPT-5: if empty and budget is small, retry once with a larger budget.
        """
        llm_params = self._build_llm_params(max_tokens)
        response = litellm.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **llm_params
        )
        text = self._extract_text_from_model_response(response) or ""
        if not text and self._is_gpt5_model() and max_tokens < 128:
            try:
                retry_params = self._build_llm_params(max(128, max_tokens * 2))
                response_retry = litellm.completion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **retry_params
                )
                text = self._extract_text_from_model_response(response_retry) or ""
            except Exception:
                pass
        return text

    def _build_extra_body(self) -> Dict:
        """Extra payload for specific model families (e.g., GPT-5 Responses API)."""
        if self._is_gpt5_model():
            # Ensure text output with Responses API
            return {'response_format': {'type': 'text'}}
        return {}

    
    
    def generate_posts_for_agents(self, topic: str, agents: List) -> List[str]:
        """
        Generate posts for multiple agents using their specific personalities.
        
        Args:
            topic: Topic to generate posts about
            agents: List of agent objects with personalities
            
        Returns:
            List of generated post texts
        """
        start_time = time.time()
        
        prompts = []
        for agent in agents:
            prompt = agent.generate_post_prompt(topic)
            prompts.append(prompt)
        
        logger.info(f"Generated {len(prompts)} prompts, calling LLM API...")
        
        try:
            responses = self._call_llm(prompts, max_completion_tokens=800)
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
                        # Use single-call with small token budget for numeric output
                        response_text = self._generate_single_text(prompt, max_completion_tokens=128)
                        response = [response_text]
                        rating = self._parse_opinion_response(response[0])
                        agent_ratings.append((neighbor_idx, rating))
                        logger.debug(f"Agent {i} rated neighbor {neighbor_idx} post: {rating}")
                    except Exception as e:
                        logger.debug(f"Failed to rate neighbor {neighbor_idx} post for agent {i}: {str(e)}")
                        # Use neutral rating as fallback
                        agent_ratings.append((neighbor_idx, 0.0))
                
                # Average the ratings of all connected neighbors' posts
                if agent_ratings:
                    avg_rating = sum(rating for _, rating in agent_ratings) / len(agent_ratings)
                    neighbor_opinions.append(avg_rating)
                    individual_ratings.append(agent_ratings)
                    logger.debug(f"Agent {i} average neighbor rating: {avg_rating}")
                else:
                    logger.debug(f"Agent {i} has no valid ratings, using neutral opinion")
                    neighbor_opinions.append(0.0)
                    individual_ratings.append([])
        
        elapsed_time = time.time() - start_time
        
        # Final validation
        if len(neighbor_opinions) != len(agents) or len(individual_ratings) != len(agents):
            logger.error(f"OUTPUT VALIDATION FAILED: opinions={len(neighbor_opinions)}, ratings={len(individual_ratings)}, expected={len(agents)}")
            raise ValueError(f"Output validation failed: opinions={len(neighbor_opinions)}, ratings={len(individual_ratings)}, expected={len(agents)}")
        
        return neighbor_opinions, individual_ratings
    

    
    def _generate_single_text(self, prompt: str, max_completion_tokens: int = 1000, temperature: float = None) -> str:
        """
        Generate single text using the LLM.
        
        Args:
            prompt: Input prompt
            max_completion_tokens: Maximum completion tokens to generate
            temperature: Sampling temperature (uses class default if None)
            
        Returns:
            Generated text
        """
        return self._completion_with_retry_text(prompt, max_completion_tokens)
    

    
    def _call_llm(self, prompts: List[str], max_completion_tokens: int = 1000) -> List[str]:
        """Call LLM per prompt (no batch), returning extracted texts."""
        return self._generate_texts(prompts, max_tokens=max_completion_tokens)

    def _generate_texts(self, prompts: List[str], max_tokens: int) -> List[str]:
        results: List[str] = []
        for prompt in prompts:
            try:
                text = self._completion_with_retry_text(prompt, max_tokens)
            except Exception as e:
                logger.warning(f"LLM single call failed: {e}")
                text = ""
            results.append(text)
        return results
    
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

    def _extract_text_from_model_response(self, response) -> Optional[str]:
        """Extract text from a LiteLLM ModelResponse across providers/models.

        Tries, in order:
        - choices[0].message.content (string or list of segments)
        - choices[0].text
        - response.output_text
        - response.content
        - deep search in dict-like dump for first non-empty string under keys 'content' or 'text'
        """
        # choices[0].message.content
        try:
            if hasattr(response, 'choices') and len(response.choices) > 0:
                message = getattr(response.choices[0], 'message', None)
                if message is not None:
                    content = getattr(message, 'content', None)
                    if isinstance(content, str) and content.strip():
                        return content
                    if isinstance(content, list):
                        # Handle content as list of strings, dicts, or objects with 'text' attr.
                        try:
                            segments: list[str] = []
                            for seg in content:
                                if isinstance(seg, str):
                                    segments.append(seg)
                                elif isinstance(seg, dict):
                                    text_val = seg.get('text') or seg.get('content') or seg.get('value')
                                    if isinstance(text_val, str):
                                        segments.append(text_val)
                                else:
                                    # pydantic/model objects
                                    text_attr = getattr(seg, 'text', None)
                                    if isinstance(text_attr, str):
                                        segments.append(text_attr)
                            joined = "".join(segments)
                        except Exception:
                            joined = None
                        if isinstance(joined, str) and joined.strip():
                            return joined
                # choices[0].text
                choice_text = getattr(response.choices[0], 'text', None)
                if isinstance(choice_text, str) and choice_text.strip():
                    return choice_text
        except Exception:
            pass

        # response.output_text
        try:
            output_text = getattr(response, 'output_text', None)
            if isinstance(output_text, str) and output_text.strip():
                return output_text
        except Exception:
            pass

        # response.content
        try:
            top_content = getattr(response, 'content', None)
            if isinstance(top_content, str) and top_content.strip():
                return top_content
        except Exception:
            pass

        # Deep search in dict-like dump
        def _to_dict_safe(obj):
            for method in ('model_dump', 'dict', 'to_dict', '__dict__'):
                try:
                    candidate = getattr(obj, method)
                    if callable(candidate):
                        d = candidate()
                    else:
                        d = candidate
                    if isinstance(d, dict):
                        return d
                except Exception:
                    continue
            return None

        def _dfs_find_text(d):
            try:
                if isinstance(d, dict):
                    for k, v in d.items():
                        if isinstance(k, str) and k.lower() in ("content", "text", "output_text"):
                            if isinstance(v, str) and v.strip():
                                return v
                            if isinstance(v, list):
                                try:
                                    parts: list[str] = []
                                    for seg in v:
                                        if isinstance(seg, str):
                                            parts.append(seg)
                                        elif isinstance(seg, dict):
                                            tv = seg.get('text') or seg.get('content') or seg.get('value')
                                            if isinstance(tv, str):
                                                parts.append(tv)
                                        else:
                                            ta = getattr(seg, 'text', None)
                                            if isinstance(ta, str):
                                                parts.append(ta)
                                    joined = "".join(parts)
                                except Exception:
                                    joined = None
                                if isinstance(joined, str) and joined.strip():
                                    return joined
                        res = _dfs_find_text(v)
                        if isinstance(res, str) and res.strip():
                            return res
                elif isinstance(d, list):
                    for item in d:
                        res = _dfs_find_text(item)
                        if isinstance(res, str) and res.strip():
                            return res
            except Exception:
                return None
            return None

        try:
            dumped = _to_dict_safe(response)
            if isinstance(dumped, dict):
                found = _dfs_find_text(dumped)
                if isinstance(found, str) and found.strip():
                    return found
        except Exception:
            pass

        return None

    def _call_llm_with_retry(self, prompts: List[str], max_completion_tokens: int = 1000, max_retries: int = 3) -> List[str]:
        """
        Call LLM with retry logic for improved reliability.
        
        Args:
            prompts: List of prompts to send
            max_completion_tokens: Maximum completion tokens to generate
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of generated responses
        """
        for attempt in range(max_retries):
            try:
                return self._call_llm(prompts, max_completion_tokens)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"LLM call failed after {max_retries} attempts: {str(e)}")
                    raise
                else:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
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
        return self._call_llm_with_retry(prompts, max_completion_tokens=800)
    
    def interpret_neighbor_posts_with_retry(self, posts: List[str], topic: str, agents: List, adjacency_matrix: np.ndarray) -> tuple[List[float], List[List[tuple[int, float]]]]:
        """Interpret posts with retry logic."""
        # This method is more complex, so we'll implement retry at the individual rating level
        return self.interpret_neighbor_posts(posts, topic, agents, adjacency_matrix) 