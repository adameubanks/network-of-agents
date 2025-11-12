"""
Simplified LLM client for post generation and interpretation.
"""

import os
from typing import List, Optional
import numpy as np
import litellm
import concurrent.futures as _f
from dotenv import load_dotenv
import logging
import re
import time

load_dotenv()

# Suppress verbose logging
os.environ["LITELLM_LOG"] = "ERROR"
logging.basicConfig(level=logging.WARNING, force=True)
for logger_name in ["litellm", "httpx", "openai", "urllib3", "requests"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

class LLMClient:
    """Client for generating and rating posts."""
    
    # Supported models
    SUPPORTED_MODELS = ["gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-5-pro", "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"]
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = None, 
                 max_workers: int = None, timeout: int = None, max_tokens: int = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("No API key provided or found in environment")
        
        if model_name is None:
            raise ValueError("model_name must be provided")
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {self.SUPPORTED_MODELS}")
        
        if max_workers is None:
            raise ValueError("max_workers must be provided")
        if timeout is None:
            raise ValueError("timeout must be provided")
        if max_tokens is None:
            raise ValueError("max_tokens must be provided")
        
        # Configure LiteLLM
        litellm.set_verbose = False
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        self.model_name = model_name
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_tokens = max_tokens
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Make a single LLM call with retry logic for rate limits"""
        max_tokens = min(4000, self.max_tokens * 4) if "gpt-5" in self.model_name.lower() else self.max_tokens
        
        completion_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 1.0
        }
        
        if "gpt-5" in self.model_name.lower():
            completion_params["reasoning_effort"] = "minimal"
        
        for attempt in range(max_retries):
            try:
                response = litellm.completion(**completion_params)
                
                # Extract content from response
                if hasattr(response, 'choices') and response.choices:
                    choice = response.choices[0]
                    content = getattr(choice.message, 'content', '') or getattr(choice, 'text', '') or getattr(choice, 'content', '') or str(choice)
                else:
                    content = getattr(response, 'content', '') or getattr(response, 'text', '') or str(response)
                
                if not content or content.strip() == "":
                    raise ValueError(f"Empty response from {self.model_name}")
                
                return content
                
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = 'rate' in error_str or '429' in error_str or 'quota' in error_str
                
                if is_rate_limit and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + (attempt * 0.5)
                    print(f"Rate limit hit, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"LLM call failed: {e}")
                    return ""
        
        return ""
    
    
    def generate_posts(self, topic, agents: List, neighbor_posts_per_agent: Optional[List[List[str]]] = None) -> List[str]:
        """Generate posts for agents"""
        prompts = []
        for i, agent in enumerate(agents):
            # Get neighbor posts for this agent
            neighbor_posts = neighbor_posts_per_agent[i] if neighbor_posts_per_agent and i < len(neighbor_posts_per_agent) else None
            p = agent.generate_post_prompt(topic, neighbor_posts)
            prompts.append(p)
        
        responses = self._call_llm_batch(prompts)
        
        # Simple output formatting - throw error for empty responses
        out = []
        for i, r in enumerate(responses):
            agent_id = getattr(agents[i], 'agent_id', i)
            text = r.strip() if r and isinstance(r, str) else ""
            if not text:
                raise ValueError(f"Empty response for agent {agent_id} on topic {topic}. Response: '{r}'")
            out.append(f"Agent {agent_id}: {text}")
        return out
    
    def _call_llm_batch(self, prompts: List[str]) -> List[str]:
        """Call LLM for multiple prompts in parallel with rate limiting"""
        if not prompts:
            return []
        
        results = [""] * len(prompts)
        
        def _one(i: int, p: str) -> None:
            try:
                results[i] = self._call_llm(p)
            except Exception as e:
                print(f"LLM call {i} failed: {e}")
                results[i] = ""
        
        # Reduce workers to avoid rate limits - use smaller batches
        # Rate limits are often per-minute, so we want to avoid too many parallel requests
        workers = min(self.max_workers, max(1, len(prompts)), 20)
        
        # Process in smaller batches to avoid overwhelming the API
        batch_size = workers * 2
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            
            with _f.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_one, batch_start + i, p) for i, p in enumerate(batch_prompts)]
                _f.wait(futures, timeout=300)
            
            # Small delay between batches to avoid rate limits
            if batch_end < len(prompts):
                time.sleep(0.5)
        
        return results
    
    def _parse_opinion_response(self, response: str) -> float:
        """Parse LLM response to extract opinion value"""
        text = (response or "").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        
        # Try last line first
        if lines and re.match(r'^-?\d+(?:\.\d+)?$', lines[-1]):
            return max(-1.0, min(1.0, float(lines[-1])))
        
        # Fallback: extract all numbers and use last one
        numbers = re.findall(r'-?\d+\.?\d*', text.replace(' ', ''))
        if not numbers:
            raise ValueError(f"No number found in response: '{response}'")
        
        return max(-1.0, min(1.0, float(numbers[-1])))
    
    def rate_posts_pairwise(self, posts: List[str], topic: str, agents: List, adjacency) -> tuple:
        """Each agent rates neighbor posts. Returns (R, per_agent)"""
        n = len(agents)
        try:
            A = np.array(adjacency)
        except Exception:
            A = np.ones((n, n)) - np.eye(n)
        
        pair_indices = []
        prompts = []
        for i in range(n):
            neighbors = np.where(A[i] == 1)[0]
            for j in neighbors:
                prompts.append(agents[i].interpret_post_prompt(posts[j], topic))
                pair_indices.append((i, j))
        
        texts = self._call_llm_batch(prompts)
        R = np.full((n, n), np.nan, dtype=float)
        per_agent = [[] for _ in range(n)]
        
        for (i, j), t in zip(pair_indices, texts):
            try:
                val = self._parse_opinion_response(t)
            except Exception:
                try:
                    val = float(getattr(agents[j], 'current_opinion', 0.0))
                except Exception:
                    val = 0.0
            
            v = max(-1.0, min(1.0, float(val)))
            R[i, j] = v
            per_agent[i].append((j, v))
        
        return R, per_agent
