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

load_dotenv()
logger = logging.getLogger(__name__)

class LLMClient:
    """Client for generating and rating posts."""
    
    # Supported models
    SUPPORTED_MODELS = ["gpt-5-nano", "gpt-5-mini", "grok-mini"]
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-5-mini"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("No API key provided or found in environment")
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {self.SUPPORTED_MODELS}")
        
        # Reduce LiteLLM logging verbosity
        litellm.set_verbose = False
        litellm.drop_params = True
        litellm.suppress_debug_info = True
        
        # Set logging levels to reduce verbosity
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        self.model_name = model_name
        self.max_workers = 5
    
    def _call_llm(self, prompt: str) -> str:
        """Make a single LLM call and return the response text"""
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return self._extract_text(response) or ""
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return ""
    
    def _extract_text(self, response) -> Optional[str]:
        """Extract text from LLM response"""
        try:
            if hasattr(response, 'choices') and len(response.choices) > 0:
                message = getattr(response.choices[0], 'message', None)
                if message and hasattr(message, 'content'):
                    return message.content
        except Exception:
            pass
        return None
    
    def generate_posts(self, topic: str, agents: List, neighbor_posts_per_agent: Optional[List[List[str]]] = None) -> List[str]:
        """Generate posts for agents"""
        prompts = []
        for i, agent in enumerate(agents):
            p = agent.generate_post_prompt(topic)
            if neighbor_posts_per_agent and i < len(neighbor_posts_per_agent) and neighbor_posts_per_agent[i]:
                neighbors = neighbor_posts_per_agent[i][-6:]  # Limit to 6 neighbors
                trimmed = []
                for txt in neighbors:
                    t = (txt or "").strip()
                    if len(t) > 220:
                        t = t[:219] + "…"
                    trimmed.append(t)
                combined = "\n".join(trimmed)
                p = f"{p}\nHere are recent posts from connected agents:\n{combined}"
            prompts.append(p)
        
        responses = self._call_llm_batch(prompts)
        
        # Prefix each generated post with agent ID
        out = []
        for i, r in enumerate(responses):
            agent_id = getattr(agents[i], 'agent_id', i)
            text = r.strip() if r and isinstance(r, str) else ""
            if not text:
                text = f"No content for agent {agent_id} on {topic}"
            out.append(f"Agent {agent_id}: {text}")
        return out
    
    def _call_llm_batch(self, prompts: List[str]) -> List[str]:
        """Call LLM for multiple prompts in parallel"""
        if not prompts:
            return []
        
        results = [""] * len(prompts)
        
        def _one(i: int, p: str) -> None:
            try:
                results[i] = self._call_llm(p)
            except Exception as e:
                logger.warning(f"LLM call {i} failed: {e}")
                results[i] = ""
        
        workers = min(self.max_workers, max(1, len(prompts)))
        with _f.ThreadPoolExecutor(max_workers=workers) as ex:
            for i, p in enumerate(prompts):
                ex.submit(_one, i, p)
            ex.shutdown(wait=True)
        return results
    
    def _parse_opinion_response(self, response: str) -> float:
        """Parse LLM response to extract opinion value"""
        text = (response or "").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        
        # Look for number in last non-empty line
        num_pattern = re.compile(r'^-?\d+(?:\.\d+)?$')
        if lines:
            last = lines[-1]
            if num_pattern.match(last):
                try:
                    val = float(last)
                    return max(-1.0, min(1.0, val))
                except Exception:
                    pass
        
        # Fallback: extract all numbers and pick the last one
        numbers = re.findall(r'-?\d+\.?\d*', text.replace(' ', ''))
        if not numbers:
            raise ValueError(f"No number found in response: '{response}'")
        
        try:
            opinion = float(numbers[-1])
            return max(-1.0, min(1.0, opinion))
        except ValueError:
            logger.error(f"Could not parse number from response: '{response}'")
            raise ValueError(f"Could not parse number from response: '{response}'")
    
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
