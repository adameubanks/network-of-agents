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

# Suppress verbose logging
os.environ["LITELLM_LOG"] = "ERROR"
logging.basicConfig(level=logging.WARNING, force=True)
for logger_name in ["litellm", "httpx", "openai", "urllib3", "requests"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

class LLMClient:
    """Client for generating and rating posts."""
    
    # Supported models
    SUPPORTED_MODELS = ["gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-5-pro", "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"]
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini", 
                 max_workers: int = 100, timeout: int = 15, max_tokens: int = 150):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("No API key provided or found in environment")
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {self.SUPPORTED_MODELS}")
        
        # Configure LiteLLM
        litellm.set_verbose = False
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        self.model_name = model_name
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_tokens = max_tokens
    
    def _call_llm(self, prompt: str, max_retries: int = 2) -> str:
        """Make a single LLM call and return the response text"""
        try:
            # Adjust max_tokens for GPT-5 models to ensure text output
            max_tokens = self.max_tokens
            if "gpt-5" in self.model_name.lower():
                max_tokens = min(4000, self.max_tokens * 4)  # Give much more room for reasoning + text
            
            # Configure parameters based on model type
            completion_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
            
            # GPT-5 models only support temperature=1 and benefit from minimal reasoning
            if "gpt-5" in self.model_name.lower():
                completion_params["temperature"] = 1.0
                completion_params["reasoning_effort"] = "minimal"  # Reduce reasoning to get more text output
            else:
                completion_params["temperature"] = 0.7
            
            response = litellm.completion(**completion_params)
            
            
            # Try different ways to extract content based on response structure
            content = ""
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content or ""
                elif hasattr(choice, 'text'):
                    content = choice.text or ""
                elif hasattr(choice, 'content'):
                    content = choice.content or ""
                else:
                    # Try to get content directly from choice
                    content = str(choice) if choice else ""
            elif hasattr(response, 'content'):
                content = response.content or ""
            elif hasattr(response, 'text'):
                content = response.text or ""
            else:
                # Last resort: convert response to string
                content = str(response)
            
            # Throw error for empty responses instead of fallback
            if not content or content.strip() == "":
                choice = response.choices[0] if hasattr(response, 'choices') and response.choices else None
                finish_reason = getattr(choice, 'finish_reason', 'N/A') if choice else 'N/A'
                
                raise ValueError(f"Empty response from {self.model_name}. Finish reason: {finish_reason}. Response: {response}")
            
            return content
        except Exception as e:
            print(f"LLM call failed: {e}")
            return ""
    
    
    def generate_posts(self, topic, agents: List, neighbor_posts_per_agent: Optional[List[List[str]]] = None) -> List[str]:
        """Generate posts for agents"""
        print(f"Generating posts for {len(agents)} agents on topic: {topic}")
        
        prompts = []
        for i, agent in enumerate(agents):
            p = agent.generate_post_prompt(topic)
            prompts.append(p)
        
        print(f"Generated {len(prompts)} prompts")
        responses = self._call_llm_batch(prompts)
        
        # Simple output formatting - throw error for empty responses
        out = []
        for i, r in enumerate(responses):
            agent_id = getattr(agents[i], 'agent_id', i)
            text = r.strip() if r and isinstance(r, str) else ""
            if not text:
                raise ValueError(f"Empty response for agent {agent_id} on topic {topic}. Response: '{r}'")
            out.append(f"Agent {agent_id}: {text}")
        
        print(f"Generated {len(out)} posts")
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
                print(f"LLM call {i} failed: {e}")
                results[i] = ""
        
        # Use parallel processing but with reasonable limits
        workers = min(self.max_workers, max(1, len(prompts)))
        
        with _f.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_one, i, p) for i, p in enumerate(prompts)]
            _f.wait(futures, timeout=300)
        
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
