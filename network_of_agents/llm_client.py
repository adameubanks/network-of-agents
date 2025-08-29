"""
Streamlined LLM client for post generation and interpretation.
"""

import os
from typing import List, Optional, Dict
import numpy as np
import litellm
import concurrent.futures as _f
from dotenv import load_dotenv
import logging
import time

load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    """Minimal client for generating and rating posts."""
    
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
        
        self.model_name = model_name
        self.max_workers = 50
        self.timeout = 60
    
    def _is_gpt5_model(self) -> bool:
        try:
            return str(self.model_name or "").startswith("gpt-5")
        except Exception:
            return False

    
    
    def _build_llm_params(self) -> Dict:
        """Build parameter dict compatible with model family (gpt-5 vs others)."""
        params: Dict = {}
        if self._is_gpt5_model():
            params['reasoning_effort'] = 'low'
            params['response_format'] = {'type': 'text'}
        return params

    def _completion_with_retry_text(self, prompt: str) -> str:
        """Make a single completion call and return extracted text."""
        llm_params = self._build_llm_params()
        response = litellm.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **llm_params
        )
        text = self._extract_text_from_model_response(response) or ""
        return text

    def _build_extra_body(self) -> Dict:
        return {'response_format': {'type': 'text'}} if self._is_gpt5_model() else {}

    
    
    def generate_posts(self, topic: str, agents: List, neighbor_posts_per_agent: Optional[List[List[str]]] = None) -> List[str]:
        # Limit neighbor context and truncate to avoid exceeding provider limits
        MAX_NEIGHBORS = 6
        MAX_NEIGHBOR_CHARS = 220
        prompts: List[str] = []
        for i, agent in enumerate(agents):
            p = agent.generate_post_prompt(topic)
            if neighbor_posts_per_agent and i < len(neighbor_posts_per_agent) and neighbor_posts_per_agent[i]:
                # Take the most recent up to MAX_NEIGHBORS
                neighbors = neighbor_posts_per_agent[i][-MAX_NEIGHBORS:]
                trimmed = []
                for txt in neighbors:
                    t = (txt or "").strip()
                    if len(t) > MAX_NEIGHBOR_CHARS:
                        t = t[:MAX_NEIGHBOR_CHARS - 1] + "…"
                    trimmed.append(t)
                combined = "\n".join(trimmed)
                p = (
                    f"{p}\nHere are recent posts from connected agents. Write a conversational, social-post reply:"
                    f"\n- Optionally respond to 1-2 agents by name (e.g., Agent 3)."
                    f"\n- You may quote/paraphrase briefly and agree, disagree, or ask a question."
                    f"\n- Keep it 1-3 sentences, ≤320 characters, first person."
                    f"\n\n{combined}"
                )
            prompts.append(p)
        responses = self._call_llm(prompts)
        # Prefix each generated post with the agent name for downstream clarity
        out: List[str] = []
        for i, r in enumerate(responses):
            text = r.strip() if r and isinstance(r, str) else ""
            if not text:
                text = f"No content for agent {i} on {topic}"
            out.append(f"Agent {i}: {text}")
        return out
    
    # Neighbor-by-neighbor rating removed for simplicity
    

    
    def _generate_single_text(self, prompt: str, temperature: float = None) -> str:
        """
        Generate single text using the LLM.
        
        Args:
            prompt: Input prompt
            max_completion_tokens: Maximum completion tokens to generate
            temperature: Sampling temperature (uses class default if None)
            
        Returns:
            Generated text
        """
        return self._completion_with_retry_text(prompt)
    

    
    def _call_llm(self, prompts: List[str]) -> List[str]:
        """Call LLM per prompt; return extracted texts."""
        return self._generate_texts(prompts)

    def _generate_texts(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        results: List[str] = [""] * len(prompts)
        def _one(i: int, p: str) -> None:
            try:
                results[i] = self._completion_with_retry_text(p)
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")
                results[i] = ""
        workers = min(self.max_workers, max(1, len(prompts)))
        with _f.ThreadPoolExecutor(max_workers=workers) as ex:
            for i, p in enumerate(prompts):
                ex.submit(_one, i, p)
            ex.shutdown(wait=True)
        return results
    
    def _parse_opinion_response(self, response: str) -> float:
        """
        Parse the LLM response to extract a single opinion value.
        
        Args:
            response: LLM response text
            
        Returns:
            Opinion value (-1 to 1)
        """
        text = (response or "").strip()
        # Prefer last non-empty line if it is numeric-like
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        import re
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

    def rate_posts_globally(self, posts: List[str], topic: str, agents: List) -> List[float]:
        """Rate each post using its posting agent's interpret prompt (per-agent rater)."""
        prompts = [agents[i].interpret_post_prompt(posts[i], topic) for i in range(len(posts))]
        texts = self._call_llm(prompts)
        ratings: List[float] = []
        for i, t in enumerate(texts):
            try:
                ratings.append(self._parse_opinion_response(t))
            except Exception:
                # Fallback to the posting agent's current opinion to avoid collapse to 0
                try:
                    ratings.append(float(getattr(agents[i], 'current_opinion', 0.0)))
                except Exception:
                    ratings.append(0.0)
        return ratings

    def rate_posts_pairwise(self, posts: List[str], topic: str, agents: List, adjacency) -> tuple:
        """Each agent i rates neighbor j's post. Returns (R, per_agent).

        R[i, j] = rating by agent i of post j in [-1,1] (nan if not rated).
        per_agent[i] = list of (neighbor_index, rating).
        """
        n = len(agents)
        try:
            A = np.array(adjacency)
        except Exception:
            A = np.ones((n, n)) - np.eye(n)
        pair_indices: List[tuple[int, int]] = []
        prompts: List[str] = []
        for i in range(n):
            neighbors = np.where(A[i] == 1)[0]
            for j in neighbors:
                prompts.append(agents[i].interpret_post_prompt(posts[j], topic))
                pair_indices.append((i, j))
        texts = self._call_llm(prompts)
        R = np.full((n, n), np.nan, dtype=float)
        per_agent: List[List[tuple[int, float]]] = [[] for _ in range(n)]
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