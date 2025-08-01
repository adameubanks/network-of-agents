"""
Opinion generator for creating initial agent opinions using LLMs.
"""

from typing import List, Dict, Any, Optional
from .litellm_client import LiteLLMClient


class OpinionGenerator:
    """
    Generates initial opinions for agents using LLM integration.
    """
    
    def __init__(self, llm_client: LiteLLMClient):
        """
        Initialize the opinion generator.
        
        Args:
            llm_client: LiteLLM client for opinion generation
        """
        self.llm_client = llm_client
    
    def generate_opinion_vector(self, topics: List[str], persona: str) -> List[float]:
        """
        Generate an opinion vector for a given persona and topics.
        
        Args:
            topics: List of topics to generate opinions for
            persona: Description of the agent persona
            
        Returns:
            List of opinion values (0-1) for each topic
        """
        return self.llm_client.generate_opinion_vector(topics, persona)
    
    def generate_diverse_opinions(self, topics: List[str], n_agents: int) -> List[List[float]]:
        """
        Generate diverse opinion vectors for multiple agents.
        
        Args:
            topics: List of topics to generate opinions for
            n_agents: Number of agents to generate opinions for
            
        Returns:
            List of opinion vectors for each agent
        """
        personas = self._generate_diverse_personas(n_agents)
        opinions = []
        
        for persona in personas:
            opinion_vector = self.generate_opinion_vector(topics, persona)
            opinions.append(opinion_vector)
        
        return opinions
    
    def _generate_diverse_personas(self, n_agents: int) -> List[str]:
        """
        Generate diverse personas for agents.
        
        Args:
            n_agents: Number of agents
            
        Returns:
            List of persona descriptions
        """
        base_personas = [
            "A conservative individual who values tradition and stability",
            "A liberal individual who values progress and change",
            "A moderate individual who seeks balance and compromise",
            "A libertarian individual who values individual freedom",
            "A progressive individual who advocates for social justice",
            "A traditionalist individual who respects established norms",
            "An activist individual who fights for causes they believe in",
            "A pragmatist individual who focuses on practical solutions",
            "An idealist individual who believes in perfect solutions",
            "A realist individual who accepts the world as it is",
            "A centrist individual who avoids extremes",
            "A populist individual who distrusts elites",
            "An intellectual individual who values evidence and reason",
            "A spiritual individual who prioritizes moral values",
            "A business-oriented individual who focuses on economic growth"
        ]
        
        # Repeat personas if needed
        while len(base_personas) < n_agents:
            base_personas.extend(base_personas[:n_agents - len(base_personas)])
        
        return base_personas[:n_agents]
    
    def generate_polarized_opinions(self, topics: List[str], n_agents: int, 
                                  polarization_strength: float = 0.8) -> List[List[float]]:
        """
        Generate polarized opinion vectors.
        
        Args:
            topics: List of topics to generate opinions for
            n_agents: Number of agents
            polarization_strength: Strength of polarization (0-1)
            
        Returns:
            List of polarized opinion vectors
        """
        # Generate two extreme personas
        conservative_persona = "A very conservative individual with traditional values and strong opposition to change"
        liberal_persona = "A very liberal individual who strongly supports progressive change and social justice"
        
        # Generate opinions for each group
        n_conservative = n_agents // 2
        n_liberal = n_agents - n_conservative
        
        conservative_opinions = []
        for _ in range(n_conservative):
            opinion_vector = self.generate_opinion_vector(topics, conservative_persona)
            # Apply polarization strength
            polarized_opinions = [max(0, min(1, op * (1 - polarization_strength))) for op in opinion_vector]
            conservative_opinions.append(polarized_opinions)
        
        liberal_opinions = []
        for _ in range(n_liberal):
            opinion_vector = self.generate_opinion_vector(topics, liberal_persona)
            # Apply polarization strength
            polarized_opinions = [max(0, min(1, op * (1 + polarization_strength))) for op in opinion_vector]
            liberal_opinions.append(polarized_opinions)
        
        return conservative_opinions + liberal_opinions
    
    def generate_biased_opinions(self, topics: List[str], n_agents: int, 
                               bias_direction: str = "conservative") -> List[List[float]]:
        """
        Generate biased opinion vectors.
        
        Args:
            topics: List of topics to generate opinions for
            n_agents: Number of agents
            bias_direction: Direction of bias ("conservative", "liberal", "moderate")
            
        Returns:
            List of biased opinion vectors
        """
        if bias_direction == "conservative":
            base_persona = "A conservative individual with traditional values"
        elif bias_direction == "liberal":
            base_persona = "A liberal individual who supports progressive change"
        else:
            base_persona = "A moderate individual who seeks balance"
        
        opinions = []
        for _ in range(n_agents):
            opinion_vector = self.generate_opinion_vector(topics, base_persona)
            opinions.append(opinion_vector)
        
        return opinions 