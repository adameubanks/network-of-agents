"""
Simplified LLM Agent class representing individual agents in the social network.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .litellm_client import LiteLLMClient


class LLMAgent:
    """
    Represents an individual LLM agent in the social network.
    """
    
    def __init__(self, 
                 agent_id: int,
                 initial_opinions: Optional[List[float]] = None,
                 topics: Optional[List[str]] = None,
                 llm_client: Optional[LiteLLMClient] = None):
        """
        Initialize an LLM agent.
        
        Args:
            agent_id: Unique identifier for the agent
            initial_opinions: Initial opinion vector (if None, will be generated using LLM)
            topics: List of topics for opinions
            llm_client: LiteLLM client for opinion interpretation/generation
        """
        self.agent_id = agent_id
        self.topics = topics or []
        self.llm_client = llm_client
        
        if initial_opinions is not None:
            self.opinions = np.array(initial_opinions, dtype=float)
        else:
            # Initialize with empty array - will be set later
            self.opinions = np.array([], dtype=float)
        
        self.opinion_history = []
    
    def initialize_opinions(self, topics: List[str], llm_client: LiteLLMClient) -> np.ndarray:
        """
        Initialize opinions using LLM if not already set.
        
        Args:
            topics: List of topics for opinions
            llm_client: LiteLLM client for opinion generation
            
        Returns:
            Initial opinion vector
        """
        if len(self.opinions) == 0:
            self.topics = topics
            self.llm_client = llm_client
            
            # Generate initial opinions using LLM
            if self.llm_client is not None:
                # Generate a random text sample to interpret
                sample_text = f"Agent {self.agent_id} initial thoughts on {topics[0]}: "
                sample_text += "I have mixed feelings about this topic."
                
                # Use LLM to interpret the text and generate opinion
                self.opinions = self.llm_client.interpret_text_to_opinions(sample_text, topics)
            else:
                # Fallback to random if no LLM client
                self.opinions = np.random.random(len(topics)).astype(float)
        
        return self.opinions
    
    def update_opinions(self, new_opinions: np.ndarray):
        """
        Update the agent's opinions.
        
        Args:
            new_opinions: New opinion vector
        """
        # Store current opinions in history
        self.opinion_history.append(self.opinions.copy())
        
        # Update current opinions
        self.opinions = new_opinions.copy()
    
    def get_opinions(self) -> np.ndarray:
        """
        Get current opinion vector.
        
        Returns:
            Current opinion vector
        """
        return self.opinions.copy()
    
    def get_opinion_history(self) -> List[np.ndarray]:
        """
        Get opinion history.
        
        Returns:
            List of opinion vectors over time
        """
        return self.opinion_history.copy()
    
    def get_similarity_to(self, other_agent: 'LLMAgent') -> float:
        """
        Calculate opinion similarity to another agent.
        
        Args:
            other_agent: Another LLM agent
            
        Returns:
            Similarity score (0-1)
        """
        if len(self.opinions) != len(other_agent.opinions):
            raise ValueError("Agents must have the same number of opinions")
        
        # Calculate L1 norm of difference
        diff = np.linalg.norm(self.opinions - other_agent.opinions, ord=1)
        
        # Convert to similarity (0 = identical, 1 = completely different)
        similarity = 1 - (diff / len(self.opinions))
        
        return max(0.0, min(1.0, similarity))
    
    def generate_content(self) -> str:
        """
        Generate content reflecting current opinions.
        
        Returns:
            Generated text content
        """
        if self.llm_client is None:
            return f"Agent {self.agent_id} opinions: {self.opinions}"
        
        return self.llm_client.generate_text_from_opinions(
            self.topics, 
            self.opinions  # Already a list, no need for .tolist()
        )
    
    def interpret_content(self, text: str) -> np.ndarray:
        """
        Interpret text content to update opinions.
        
        Args:
            text: Text content to interpret
            
        Returns:
            Updated opinion vector
        """
        if self.llm_client is None:
            return self.opinions
        
        # Use LLM to interpret the text and get opinion values
        new_opinions = self.llm_client.interpret_text_to_opinions(text, self.topics)
        
        # Store the interpreted opinions in history
        self.opinion_history.append(self.opinions.copy())
        
        # Update current opinions
        self.opinions = new_opinions
        
        return new_opinions
    
    def get_degree(self, adjacency_matrix: np.ndarray) -> int:
        """
        Get the degree of this agent in the network.
        
        Args:
            adjacency_matrix: Network adjacency matrix
            
        Returns:
            Agent's degree (number of connections)
        """
        if self.agent_id >= len(adjacency_matrix):
            return 0
        
        return int(np.sum(adjacency_matrix[self.agent_id, :]))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent to dictionary for serialization.
        
        Returns:
            Dictionary representation of agent
        """
        return {
            'agent_id': self.agent_id,
            'topics': self.topics,
            'opinions': self.opinions.tolist(),
            'opinion_history': [op.tolist() for op in self.opinion_history]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMAgent':
        """
        Create agent from dictionary.
        
        Args:
            data: Dictionary representation of agent
            
        Returns:
            LLMAgent instance
        """
        agent = cls(
            agent_id=data['agent_id'],
            initial_opinions=data.get('opinions'),
            topics=data.get('topics', [])
        )
        
        # Restore opinion history
        agent.opinion_history = [np.array(op) for op in data.get('opinion_history', [])]
        
        return agent 