"""
Agent class for post generation and interpretation.
"""

import numpy as np
import random
from typing import List, Optional


class Agent:
    """
    Represents an individual agent in the social network.
    """
    
    def __init__(self, agent_id: int, random_seed: Optional[int] = None):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            random_seed: Random seed for reproducible opinion generation
        """
        self.agent_id = agent_id
        
        # Initialize with random opinion
        self.current_opinion = np.random.uniform(-1, 1)
    
    def generate_post_prompt(self, topic: str) -> str:
        """
        Generate agent-specific prompt for post generation.
        
        Args:
            topic: Topic to generate post about
            
        Returns:
            Agent-specific prompt
        """
        # Support either a single topic or a vs-topic formatted as "Topic A vs Topic B"
        s = topic.strip(); low = s.lower(); sep = " vs "
        i = low.find(sep)
        if i != -1:
            a = s[:i].strip(); b = s[i + len(sep):].strip()
            prompt = f"""
Write a concise argument about {a} vs {b} (3-6 sentences).
Your current opinion: {self.current_opinion:.3f} (-1=strongly favors {a} over {b}, 1=strongly favors {b} over {a}).

Express your stance along this axis with reasoning and nuance. Use prose only, no numeric score. Keep it under 800 characters.
"""
            return prompt

        prompt = f"""
Generate a comprehensive statement about {topic} (3-6 sentences, multiple paragraphs if needed).
Your current opinion intensity: {self.current_opinion:.3f} (-1=strongly oppose, 1=strongly support)

Express your opinion with detailed reasoning, context, and nuance. Examples of opinion levels:
- Opinion -1.000: "I strongly oppose this and believe it's completely wrong. The negative consequences far outweigh any potential benefits, and I cannot support it under any circumstances. The evidence clearly shows this causes more harm than good, and I stand firmly against it."
- Opinion -0.750: "I'm strongly against this but acknowledge some complexity. While I see some valid arguments on the other side, the overall impact is too negative to support. There are legitimate concerns that need to be addressed, though I understand why some people might disagree."
- Opinion -0.500: "I'm mostly against this but see some valid points. There are legitimate concerns that need to be addressed, though I understand some people might disagree. While I have reservations, I can see why others might feel differently."
- Opinion -0.250: "I'm somewhat against this but open to discussion. I have reservations but could be persuaded with better arguments or evidence. I'm not completely closed off to the idea, but I need more convincing."
- Opinion 0.000: "I'm completely neutral - I don't have strong feelings either way. I can see both sides and don't feel strongly enough to take a position. There are valid arguments on both sides, and I'm not convinced either way."
- Opinion 0.250: "I'm somewhat supportive but have reservations. I generally support this but have some concerns that need to be addressed. While I think it's mostly good, there are issues that should be considered."
- Opinion 0.500: "I mostly support this but have some concerns. While I think it's generally good, there are issues that should be considered. I'm supportive but not without reservations."
- Opinion 0.750: "I'm strongly supportive but acknowledge some issues. I think this is mostly positive, though there are some problems that need attention. The benefits clearly outweigh the drawbacks."
- Opinion 1.000: "I fully support this and believe it's completely right. The benefits are clear and I cannot see any significant drawbacks. This is exactly what we need, and I'm fully behind it."

Your opinion is {self.current_opinion:.3f}, so your post should reflect this precise level of support/opposition for {topic}.
Include detailed reasoning, personal context, specific examples, qualifications, and nuanced thoughts.
Feel free to use multiple paragraphs if needed to fully express your position.
Do not include any numeric score for your opinion; express your stance in prose.
Keep it under 800 characters.
"""
        return prompt


    

    
    def update_opinion(self, new_opinion: float):
        """
        Update the agent's opinion.
        
        Args:
            new_opinion: New opinion value (-1 to 1)
        """
        self.current_opinion = new_opinion
    
    def get_opinion(self) -> float:
        """
        Get current opinion value.
        
        Returns:
            Current opinion value (-1 to 1)
        """
        return self.current_opinion 

    def interpret_post_prompt(self, post: str, topic: str) -> str:
        """
        Generate agent-specific interpretation prompt for a single post.
        
        Args:
            post: Single post to interpret
            topic: Topic the post is about
            
        Returns:
            Agent-specific interpretation prompt
        """
        # Support either a single topic or a vs-topic formatted as "Topic A vs Topic B"
        s = topic.strip(); low = s.lower(); sep = " vs "
        i = low.find(sep)
        if i != -1:
            a = s[:i].strip(); b = s[i + len(sep):].strip()
            prompt = f"""
Rate the post on the axis {a} (-1) ↔ {b} (1).
-1.000 = strongly favors {a} over {b}
 1.000 = strongly favors {b} over {a}

Post: "{post}"
Respond with ONLY one number in [-1.000, 1.000] on its own line. Use 0.000 if neutral.
"""
            return prompt
        prompt = f"""
Rate the post about "{topic}" on a -1 to 1 scale.
-1.000 = strongly oppose, 1.000 = strongly support

Post: "{post}"
Respond with ONLY one number in [-1.000, 1.000] on its own line. Use 0.000 if neutral.
"""
        return prompt 

 