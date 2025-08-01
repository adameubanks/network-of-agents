"""
Persona generator for creating diverse agent personas.
"""

from typing import List, Dict, Any, Optional
import random


class PersonaGenerator:
    """
    Generates diverse personas for LLM agents.
    """
    
    def __init__(self):
        """Initialize the persona generator."""
        self.base_personas = self._initialize_base_personas()
    
    def _initialize_base_personas(self) -> Dict[str, List[str]]:
        """Initialize base persona templates."""
        return {
            'political': [
                "A conservative individual who values tradition and stability",
                "A liberal individual who values progress and change",
                "A moderate individual who seeks balance and compromise",
                "A libertarian individual who values individual freedom",
                "A progressive individual who advocates for social justice",
                "A traditionalist individual who respects established norms",
                "A populist individual who distrusts elites",
                "A centrist individual who avoids extremes"
            ],
            'professional': [
                "A business executive focused on economic growth and efficiency",
                "A healthcare worker who prioritizes public health and safety",
                "An educator who values knowledge and critical thinking",
                "A scientist who relies on evidence and research",
                "A lawyer who emphasizes justice and legal precedent",
                "A journalist who values truth and transparency",
                "A social worker who advocates for vulnerable populations",
                "A military veteran who values discipline and national security"
            ],
            'demographic': [
                "A young urban professional in their 20s",
                "A middle-aged suburban parent in their 40s",
                "A senior citizen who has seen many changes over decades",
                "A college student exploring different viewpoints",
                "A rural resident who values community and tradition",
                "An immigrant who brings diverse cultural perspectives",
                "A religious person who grounds decisions in faith",
                "A secular individual who relies on reason and evidence"
            ],
            'personality': [
                "An optimist who believes in positive change",
                "A pessimist who is cautious about change",
                "An idealist who believes in perfect solutions",
                "A pragmatist who focuses on practical solutions",
                "An activist who fights for causes they believe in",
                "A quiet observer who prefers to listen and learn",
                "An intellectual who values deep analysis",
                "An emotional person who follows their heart"
            ]
        }
    
    def generate_diverse_personas(self, n_agents: int) -> List[str]:
        """
        Generate diverse personas for a given number of agents.
        
        Args:
            n_agents: Number of agents to generate personas for
            
        Returns:
            List of persona descriptions
        """
        personas = []
        
        # Generate personas by combining different categories
        for i in range(n_agents):
            persona = self._generate_single_persona()
            personas.append(persona)
        
        return personas
    
    def _generate_single_persona(self) -> str:
        """
        Generate a single persona by combining different aspects.
        
        Returns:
            Persona description
        """
        # Select one aspect from each category
        political = random.choice(self.base_personas['political'])
        professional = random.choice(self.base_personas['professional'])
        demographic = random.choice(self.base_personas['demographic'])
        personality = random.choice(self.base_personas['personality'])
        
        # Combine aspects into a coherent persona
        persona = f"{demographic}, {professional}, {political}, {personality}."
        
        return persona
    
    def generate_polarized_personas(self, n_agents: int) -> List[str]:
        """
        Generate polarized personas (conservative vs liberal).
        
        Args:
            n_agents: Number of agents to generate personas for
            
        Returns:
            List of polarized persona descriptions
        """
        personas = []
        
        # Split agents into two groups
        n_conservative = n_agents // 2
        n_liberal = n_agents - n_conservative
        
        # Generate conservative personas
        conservative_templates = [
            "A conservative individual who values tradition and stability",
            "A traditionalist individual who respects established norms",
            "A libertarian individual who values individual freedom",
            "A religious person who grounds decisions in faith",
            "A rural resident who values community and tradition"
        ]
        
        for _ in range(n_conservative):
            base = random.choice(conservative_templates)
            professional = random.choice(self.base_personas['professional'])
            demographic = random.choice(self.base_personas['demographic'])
            personality = random.choice(self.base_personas['personality'])
            
            persona = f"{demographic}, {professional}, {base}, {personality}."
            personas.append(persona)
        
        # Generate liberal personas
        liberal_templates = [
            "A liberal individual who values progress and change",
            "A progressive individual who advocates for social justice",
            "A secular individual who relies on reason and evidence",
            "A young urban professional in their 20s",
            "An activist who fights for causes they believe in"
        ]
        
        for _ in range(n_liberal):
            base = random.choice(liberal_templates)
            professional = random.choice(self.base_personas['professional'])
            demographic = random.choice(self.base_personas['demographic'])
            personality = random.choice(self.base_personas['personality'])
            
            persona = f"{demographic}, {professional}, {base}, {personality}."
            personas.append(persona)
        
        # Shuffle the personas
        random.shuffle(personas)
        
        return personas
    
    def generate_biased_personas(self, n_agents: int, bias_direction: str = "conservative") -> List[str]:
        """
        Generate personas with a specific bias direction.
        
        Args:
            n_agents: Number of agents to generate personas for
            bias_direction: Direction of bias ("conservative", "liberal", "moderate")
            
        Returns:
            List of biased persona descriptions
        """
        if bias_direction == "conservative":
            base_templates = [
                "A conservative individual who values tradition and stability",
                "A traditionalist individual who respects established norms",
                "A libertarian individual who values individual freedom",
                "A religious person who grounds decisions in faith",
                "A rural resident who values community and tradition"
            ]
        elif bias_direction == "liberal":
            base_templates = [
                "A liberal individual who values progress and change",
                "A progressive individual who advocates for social justice",
                "A secular individual who relies on reason and evidence",
                "A young urban professional in their 20s",
                "An activist who fights for causes they believe in"
            ]
        else:  # moderate
            base_templates = [
                "A moderate individual who seeks balance and compromise",
                "A centrist individual who avoids extremes",
                "A pragmatist individual who focuses on practical solutions",
                "A middle-aged suburban parent in their 40s",
                "A quiet observer who prefers to listen and learn"
            ]
        
        personas = []
        for _ in range(n_agents):
            base = random.choice(base_templates)
            professional = random.choice(self.base_personas['professional'])
            demographic = random.choice(self.base_personas['demographic'])
            personality = random.choice(self.base_personas['personality'])
            
            persona = f"{demographic}, {professional}, {base}, {personality}."
            personas.append(persona)
        
        return personas
    
    def generate_custom_personas(self, persona_templates: List[str], n_agents: int) -> List[str]:
        """
        Generate personas from custom templates.
        
        Args:
            persona_templates: List of custom persona templates
            n_agents: Number of agents to generate personas for
            
        Returns:
            List of custom persona descriptions
        """
        personas = []
        
        for _ in range(n_agents):
            if persona_templates:
                persona = random.choice(persona_templates)
            else:
                persona = self._generate_single_persona()
            
            personas.append(persona)
        
        return personas
    
    def add_demographic_diversity(self, personas: List[str]) -> List[str]:
        """
        Add demographic diversity to existing personas.
        
        Args:
            personas: List of existing personas
            
        Returns:
            List of personas with added demographic diversity
        """
        demographics = self.base_personas['demographic']
        enhanced_personas = []
        
        for persona in personas:
            # Add demographic if not already present
            if not any(demo in persona for demo in demographics):
                demographic = random.choice(demographics)
                enhanced_persona = f"{demographic}, {persona}"
            else:
                enhanced_persona = persona
            
            enhanced_personas.append(enhanced_persona)
        
        return enhanced_personas
    
    def validate_persona_diversity(self, personas: List[str]) -> Dict[str, Any]:
        """
        Validate the diversity of generated personas.
        
        Args:
            personas: List of personas to validate
            
        Returns:
            Dictionary containing diversity metrics
        """
        # Count occurrences of different aspects
        political_aspects = {aspect: 0 for aspect in self.base_personas['political']}
        professional_aspects = {aspect: 0 for aspect in self.base_personas['professional']}
        demographic_aspects = {aspect: 0 for aspect in self.base_personas['demographic']}
        personality_aspects = {aspect: 0 for aspect in self.base_personas['personality']}
        
        for persona in personas:
            # Count political aspects
            for aspect in political_aspects:
                if aspect in persona:
                    political_aspects[aspect] += 1
            
            # Count professional aspects
            for aspect in professional_aspects:
                if aspect in persona:
                    professional_aspects[aspect] += 1
            
            # Count demographic aspects
            for aspect in demographic_aspects:
                if aspect in persona:
                    demographic_aspects[aspect] += 1
            
            # Count personality aspects
            for aspect in personality_aspects:
                if aspect in persona:
                    personality_aspects[aspect] += 1
        
        # Calculate diversity metrics
        total_agents = len(personas)
        
        political_diversity = len([count for count in political_aspects.values() if count > 0]) / len(political_aspects)
        professional_diversity = len([count for count in professional_aspects.values() if count > 0]) / len(professional_aspects)
        demographic_diversity = len([count for count in demographic_aspects.values() if count > 0]) / len(demographic_aspects)
        personality_diversity = len([count for count in personality_aspects.values() if count > 0]) / len(personality_aspects)
        
        return {
            'total_agents': total_agents,
            'political_diversity': political_diversity,
            'professional_diversity': professional_diversity,
            'demographic_diversity': demographic_diversity,
            'personality_diversity': personality_diversity,
            'overall_diversity': (political_diversity + professional_diversity + demographic_diversity + personality_diversity) / 4,
            'political_distribution': political_aspects,
            'professional_distribution': professional_aspects,
            'demographic_distribution': demographic_aspects,
            'personality_distribution': personality_aspects
        } 