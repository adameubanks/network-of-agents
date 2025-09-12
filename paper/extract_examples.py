#!/usr/bin/env python3
"""
Script to extract interesting examples and patterns from the simulation results
for inclusion in the paper.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import re

def load_simulation_data(results_dir: str) -> Dict:
    """Load all simulation data from JSON files."""
    data = {}
    
    # Load DeGroot results
    degroot_dir = Path(results_dir) / "degroot"
    for topic_dir in degroot_dir.iterdir():
        if topic_dir.is_dir() and topic_dir.name != "test_topic":
            for subtopic_dir in topic_dir.iterdir():
                if subtopic_dir.is_dir():
                    for json_file in subtopic_dir.glob("*.json"):
                        if "gpt-5-nano" in json_file.name:
                            topic_name = json_file.stem.split("_50_50_gpt-5-nano")[0]
                            with open(json_file, 'r') as f:
                                data[topic_name] = json.load(f)
    
    return data

def extract_persona_examples(data: Dict) -> Dict:
    """Extract examples showing persona differences (teachers vs non-teachers)."""
    persona_examples = {}
    
    # Look for Pride Flag vs 10 Commandments topics
    pride_topics = [k for k in data.keys() if 'pride' in k.lower() and 'flag' in k.lower()]
    commandment_topics = [k for k in data.keys() if 'commandment' in k.lower()]
    
    for topic in pride_topics + commandment_topics:
        if topic in data:
            posts = []
            # Extract posts from first timestep
            if 'timesteps' in data[topic] and '0' in data[topic]['timesteps']:
                agents = data[topic]['timesteps']['0']['agents']
                for agent in agents[:10]:  # First 10 agents
                    if 'post' in agent:
                        posts.append({
                            'agent_id': agent['agent_id'],
                            'post': agent['post'],
                            'opinion': agent['opinion']
                        })
            persona_examples[topic] = posts
    
    return persona_examples

def extract_bias_examples(data: Dict) -> Dict:
    """Extract examples showing different types of bias."""
    bias_examples = {}
    
    # Topics with different bias patterns
    interesting_topics = [
        'israel_vs_palestine',
        'palestine_vs_israel', 
        'lebron_james_is_the_goat_vs_michael_jordan_is_the_goat',
        'chocolate_ice_cream_vs_vanilla_ice_cream',
        'circles_vs_triangles'
    ]
    
    for topic in interesting_topics:
        if topic in data:
            posts = []
            if 'timesteps' in data[topic] and '0' in data[topic]['timesteps']:
                agents = data[topic]['timesteps']['0']['agents']
                for agent in agents[:5]:  # First 5 agents
                    if 'post' in agent:
                        posts.append({
                            'agent_id': agent['agent_id'],
                            'post': agent['post'],
                            'opinion': agent['opinion']
                        })
            bias_examples[topic] = posts
    
    return bias_examples

def analyze_rating_patterns(data: Dict) -> Dict:
    """Analyze rating patterns to show inconsistencies."""
    rating_analysis = {}
    
    for topic, sim_data in data.items():
        if 'timesteps' in sim_data and '0' in sim_data['timesteps']:
            agents = sim_data['timesteps']['0']['agents']
            rating_examples = []
            
            for agent in agents[:3]:  # First 3 agents
                if 'outgoing_ratings' in agent and 'incoming_ratings' in agent:
                    rating_examples.append({
                        'agent_id': agent['agent_id'],
                        'post': agent['post'],
                        'outgoing_ratings': agent['outgoing_ratings'],
                        'incoming_ratings': agent['incoming_ratings']
                    })
            
            rating_analysis[topic] = rating_examples
    
    return rating_analysis

def find_extreme_opinions(data: Dict) -> Dict:
    """Find examples of extreme opinions and their posts."""
    extreme_examples = {}
    
    for topic, sim_data in data.items():
        if 'final_opinions' in sim_data:
            final_opinions = np.array(sim_data['final_opinions'])
            
            # Find most extreme positive and negative opinions
            max_idx = np.argmax(final_opinions)
            min_idx = np.argmin(final_opinions)
            
            extreme_examples[topic] = {
                'most_positive': {
                    'agent_id': max_idx,
                    'opinion': final_opinions[max_idx]
                },
                'most_negative': {
                    'agent_id': min_idx,
                    'opinion': final_opinions[min_idx]
                }
            }
    
    return extreme_examples

def generate_latex_examples(persona_examples: Dict, bias_examples: Dict, rating_analysis: Dict) -> str:
    """Generate LaTeX code with examples for the paper."""
    
    latex_content = "% Generated examples for algorithmic fidelity paper\n"
    latex_content += "% This file is automatically generated by extract_examples.py\n\n"
    
    # Persona examples
    latex_content += "\\newcommand{\\personaexamples}{\n"
    latex_content += "\\begin{itemize}\n"
    
    for topic, posts in persona_examples.items():
        if posts:
            latex_content += f"\\item \\textbf{{{topic.replace('_', ' ').title()}:}}\n"
            for post in posts[:3]:  # First 3 posts
                # Clean the post text
                clean_post = post['post'].replace('Agent ' + str(post['agent_id']) + ': ', '')
                clean_post = clean_post.replace('&', '\\&').replace('%', '\\%')
                latex_content += f"  \\begin{{quote}}\n"
                latex_content += f"  ``{clean_post}''\n"
                latex_content += f"  \\end{{quote}}\n"
    
    latex_content += "\\end{itemize}\n"
    latex_content += "}\n\n"
    
    # Bias examples
    latex_content += "\\newcommand{\\biasexamples}{\n"
    for topic, posts in bias_examples.items():
        if posts:
            latex_content += f"\\textbf{{{topic.replace('_', ' ').title()}:}}\n"
            for post in posts[:2]:  # First 2 posts
                clean_post = post['post'].replace('Agent ' + str(post['agent_id']) + ': ', '')
                clean_post = clean_post.replace('&', '\\&').replace('%', '\\%')
                latex_content += f"\\begin{{quote}}\n"
                latex_content += f"``{clean_post}''\n"
                latex_content += f"\\end{{quote}}\n"
            latex_content += "\\\\\n"
    latex_content += "}\n\n"
    
    return latex_content

def main():
    """Main function to extract examples and generate LaTeX."""
    
    results_dir = "/home/adam/Projects/IDeA/network-of-agents/results"
    
    print("Loading simulation data...")
    data = load_simulation_data(results_dir)
    print(f"Loaded data for {len(data)} topics")
    
    print("Extracting persona examples...")
    persona_examples = extract_persona_examples(data)
    
    print("Extracting bias examples...")
    bias_examples = extract_bias_examples(data)
    
    print("Analyzing rating patterns...")
    rating_analysis = analyze_rating_patterns(data)
    
    print("Finding extreme opinions...")
    extreme_examples = find_extreme_opinions(data)
    
    print("Generating LaTeX examples...")
    latex_content = generate_latex_examples(persona_examples, bias_examples, rating_analysis)
    
    # Save to file
    with open("/home/adam/Projects/IDeA/network-of-agents/paper/examples.tex", "w") as f:
        f.write(latex_content)
    
    # Print some interesting findings
    print("\n=== INTERESTING FINDINGS ===")
    
    print("\n1. PERSONA DIFFERENCES (Pride Flag vs 10 Commandments):")
    for topic, posts in persona_examples.items():
        if 'pride' in topic.lower():
            print(f"\n{topic}:")
            for post in posts[:3]:
                print(f"  Agent {post['agent_id']}: {post['post']}")
    
    print("\n2. BIAS PATTERNS:")
    for topic, posts in bias_examples.items():
        if posts:
            print(f"\n{topic}:")
            for post in posts[:2]:
                print(f"  Agent {post['agent_id']}: {post['post']}")
    
    print("\n3. EXTREME OPINIONS:")
    for topic, extremes in extreme_examples.items():
        print(f"{topic}: Most positive = {extremes['most_positive']['opinion']:.3f}, Most negative = {extremes['most_negative']['opinion']:.3f}")
    
    print(f"\nExamples saved to examples.tex")

if __name__ == "__main__":
    main()
