"""
Content analyzer for analyzing converged opinions and generating content.
"""

from typing import List, Dict, Any, Optional
from .litellm_client import LiteLLMClient


class ContentAnalyzer:
    """
    Analyzes converged opinions and generates content reflecting those opinions.
    """
    
    def __init__(self, llm_client: LiteLLMClient):
        """
        Initialize the content analyzer.
        
        Args:
            llm_client: LiteLLM client for content generation
        """
        self.llm_client = llm_client
    
    def analyze_converged_opinions(self, topics: List[str], opinion_vector: List[float]) -> str:
        """
        Generate content that reflects the converged opinion vector.
        
        Args:
            topics: List of topics
            opinion_vector: Opinion values for each topic
            
        Returns:
            Generated content reflecting the opinions
        """
        return self.llm_client.analyze_converged_opinions(topics, opinion_vector)
    
    def analyze_echo_chamber_content(self, topics: List[str], echo_chamber_opinions: List[List[float]]) -> Dict[str, Any]:
        """
        Analyze content for an echo chamber.
        
        Args:
            topics: List of topics
            echo_chamber_opinions: List of opinion vectors for agents in the echo chamber
            
        Returns:
            Dictionary containing echo chamber analysis
        """
        # Calculate average opinions for the echo chamber
        avg_opinions = []
        for topic_idx in range(len(topics)):
            topic_opinions = [opinions[topic_idx] for opinions in echo_chamber_opinions]
            avg_opinions.append(sum(topic_opinions) / len(topic_opinions))
        
        # Generate content reflecting the echo chamber's average opinions
        content = self.analyze_converged_opinions(topics, avg_opinions)
        
        # Calculate opinion homogeneity
        opinion_variance = []
        for topic_idx in range(len(topics)):
            topic_opinions = [opinions[topic_idx] for opinions in echo_chamber_opinions]
            variance = sum((op - avg_opinions[topic_idx]) ** 2 for op in topic_opinions) / len(topic_opinions)
            opinion_variance.append(variance)
        
        avg_variance = sum(opinion_variance) / len(opinion_variance)
        homogeneity = 1 - avg_variance  # Higher homogeneity means lower variance
        
        return {
            'content': content,
            'average_opinions': avg_opinions,
            'opinion_variance': opinion_variance,
            'homogeneity': homogeneity,
            'chamber_size': len(echo_chamber_opinions)
        }
    
    def compare_content_bias(self, topics: List[str], opinion_vectors: List[List[float]], 
                           group_labels: List[str]) -> Dict[str, Any]:
        """
        Compare content bias between different groups.
        
        Args:
            topics: List of topics
            opinion_vectors: List of opinion vectors for each group
            group_labels: Labels for each group
            
        Returns:
            Dictionary containing bias comparison analysis
        """
        group_analyses = {}
        
        for i, (opinions, label) in enumerate(zip(opinion_vectors, group_labels)):
            # Calculate average opinions for this group
            avg_opinions = []
            for topic_idx in range(len(topics)):
                topic_opinions = [op[topic_idx] for op in opinions]
                avg_opinions.append(sum(topic_opinions) / len(topic_opinions))
            
            # Generate content for this group
            content = self.analyze_converged_opinions(topics, avg_opinions)
            
            group_analyses[label] = {
                'content': content,
                'average_opinions': avg_opinions,
                'group_size': len(opinions)
            }
        
        # Compare content between groups
        comparisons = {}
        for i, label1 in enumerate(group_labels):
            for j, label2 in enumerate(group_labels):
                if i < j:  # Avoid duplicate comparisons
                    comparison_key = f"{label1}_vs_{label2}"
                    
                    # Calculate opinion differences
                    opinions1 = group_analyses[label1]['average_opinions']
                    opinions2 = group_analyses[label2]['average_opinions']
                    
                    opinion_diffs = [abs(op1 - op2) for op1, op2 in zip(opinions1, opinions2)]
                    avg_opinion_diff = sum(opinion_diffs) / len(opinion_diffs)
                    
                    comparisons[comparison_key] = {
                        'opinion_differences': opinion_diffs,
                        'average_opinion_difference': avg_opinion_diff,
                        'content1': group_analyses[label1]['content'],
                        'content2': group_analyses[label2]['content']
                    }
        
        return {
            'group_analyses': group_analyses,
            'comparisons': comparisons
        }
    
    def detect_content_bias_patterns(self, topics: List[str], 
                                   topic_pairs: List[List[str]], 
                                   opinion_data: Dict[str, List[List[float]]]) -> Dict[str, Any]:
        """
        Detect bias patterns in content across topic pairs.
        
        Args:
            topics: List of topics
            topic_pairs: List of topic pairs to compare
            opinion_data: Dictionary mapping topic names to opinion vectors
            
        Returns:
            Dictionary containing bias pattern analysis
        """
        bias_patterns = {}
        
        for term1, term2 in topic_pairs:
            if term1 in opinion_data and term2 in opinion_data:
                # Get opinion vectors for each term
                opinions1 = opinion_data[term1]
                opinions2 = opinion_data[term2]
                
                # Calculate average opinions for each term
                avg_opinions1 = []
                avg_opinions2 = []
                
                for topic_idx in range(len(topics)):
                    topic_opinions1 = [op[topic_idx] for op in opinions1]
                    topic_opinions2 = [op[topic_idx] for op in opinions2]
                    
                    avg_opinions1.append(sum(topic_opinions1) / len(topic_opinions1))
                    avg_opinions2.append(sum(topic_opinions2) / len(topic_opinions2))
                
                # Generate content for each term
                content1 = self.analyze_converged_opinions(topics, avg_opinions1)
                content2 = self.analyze_converged_opinions(topics, avg_opinions2)
                
                # Calculate bias metrics
                opinion_diffs = [abs(op1 - op2) for op1, op2 in zip(avg_opinions1, avg_opinions2)]
                avg_opinion_diff = sum(opinion_diffs) / len(opinion_diffs)
                
                # Determine if bias is detected
                bias_threshold = 0.1
                bias_detected = avg_opinion_diff > bias_threshold
                
                bias_patterns[f"{term1}_vs_{term2}"] = {
                    'term1': term1,
                    'term2': term2,
                    'content1': content1,
                    'content2': content2,
                    'average_opinion_difference': avg_opinion_diff,
                    'opinion_differences': opinion_diffs,
                    'bias_detected': bias_detected,
                    'bias_magnitude': avg_opinion_diff
                }
        
        return bias_patterns
    
    def generate_bias_report(self, bias_patterns: Dict[str, Any]) -> str:
        """
        Generate a human-readable bias report.
        
        Args:
            bias_patterns: Dictionary containing bias pattern analysis
            
        Returns:
            Formatted bias report
        """
        report = "CONTENT BIAS ANALYSIS REPORT\n"
        report += "=" * 40 + "\n\n"
        
        for comparison_key, analysis in bias_patterns.items():
            report += f"COMPARISON: {analysis['term1']} vs {analysis['term2']}\n"
            report += "-" * 30 + "\n"
            report += f"Average opinion difference: {analysis['average_opinion_difference']:.4f}\n"
            report += f"Bias detected: {analysis['bias_detected']}\n"
            report += f"Bias magnitude: {analysis['bias_magnitude']:.4f}\n\n"
            
            report += f"Content for '{analysis['term1']}':\n"
            report += f"{analysis['content1']}\n\n"
            
            report += f"Content for '{analysis['term2']}':\n"
            report += f"{analysis['content2']}\n\n"
            
            report += "Opinion differences by topic:\n"
            for i, diff in enumerate(analysis['opinion_differences']):
                report += f"  Topic {i+1}: {diff:.4f}\n"
            report += "\n" + "=" * 40 + "\n\n"
        
        # Summary statistics
        bias_detected_count = sum(1 for analysis in bias_patterns.values() if analysis['bias_detected'])
        total_comparisons = len(bias_patterns)
        
        report += "SUMMARY:\n"
        report += f"Total comparisons: {total_comparisons}\n"
        report += f"Comparisons with bias detected: {bias_detected_count}\n"
        report += f"Bias detection rate: {bias_detected_count/total_comparisons*100:.1f}%\n"
        
        if bias_patterns:
            avg_bias_magnitude = sum(analysis['bias_magnitude'] for analysis in bias_patterns.values()) / len(bias_patterns)
            report += f"Average bias magnitude: {avg_bias_magnitude:.4f}\n"
        
        return report 