"""
Bias analyzer for detecting and analyzing bias patterns in simulation results.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class BiasAnalyzer:
    """
    Analyzes bias patterns in simulation results.
    """
    
    def __init__(self):
        """Initialize the bias analyzer."""
        pass
    
    def analyze_convergence_patterns(self, opinion_history: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze convergence patterns in opinion evolution.
        
        Args:
            opinion_history: List of opinion matrices over time
            
        Returns:
            Dictionary containing convergence analysis metrics
        """
        if not opinion_history or len(opinion_history) < 2:
            return {}
        
        # Calculate opinion variance over time
        variances = []
        for opinions in opinion_history:
            if opinions is not None:
                variance = np.var(opinions)
                variances.append(variance)
            else:
                variances.append(0)
        
        # Calculate convergence metrics
        initial_variance = variances[0]
        final_variance = variances[-1]
        
        # Convergence speed (rate of variance reduction)
        if initial_variance > 0:
            convergence_speed = (initial_variance - final_variance) / initial_variance
        else:
            convergence_speed = 0
        
        # Convergence stability (how stable the final state is)
        if len(variances) > 10:
            final_stability = np.std(variances[-10:])  # Standard deviation of last 10 timesteps
        else:
            final_stability = np.std(variances)
        
        # Convergence time (when variance drops below threshold)
        threshold = initial_variance * 0.1  # 10% of initial variance
        convergence_time = len(variances)
        for i, variance in enumerate(variances):
            if variance <= threshold:
                convergence_time = i
                break
        
        return {
            'convergence_speed': convergence_speed,
            'convergence_stability': final_stability,
            'convergence_time': convergence_time,
            'initial_variance': initial_variance,
            'final_variance': final_variance,
            'variance_reduction': initial_variance - final_variance
        }
    
    def analyze_echo_chambers(self, adjacency_matrix: np.ndarray, opinion_matrix: np.ndarray, 
                            similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Analyze echo chamber formation and characteristics.
        
        Args:
            adjacency_matrix: Network adjacency matrix
            opinion_matrix: Current opinion matrix
            similarity_threshold: Threshold for echo chamber detection
            
        Returns:
            Dictionary containing echo chamber analysis
        """
        # Calculate opinion similarity matrix
        n_agents = len(opinion_matrix)
        similarity_matrix = np.zeros((n_agents, n_agents))
        
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    # Calculate opinion similarity (1 - normalized L1 distance)
                    opinion_diff = np.linalg.norm(opinion_matrix[i] - opinion_matrix[j], ord=1)
                    max_possible_diff = len(opinion_matrix[i])
                    similarity = 1 - (opinion_diff / max_possible_diff)
                    similarity_matrix[i, j] = max(0, similarity)
        
        # Find echo chambers (connected components with high similarity)
        echo_chamber_graph = (adjacency_matrix > 0) & (similarity_matrix > similarity_threshold)
        
        # Use networkx to find connected components
        import networkx as nx
        G = nx.from_numpy_array(echo_chamber_graph.astype(int))
        components = list(nx.connected_components(G))
        
        # Filter out single-node components
        echo_chambers = [list(comp) for comp in components if len(comp) > 1]
        
        # Analyze echo chamber characteristics
        chamber_sizes = [len(chamber) for chamber in echo_chambers]
        chamber_opinions = []
        
        for chamber in echo_chambers:
            chamber_opinion = np.mean([opinion_matrix[i] for i in chamber], axis=0)
            chamber_opinions.append(chamber_opinion)
        
        # Calculate echo chamber metrics
        total_echo_chamber_agents = sum(chamber_sizes)
        echo_chamber_coverage = total_echo_chamber_agents / n_agents if n_agents > 0 else 0
        
        # Calculate opinion polarization within echo chambers
        polarization_scores = []
        for chamber in echo_chambers:
            if len(chamber) > 1:
                chamber_opinions_subset = [opinion_matrix[i] for i in chamber]
                # Calculate opinion variance within chamber
                variance = np.var(chamber_opinions_subset)
                polarization_scores.append(variance)
        
        avg_polarization = np.mean(polarization_scores) if polarization_scores else 0
        
        return {
            'num_echo_chambers': len(echo_chambers),
            'echo_chamber_sizes': chamber_sizes,
            'echo_chamber_coverage': echo_chamber_coverage,
            'avg_echo_chamber_size': np.mean(chamber_sizes) if chamber_sizes else 0,
            'max_echo_chamber_size': max(chamber_sizes) if chamber_sizes else 0,
            'chamber_opinions': chamber_opinions,
            'avg_polarization': avg_polarization,
            'echo_chambers': echo_chambers
        }
    
    def analyze_opinion_distribution(self, opinion_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the distribution of opinions across agents.
        
        Args:
            opinion_matrix: Current opinion matrix
            
        Returns:
            Dictionary containing opinion distribution analysis
        """
        n_agents, n_topics = opinion_matrix.shape
        
        # Calculate basic statistics for each topic
        topic_stats = {}
        for topic_idx in range(n_topics):
            opinions = opinion_matrix[:, topic_idx]
            topic_stats[f'topic_{topic_idx}'] = {
                'mean': np.mean(opinions),
                'std': np.std(opinions),
                'min': np.min(opinions),
                'max': np.max(opinions),
                'median': np.median(opinions),
                'skewness': self._calculate_skewness(opinions),
                'kurtosis': self._calculate_kurtosis(opinions)
            }
        
        # Calculate overall opinion diversity
        overall_variance = np.var(opinion_matrix)
        overall_entropy = self._calculate_opinion_entropy(opinion_matrix)
        
        # Detect opinion clusters
        opinion_clusters = self._detect_opinion_clusters(opinion_matrix)
        
        return {
            'topic_statistics': topic_stats,
            'overall_variance': overall_variance,
            'overall_entropy': overall_entropy,
            'opinion_clusters': opinion_clusters,
            'n_agents': n_agents,
            'n_topics': n_topics
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of a dataset."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of a dataset."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return kurtosis
    
    def _calculate_opinion_entropy(self, opinion_matrix: np.ndarray) -> float:
        """Calculate entropy of opinion distribution."""
        # Discretize opinions into bins for entropy calculation
        n_bins = 10
        binned_opinions = np.digitize(opinion_matrix.flatten(), bins=np.linspace(0, 1, n_bins))
        
        # Calculate entropy
        unique, counts = np.unique(binned_opinions, return_counts=True)
        probabilities = counts / len(binned_opinions)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy
    
    def _detect_opinion_clusters(self, opinion_matrix: np.ndarray, n_clusters: int = 3) -> Dict[str, Any]:
        """Detect opinion clusters using k-means clustering."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            
            # Reduce dimensionality if needed
            if opinion_matrix.shape[1] > 2:
                pca = PCA(n_components=2)
                reduced_opinions = pca.fit_transform(opinion_matrix)
            else:
                reduced_opinions = opinion_matrix
            
            # Perform clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(opinion_matrix)), random_state=42)
            cluster_labels = kmeans.fit_predict(reduced_opinions)
            
            # Analyze clusters
            cluster_sizes = []
            cluster_centers = []
            cluster_variances = []
            
            for i in range(kmeans.n_clusters_):
                cluster_mask = cluster_labels == i
                cluster_opinions = opinion_matrix[cluster_mask]
                
                cluster_sizes.append(np.sum(cluster_mask))
                cluster_centers.append(np.mean(cluster_opinions, axis=0))
                cluster_variances.append(np.var(cluster_opinions))
            
            return {
                'cluster_labels': cluster_labels.tolist(),
                'cluster_sizes': cluster_sizes,
                'cluster_centers': [center.tolist() for center in cluster_centers],
                'cluster_variances': cluster_variances,
                'n_clusters': kmeans.n_clusters_
            }
        except ImportError:
            # Fallback if sklearn is not available
            return {
                'cluster_labels': [0] * len(opinion_matrix),
                'cluster_sizes': [len(opinion_matrix)],
                'cluster_centers': [np.mean(opinion_matrix, axis=0).tolist()],
                'cluster_variances': [np.var(opinion_matrix)],
                'n_clusters': 1
            }
    
    def compare_simulations(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two simulation results to detect bias.
        
        Args:
            results1: Results from first simulation
            results2: Results from second simulation
            
        Returns:
            Dictionary containing comparison analysis
        """
        comparison = {}
        
        # Compare convergence patterns
        conv1 = results1.get('convergence_analysis', {})
        conv2 = results2.get('convergence_analysis', {})
        
        comparison['convergence_speed_diff'] = conv1.get('convergence_speed', 0) - conv2.get('convergence_speed', 0)
        comparison['convergence_time_diff'] = conv1.get('convergence_time', 0) - conv2.get('convergence_time', 0)
        comparison['final_variance_diff'] = conv1.get('final_variance', 0) - conv2.get('final_variance', 0)
        
        # Compare echo chamber formation
        echo1 = results1.get('echo_chamber_analysis', {})
        echo2 = results2.get('echo_chamber_analysis', {})
        
        comparison['echo_chamber_count_diff'] = echo1.get('num_echo_chambers', 0) - echo2.get('num_echo_chambers', 0)
        comparison['echo_chamber_coverage_diff'] = echo1.get('echo_chamber_coverage', 0) - echo2.get('echo_chamber_coverage', 0)
        comparison['avg_echo_chamber_size_diff'] = echo1.get('avg_echo_chamber_size', 0) - echo2.get('avg_echo_chamber_size', 0)
        
        # Compare opinion distribution
        dist1 = results1.get('opinion_distribution_analysis', {})
        dist2 = results2.get('opinion_distribution_analysis', {})
        
        comparison['overall_variance_diff'] = dist1.get('overall_variance', 0) - dist2.get('overall_variance', 0)
        comparison['overall_entropy_diff'] = dist1.get('overall_entropy', 0) - dist2.get('overall_entropy', 0)
        
        # Determine if bias is detected
        bias_threshold = 0.1  # Configurable threshold
        bias_indicators = [
            abs(comparison['convergence_speed_diff']) > bias_threshold,
            abs(comparison['echo_chamber_coverage_diff']) > bias_threshold,
            abs(comparison['overall_variance_diff']) > bias_threshold
        ]
        
        comparison['bias_detected'] = any(bias_indicators)
        comparison['bias_confidence'] = sum(bias_indicators) / len(bias_indicators)
        
        return comparison
    
    def generate_bias_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable bias report.
        
        Args:
            comparison_results: Results from bias comparison
            
        Returns:
            Formatted report string
        """
        report = "BIAS ANALYSIS REPORT\n"
        report += "=" * 30 + "\n\n"
        
        # Convergence analysis
        report += "CONVERGENCE ANALYSIS:\n"
        report += f"  Convergence speed difference: {comparison_results.get('convergence_speed_diff', 0):.4f}\n"
        report += f"  Convergence time difference: {comparison_results.get('convergence_time_diff', 0):.0f} timesteps\n"
        report += f"  Final variance difference: {comparison_results.get('final_variance_diff', 0):.4f}\n\n"
        
        # Echo chamber analysis
        report += "ECHO CHAMBER ANALYSIS:\n"
        report += f"  Echo chamber count difference: {comparison_results.get('echo_chamber_count_diff', 0):.0f}\n"
        report += f"  Echo chamber coverage difference: {comparison_results.get('echo_chamber_coverage_diff', 0):.4f}\n"
        report += f"  Average chamber size difference: {comparison_results.get('avg_echo_chamber_size_diff', 0):.2f}\n\n"
        
        # Opinion distribution analysis
        report += "OPINION DISTRIBUTION ANALYSIS:\n"
        report += f"  Overall variance difference: {comparison_results.get('overall_variance_diff', 0):.4f}\n"
        report += f"  Overall entropy difference: {comparison_results.get('overall_entropy_diff', 0):.4f}\n\n"
        
        # Bias detection
        report += "BIAS DETECTION:\n"
        report += f"  Bias detected: {comparison_results.get('bias_detected', False)}\n"
        report += f"  Bias confidence: {comparison_results.get('bias_confidence', 0):.2f}\n"
        
        return report 