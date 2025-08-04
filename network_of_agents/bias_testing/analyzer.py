"""
Bias testing analyzer for detecting and analyzing biases in simulation results.
"""

import numpy as np
from typing import Dict, List, Any


class BiasAnalyzer:
    """
    Analyzes simulation results for bias detection and analysis.
    """
    
    def __init__(self):
        pass
    
    def analyze_convergence_patterns(self, opinion_history: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze opinion convergence patterns.
        
        Args:
            opinion_history: List of opinion vectors over time
            
        Returns:
            Dictionary containing convergence analysis
        """
        if not opinion_history or len(opinion_history) < 2:
            return {}
        
        # Calculate variance over time
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
        
        # Convergence speed
        if initial_variance > 0:
            convergence_speed = (initial_variance - final_variance) / initial_variance
        else:
            convergence_speed = 0
        
        # Convergence time (when variance drops below 10% of initial)
        threshold = initial_variance * 0.1
        convergence_time = len(variances)
        for i, variance in enumerate(variances):
            if variance <= threshold:
                convergence_time = i
                break
        
        # Stability analysis
        if len(variances) > 10:
            final_stability = np.std(variances[-10:])
        else:
            final_stability = np.std(variances)
        
        return {
            'convergence_speed': convergence_speed,
            'convergence_time': convergence_time,
            'final_stability': final_stability,
            'initial_variance': initial_variance,
            'final_variance': final_variance,
            'variance_history': variances
        }
    
    def analyze_llm_consistency(self, loss_metrics_history: List[Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze LLM round-trip consistency over time.
        
        Args:
            loss_metrics_history: History of loss metrics from data storage
            
        Returns:
            Dictionary containing LLM consistency analysis
        """
        if not loss_metrics_history or len(loss_metrics_history) < 2:
            return {}
        
        # Extract loss trends over time
        avg_losses = []
        max_losses = []
        min_losses = []
        
        for timestep_losses in loss_metrics_history:
            if timestep_losses is not None:
                timestep_avg_losses = [data['average_loss'] for data in timestep_losses.values()]
                timestep_max_losses = [data['max_loss'] for data in timestep_losses.values()]
                timestep_min_losses = [data['min_loss'] for data in timestep_losses.values()]
                
                avg_losses.append(np.mean(timestep_avg_losses))
                max_losses.append(np.max(timestep_max_losses))
                min_losses.append(np.min(timestep_min_losses))
            else:
                avg_losses.append(0.0)
                max_losses.append(0.0)
                min_losses.append(0.0)
        
        # Calculate consistency metrics
        initial_avg_loss = avg_losses[0]
        final_avg_loss = avg_losses[-1]
        
        # Consistency trend
        if initial_avg_loss > 0:
            consistency_trend = (initial_avg_loss - final_avg_loss) / initial_avg_loss
        else:
            consistency_trend = 0
        
        # Loss stability
        if len(avg_losses) > 10:
            final_stability = np.std(avg_losses[-10:])
        else:
            final_stability = np.std(avg_losses)
        
        # Overall consistency score (lower is better)
        overall_consistency = 1.0 - np.mean(avg_losses)
        
        return {
            'overall_consistency': overall_consistency,
            'consistency_trend': consistency_trend,
            'final_stability': final_stability,
            'initial_avg_loss': initial_avg_loss,
            'final_avg_loss': final_avg_loss,
            'avg_loss_history': avg_losses,
            'max_loss_history': max_losses,
            'min_loss_history': min_losses
        }
    
    def assess_opinion_dynamics_performance(self, opinion_history: List[np.ndarray], 
                                          loss_metrics_history: List[Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Evaluate how well LLM maintains opinion dynamics.
        
        Args:
            opinion_history: History of opinion vectors over time
            loss_metrics_history: History of loss metrics
            
        Returns:
            Dictionary containing opinion dynamics performance analysis
        """
        if not opinion_history or len(opinion_history) < 2:
            return {}
        
        # Calculate opinion change rates
        opinion_changes = []
        for i in range(1, len(opinion_history)):
            if opinion_history[i] is not None and opinion_history[i-1] is not None:
                change = np.mean(np.abs(opinion_history[i] - opinion_history[i-1]))
                opinion_changes.append(change)
            else:
                opinion_changes.append(0.0)
        
        # Calculate dynamics metrics
        avg_change_rate = np.mean(opinion_changes)
        change_variance = np.var(opinion_changes)
        
        # Assess if opinions are actually changing
        significant_changes = [c for c in opinion_changes if c > 0.01]  # Threshold for significant change
        change_frequency = len(significant_changes) / len(opinion_changes) if opinion_changes else 0
        
        # Calculate loss-consistency correlation
        loss_consistency_correlation = 0.0
        if loss_metrics_history and len(loss_metrics_history) == len(opinion_history):
            avg_losses = []
            for timestep_losses in loss_metrics_history:
                if timestep_losses is not None:
                    timestep_avg_losses = [data['average_loss'] for data in timestep_losses.values()]
                    avg_losses.append(np.mean(timestep_avg_losses))
                else:
                    avg_losses.append(0.0)
            
            if len(avg_losses) == len(opinion_changes):
                # Calculate correlation between opinion changes and loss
                correlation_matrix = np.corrcoef(opinion_changes, avg_losses)
                loss_consistency_correlation = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
        
        # Overall performance score
        # Higher score = better performance (more changes, lower loss, higher consistency)
        performance_score = (change_frequency * 0.4 + 
                           (1.0 - np.mean(avg_losses)) * 0.4 + 
                           (1.0 + loss_consistency_correlation) * 0.2)
        
        return {
            'avg_change_rate': avg_change_rate,
            'change_variance': change_variance,
            'change_frequency': change_frequency,
            'significant_changes_count': len(significant_changes),
            'loss_consistency_correlation': loss_consistency_correlation,
            'performance_score': performance_score,
            'opinion_changes': opinion_changes
        }
    
    def analyze_echo_chambers(self, adjacency_matrix: np.ndarray, opinion_matrix: np.ndarray, 
                            similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Analyze echo chambers in the network.
        
        Args:
            adjacency_matrix: Network adjacency matrix
            opinion_matrix: Opinion vector (single topic)
            similarity_threshold: Threshold for echo chamber detection
            
        Returns:
            Dictionary containing echo chamber analysis
        """
        n_agents = len(opinion_matrix)
        
        # Calculate opinion similarity matrix
        similarity_matrix = np.zeros((n_agents, n_agents))
        
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    # Calculate opinion similarity (1 - normalized L1 distance)
                    # For single topic, just use absolute difference
                    opinion_diff = abs(opinion_matrix[i] - opinion_matrix[j])
                    max_possible_diff = 1.0  # Single topic, max difference is 1
                    similarity = 1 - (opinion_diff / max_possible_diff)
                    similarity_matrix[i, j] = max(0, similarity)
        
        # Find echo chambers using connected components with high similarity
        import networkx as nx
        
        # Create graph where edges exist if both connected and similar
        echo_chamber_graph = (adjacency_matrix > 0) & (similarity_matrix > similarity_threshold)
        G = nx.from_numpy_array(echo_chamber_graph.astype(int))
        components = list(nx.connected_components(G))
        
        # Filter out single-node components
        echo_chambers = [list(comp) for comp in components if len(comp) > 1]
        
        # Analyze echo chambers
        chamber_sizes = [len(chamber) for chamber in echo_chambers]
        chamber_opinions = []
        
        for chamber in echo_chambers:
            chamber_opinion = np.mean([opinion_matrix[i] for i in chamber])
            chamber_opinions.append(chamber_opinion)
        
        return {
            'echo_chambers': echo_chambers,
            'chamber_sizes': chamber_sizes,
            'chamber_opinions': chamber_opinions,
            'num_echo_chambers': len(echo_chambers),
            'total_echo_chamber_agents': sum(chamber_sizes),
            'avg_chamber_size': np.mean(chamber_sizes) if chamber_sizes else 0,
            'max_chamber_size': max(chamber_sizes) if chamber_sizes else 0
        }
    
    def analyze_opinion_distribution(self, opinion_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analyze opinion distribution.
        
        Args:
            opinion_matrix: Opinion vector (single topic)
            
        Returns:
            Dictionary containing opinion distribution analysis
        """
        n_agents = len(opinion_matrix)
        
        # Calculate basic statistics
        opinion_stats = {
            'mean': np.mean(opinion_matrix),
            'std': np.std(opinion_matrix),
            'min': np.min(opinion_matrix),
            'max': np.max(opinion_matrix),
            'median': np.median(opinion_matrix),
            'skewness': self._calculate_skewness(opinion_matrix),
            'kurtosis': self._calculate_kurtosis(opinion_matrix)
        }
        
        # Calculate overall opinion diversity
        overall_variance = np.var(opinion_matrix)
        overall_entropy = self._calculate_opinion_entropy(opinion_matrix)
        
        # Detect opinion clusters
        opinion_clusters = self._detect_opinion_clusters(opinion_matrix)
        
        return {
            'opinion_statistics': opinion_stats,
            'overall_variance': overall_variance,
            'overall_entropy': overall_entropy,
            'opinion_clusters': opinion_clusters,
            'n_agents': n_agents
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
        """
        Calculate entropy of opinion distribution.
        
        Args:
            opinion_matrix: Opinion vector (single topic)
            
        Returns:
            Entropy value
        """
        # Discretize opinions into bins for entropy calculation
        n_bins = 10
        binned_opinions = np.digitize(opinion_matrix, bins=np.linspace(0, 1, n_bins))
        
        # Calculate entropy
        unique, counts = np.unique(binned_opinions, return_counts=True)
        probabilities = counts / len(binned_opinions)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy
    
    def _detect_opinion_clusters(self, opinion_matrix: np.ndarray, n_clusters: int = 3) -> Dict[str, Any]:
        """
        Detect opinion clusters using K-means clustering.
        
        Args:
            opinion_matrix: Opinion vector (single topic)
            n_clusters: Number of clusters to detect
            
        Returns:
            Dictionary containing cluster analysis
        """
        from sklearn.cluster import KMeans
        
        # Reshape for clustering (single dimension)
        X = opinion_matrix.reshape(-1, 1)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(opinion_matrix)), random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate cluster statistics
        cluster_sizes = []
        cluster_centers = []
        cluster_variances = []
        
        for i in range(kmeans.n_clusters):
            cluster_mask = cluster_labels == i
            cluster_data = opinion_matrix[cluster_mask]
            
            cluster_sizes.append(len(cluster_data))
            cluster_centers.append(np.mean(cluster_data))
            cluster_variances.append(np.var(cluster_data))
        
        return {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_sizes': cluster_sizes,
            'cluster_centers': cluster_centers,
            'cluster_variances': cluster_variances,
            'n_clusters': kmeans.n_clusters
        }
    
    def compare_simulations(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two simulation results.
        
        Args:
            results1: Results from first simulation
            results2: Results from second simulation
            
        Returns:
            Dictionary containing comparison analysis
        """
        comparison = {}
        
        # Compare convergence patterns
        if 'convergence_analysis' in results1 and 'convergence_analysis' in results2:
            conv1 = results1['convergence_analysis']
            conv2 = results2['convergence_analysis']
            
            comparison['convergence_comparison'] = {
                'speed_difference': conv1.get('convergence_speed', 0) - conv2.get('convergence_speed', 0),
                'time_difference': conv1.get('convergence_time', 0) - conv2.get('convergence_time', 0),
                'stability_difference': conv1.get('final_stability', 0) - conv2.get('final_stability', 0)
            }
        
        # Compare echo chambers
        if 'echo_chamber_analysis' in results1 and 'echo_chamber_analysis' in results2:
            echo1 = results1['echo_chamber_analysis']
            echo2 = results2['echo_chamber_analysis']
            
            comparison['echo_chamber_comparison'] = {
                'num_chambers_difference': echo1.get('num_echo_chambers', 0) - echo2.get('num_echo_chambers', 0),
                'total_agents_difference': echo1.get('total_echo_chamber_agents', 0) - echo2.get('total_echo_chamber_agents', 0),
                'avg_size_difference': echo1.get('avg_chamber_size', 0) - echo2.get('avg_chamber_size', 0)
            }
        
        # Compare opinion distributions
        if 'opinion_distribution' in results1 and 'opinion_distribution' in results2:
            dist1 = results1['opinion_distribution']
            dist2 = results2['opinion_distribution']
            
            stats1 = dist1.get('opinion_statistics', {})
            stats2 = dist2.get('opinion_statistics', {})
            
            comparison['opinion_comparison'] = {
                'mean_difference': stats1.get('mean', 0) - stats2.get('mean', 0),
                'variance_difference': dist1.get('overall_variance', 0) - dist2.get('overall_variance', 0),
                'entropy_difference': dist1.get('overall_entropy', 0) - dist2.get('overall_entropy', 0)
            }
        
        return comparison 