"""
Network and opinion dynamics visualization module.

This module provides comprehensive visualizations for:
- Network topology evolution
- Opinion dynamics over time
- Statistical summaries of agent behavior
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
from datetime import datetime
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NetworkOpinionVisualizer:
    """
    Comprehensive visualizer for network topology and opinion dynamics.
    """
    
    def __init__(self, output_dir: str = "results/visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up consistent styling
        self.colors = {
            'network': '#2E86AB',
            'opinions': '#A23B72', 
            'consensus': '#F18F01',
            'stubborn': '#C73E1D',
            'background': '#F5F5F5'
        }
        
    def create_comprehensive_visualization(self, 
                                         config_name: str,
                                         config: Dict[str, Any],
                                         opinion_history: List[np.ndarray],
                                         adjacency_matrix: np.ndarray,
                                         convergence_timestep: int,
                                         network_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Create separate visualizations for each component.
        
        Args:
            config_name: Name of the configuration
            config: Configuration parameters
            opinion_history: List of opinion vectors over time
            adjacency_matrix: Final adjacency matrix
            convergence_timestep: When convergence occurred
            network_info: Network statistics
            
        Returns:
            Dictionary mapping component names to file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config.get('model', 'Unknown').replace('_', ' ').title()
        topology_name = config.get('topology', 'Unknown').replace('_', ' ').title()
        
        visualization_paths = {}
        
        # 1. Initial Network State
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        self._plot_network_state(ax, adjacency_matrix, opinion_history[0], 
                               "Initial Network State", "Initial")
        initial_path = os.path.join(self.output_dir, f"{config_name}_initial_network_{timestamp}.png")
        plt.savefig(initial_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        visualization_paths['initial_network'] = initial_path
        
        # 2. Final Network State
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        self._plot_network_state(ax, adjacency_matrix, opinion_history[-1],
                               "Final Network State", "Final")
        final_path = os.path.join(self.output_dir, f"{config_name}_final_network_{timestamp}.png")
        plt.savefig(final_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        visualization_paths['final_network'] = final_path
        
        # 3. Opinion Trajectories
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        self._plot_opinion_trajectories(ax, opinion_history, convergence_timestep)
        trajectories_path = os.path.join(self.output_dir, f"{config_name}_opinion_trajectories_{timestamp}.png")
        plt.savefig(trajectories_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        visualization_paths['opinion_trajectories'] = trajectories_path
        
        # 4. Mean Opinion Evolution
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        self._plot_mean_opinion_evolution(ax, opinion_history, convergence_timestep)
        mean_evolution_path = os.path.join(self.output_dir, f"{config_name}_mean_opinion_evolution_{timestamp}.png")
        plt.savefig(mean_evolution_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        visualization_paths['mean_opinion_evolution'] = mean_evolution_path
        
        return visualization_paths
    
    def _plot_network_state(self, ax, adjacency_matrix: np.ndarray, 
                          opinions: np.ndarray, title: str, state: str):
        """Plot network topology with node colors based on opinions."""
        n_agents = len(opinions)
        
        # Create networkx graph
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        # Draw network edges first
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6, edge_color='gray', width=0.5)
        
        # Draw nodes with colors based on opinions
        # Use the actual opinion values for color mapping
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, 
                                     node_color=opinions,  # Use actual opinion values
                                     cmap='RdYlBu_r', 
                                     vmin=-1, vmax=1,  # Set proper range
                                     node_size=150,  # Larger nodes for better visibility
                                     alpha=0.9)
        
        # Add colorbar with proper range
        sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Opinion Value', rotation=270, labelpad=15)
        
        # Add title with statistics
        mean_opinion = np.mean(opinions)
        std_opinion = np.std(opinions)
        ax.set_title(f"{title}\nMean: {mean_opinion:.3f}, Std: {std_opinion:.3f}", 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add text box with opinion range
        min_opinion = np.min(opinions)
        max_opinion = np.max(opinions)
        textstr = f'Range: [{min_opinion:.3f}, {max_opinion:.3f}]'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    def _plot_opinion_trajectories(self, ax, opinion_history: List[np.ndarray], 
                                 convergence_timestep: int):
        """Plot individual agent opinion trajectories over time."""
        opinion_array = np.array(opinion_history)
        timesteps = range(len(opinion_history))
        
        # Plot trajectories for all agents
        for i in range(opinion_array.shape[1]):
            alpha = 0.3 if i < opinion_array.shape[1] - 5 else 0.7  # Highlight last 5 agents
            ax.plot(timesteps, opinion_array[:, i], alpha=alpha, linewidth=0.8)
        
        # Add convergence line
        if convergence_timestep < len(opinion_history):
            ax.axvline(x=convergence_timestep, color='red', linestyle='--', 
                      alpha=0.8, label=f'Convergence (t={convergence_timestep})')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Opinion Value')
        ax.set_title('Individual Agent Opinion Trajectories')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-1.1, 1.1)
    
    def _plot_mean_opinion_evolution(self, ax, opinion_history: List[np.ndarray],
                                   convergence_timestep: int):
        """Plot mean opinion evolution with standard deviation bands."""
        opinion_array = np.array(opinion_history)
        timesteps = range(len(opinion_history))
        
        mean_opinions = np.mean(opinion_array, axis=1)
        std_opinions = np.std(opinion_array, axis=1)
        
        # Plot mean with std bands
        ax.plot(timesteps, mean_opinions, color=self.colors['opinions'], 
               linewidth=2, label='Mean Opinion')
        ax.fill_between(timesteps, 
                       mean_opinions - std_opinions,
                       mean_opinions + std_opinions,
                       alpha=0.3, color=self.colors['opinions'], 
                       label='±1 Std Dev')
        
        # Add convergence line
        if convergence_timestep < len(opinion_history):
            ax.axvline(x=convergence_timestep, color='red', linestyle='--',
                      alpha=0.8, label=f'Convergence (t={convergence_timestep})')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Opinion Value')
        ax.set_title('Mean Opinion Evolution with Standard Deviation')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-1.1, 1.1)
    
    
    def create_animation(self, config_name: str, opinion_history: List[np.ndarray],
                        adjacency_matrix: np.ndarray, save_path: str = None) -> str:
        """
        Create an animated visualization showing network evolution over time.
        
        Args:
            config_name: Name of the configuration
            opinion_history: List of opinion vectors over time
            adjacency_matrix: Adjacency matrix (assumed constant)
            save_path: Path to save animation (optional)
            
        Returns:
            Path to saved animation file
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"{config_name}_animation_{timestamp}.gif")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Create networkx graph
        G = nx.from_numpy_array(adjacency_matrix)
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            opinions = opinion_history[frame]
            
            # Plot network
            nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.6, edge_color='gray', width=0.5)
            normalized_opinions = (opinions + 1) / 2
            nx.draw_networkx_nodes(G, pos, ax=ax1, 
                                 node_color=normalized_opinions,
                                 cmap='RdYlBu_r', vmin=0, vmax=1,
                                 node_size=100, alpha=0.8)
            
            ax1.set_title(f'Network State (t={frame})')
            ax1.axis('off')
            
            # Plot opinion distribution
            ax2.hist(opinions, bins=20, alpha=0.7, color=self.colors['opinions'])
            ax2.set_xlabel('Opinion Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Opinion Distribution (t={frame})')
            ax2.set_xlim(-1.1, 1.1)
            ax2.grid(True, alpha=0.3)
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(opinion_history),
                           interval=200, repeat=True)
        
        # Save animation
        anim.save(save_path, writer='pillow', fps=5)
        plt.close()
        
        return save_path
