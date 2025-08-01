"""
Network graph visualization component.
"""

import plotly.graph_objs as go
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple


class NetworkGraph:
    """
    Handles network graph visualization using Plotly.
    """
    
    def __init__(self):
        """Initialize the network graph component."""
        pass
    
    def create_network_graph(self, 
                           adjacency_matrix: np.ndarray,
                           opinion_matrix: np.ndarray,
                           metrics: Dict[str, float]) -> go.Figure:
        """
        Create an interactive network graph.
        
        Args:
            adjacency_matrix: Network adjacency matrix
            opinion_matrix: Agent opinion matrix
            metrics: Network metrics
            
        Returns:
            Plotly figure object
        """
        # Create NetworkX graph
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Extract node positions
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Calculate node sizes based on degree
        degrees = [G.degree(node) for node in G.nodes()]
        max_degree = max(degrees) if degrees else 1
        node_sizes = [10 + 40 * (deg / max_degree) for deg in degrees]
        
        # Calculate node colors based on opinions (RGB for 3 topics)
        node_colors = []
        for i in range(len(G.nodes())):
            if i < len(opinion_matrix):
                opinions = opinion_matrix[i]
                # Map opinions to RGB colors
                if len(opinions) >= 3:
                    color = f'rgb({int(opinions[0]*255)}, {int(opinions[1]*255)}, {int(opinions[2]*255)})'
                else:
                    # Use grayscale for fewer topics
                    avg_opinion = np.mean(opinions)
                    color = f'rgb({int(avg_opinion*255)}, {int(avg_opinion*255)}, {int(avg_opinion*255)})'
                node_colors.append(color)
            else:
                node_colors.append('rgb(128, 128, 128)')
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=node_sizes,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        
        # Color nodes by number of connections
        node_trace.marker.color = degrees
        
        # Create hover text
        node_text = []
        for i, node in enumerate(G.nodes()):
            if i < len(opinion_matrix):
                opinions = opinion_matrix[i]
                opinion_text = ', '.join([f'Topic {j+1}: {op:.2f}' for j, op in enumerate(opinions)])
                node_text.append(f'Agent {node}<br>Degree: {degrees[i]}<br>Opinions: {opinion_text}')
            else:
                node_text.append(f'Agent {node}<br>Degree: {degrees[i]}')
        
        node_trace.text = node_text
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Network of Agents',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[dict(
                               text=f"Network Density: {metrics.get('density', 0):.3f}<br>"
                                    f"Clustering Coefficient: {metrics.get('clustering_coefficient', 0):.3f}<br>"
                                    f"Echo Chambers: {metrics.get('echo_chambers', 0)}",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(size=10)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig
    
    def create_opinion_heatmap(self, opinion_matrix: np.ndarray, topics: List[str]) -> go.Figure:
        """
        Create a heatmap of opinions across agents and topics.
        
        Args:
            opinion_matrix: Agent opinion matrix
            topics: List of topic names
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=opinion_matrix,
            x=topics,
            y=[f'Agent {i}' for i in range(len(opinion_matrix))],
            colorscale='RdBu',
            zmid=0.5
        ))
        
        fig.update_layout(
            title='Opinion Heatmap',
            xaxis_title='Topics',
            yaxis_title='Agents',
            height=600
        )
        
        return fig
    
    def create_opinion_distribution(self, opinion_matrix: np.ndarray, topics: List[str]) -> go.Figure:
        """
        Create distribution plots for each topic.
        
        Args:
            opinion_matrix: Agent opinion matrix
            topics: List of topic names
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        for i, topic in enumerate(topics):
            if i < opinion_matrix.shape[1]:
                opinions = opinion_matrix[:, i]
                fig.add_trace(go.Histogram(
                    x=opinions,
                    name=topic,
                    opacity=0.7
                ))
        
        fig.update_layout(
            title='Opinion Distributions by Topic',
            xaxis_title='Opinion Value',
            yaxis_title='Frequency',
            barmode='overlay'
        )
        
        return fig
    
    def create_network_evolution(self, adjacency_history: List[np.ndarray]) -> go.Figure:
        """
        Create network evolution visualization.
        
        Args:
            adjacency_history: List of adjacency matrices over time
            
        Returns:
            Plotly figure object
        """
        # Calculate network metrics over time
        timesteps = list(range(len(adjacency_history)))
        densities = []
        clustering_coeffs = []
        
        for adj_matrix in adjacency_history:
            if adj_matrix is not None:
                G = nx.from_numpy_array(adj_matrix)
                density = nx.density(G)
                clustering = nx.average_clustering(G)
                densities.append(density)
                clustering_coeffs.append(clustering)
            else:
                densities.append(0)
                clustering_coeffs.append(0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=densities,
            mode='lines',
            name='Network Density',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=clustering_coeffs,
            mode='lines',
            name='Clustering Coefficient',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Network Evolution',
            xaxis_title='Timestep',
            yaxis_title='Metric Value',
            showlegend=True
        )
        
        return fig
    
    def create_echo_chamber_visualization(self, 
                                        adjacency_matrix: np.ndarray,
                                        opinion_matrix: np.ndarray,
                                        echo_chambers: List[List[int]]) -> go.Figure:
        """
        Create visualization highlighting echo chambers.
        
        Args:
            adjacency_matrix: Network adjacency matrix
            opinion_matrix: Agent opinion matrix
            echo_chambers: List of echo chamber agent groups
            
        Returns:
            Plotly figure object
        """
        G = nx.from_numpy_array(adjacency_matrix)
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Extract node positions
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Color nodes by echo chamber membership
        node_colors = []
        for i in range(len(G.nodes())):
            in_echo_chamber = False
            for chamber in echo_chambers:
                if i in chamber:
                    in_echo_chamber = True
                    break
            node_colors.append('red' if in_echo_chamber else 'blue')
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=node_colors,
                size=15,
                line_width=2))
        
        # Create hover text
        node_text = []
        for i, node in enumerate(G.nodes()):
            in_echo_chamber = any(node in chamber for chamber in echo_chambers)
            status = "Echo Chamber" if in_echo_chamber else "Regular Agent"
            node_text.append(f'Agent {node}<br>Status: {status}')
        
        node_trace.text = node_text
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Echo Chamber Detection',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig 