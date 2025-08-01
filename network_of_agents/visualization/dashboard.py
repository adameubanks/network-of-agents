"""
Real-time interactive dashboard for the network of agents simulation.
"""

import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from typing import Dict, Any, List, Optional
import threading
import time

from .network_graph import NetworkGraph


class Dashboard:
    """
    Real-time interactive dashboard for simulation visualization.
    """
    
    def __init__(self, simulation_controller, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the dashboard.
        
        Args:
            simulation_controller: Simulation controller instance
            config: Dashboard configuration
        """
        self.simulation_controller = simulation_controller
        self.config = config or {}
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.network_graph = NetworkGraph()
        
        # Setup layout
        self.setup_layout()
        self.setup_callbacks()
        
        # Dashboard state
        self.is_running = False
        self.update_interval = self.config.get('update_interval', 100)
        
    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Network of Agents Simulation", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Simulation Controls"),
                        dbc.CardBody([
                            dbc.Button("Start Simulation", id="start-btn", color="success", className="me-2"),
                            dbc.Button("Stop Simulation", id="stop-btn", color="danger", className="me-2"),
                            dbc.Button("Reset Simulation", id="reset-btn", color="warning", className="me-2"),
                            html.Br(),
                            html.Br(),
                            dbc.Label("Update Interval (ms):"),
                            dcc.Slider(
                                id="update-interval-slider",
                                min=50,
                                max=1000,
                                step=50,
                                value=self.update_interval,
                                marks={i: str(i) for i in [50, 200, 500, 1000]}
                            )
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Simulation Metrics"),
                        dbc.CardBody([
                            html.Div(id="metrics-display")
                        ])
                    ])
                ], width=9)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Network Graph"),
                        dbc.CardBody([
                            dcc.Graph(id="network-graph", style={'height': '600px'})
                        ])
                    ])
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Opinion Evolution"),
                        dbc.CardBody([
                            dcc.Graph(id="opinion-evolution", style={'height': '300px'})
                        ])
                    ]),
                    dbc.Card([
                        dbc.CardHeader("Network Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id="network-metrics", style={'height': '300px'})
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Bias Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="bias-analysis", style={'height': '400px'})
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Echo Chamber Detection"),
                        dbc.CardBody([
                            dcc.Graph(id="echo-chambers", style={'height': '400px'})
                        ])
                    ])
                ], width=6)
            ]),
            
            # Hidden div for storing simulation data
            html.Div(id="simulation-data", style={'display': 'none'}),
            
            # Interval component for updates
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0,
                disabled=True
            )
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output("start-btn", "disabled"),
             Output("stop-btn", "disabled"),
             Output("interval-component", "disabled")],
            [Input("start-btn", "n_clicks"),
             Input("stop-btn", "n_clicks"),
             Input("reset-btn", "n_clicks")]
        )
        def update_controls(start_clicks, stop_clicks, reset_clicks):
            ctx = callback_context
            if not ctx.triggered:
                return False, True, True
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == "start-btn":
                self.is_running = True
                return True, False, False
            elif button_id == "stop-btn":
                self.is_running = False
                return False, True, True
            elif button_id == "reset-btn":
                self.is_running = False
                # Reset simulation logic here
                return False, True, True
            
            return False, True, True
        
        @self.app.callback(
            Output("update-interval-slider", "value"),
            [Input("update-interval-slider", "value")]
        )
        def update_interval(value):
            self.update_interval = value
            return value
        
        @self.app.callback(
            [Output("network-graph", "figure"),
             Output("opinion-evolution", "figure"),
             Output("network-metrics", "figure"),
             Output("metrics-display", "children"),
             Output("bias-analysis", "figure"),
             Output("echo-chambers", "figure")],
            [Input("interval-component", "n_intervals")]
        )
        def update_dashboard(n_intervals):
            if not self.is_running:
                return self.get_empty_figures()
            
            # Get current simulation state
            state = self.simulation_controller.get_current_state()
            
            # Update network graph
            network_fig = self.network_graph.create_network_graph(
                state['adjacency'],
                state['opinions'],
                state['network_metrics']
            )
            
            # Update opinion evolution
            opinion_fig = self.create_opinion_evolution_figure()
            
            # Update network metrics
            metrics_fig = self.create_network_metrics_figure()
            
            # Update metrics display
            metrics_display = self.create_metrics_display(state['network_metrics'])
            
            # Update bias analysis
            bias_fig = self.create_bias_analysis_figure()
            
            # Update echo chambers
            echo_fig = self.create_echo_chambers_figure()
            
            return network_fig, opinion_fig, metrics_fig, metrics_display, bias_fig, echo_fig
    
    def create_opinion_evolution_figure(self):
        """Create opinion evolution figure."""
        # Get opinion history from data storage
        opinion_history = self.simulation_controller.data_storage.get_opinion_history()
        
        if not opinion_history or len(opinion_history) < 2:
            return self.get_empty_figure("Opinion Evolution")
        
        # Calculate average opinions over time
        timesteps = list(range(len(opinion_history)))
        avg_opinions = []
        
        for opinions in opinion_history:
            if opinions is not None:
                avg_opinions.append(np.mean(opinions, axis=0))
            else:
                avg_opinions.append([0] * self.simulation_controller.n_topics)
        
        # Create traces for each topic
        traces = []
        for topic_idx in range(self.simulation_controller.n_topics):
            topic_opinions = [avg[topic_idx] for avg in avg_opinions]
            traces.append(go.Scatter(
                x=timesteps,
                y=topic_opinions,
                mode='lines',
                name=f'Topic {topic_idx + 1}',
                line=dict(width=2)
            ))
        
        fig = go.Figure(data=traces)
        fig.update_layout(
            title="Opinion Evolution Over Time",
            xaxis_title="Timestep",
            yaxis_title="Average Opinion",
            yaxis=dict(range=[0, 1]),
            showlegend=True
        )
        
        return fig
    
    def create_network_metrics_figure(self):
        """Create network metrics figure."""
        metrics_history = self.simulation_controller.data_storage.get_metrics_history()
        
        if not metrics_history or len(metrics_history) < 2:
            return self.get_empty_figure("Network Metrics")
        
        timesteps = list(range(len(metrics_history)))
        
        # Extract metrics
        density = [m['density'] if m else 0 for m in metrics_history]
        clustering = [m['clustering_coefficient'] if m else 0 for m in metrics_history]
        components = [m['num_components'] if m else 0 for m in metrics_history]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=density,
            mode='lines',
            name='Network Density',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=clustering,
            mode='lines',
            name='Clustering Coefficient',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=components,
            mode='lines',
            name='Number of Components',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title="Network Metrics Over Time",
            xaxis_title="Timestep",
            yaxis_title="Metric Value",
            showlegend=True
        )
        
        return fig
    
    def create_metrics_display(self, metrics: Dict[str, float]):
        """Create metrics display."""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{metrics.get('density', 0):.3f}", className="text-primary"),
                        html.P("Network Density", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{metrics.get('average_degree', 0):.1f}", className="text-success"),
                        html.P("Average Degree", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{metrics.get('clustering_coefficient', 0):.3f}", className="text-warning"),
                        html.P("Clustering Coefficient", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{metrics.get('echo_chambers', 0)}", className="text-danger"),
                        html.P("Echo Chambers", className="text-muted")
                    ])
                ])
            ], width=3)
        ])
    
    def create_bias_analysis_figure(self):
        """Create bias analysis figure."""
        # This would show bias patterns in convergence
        # For now, return empty figure
        return self.get_empty_figure("Bias Analysis")
    
    def create_echo_chambers_figure(self):
        """Create echo chambers figure."""
        # Get echo chambers from network
        echo_chambers = self.simulation_controller.network.get_echo_chambers()
        
        if not echo_chambers:
            return self.get_empty_figure("Echo Chambers")
        
        # Create bar chart of echo chamber sizes
        chamber_sizes = [len(chamber) for chamber in echo_chambers]
        
        fig = go.Figure(data=[
            go.Bar(x=list(range(len(chamber_sizes))), y=chamber_sizes)
        ])
        
        fig.update_layout(
            title="Echo Chamber Sizes",
            xaxis_title="Echo Chamber ID",
            yaxis_title="Number of Agents",
            showlegend=False
        )
        
        return fig
    
    def get_empty_figure(self, title: str):
        """Create an empty figure with title."""
        fig = go.Figure()
        fig.update_layout(
            title=title,
            xaxis_title="",
            yaxis_title="",
            annotations=[
                dict(
                    text="No data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False
                )
            ]
        )
        return fig
    
    def get_empty_figures(self):
        """Get empty figures for all plots."""
        empty_fig = self.get_empty_figure("No Data")
        empty_metrics = self.create_metrics_display({})
        
        return empty_fig, empty_fig, empty_fig, empty_metrics, empty_fig, empty_fig
    
    def run(self, host: str = "localhost", port: int = 8050, debug: bool = False):
        """
        Run the dashboard.
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        print(f"Starting dashboard at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)
    
    def start_simulation_thread(self):
        """Start simulation in a separate thread."""
        def run_simulation():
            self.simulation_controller.run_simulation(progress_bar=False)
        
        thread = threading.Thread(target=run_simulation)
        thread.daemon = True
        thread.start()
        return thread 