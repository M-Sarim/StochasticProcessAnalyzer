"""
Charts and Plotting Module

This module provides functions for creating various charts and visualizations
for stochastic process analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap


def plot_heatmap(matrix, title="Heatmap", figsize=(8, 6)):
    """
    Create a heatmap visualization of a matrix.
    
    Args:
        matrix (pd.DataFrame): Matrix to visualize
        title (str): Title for the plot
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(matrix, annot=True, cmap='Blues', fmt='.3f', ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_network_graph(transition_matrix, title="Network Graph", figsize=(10, 8)):
    """
    Create a network graph visualization of state transitions.
    
    Args:
        transition_matrix (pd.DataFrame): Transition matrix
        title (str): Title for the plot
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for state in transition_matrix.index:
        G.add_node(state)
    
    # Add edges with weights
    for i, row in transition_matrix.iterrows():
        for j, prob in enumerate(row):
            if prob > 0.01:  # Only show significant transitions
                G.add_edge(i, transition_matrix.columns[j], weight=prob)
    
    # Position nodes
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, ax=ax)
    
    # Draw edges with varying thickness based on probability
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w*5 for w in weights],
                          alpha=0.6, edge_color='gray', arrows=True, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    # Draw edge labels
    edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
    
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    return fig


def plot_state_probabilities(probabilities, title="State Probabilities", figsize=(8, 6)):
    """
    Create a bar plot of state probabilities.
    
    Args:
        probabilities (pd.Series or pd.DataFrame): State probabilities
        title (str): Title for the plot
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(probabilities, pd.DataFrame):
        probabilities = probabilities.iloc[:, 0]
    
    bars = ax.bar(range(len(probabilities)), probabilities.values, 
                  color='skyblue', alpha=0.7)
    ax.set_xlabel('States')
    ax.set_ylabel('Probability')
    ax.set_title(title)
    ax.set_xticks(range(len(probabilities)))
    ax.set_xticklabels(probabilities.index, rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, probabilities.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def plot_time_series(data, title="Time Series", xlabel="Time", ylabel="Value", figsize=(10, 6)):
    """
    Create a time series plot.
    
    Args:
        data (list or pd.Series): Time series data
        title (str): Title for the plot
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(data, list):
        ax.plot(range(len(data)), data, 'o-', linewidth=2, markersize=4)
    else:
        ax.plot(data.index, data.values, 'o-', linewidth=2, markersize=4)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_distribution(data, title="Distribution", bins=30, figsize=(8, 6)):
    """
    Create a histogram of data distribution.
    
    Args:
        data (list or pd.Series): Data to plot
        title (str): Title for the plot
        bins (int): Number of histogram bins
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax.axvline(mean_val, color='red', linestyle='--', 
               label=f'Mean: {mean_val:.2f}')
    ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7,
               label=f'±1 Std: {std_val:.2f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    return fig


def create_plotly_heatmap(matrix, title="Interactive Heatmap"):
    """
    Create an interactive heatmap using Plotly.
    
    Args:
        matrix (pd.DataFrame): Matrix to visualize
        title (str): Title for the plot
    
    Returns:
        plotly.graph_objects.Figure: The created figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale='Blues',
        text=matrix.values,
        texttemplate="%{text:.3f}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="To State",
        yaxis_title="From State"
    )
    
    return fig
