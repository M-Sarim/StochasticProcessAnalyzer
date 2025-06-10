"""
Diagrams Module

This module provides functions for creating graphviz diagrams
for stochastic process visualization.
"""

import pandas as pd


def create_diagram(transition_matrix, highlight_state=None):
    """
    Creates a dictionary representation of the Markov chain for use with graphviz.
    Optionally highlights a specific state.
    
    Args:
        transition_matrix (pd.DataFrame): Transition probability matrix
        highlight_state (str): State to highlight (optional)
    
    Returns:
        str: Graphviz DOT notation string
    """
    dot_graph = "digraph MarkovChain {\n"
    dot_graph += "  rankdir=LR;\n"
    dot_graph += "  bgcolor=\"transparent\";\n"
    dot_graph += "  node [shape=circle, style=filled, fontname=Helvetica];\n"
    dot_graph += "  edge [fontname=Helvetica, fontsize=10];\n"

    for state in transition_matrix.index:
        if highlight_state and state == highlight_state:
            dot_graph += f'  "{state}" [label="{state}", fillcolor="#ff7f0e", fontcolor="white"];\n'
        else:
            dot_graph += f'  "{state}" [label="{state}", fillcolor="lightblue"];\n'

    for i, row in transition_matrix.iterrows():
        for j, prob in enumerate(row):
            if prob > 0.01:  # Only show transitions with probability > 1%
                color = "red" if prob > 0.5 else "blue" if prob > 0.2 else "gray"
                dot_graph += f'  "{i}" -> "{transition_matrix.columns[j]}" [label="{prob:.2f}", color="{color}"];\n'

    dot_graph += "}"
    return dot_graph


def create_hmm_diagram(transition_probs, emission_probs):
    """
    Creates a graphviz diagram for Hidden Markov Model.
    
    Args:
        transition_probs (pd.DataFrame): Hidden state transition probabilities
        emission_probs (pd.DataFrame): Emission probabilities
    
    Returns:
        str: Graphviz DOT notation string
    """
    dot_graph = "digraph HMM {\n"
    dot_graph += "  rankdir=LR;\n"
    dot_graph += "  bgcolor=\"transparent\";\n"
    dot_graph += "  node [fontname=Helvetica];\n"
    dot_graph += "  edge [fontname=Helvetica, fontsize=9];\n"
    
    # Subgraph for hidden states
    dot_graph += "  subgraph cluster_hidden {\n"
    dot_graph += "    label=\"Hidden States\";\n"
    dot_graph += "    style=dashed;\n"
    dot_graph += "    color=blue;\n"
    
    for state in transition_probs.index:
        dot_graph += f'    "{state}" [shape=circle, style=filled, fillcolor="lightblue"];\n'
    
    # Add transitions between hidden states
    for i, row in transition_probs.iterrows():
        for j, prob in enumerate(row):
            if prob > 0.01:
                dot_graph += f'    "{i}" -> "{transition_probs.columns[j]}" [label="{prob:.2f}"];\n'
    
    dot_graph += "  }\n"
    
    # Subgraph for observed events
    dot_graph += "  subgraph cluster_observed {\n"
    dot_graph += "    label=\"Observed Events\";\n"
    dot_graph += "    style=dashed;\n"
    dot_graph += "    color=green;\n"
    
    for event in emission_probs.columns:
        dot_graph += f'    "{event}_obs" [label="{event}", shape=box, style=filled, fillcolor="lightgreen"];\n'
    
    dot_graph += "  }\n"
    
    # Add emission edges
    for state in emission_probs.index:
        for event in emission_probs.columns:
            prob = emission_probs.loc[state, event]
            if prob > 0.01:
                dot_graph += f'  "{state}" -> "{event}_obs" [label="{prob:.2f}", style=dashed, color=green];\n'
    
    dot_graph += "}"
    return dot_graph


def create_queue_diagram(servers=1, queue_capacity=None):
    """
    Creates a graphviz diagram for a queuing system.
    
    Args:
        servers (int): Number of servers
        queue_capacity (int): Queue capacity (None for infinite)
    
    Returns:
        str: Graphviz DOT notation string
    """
    dot_graph = "digraph Queue {\n"
    dot_graph += "  rankdir=LR;\n"
    dot_graph += "  bgcolor=\"transparent\";\n"
    dot_graph += "  node [fontname=Helvetica];\n"
    dot_graph += "  edge [fontname=Helvetica];\n"
    
    # Arrival process
    dot_graph += '  "Arrivals" [shape=box, style=filled, fillcolor="yellow", label="Arrivals\\n(λ)"];\n'
    
    # Queue
    if queue_capacity:
        queue_label = f"Queue\\n(Capacity: {queue_capacity})"
    else:
        queue_label = "Queue\\n(Infinite)"
    dot_graph += f'  "Queue" [shape=box, style=filled, fillcolor="orange", label="{queue_label}"];\n'
    
    # Servers
    for i in range(servers):
        dot_graph += f'  "Server{i+1}" [shape=circle, style=filled, fillcolor="lightblue", label="Server {i+1}\\n(μ)"];\n'
    
    # Departures
    dot_graph += '  "Departures" [shape=box, style=filled, fillcolor="lightgreen", label="Departures"];\n'
    
    # Connections
    dot_graph += '  "Arrivals" -> "Queue";\n'
    
    for i in range(servers):
        dot_graph += f'  "Queue" -> "Server{i+1}";\n'
        dot_graph += f'  "Server{i+1}" -> "Departures";\n'
    
    dot_graph += "}"
    return dot_graph


def create_state_transition_diagram(states, transitions, title="State Transition Diagram"):
    """
    Creates a general state transition diagram.
    
    Args:
        states (list): List of state names
        transitions (list): List of tuples (from_state, to_state, probability)
        title (str): Diagram title
    
    Returns:
        str: Graphviz DOT notation string
    """
    dot_graph = f"digraph \"{title}\" {{\n"
    dot_graph += "  rankdir=LR;\n"
    dot_graph += "  bgcolor=\"transparent\";\n"
    dot_graph += "  node [shape=circle, style=filled, fontname=Helvetica];\n"
    dot_graph += "  edge [fontname=Helvetica, fontsize=10];\n"
    dot_graph += f"  label=\"{title}\";\n"
    dot_graph += "  labelloc=t;\n"
    
    # Add states
    for state in states:
        dot_graph += f'  "{state}" [fillcolor="lightblue"];\n'
    
    # Add transitions
    for from_state, to_state, prob in transitions:
        if prob > 0.01:
            color = "red" if prob > 0.5 else "blue" if prob > 0.2 else "gray"
            dot_graph += f'  "{from_state}" -> "{to_state}" [label="{prob:.2f}", color="{color}"];\n'
    
    dot_graph += "}"
    return dot_graph
