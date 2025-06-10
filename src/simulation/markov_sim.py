"""
Markov Chain Simulation Module

This module provides functions for simulating Markov chains.
"""

import numpy as np
import pandas as pd


def simulate_markov_chain(transition_matrix, steps=50, start_state=None):
    """
    Simulates a Markov chain for the given number of steps.
    
    Args:
        transition_matrix (pd.DataFrame): Transition probability matrix
        steps (int): Number of simulation steps
        start_state (str): Starting state (random if None)
    
    Returns:
        list: Sequence of states
    """
    states = transition_matrix.index.tolist()

    if start_state is None:
        # Start with a random state
        start_state = np.random.choice(states)

    if start_state not in states:
        return None

    # Initialize the simulation
    current_state = start_state
    state_sequence = [current_state]

    # Simulate the chain
    for _ in range(steps - 1):
        # Get transition probabilities for current state
        transition_probs = transition_matrix.loc[current_state].values

        # Choose next state based on transition probabilities
        next_state = np.random.choice(states, p=transition_probs)
        state_sequence.append(next_state)
        current_state = next_state

    return state_sequence


def simulate_multiple_chains(transition_matrix, num_simulations=100, steps=50, start_state=None):
    """
    Simulates multiple Markov chains and returns statistics.
    
    Args:
        transition_matrix (pd.DataFrame): Transition probability matrix
        num_simulations (int): Number of simulation runs
        steps (int): Number of steps per simulation
        start_state (str): Starting state (random if None)
    
    Returns:
        dict: Simulation results and statistics
    """
    states = transition_matrix.index.tolist()
    all_sequences = []
    state_frequencies = {state: [] for state in states}
    
    for _ in range(num_simulations):
        sequence = simulate_markov_chain(transition_matrix, steps, start_state)
        if sequence:
            all_sequences.append(sequence)
            
            # Count state frequencies for this simulation
            for state in states:
                frequency = sequence.count(state) / len(sequence)
                state_frequencies[state].append(frequency)
    
    # Calculate statistics
    avg_frequencies = {}
    std_frequencies = {}
    
    for state in states:
        if state_frequencies[state]:
            avg_frequencies[state] = np.mean(state_frequencies[state])
            std_frequencies[state] = np.std(state_frequencies[state])
        else:
            avg_frequencies[state] = 0
            std_frequencies[state] = 0
    
    return {
        'sequences': all_sequences,
        'average_frequencies': avg_frequencies,
        'frequency_std': std_frequencies,
        'num_simulations': num_simulations,
        'steps_per_simulation': steps
    }


def analyze_convergence(transition_matrix, steps=1000, start_state=None):
    """
    Analyzes how quickly the chain converges to steady state.
    
    Args:
        transition_matrix (pd.DataFrame): Transition probability matrix
        steps (int): Number of simulation steps
        start_state (str): Starting state (random if None)
    
    Returns:
        dict: Convergence analysis results
    """
    states = transition_matrix.index.tolist()
    sequence = simulate_markov_chain(transition_matrix, steps, start_state)
    
    if not sequence:
        return None
    
    # Calculate running frequencies
    running_frequencies = {state: [] for state in states}
    
    for i in range(1, len(sequence) + 1):
        partial_sequence = sequence[:i]
        for state in states:
            frequency = partial_sequence.count(state) / len(partial_sequence)
            running_frequencies[state].append(frequency)
    
    # Calculate theoretical steady state (if possible)
    try:
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        closest_to_one_idx = np.argmin(np.abs(eigenvalues - 1))
        steady_state_eigenvector = eigenvectors[:, closest_to_one_idx]
        steady_state_probs = np.abs(steady_state_eigenvector) / np.sum(np.abs(steady_state_eigenvector))
        theoretical_steady_state = dict(zip(states, steady_state_probs))
    except:
        theoretical_steady_state = None
    
    return {
        'sequence': sequence,
        'running_frequencies': running_frequencies,
        'theoretical_steady_state': theoretical_steady_state,
        'final_frequencies': {state: running_frequencies[state][-1] for state in states}
    }


def simulate_absorption_time(transition_matrix, absorbing_states, start_state=None, max_steps=10000):
    """
    Simulates time to absorption for absorbing Markov chains.
    
    Args:
        transition_matrix (pd.DataFrame): Transition probability matrix
        absorbing_states (list): List of absorbing states
        start_state (str): Starting state (random transient state if None)
        max_steps (int): Maximum simulation steps
    
    Returns:
        dict: Absorption time results
    """
    states = transition_matrix.index.tolist()
    transient_states = [s for s in states if s not in absorbing_states]
    
    if start_state is None and transient_states:
        start_state = np.random.choice(transient_states)
    elif start_state in absorbing_states:
        return {'absorption_time': 0, 'absorbed_state': start_state, 'sequence': [start_state]}
    
    current_state = start_state
    sequence = [current_state]
    
    for step in range(max_steps):
        if current_state in absorbing_states:
            return {
                'absorption_time': step,
                'absorbed_state': current_state,
                'sequence': sequence
            }
        
        # Get transition probabilities
        transition_probs = transition_matrix.loc[current_state].values
        next_state = np.random.choice(states, p=transition_probs)
        sequence.append(next_state)
        current_state = next_state
    
    # If we reach here, absorption didn't occur within max_steps
    return {
        'absorption_time': max_steps,
        'absorbed_state': None,
        'sequence': sequence
    }
