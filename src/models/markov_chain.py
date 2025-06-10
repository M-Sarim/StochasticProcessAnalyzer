"""
Markov Chain Analysis Module

This module provides functions for analyzing Markov chains including:
- Transition matrix calculation
- Steady-state probabilities
- Average passage times
- Recurrence times
- State properties
"""

import pandas as pd
import numpy as np
import networkx as nx


def markov_chain_analysis(df, state_col='current_state', next_state_col='next_state'):
    """
    Analyzes a Markov Chain from the given dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe with state transitions
        state_col (str): Column name for current states
        next_state_col (str): Column name for next states
    
    Returns:
        dict: Analysis results including transition matrix, steady states, etc.
    """
    if df.empty:
        return None

    if state_col not in df.columns or next_state_col not in df.columns:
        return None

    # 1. Identify States
    states = sorted(list(set(df[state_col]) | set(df[next_state_col])))
    num_states = len(states)

    if num_states == 0:
        return None

    # 2. Create Transition Matrix
    transition_matrix = pd.DataFrame(0, index=states, columns=states)
    for _, row in df.iterrows():
        current_state = row[state_col]
        next_state = row[next_state_col]
        transition_matrix.loc[current_state, next_state] += 1

    # Convert counts to probabilities
    for state in states:
        row_sum = transition_matrix.loc[state].sum()
        if row_sum > 0:
            transition_matrix.loc[state] = transition_matrix.loc[state] / row_sum
        else:
             transition_matrix.loc[state] = 0  # handle absorbing states

    # 3. Steady-State Probabilities
    try:
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        closest_to_one_idx = np.argmin(np.abs(eigenvalues - 1))
        steady_state_eigenvector = eigenvectors[:, closest_to_one_idx]
        steady_state_probs = np.abs(steady_state_eigenvector) / np.sum(np.abs(steady_state_eigenvector))
        steady_state_probs_df = pd.DataFrame(steady_state_probs, index=states, columns=['Probability'])
    except np.linalg.LinAlgError:
        steady_state_probs_df = pd.DataFrame(np.nan, index=states, columns=['Probability'])

    # 4. Average Passage Time (Mean First Passage Time)
    passage_times = pd.DataFrame(np.inf, index=states, columns=states)
    for i, origin_state in enumerate(states):
        for j, destination_state in enumerate(states):
            if origin_state == destination_state:
                passage_times.loc[origin_state, destination_state] = 0
            else:
                # solve system of equations
                A = transition_matrix.copy()
                A.loc[destination_state, :] = 0
                A.loc[destination_state, destination_state] = 1
                b = -np.ones(len(states))
                b[states.index(destination_state)] = 0
                try:
                    passage_times.loc[origin_state, destination_state] = np.linalg.solve(A,b)[states.index(origin_state)]
                except np.linalg.LinAlgError:
                     passage_times.loc[origin_state, destination_state] = np.inf

    # 5. Average Recurrence Time
    recurrence_times = pd.Series(np.inf, index=states)
    for state in states:
        if steady_state_probs_df.loc[state, 'Probability'] > 0:
            recurrence_times[state] = 1 / steady_state_probs_df.loc[state, 'Probability']
        else:
            recurrence_times[state] = np.inf

    # 6. Calculate Expected Number of Steps for Absorption
    # Only applicable if there are absorbing states
    absorbing_states = [state for state in states if transition_matrix.loc[state, state] == 1 and
                        transition_matrix.loc[state].sum() == 1]

    absorption_steps = {}
    if absorbing_states:
        transient_states = [state for state in states if state not in absorbing_states]
        if transient_states:
            # Create Q matrix (transitions between transient states)
            Q = transition_matrix.loc[transient_states, transient_states]
            # Calculate N = (I-Q)^(-1)
            I = np.identity(len(transient_states))
            try:
                N = np.linalg.inv(I - Q.values)
                # Expected steps to absorption for each transient state
                absorption_steps = {state: N[i, :].sum() for i, state in enumerate(transient_states)}
            except np.linalg.LinAlgError:
                absorption_steps = {state: np.inf for state in transient_states}

    # 7. Calculate state transition frequencies
    state_counts = df[state_col].value_counts().to_dict()
    total_transitions = len(df)
    state_frequencies = {state: count/total_transitions for state, count in state_counts.items()}

    # 8. Calculate period of each state if possible
    periods = {}
    G = nx.DiGraph()
    for i, row in transition_matrix.iterrows():
        for j, prob in enumerate(row):
            if prob > 0:
                G.add_edge(i, states[j])

    for state in states:
        try:
            # Check if state is part of a cycle
            if nx.has_path(G, state, state):
                cycles = []
                for path in nx.all_simple_paths(G, state, state):
                    if len(path) > 1:  # Ignore self-loops
                        cycles.append(len(path))
                if cycles:
                    periods[state] = np.gcd.reduce(cycles)
                else:
                    periods[state] = 1 if transition_matrix.loc[state, state] > 0 else float('inf')
            else:
                periods[state] = float('inf')
        except nx.NetworkXNoPath:
            periods[state] = float('inf')

    return {
        'transition_matrix': transition_matrix,
        'steady_state_probabilities': steady_state_probs_df,
        'average_passage_times': passage_times,
        'average_recurrence_times': recurrence_times,
        'states': states,
        'absorption_steps': absorption_steps,
        'absorbing_states': absorbing_states,
        'state_frequencies': state_frequencies,
        'periods': periods
    }
