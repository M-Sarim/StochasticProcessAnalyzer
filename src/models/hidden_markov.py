"""
Hidden Markov Model Analysis Module

This module provides functions for analyzing Hidden Markov Models including:
- Transition and emission probability calculation
- Forward algorithm
- Viterbi algorithm
- Steady-state analysis
"""

import pandas as pd
import numpy as np


def hidden_markov_model_analysis(df, hidden_state_col='hidden_state', observed_event_col='observed_event'):
    """
    Analyzes a Hidden Markov Model from the given dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe with hidden states and observed events
        hidden_state_col (str): Column name for hidden states
        observed_event_col (str): Column name for observed events
    
    Returns:
        dict: Analysis results including transition/emission probabilities, algorithms
    """
    if df.empty:
        return None

    if hidden_state_col not in df.columns or observed_event_col not in df.columns:
        return None

    # 1. Identify Hidden States and Observed Events
    hidden_states = sorted(list(set(df[hidden_state_col])))
    observed_events = sorted(list(set(df[observed_event_col])))
    num_hidden_states = len(hidden_states)
    num_observed_events = len(observed_events)

    if num_hidden_states == 0 or num_observed_events == 0:
        return None

    # 2. Calculate Transition Probabilities (P(S_t+1 | S_t))
    transition_counts = pd.DataFrame(0, index=hidden_states, columns=hidden_states)
    for i in range(len(df) - 1):
        current_state = df.loc[i, hidden_state_col]
        next_state = df.loc[i + 1, hidden_state_col]
        transition_counts.loc[current_state, next_state] += 1
    transition_probabilities = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)

    # 3. Calculate Emission Probabilities (P(O_t | S_t))
    emission_counts = pd.DataFrame(0, index=hidden_states, columns=observed_events)
    for _, row in df.iterrows():
        state = row[hidden_state_col]
        event = row[observed_event_col]
        emission_counts.loc[state, event] += 1
    emission_probabilities = emission_counts.div(emission_counts.sum(axis=1), axis=0).fillna(0)

    # 4. Steady-State Probabilities of Hidden States
    try:
        eigenvalues, eigenvectors = np.linalg.eig(transition_probabilities.T)
        closest_to_one_idx = np.argmin(np.abs(eigenvalues - 1))
        steady_state_eigenvector = eigenvectors[:, closest_to_one_idx]
        steady_state_probs = np.abs(steady_state_eigenvector) / np.sum(np.abs(steady_state_eigenvector))
        steady_state_probs_df = pd.DataFrame(steady_state_probs, index=hidden_states, columns=['Probability'])
    except np.linalg.LinAlgError:
        steady_state_probs_df = pd.DataFrame(np.nan, index=hidden_states, columns=['Probability'])

    # 5. Prepare initial probabilities for the forward and viterbi algorithms
    initial_state_counts = df.groupby(hidden_state_col).size()
    initial_probs = initial_state_counts / initial_state_counts.sum()
    initial_probs = initial_probs.reindex(transition_probabilities.index, fill_value=0)

    # 6. Forward Algorithm
    def forward_algorithm(observations, initial_probs, transition_probs, emission_probs):
        num_states = len(transition_probs)
        T = len(observations)
        alpha = np.zeros((num_states, T))

        # For visualization: store all alphas
        all_alphas = []

        # Initialization
        for s_idx, state in enumerate(transition_probs.index):
            if observations[0] in emission_probs.columns:
                alpha[s_idx, 0] = initial_probs[state] * emission_probs.loc[state, observations[0]]
            else:
                alpha[s_idx, 0] = 0

        all_alphas.append(alpha.copy())

        # Induction
        for t in range(1, T):
            for s_idx, state in enumerate(transition_probs.index):
                if observations[t] in emission_probs.columns:
                    alpha[s_idx, t] = sum(alpha[prev_s_idx, t - 1] * transition_probs.iloc[prev_s_idx, s_idx] * emission_probs.loc[state, observations[t]]
                                            for prev_s_idx in range(num_states))
                else:
                    alpha[s_idx, t] = 0
            all_alphas.append(alpha.copy())

        # Termination
        probability_of_sequence = np.sum(alpha[:, T - 1])
        return probability_of_sequence, all_alphas

    # 7. Viterbi Algorithm
    def viterbi_algorithm(observations, initial_probs, transition_probs, emission_probs):
        num_states = len(transition_probs)
        T = len(observations)
        viterbi = np.zeros((num_states, T))
        backpointer = np.zeros((num_states, T), dtype=int)

        # For visualization
        all_viterbi_values = []
        all_backpointers = []

        # Initialization
        for s_idx, state in enumerate(transition_probs.index):
            if observations[0] in emission_probs.columns:
                viterbi[s_idx, 0] = initial_probs[state] * emission_probs.loc[state, observations[0]]
            else:
                viterbi[s_idx, 0] = 0

        all_viterbi_values.append(viterbi.copy())
        all_backpointers.append(backpointer[:, 0].copy())

        # Recursion
        for t in range(1, T):
            for s_idx, state in enumerate(transition_probs.index):
                if observations[t] in emission_probs.columns:
                    probs = [viterbi[prev_s_idx, t - 1] * transition_probs.iloc[prev_s_idx, s_idx] for prev_s_idx in range(num_states)]
                    max_val = max(probs) * emission_probs.loc[state, observations[t]] if max(probs) > 0 else 0
                    viterbi[s_idx, t] = max_val
                    backpointer[s_idx, t] = np.argmax(probs) if max(probs) > 0 else 0
                else:
                    viterbi[s_idx, t] = 0
                    backpointer[s_idx, t] = 0

            all_viterbi_values.append(viterbi.copy())
            all_backpointers.append(backpointer[:, t].copy())

        # Termination
        best_path_end = np.argmax(viterbi[:, T - 1])
        best_path = [hidden_states[best_path_end]]
        best_path_indices = [best_path_end]

        # Backtracking
        for t in range(T - 1, 0, -1):
            best_path_end = backpointer[best_path_end, t]
            best_path.insert(0, hidden_states[best_path_end])
            best_path_indices.insert(0, best_path_end)

        return best_path, all_viterbi_values, all_backpointers, best_path_indices

    # 8. Additional - Joint probability table
    joint_probability = pd.DataFrame(0, index=hidden_states, columns=observed_events)
    for state in hidden_states:
        state_prob = steady_state_probs_df.loc[state, 'Probability']
        for event in observed_events:
            if state in emission_probabilities.index and event in emission_probabilities.columns:
                joint_probability.loc[state, event] = state_prob * emission_probabilities.loc[state, event]

    # 9. Calculate observation probabilities
    observation_probabilities = {}
    for event in observed_events:
        prob = 0
        for state in hidden_states:
            if state in emission_probabilities.index and event in emission_probabilities.columns:
                prob += steady_state_probs_df.loc[state, 'Probability'] * emission_probabilities.loc[state, event]
        observation_probabilities[event] = prob

    return {
        'transition_probabilities': transition_probabilities,
        'emission_probabilities': emission_probabilities,
        'steady_state_probabilities': steady_state_probs_df,
        'forward_algorithm': forward_algorithm,
        'viterbi_algorithm': viterbi_algorithm,
        'hidden_states': hidden_states,
        'observed_events': observed_events,
        'initial_probs': initial_probs,
        'joint_probability': joint_probability,
        'observation_probabilities': observation_probabilities
    }
