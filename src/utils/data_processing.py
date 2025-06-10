"""
Data Processing Utilities

This module provides utility functions for data processing and validation.
"""

import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO


def load_csv_data(uploaded_file):
    """
    Load and validate CSV data from uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        pd.DataFrame or None: Loaded dataframe or None if error
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None


def validate_markov_data(df, state_col, next_state_col):
    """
    Validate data for Markov chain analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe
        state_col (str): Current state column name
        next_state_col (str): Next state column name
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    if state_col not in df.columns:
        return False, f"Column '{state_col}' not found in data"
    
    if next_state_col not in df.columns:
        return False, f"Column '{next_state_col}' not found in data"
    
    if df[state_col].isnull().any():
        return False, f"Column '{state_col}' contains null values"
    
    if df[next_state_col].isnull().any():
        return False, f"Column '{next_state_col}' contains null values"
    
    return True, "Data is valid"


def validate_hmm_data(df, hidden_state_col, observed_event_col):
    """
    Validate data for Hidden Markov Model analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe
        hidden_state_col (str): Hidden state column name
        observed_event_col (str): Observed event column name
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    if hidden_state_col not in df.columns:
        return False, f"Column '{hidden_state_col}' not found in data"
    
    if observed_event_col not in df.columns:
        return False, f"Column '{observed_event_col}' not found in data"
    
    if df[hidden_state_col].isnull().any():
        return False, f"Column '{hidden_state_col}' contains null values"
    
    if df[observed_event_col].isnull().any():
        return False, f"Column '{observed_event_col}' contains null values"
    
    return True, "Data is valid"


def validate_queue_data(df, arrival_time_col, service_time_col):
    """
    Validate data for queuing theory analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe
        arrival_time_col (str): Arrival time column name
        service_time_col (str): Service time column name
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    if arrival_time_col not in df.columns:
        return False, f"Column '{arrival_time_col}' not found in data"
    
    if service_time_col not in df.columns:
        return False, f"Column '{service_time_col}' not found in data"
    
    if df[arrival_time_col].isnull().any():
        return False, f"Column '{arrival_time_col}' contains null values"
    
    if df[service_time_col].isnull().any():
        return False, f"Column '{service_time_col}' contains null values"
    
    # Check for non-negative values
    if (df[arrival_time_col] < 0).any():
        return False, f"Column '{arrival_time_col}' contains negative values"
    
    if (df[service_time_col] <= 0).any():
        return False, f"Column '{service_time_col}' contains non-positive values"
    
    return True, "Data is valid"


def create_example_markov_data():
    """
    Create example Markov chain data.
    
    Returns:
        pd.DataFrame: Example dataframe
    """
    example_data = {
        'current_state': ['Sunny', 'Sunny', 'Rainy', 'Rainy', 'Sunny', 'Rainy', 'Sunny', 'Sunny', 'Rainy', 'Sunny'],
        'next_state': ['Sunny', 'Rainy', 'Rainy', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Rainy', 'Sunny', 'Sunny']
    }
    return pd.DataFrame(example_data)


def create_example_hmm_data():
    """
    Create example Hidden Markov Model data.
    
    Returns:
        pd.DataFrame: Example dataframe
    """
    example_data = {
        'hidden_state': ['Sunny', 'Sunny','Rainy', 'Rainy', 'Sunny', 'Rainy', 'Sunny', 'Sunny', 'Rainy', 'Sunny'],
        'observed_event': ['Dry', 'Dry', 'Wet', 'Wet', 'Dry', 'Wet', 'Dry', 'Dry', 'Wet', 'Dry']
    }
    return pd.DataFrame(example_data)


def create_example_queue_data():
    """
    Create example queuing system data.
    
    Returns:
        pd.DataFrame: Example dataframe
    """
    np.random.seed(42)  # For reproducible results
    n_customers = 50
    
    # Generate inter-arrival times (exponential distribution)
    inter_arrival_times = np.random.exponential(2.0, n_customers)
    arrival_times = np.cumsum(inter_arrival_times)
    
    # Generate service times (exponential distribution)
    service_times = np.random.exponential(1.5, n_customers)
    
    example_data = {
        'arrival_time_minutes': arrival_times,
        'service_time_minutes': service_times
    }
    return pd.DataFrame(example_data)


def format_matrix_display(matrix, precision=3):
    """
    Format matrix for better display in Streamlit.
    
    Args:
        matrix (pd.DataFrame): Matrix to format
        precision (int): Number of decimal places
    
    Returns:
        pd.DataFrame: Formatted matrix
    """
    return matrix.round(precision)


def get_data_summary(df):
    """
    Get summary statistics for a dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    # Add numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    return summary
