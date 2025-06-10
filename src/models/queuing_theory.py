"""
Queuing Theory Analysis Module

This module provides functions for analyzing queuing systems including:
- M/M/s queue analysis
- Performance metrics calculation
- System stability analysis
"""

import pandas as pd
import numpy as np
from math import factorial


def queuing_theory_analysis(df, arrival_time_col='arrival_time_minutes', service_time_col='service_time_minutes', servers=1):
    """
    Analyzes a queuing system (M/M/s) from the given dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe with arrival and service times
        arrival_time_col (str): Column name for arrival times
        service_time_col (str): Column name for service times
        servers (int): Number of servers in the system
    
    Returns:
        dict: Analysis results including rates, utilization, and performance metrics
    """
    if df.empty:
        return None

    if arrival_time_col not in df.columns or service_time_col not in df.columns:
        return None

    # 1. Calculate Arrival Rate (lambda)
    arrival_times = df[arrival_time_col].sort_values().reset_index(drop=True)
    inter_arrival_times = arrival_times.diff().dropna()
    mean_inter_arrival_time = inter_arrival_times.mean()
    arrival_rate = 1 / mean_inter_arrival_time if mean_inter_arrival_time > 0 else 0

    # Calculate variance of inter-arrival times for coefficient of variation
    var_inter_arrival_time = inter_arrival_times.var() if len(inter_arrival_times) > 1 else 0
    cv_inter_arrival = np.sqrt(var_inter_arrival_time) / mean_inter_arrival_time if mean_inter_arrival_time > 0 else 0

    # 2. Calculate Service Rate (mu)
    mean_service_time = df[service_time_col].mean()
    service_rate = 1 / mean_service_time if mean_service_time > 0 else 0

    # Calculate variance of service times for coefficient of variation
    var_service_time = df[service_time_col].var() if len(df) > 1 else 0
    cv_service = np.sqrt(var_service_time) / mean_service_time if mean_service_time > 0 else 0

    # 3. Calculate System Metrics for M/M/s
    if service_rate <= 0 or arrival_rate <= 0:
        return None

    # Traffic intensity (rho)
    rho = arrival_rate / (servers * service_rate)

    if rho >= 1:  # System is unstable
        return {
            'arrival_rate': arrival_rate,
            'service_rate': service_rate,
            'utilization': rho,
            'servers': servers,
            'mean_inter_arrival_time': mean_inter_arrival_time,
            'mean_service_time': mean_service_time,
            'cv_inter_arrival': cv_inter_arrival,
            'cv_service': cv_service,
            'stable': False
        }

    # Calculate P0 (probability of zero customers)
    sum_term = sum([(servers * rho)**n / factorial(n) for n in range(servers)])
    last_term = (servers * rho)**servers / (factorial(servers) * (1 - rho))
    p0 = 1 / (sum_term + last_term)

    # Calculate Lq (average customers in queue)
    lq = (p0 * (servers * rho)**servers * rho) / (factorial(servers) * (1 - rho)**2)

    # Calculate L (average customers in system)
    l = lq + arrival_rate / service_rate

    # Calculate Wq (average waiting time in queue)
    wq = lq / arrival_rate if arrival_rate > 0 else 0

    # Calculate W (average time in system)
    w = l / arrival_rate if arrival_rate > 0 else 0

    # Calculate probability of waiting (Pw)
    pw = (p0 * (servers * rho)**servers) / (factorial(servers) * (1 - rho))

    # Calculate server utilization
    server_utilization = arrival_rate / (servers * service_rate)

    # Calculate probability distribution for number of customers
    prob_distribution = {}
    for n in range(min(50, servers + 20)):  # Calculate first 50 or servers+20 probabilities
        if n < servers:
            prob_distribution[n] = p0 * (servers * rho)**n / factorial(n)
        else:
            prob_distribution[n] = p0 * (servers * rho)**servers * rho**(n - servers) / factorial(servers)

    return {
        'arrival_rate': arrival_rate,
        'service_rate': service_rate,
        'servers': servers,
        'utilization': rho,
        'server_utilization': server_utilization,
        'mean_inter_arrival_time': mean_inter_arrival_time,
        'mean_service_time': mean_service_time,
        'cv_inter_arrival': cv_inter_arrival,
        'cv_service': cv_service,
        'stable': True,
        'p0': p0,
        'lq': lq,
        'l': l,
        'wq': wq,
        'w': w,
        'pw': pw,
        'prob_distribution': prob_distribution
    }
