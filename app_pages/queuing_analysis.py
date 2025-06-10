"""
Queuing Theory Analysis Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from math import factorial


def queuing_theory_analysis(df, arrival_time_col='arrival_time_minutes', service_time_col='service_time_minutes', servers=1):
    """
    Analyzes a queuing system (M/M/s) from the given dataframe.
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
    lq = p0 * (servers * rho)**servers * rho / (factorial(servers) * (1 - rho)**2)

    # Calculate L (average customers in system)
    l = lq + servers * rho

    # Calculate Wq (average time in queue)
    wq = lq / arrival_rate

    # Calculate W (average time in system)
    w = wq + 1/service_rate

    # Calculate probability of waiting
    p_wait = p0 * (servers * rho)**servers / (factorial(servers) * (1 - rho))

    # Calculate probability of n or more customers
    def p_n_or_more(n):
        if n < servers:
            return sum([p0 * (servers * rho)**k / factorial(k) for k in range(n, servers)]) + \
                   p0 * (servers * rho)**servers / (factorial(servers) * (1 - rho))
        else:
            return p0 * (servers * rho)**servers * rho**(n-servers) / (factorial(servers) * (1 - rho))

    # Calculate probabilities for different queue lengths
    queue_probs = {i: p_n_or_more(i) - p_n_or_more(i+1) for i in range(10)}

    # Calculate server utilization (different from traffic intensity for multi-server)
    utilization = rho

    return {
        'arrival_rate': arrival_rate,
        'service_rate': service_rate,
        'utilization': utilization,
        'servers': servers,
        'probability_of_zero_customers': p0,
        'average_customers_in_system': l,
        'average_customers_in_queue': lq,
        'average_time_in_system': w,
        'average_time_in_queue': wq,
        'probability_of_waiting': p_wait,
        'queue_length_probabilities': queue_probs,
        'mean_inter_arrival_time': mean_inter_arrival_time,
        'mean_service_time': mean_service_time,
        'cv_inter_arrival': cv_inter_arrival,
        'cv_service': cv_service,
        'stable': True
    }


def plot_queuing_metrics(queuing_data):
    """
    Creates visualizations for queuing theory metrics.
    """
    primary_color = "#1f77b4"
    secondary_color = "#ff7f0e"
    accent_color = "#2ca02c"
    dark_color = "#2c3e50"

    if not queuing_data['stable']:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "System is unstable (utilization ≥ 1)",
                ha='center', va='center', fontsize=14, color='red')
        ax.axis('off')
        return fig

    # Create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Queue length distribution
    probs = queuing_data['queue_length_probabilities']
    keys = list(probs.keys())
    values = list(probs.values())

    axs[0, 0].bar(keys, values, color=primary_color)
    axs[0, 0].set_title('Queue Length Distribution')
    axs[0, 0].set_xlabel('Number of Customers')
    axs[0, 0].set_ylabel('Probability')
    axs[0, 0].grid(True, alpha=0.3)

    # Plot 2: System metrics
    metrics = ['probability_of_zero_customers', 'probability_of_waiting', 'utilization']
    metric_names = ['P(0)', 'P(wait)', 'Utilization']
    metric_values = [queuing_data[m] for m in metrics]

    axs[0, 1].bar(metric_names, metric_values, color=secondary_color)
    axs[0, 1].set_title('System Metrics')
    axs[0, 1].set_ylabel('Probability')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].set_ylim([0, 1])

    # Plot 3: Average number of customers
    l_metrics = ['average_customers_in_system', 'average_customers_in_queue']
    l_names = ['In System (L)', 'In Queue (Lq)']
    l_values = [queuing_data[m] for m in l_metrics]

    axs[1, 0].bar(l_names, l_values, color=accent_color)
    axs[1, 0].set_title('Average Number of Customers')
    axs[1, 0].grid(True, alpha=0.3)

    # Plot 4: Average waiting times
    w_metrics = ['average_time_in_system', 'average_time_in_queue']
    w_names = ['In System (W)', 'In Queue (Wq)']
    w_values = [queuing_data[m] for m in w_metrics]

    axs[1, 1].bar(w_names, w_values, color=dark_color)
    axs[1, 1].set_title('Average Time')
    axs[1, 1].set_ylabel('Time Units')
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def show_queuing_analysis_page():
    """Display the Queuing Theory analysis page."""
    st.title("Queuing Theory Analysis 🚶‍♂️")

    st.markdown("""
    <div class="info-text">
        <p>Analyze M/M/s queuing systems to understand performance metrics and system behavior.</p>
    </div>
    """, unsafe_allow_html=True)

    # Data upload section
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.header("Upload Data")

    # Options for data input
    data_input = st.radio("Choose data input method:", ["Upload CSV", "Use Example Data", "Manual Entry"])

    if data_input == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error: {e}")
                df = None
        else:
            df = None

    elif data_input == "Use Example Data":
        st.info("Using example Queuing data")
        # Generate example queuing data
        np.random.seed(42)
        n_customers = 100
        arrival_times = np.cumsum(np.random.exponential(2.0, n_customers))  # Mean inter-arrival time = 2 minutes
        service_times = np.random.exponential(1.5, n_customers)  # Mean service time = 1.5 minutes

        example_data = {
            'arrival_time_minutes': arrival_times,
            'service_time_minutes': service_times
        }
        df = pd.DataFrame(example_data)
        st.dataframe(df.head(10))

    elif data_input == "Manual Entry":
        st.subheader("Enter queuing data")

        # Dynamic inputs for queuing data
        num_customers = st.number_input("Number of customers:", min_value=1, value=10)

        data = {'arrival_time_minutes': [], 'service_time_minutes': []}
        for i in range(num_customers):
            cols = st.columns(2)
            with cols[0]:
                arrival = st.number_input(f"Arrival Time {i+1} (minutes)", value=float(i*2), step=0.1)
            with cols[1]:
                service = st.number_input(f"Service Time {i+1} (minutes)", value=1.5, step=0.1)

            data['arrival_time_minutes'].append(arrival)
            data['service_time_minutes'].append(service)

        df = pd.DataFrame(data)
        st.dataframe(df)

    # Column mapping and server configuration
    if df is not None and not df.empty:
        st.subheader("Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            arrival_time_col = st.selectbox("Select arrival time column:", df.columns,
                                          index=df.columns.get_loc('arrival_time_minutes') if 'arrival_time_minutes' in df.columns else 0)

        with col2:
            service_time_col = st.selectbox("Select service time column:", df.columns,
                                          index=df.columns.get_loc('service_time_minutes') if 'service_time_minutes' in df.columns else min(1, len(df.columns) - 1))

        with col3:
            servers = st.number_input("Number of servers:", min_value=1, value=1, max_value=10)

        # Analyze button
        if st.button("Analyze Queuing System"):
            with st.spinner("Analyzing..."):
                results = queuing_theory_analysis(df, arrival_time_col=arrival_time_col,
                                                service_time_col=service_time_col, servers=servers)

            if results:
                if not results['stable']:
                    st.error("⚠️ System is unstable! The utilization is ≥ 1, meaning the arrival rate exceeds the service capacity.")
                    st.write(f"**Utilization:** {results['utilization']:.3f}")
                    st.write(f"**Arrival Rate:** {results['arrival_rate']:.3f} customers/minute")
                    st.write(f"**Service Rate:** {results['service_rate']:.3f} customers/minute per server")
                    st.write(f"**Total Service Capacity:** {results['service_rate'] * servers:.3f} customers/minute")

                    st.markdown("""
                    **Recommendations:**
                    - Increase the number of servers
                    - Improve service efficiency to reduce service times
                    - Implement arrival rate control mechanisms
                    """)
                else:
                    # Display results in organized tabs
                    tabs = st.tabs(["System Metrics", "Performance Analysis", "Visualizations", "Detailed Results"])

                    with tabs[0]:
                        st.subheader("Key Performance Indicators")

                        # Create metrics display
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Utilization", f"{results['utilization']:.3f}",
                                    delta=f"{'Stable' if results['utilization'] < 0.8 else 'High'}")

                        with col2:
                            st.metric("Avg. Customers in System", f"{results['average_customers_in_system']:.2f}")

                        with col3:
                            st.metric("Avg. Wait Time", f"{results['average_time_in_queue']:.2f} min")

                        with col4:
                            st.metric("Probability of Waiting", f"{results['probability_of_waiting']:.3f}")

                        # System rates
                        st.subheader("System Rates")
                        rate_col1, rate_col2, rate_col3 = st.columns(3)

                        with rate_col1:
                            st.write(f"**Arrival Rate (λ):** {results['arrival_rate']:.3f} customers/min")

                        with rate_col2:
                            st.write(f"**Service Rate (μ):** {results['service_rate']:.3f} customers/min per server")

                        with rate_col3:
                            st.write(f"**Total Capacity:** {results['service_rate'] * servers:.3f} customers/min")

                    with tabs[1]:
                        st.subheader("Performance Analysis")

                        # Performance metrics table
                        metrics_data = {
                            'Metric': [
                                'Average Customers in System (L)',
                                'Average Customers in Queue (Lq)',
                                'Average Time in System (W)',
                                'Average Time in Queue (Wq)',
                                'Probability of Zero Customers (P₀)',
                                'Probability of Waiting',
                                'Server Utilization (ρ)'
                            ],
                            'Value': [
                                f"{results['average_customers_in_system']:.4f}",
                                f"{results['average_customers_in_queue']:.4f}",
                                f"{results['average_time_in_system']:.4f} minutes",
                                f"{results['average_time_in_queue']:.4f} minutes",
                                f"{results['probability_of_zero_customers']:.4f}",
                                f"{results['probability_of_waiting']:.4f}",
                                f"{results['utilization']:.4f}"
                            ]
                        }

                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, use_container_width=True)

                        # Little's Law verification
                        st.subheader("Little's Law Verification")
                        st.write("Little's Law states that L = λW and Lq = λWq")

                        l_calculated = results['arrival_rate'] * results['average_time_in_system']
                        lq_calculated = results['arrival_rate'] * results['average_time_in_queue']

                        verification_data = {
                            'Relationship': ['L = λW', 'Lq = λWq'],
                            'Theoretical': [f"{results['average_customers_in_system']:.4f}",
                                          f"{results['average_customers_in_queue']:.4f}"],
                            'Calculated': [f"{l_calculated:.4f}", f"{lq_calculated:.4f}"],
                            'Match': ['✓' if abs(results['average_customers_in_system'] - l_calculated) < 0.001 else '✗',
                                    '✓' if abs(results['average_customers_in_queue'] - lq_calculated) < 0.001 else '✗']
                        }

                        verification_df = pd.DataFrame(verification_data)
                        st.dataframe(verification_df, use_container_width=True)

                    with tabs[2]:
                        st.subheader("System Visualizations")

                        # Plot queuing metrics
                        fig = plot_queuing_metrics(results)
                        st.pyplot(fig)

                        # Queue length probability distribution
                        st.subheader("Queue Length Distribution")
                        prob_data = results['queue_length_probabilities']

                        fig_prob = go.Figure()
                        fig_prob.add_trace(go.Bar(
                            x=list(prob_data.keys()),
                            y=list(prob_data.values()),
                            name='Probability',
                            marker_color='lightblue'
                        ))

                        fig_prob.update_layout(
                            title='Probability Distribution of Queue Length',
                            xaxis_title='Number of Customers in Queue',
                            yaxis_title='Probability',
                            showlegend=False
                        )

                        st.plotly_chart(fig_prob, use_container_width=True)

                    with tabs[3]:
                        st.subheader("Detailed Analysis Results")

                        # Raw data statistics
                        st.write("**Input Data Statistics:**")
                        stats_data = {
                            'Statistic': [
                                'Mean Inter-arrival Time',
                                'Mean Service Time',
                                'CV of Inter-arrival Times',
                                'CV of Service Times',
                                'Number of Customers',
                                'Number of Servers'
                            ],
                            'Value': [
                                f"{results['mean_inter_arrival_time']:.4f} minutes",
                                f"{results['mean_service_time']:.4f} minutes",
                                f"{results['cv_inter_arrival']:.4f}",
                                f"{results['cv_service']:.4f}",
                                f"{len(df)}",
                                f"{servers}"
                            ]
                        }

                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)

                        # Queue length probabilities table
                        st.write("**Queue Length Probabilities:**")
                        prob_df = pd.DataFrame({
                            'Queue Length': list(results['queue_length_probabilities'].keys()),
                            'Probability': [f"{p:.6f}" for p in results['queue_length_probabilities'].values()]
                        })
                        st.dataframe(prob_df, use_container_width=True)

                        # System recommendations
                        st.subheader("System Recommendations")

                        if results['utilization'] < 0.5:
                            st.success("✅ System is under-utilized. Consider reducing servers or increasing arrival rate.")
                        elif results['utilization'] < 0.8:
                            st.info("ℹ️ System utilization is optimal.")
                        elif results['utilization'] < 0.95:
                            st.warning("⚠️ System utilization is high. Consider adding servers or improving service rate.")
                        else:
                            st.error("🚨 System utilization is very high. Immediate action required!")

                        if results['average_time_in_queue'] > results['mean_service_time']:
                            st.warning("⚠️ Average waiting time exceeds service time. Consider system improvements.")

            else:
                st.error("Unable to analyze the data. Please check your input data format.")

    # Information section
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📚 About Queuing Theory"):
        st.markdown("""
        ### M/M/s Queuing System

        **M/M/s** notation describes:
        - **First M**: Markovian (exponential) arrival process
        - **Second M**: Markovian (exponential) service times
        - **s**: Number of servers

        ### Key Metrics

        - **λ (lambda)**: Arrival rate (customers per unit time)
        - **μ (mu)**: Service rate per server (customers per unit time)
        - **ρ (rho)**: Utilization = λ/(sμ)
        - **L**: Average number of customers in the system
        - **Lq**: Average number of customers in the queue
        - **W**: Average time a customer spends in the system
        - **Wq**: Average time a customer spends waiting in the queue

        ### Stability Condition
        For the system to be stable: **ρ < 1** (utilization < 100%)

        ### Little's Law
        - L = λW (customers in system = arrival rate × time in system)
        - Lq = λWq (customers in queue = arrival rate × time in queue)
        """)

    with st.expander("📊 Data Format Requirements"):
        st.markdown("""
        ### Required Columns
        Your CSV file should contain:

        1. **Arrival Times**: Time when each customer arrives (in minutes)
        2. **Service Times**: Time required to serve each customer (in minutes)

        ### Example Data Format
        ```
        arrival_time_minutes,service_time_minutes
        0.5,2.3
        1.2,1.7
        3.6,3.1
        5.1,2.8
        ```

        ### Notes
        - Arrival times should be cumulative (not inter-arrival times)
        - Service times are the duration of service for each customer
        - Times can be in any consistent unit (minutes, seconds, hours)
        """)
