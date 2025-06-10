"""
Simulation Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def simulate_markov_chain(transition_matrix, steps=50, start_state=None):
    """
    Simulates a Markov chain for the given number of steps.
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


def simulate_hmm(transition_probs, emission_probs, steps=50, start_state=None):
    """
    Simulates a Hidden Markov Model for the given number of steps.
    """
    hidden_states = transition_probs.index.tolist()
    observed_events = emission_probs.columns.tolist()

    if start_state is None:
        # Start with a random state
        start_state = np.random.choice(hidden_states)

    if start_state not in hidden_states:
        return None

    # Initialize the simulation
    current_state = start_state
    hidden_sequence = [current_state]
    observed_sequence = []

    # Simulate the HMM
    for _ in range(steps):
        # Generate observation based on current hidden state
        observation_probs = emission_probs.loc[current_state].values
        observation = np.random.choice(observed_events, p=observation_probs)
        observed_sequence.append(observation)

        if _ < steps - 1:  # Don't transition after the last step
            # Choose next hidden state based on transition probabilities
            transition_probs_current = transition_probs.loc[current_state].values
            next_state = np.random.choice(hidden_states, p=transition_probs_current)
            hidden_sequence.append(next_state)
            current_state = next_state

    return hidden_sequence, observed_sequence


def simulate_queue(arrival_rate, service_rate, servers=1, duration=1000):
    """
    Simulates an M/M/s queuing system.
    """
    # Initialize simulation
    time = 0
    queue = []
    server_busy_until = [0] * servers

    events = []  # (time, event_type, customer_id)
    customer_id = 0

    # Generate first arrival
    next_arrival = np.random.exponential(1/arrival_rate)
    events.append((next_arrival, "arrival", customer_id))

    # Simulation statistics
    customer_stats = {}  # {id: (arrival_time, service_start, departure_time)}
    queue_length_over_time = [(0, 0)]  # (time, queue_length)
    customers_in_system_over_time = [(0, 0)]  # (time, customers_in_system)

    # Run simulation
    while time < duration:
        # Get next event
        events.sort()
        event_time, event_type, cust_id = events.pop(0)

        # Update time
        time = event_time

        if time > duration:
            break

        # Handle event
        if event_type == "arrival":
            # Schedule next arrival
            customer_id += 1
            next_arrival = time + np.random.exponential(1/arrival_rate)
            events.append((next_arrival, "arrival", customer_id))

            # Record arrival
            customer_stats[cust_id] = (time, None, None)

            # Find available server
            available_server = None
            for i in range(servers):
                if server_busy_until[i] <= time:
                    available_server = i
                    break

            if available_server is not None:
                # Server available, start service
                service_time = np.random.exponential(1/service_rate)
                server_busy_until[available_server] = time + service_time
                events.append((time + service_time, "departure", cust_id))

                # Update customer stats
                arrival_time, _, _ = customer_stats[cust_id]
                customer_stats[cust_id] = (arrival_time, time, time + service_time)
            else:
                # All servers busy, join queue
                queue.append(cust_id)

        elif event_type == "departure":
            # Customer leaves
            arrival_time, service_start, _ = customer_stats[cust_id]
            customer_stats[cust_id] = (arrival_time, service_start, time)

            # If queue not empty, start service for next customer
            if queue:
                next_cust = queue.pop(0)

                # Find server that just became available
                for i in range(servers):
                    if server_busy_until[i] <= time:
                        available_server = i
                        break

                service_time = np.random.exponential(1/service_rate)
                server_busy_until[available_server] = time + service_time
                events.append((time + service_time, "departure", next_cust))

                # Update customer stats
                arrival_time, _, _ = customer_stats[next_cust]
                customer_stats[next_cust] = (arrival_time, time, time + service_time)

        # Update statistics
        queue_length_over_time.append((time, len(queue)))
        customers_in_system = len(queue) + sum(1 for s in server_busy_until if s > time)
        customers_in_system_over_time.append((time, customers_in_system))

    # Calculate statistics
    waiting_times = []
    system_times = []

    for cust_id, (arrival, service_start, departure) in customer_stats.items():
        if service_start is not None and departure is not None:
            waiting_times.append(service_start - arrival)
            system_times.append(departure - arrival)

    stats = {
        'average_waiting_time': np.mean(waiting_times) if waiting_times else 0,
        'average_system_time': np.mean(system_times) if system_times else 0,
        'queue_length_over_time': queue_length_over_time,
        'customers_in_system_over_time': customers_in_system_over_time,
        'utilization': sum(server_busy_until) / (duration * servers),
        'customer_stats': customer_stats
    }

    return stats


def show_simulation_page():
    """Display the simulation page."""
    st.title("Process Simulation 🎮")

    st.markdown("""
    <div class="info-text">
        <p>Simulate stochastic processes to understand their behavior under different parameters.</p>
    </div>
    """, unsafe_allow_html=True)

    # Simulation type selection
    sim_type = st.selectbox("Select simulation type:",
                           ["Markov Chain", "Hidden Markov Model", "Queuing System"])

    if sim_type == "Markov Chain":
        st.header("Markov Chain Simulation")

        # Options for simulation input
        sim_input = st.radio("Choose input method:", ["Define Matrix", "Upload Matrix", "Use Example"], key="mc_sim_input")

        transition_matrix = None

        if sim_input == "Define Matrix":
            st.subheader("Define Transition Matrix")

            num_states = st.number_input("Number of states:", min_value=2, value=3)
            state_names = [st.text_input(f"Name for State {i+1}:", value=f"S{i+1}") for i in range(num_states)]

            st.write("Enter transition probabilities (rows must sum to 1):")

            # Create empty matrix
            data = {}
            for i, from_state in enumerate(state_names):
                row = []
                for j, to_state in enumerate(state_names):
                    prob = st.number_input(f"P({from_state}→{to_state})",
                                         min_value=0.0, max_value=1.0,
                                         value=0.5 if i == j else 0.5/(num_states-1),
                                         key=f"prob_{i}_{j}")
                    row.append(prob)
                data[from_state] = row

            transition_matrix = pd.DataFrame(data, index=state_names, columns=state_names)

            # Check row sums
            row_sums = transition_matrix.sum(axis=1)
            if not all(np.isclose(row_sums, 1)):
                st.error("Error: Each row must sum to 1. Current row sums: " + str(row_sums))
                transition_matrix = None
            else:
                st.success("Valid transition matrix!")
                st.dataframe(transition_matrix)

        elif sim_input == "Upload Matrix":
            uploaded_file = st.file_uploader("Upload transition matrix (CSV)", type="csv", key="mc_matrix_upload")
            if uploaded_file is not None:
                try:
                    transition_matrix = pd.read_csv(uploaded_file, index_col=0)
                    # Validate
                    row_sums = transition_matrix.sum(axis=1)
                    if not all(np.isclose(row_sums, 1)):
                        st.error("Error: Each row must sum to 1. Current row sums: " + str(row_sums))
                        transition_matrix = None
                    else:
                        st.success("Valid transition matrix!")
                        st.dataframe(transition_matrix)
                except Exception as e:
                    st.error(f"Error: {e}")

        elif sim_input == "Use Example":
            st.info("Using example transition matrix")
            example_matrix = pd.DataFrame({
                'Sunny': [0.8, 0.2, 0.1],
                'Rainy': [0.15, 0.6, 0.3],
                'Cloudy': [0.05, 0.2, 0.6]
            }, index=['Sunny', 'Rainy', 'Cloudy'])
            transition_matrix = example_matrix
            st.dataframe(transition_matrix)

        # Run simulation if matrix is valid
        if transition_matrix is not None:
            st.subheader("Simulation Parameters")
            num_steps = st.number_input("Number of steps:", min_value=10, value=50)
            start_state = st.selectbox("Starting state:", transition_matrix.index)

            if st.button("Run Simulation"):
                state_sequence = simulate_markov_chain(transition_matrix, steps=num_steps, start_state=start_state)

                st.subheader("Simulation Results")

                # Display sequence
                st.write("State sequence:")
                st.write(", ".join(state_sequence))

                # Plot sequence over time
                fig, ax = plt.subplots(figsize=(10, 4))
                state_indices = [transition_matrix.index.get_loc(state) for state in state_sequence]
                ax.plot(range(num_steps), state_indices, 'o-')
                ax.set_yticks(range(len(transition_matrix.index)))
                ax.set_yticklabels(transition_matrix.index)
                ax.set_title("Markov Chain Simulation")
                ax.set_xlabel("Step")
                ax.set_ylabel("State")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Calculate empirical probabilities
                st.subheader("Empirical State Frequencies")
                state_counts = pd.Series(state_sequence).value_counts(normalize=True)
                state_counts = state_counts.reindex(transition_matrix.index, fill_value=0)
                st.bar_chart(state_counts)

                # Compare with theoretical steady state
                try:
                    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
                    closest_to_one_idx = np.argmin(np.abs(eigenvalues - 1))
                    steady_state = np.abs(eigenvectors[:, closest_to_one_idx])
                    steady_state = steady_state / steady_state.sum()
                    steady_state = pd.Series(steady_state, index=transition_matrix.index)

                    st.write("Comparison with theoretical steady state:")
                    compare_df = pd.DataFrame({
                        'Empirical': state_counts,
                        'Theoretical': steady_state
                    })
                    st.dataframe(compare_df)

                    # Plot comparison
                    fig, ax = plt.subplots()
                    compare_df.plot(kind='bar', ax=ax)
                    ax.set_title("Empirical vs Theoretical State Frequencies")
                    ax.set_ylabel("Probability")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                except:
                    st.warning("Could not calculate steady state probabilities")

    elif sim_type == "Hidden Markov Model":
        st.header("Hidden Markov Model Simulation")

        # Options for simulation input
        sim_input = st.radio("Choose input method:", ["Define Parameters", "Upload Parameters", "Use Example"], key="hmm_sim_input")

        transition_probs = None
        emission_probs = None

        if sim_input == "Define Parameters":
            st.subheader("Define HMM Parameters")

            # Hidden states
            num_hidden = st.number_input("Number of hidden states:", min_value=2, value=2)
            hidden_names = [st.text_input(f"Name for Hidden State {i+1}:", value=f"HS{i+1}") for i in range(num_hidden)]

            # Observed events
            num_observed = st.number_input("Number of observed events:", min_value=2, value=2)
            observed_names = [st.text_input(f"Name for Observed Event {i+1}:", value=f"E{i+1}") for i in range(num_observed)]

            st.subheader("Transition Probabilities")
            st.write("Enter transition probabilities between hidden states (rows must sum to 1):")

            # Create empty transition matrix
            trans_data = {}
            for i, from_state in enumerate(hidden_names):
                row = []
                for j, to_state in enumerate(hidden_names):
                    prob = st.number_input(f"P({from_state}→{to_state})",
                                         min_value=0.0, max_value=1.0,
                                         value=0.5 if i == j else 0.5/(num_hidden-1),
                                         key=f"trans_{i}_{j}")
                    row.append(prob)
                trans_data[from_state] = row

            transition_probs = pd.DataFrame(trans_data, index=hidden_names, columns=hidden_names)

            # Check row sums
            row_sums = transition_probs.sum(axis=1)
            if not all(np.isclose(row_sums, 1)):
                st.error("Error: Each row must sum to 1. Current row sums: " + str(row_sums))
                transition_probs = None
            else:
                st.success("Valid transition matrix!")
                st.dataframe(transition_probs)

            st.subheader("Emission Probabilities")
            st.write("Enter emission probabilities (rows must sum to 1):")

            # Create empty emission matrix
            emit_data = {}
            for i, state in enumerate(hidden_names):
                row = []
                for j, event in enumerate(observed_names):
                    prob = st.number_input(f"P({event}|{state})",
                                         min_value=0.0, max_value=1.0,
                                         value=0.5 if i == j else 0.5/(num_observed-1),
                                         key=f"emit_{i}_{j}")
                    row.append(prob)
                emit_data[state] = row

            emission_probs = pd.DataFrame(emit_data, index=hidden_names, columns=observed_names)

            # Check row sums
            row_sums = emission_probs.sum(axis=1)
            if not all(np.isclose(row_sums, 1)):
                st.error("Error: Each row must sum to 1. Current row sums: " + str(row_sums))
                emission_probs = None
            else:
                st.success("Valid emission matrix!")
                st.dataframe(emission_probs)

        elif sim_input == "Upload Parameters":
            st.subheader("Upload Transition Matrix")
            trans_file = st.file_uploader("Upload transition matrix (CSV)", type="csv", key="hmm_trans_upload")
            if trans_file is not None:
                try:
                    transition_probs = pd.read_csv(trans_file, index_col=0)
                    # Validate
                    row_sums = transition_probs.sum(axis=1)
                    if not all(np.isclose(row_sums, 1)):
                        st.error("Error: Each row must sum to 1. Current row sums: " + str(row_sums))
                        transition_probs = None
                    else:
                        st.success("Valid transition matrix!")
                        st.dataframe(transition_probs)
                except Exception as e:
                    st.error(f"Error: {e}")

            st.subheader("Upload Emission Matrix")
            emit_file = st.file_uploader("Upload emission matrix (CSV)", type="csv", key="hmm_emit_upload")
            if emit_file is not None:
                try:
                    emission_probs = pd.read_csv(emit_file, index_col=0)
                    # Validate
                    row_sums = emission_probs.sum(axis=1)
                    if not all(np.isclose(row_sums, 1)):
                        st.error("Error: Each row must sum to 1. Current row sums: " + str(row_sums))
                        emission_probs = None
                    else:
                        st.success("Valid emission matrix!")
                        st.dataframe(emission_probs)
                except Exception as e:
                    st.error(f"Error: {e}")

        elif sim_input == "Use Example":
            st.info("Using example HMM parameters")

            # Weather example
            transition_probs = pd.DataFrame({
                'Sunny': [0.8, 0.2],
                'Rainy': [0.3, 0.7]
            }, index=['Sunny', 'Rainy'])

            emission_probs = pd.DataFrame({
                'Dry': [0.9, 0.1],
                'Wet': [0.1, 0.9]
            }, index=['Sunny', 'Rainy'])

            st.write("Transition Probabilities:")
            st.dataframe(transition_probs)
            st.write("Emission Probabilities:")
            st.dataframe(emission_probs)

        # Run simulation if both matrices are valid
        if transition_probs is not None and emission_probs is not None:
            st.subheader("Simulation Parameters")
            num_steps = st.number_input("Number of steps:", min_value=10, value=20, key="hmm_steps")
            start_state = st.selectbox("Starting state:", transition_probs.index, key="hmm_start")

            if st.button("Run HMM Simulation"):
                hidden_seq, observed_seq = simulate_hmm(transition_probs, emission_probs, steps=num_steps, start_state=start_state)

                st.subheader("Simulation Results")

                # Display sequences
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Hidden state sequence:")
                    st.write(", ".join(hidden_seq))
                with col2:
                    st.write("Observed event sequence:")
                    st.write(", ".join(observed_seq))

                # Plot sequences
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

                # Hidden states
                hidden_indices = [transition_probs.index.get_loc(state) for state in hidden_seq]
                ax1.plot(range(num_steps), hidden_indices, 'o-', color='#1f77b4')
                ax1.set_yticks(range(len(transition_probs.index)))
                ax1.set_yticklabels(transition_probs.index)
                ax1.set_ylabel("Hidden State")
                ax1.grid(True, alpha=0.3)

                # Observed events
                observed_indices = [emission_probs.columns.get_loc(event) for event in observed_seq]
                ax2.plot(range(num_steps), observed_indices, 'o-', color='#ff7f0e')
                ax2.set_yticks(range(len(emission_probs.columns)))
                ax2.set_yticklabels(emission_probs.columns)
                ax2.set_xlabel("Step")
                ax2.set_ylabel("Observed Event")
                ax2.grid(True, alpha=0.3)

                plt.suptitle("HMM Simulation")
                plt.tight_layout()
                st.pyplot(fig)

                # Calculate empirical probabilities
                st.subheader("Empirical Statistics")

                # Hidden state frequencies
                st.write("Hidden State Frequencies:")
                hidden_counts = pd.Series(hidden_seq).value_counts(normalize=True)
                hidden_counts = hidden_counts.reindex(transition_probs.index, fill_value=0)
                st.bar_chart(hidden_counts)

                # Observed event frequencies
                st.write("Observed Event Frequencies:")
                observed_counts = pd.Series(observed_seq).value_counts(normalize=True)
                observed_counts = observed_counts.reindex(emission_probs.columns, fill_value=0)
                st.bar_chart(observed_counts)

                # Compare with theoretical steady state
                try:
                    eigenvalues, eigenvectors = np.linalg.eig(transition_probs.T)
                    closest_to_one_idx = np.argmin(np.abs(eigenvalues - 1))
                    steady_state = np.abs(eigenvectors[:, closest_to_one_idx])
                    steady_state = steady_state / steady_state.sum()
                    steady_state = pd.Series(steady_state, index=transition_probs.index)

                    st.write("Hidden State Comparison:")
                    compare_df = pd.DataFrame({
                        'Empirical': hidden_counts,
                        'Theoretical': steady_state
                    })
                    st.dataframe(compare_df)
                except:
                    st.warning("Could not calculate steady state probabilities")

    elif sim_type == "Queuing System":
        st.header("Queuing System Simulation")

        # Simulation parameters
        st.subheader("System Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            arrival_rate = st.number_input("Arrival rate (λ, customers/time unit):",
                                         min_value=0.01, value=0.5)
        with col2:
            service_rate = st.number_input("Service rate (μ, customers/time unit):",
                                         min_value=0.01, value=1.0)
        with col3:
            servers = st.number_input("Number of servers (s):",
                                    min_value=1, value=1)

        st.subheader("Simulation Parameters")
        sim_duration = st.number_input("Simulation duration (time units):",
                                     min_value=10, value=100)

        if st.button("Run Queuing Simulation"):
            with st.spinner("Simulating..."):
                sim_results = simulate_queue(arrival_rate, service_rate,
                                           servers=servers, duration=sim_duration)

            st.subheader("Simulation Results")

            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Waiting Time", f"{sim_results['average_waiting_time']:.2f}")
            with col2:
                st.metric("Average System Time", f"{sim_results['average_system_time']:.2f}")
            with col3:
                st.metric("Server Utilization", f"{sim_results['utilization']:.2f}")

            # Plot queue length over time
            st.subheader("Queue Length Over Time")
            fig, ax = plt.subplots(figsize=(10, 5))
            times, lengths = zip(*sim_results['queue_length_over_time'])
            ax.step(times, lengths, where='post')
            ax.set_title("Queue Length Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Queue Length")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Plot customers in system over time
            st.subheader("Customers in System Over Time")
            fig, ax = plt.subplots(figsize=(10, 5))
            times, counts = zip(*sim_results['customers_in_system_over_time'])
            ax.step(times, counts, where='post')
            ax.set_title("Customers in System Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Customers")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # System performance analysis
            st.subheader("Performance Analysis")

            # Calculate theoretical values for comparison
            rho = arrival_rate / (servers * service_rate)

            if rho < 1:
                st.success(f"✅ System is stable (ρ = {rho:.3f} < 1)")

                # For M/M/s system theoretical calculations
                if servers == 1:
                    # M/M/1 formulas
                    theoretical_L = rho / (1 - rho)
                    theoretical_W = 1 / (service_rate - arrival_rate)
                    theoretical_Lq = rho**2 / (1 - rho)
                    theoretical_Wq = rho / (service_rate - arrival_rate)
                else:
                    # M/M/s approximation (simplified)
                    theoretical_L = rho + (rho**(servers+1)) / ((1-rho) * np.math.factorial(servers))
                    theoretical_W = theoretical_L / arrival_rate
                    theoretical_Lq = theoretical_L - rho
                    theoretical_Wq = theoretical_Lq / arrival_rate

                # Comparison table
                comparison_data = {
                    'Metric': [
                        'Average Customers in System (L)',
                        'Average Time in System (W)',
                        'Average Customers in Queue (Lq)',
                        'Average Time in Queue (Wq)',
                        'Utilization (ρ)'
                    ],
                    'Theoretical': [
                        f"{theoretical_L:.3f}",
                        f"{theoretical_W:.3f}",
                        f"{theoretical_Lq:.3f}",
                        f"{theoretical_Wq:.3f}",
                        f"{rho:.3f}"
                    ],
                    'Simulated': [
                        f"{np.mean([x[1] for x in sim_results['customers_in_system_over_time']]):.3f}",
                        f"{sim_results['average_system_time']:.3f}",
                        f"{np.mean([x[1] for x in sim_results['queue_length_over_time']]):.3f}",
                        f"{sim_results['average_waiting_time']:.3f}",
                        f"{sim_results['utilization']:.3f}"
                    ]
                }

                st.dataframe(pd.DataFrame(comparison_data))

            else:
                st.error(f"⚠️ System is unstable (ρ = {rho:.3f} ≥ 1). Queue will grow indefinitely!")
                st.write("Consider:")
                st.write("- Increasing the number of servers")
                st.write("- Increasing the service rate")
                st.write("- Reducing the arrival rate")

    # Information sections
    with st.expander("📚 About Simulation"):
        st.markdown("""
        ### Simulation Types

        **Markov Chain Simulation**
        - Simulates discrete-time state transitions
        - Useful for understanding long-term behavior
        - Compares empirical vs theoretical steady-state probabilities

        **Hidden Markov Model Simulation**
        - Simulates both hidden states and observable events
        - Useful for understanding the relationship between hidden and observed processes
        - Helps validate HMM parameter estimation

        **Queuing System Simulation**
        - Discrete-event simulation of M/M/s queues
        - Generates arrival and service events
        - Tracks system performance over time
        - Compares simulation results with theoretical predictions

        ### Benefits of Simulation
        - Validates theoretical models
        - Explores system behavior under different parameters
        - Provides insights into transient behavior
        - Helps with system design and optimization
        """)

    with st.expander("⚙️ Simulation Parameters"):
        st.markdown("""
        ### Key Parameters

        **Markov Chain**
        - **Transition Matrix**: Probabilities of moving between states
        - **Number of Steps**: Length of simulation
        - **Starting State**: Initial state of the chain

        **Hidden Markov Model**
        - **Transition Probabilities**: Hidden state transitions
        - **Emission Probabilities**: Observable event generation
        - **Number of Steps**: Length of simulation
        - **Starting State**: Initial hidden state

        **Queuing System**
        - **Arrival Rate (λ)**: Average customers arriving per time unit
        - **Service Rate (μ)**: Average customers served per time unit per server
        - **Number of Servers**: Parallel service channels
        - **Simulation Duration**: Total time to simulate

        ### Tips for Good Simulations
        - Use sufficient simulation length for stable results
        - Run multiple replications to assess variability
        - Validate results against theoretical predictions when available
        - Consider warm-up periods for steady-state analysis
        """)
