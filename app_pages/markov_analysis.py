"""
Markov Chain Analysis Page
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.markov_chain import markov_chain_analysis
from visualization.charts import plot_heatmap, plot_network_graph, plot_state_probabilities
from visualization.diagrams import create_diagram
from utils.data_processing import (
    load_csv_data, validate_markov_data, create_example_markov_data,
    format_matrix_display, get_data_summary
)


def create_analysis_header():
    """Create an enhanced header for the analysis page."""
    st.markdown("""
    <div class="dashboard-card fade-in-up">
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">🔗 Markov Chain Analysis</h1>
            <p style="font-size: 1.1rem; color: #718096; margin-bottom: 1rem;">
                Discover patterns in state transitions and predict long-term system behavior
            </p>
            <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.4rem 0.8rem; border-radius: 15px; font-size: 0.9rem;">⚖️ Steady States</span>
                <span style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); color: white; padding: 0.4rem 0.8rem; border-radius: 15px; font-size: 0.9rem;">⏱️ Passage Times</span>
                <span style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 0.4rem 0.8rem; border-radius: 15px; font-size: 0.9rem;">🌐 Network Graphs</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_advanced_data_input():
    """Create an advanced data input section with validation and preview."""
    st.markdown("## 📁 **Data Input & Configuration**")

    # Create columns for better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)

        # Data source selection with enhanced UI
        data_input = st.radio(
            "Choose your data source:",
            ["📁 Upload CSV File", "🎯 Use Example Dataset", "✏️ Manual Input"],
            horizontal=True
        )

        df = None

        if data_input == "📁 Upload CSV File":
            uploaded_file = st.file_uploader(
                "Upload your CSV file",
                type="csv",
                help="Upload a CSV file with state transition data"
            )
            if uploaded_file is not None:
                df = load_csv_data(uploaded_file)
                if df is not None:
                    st.success("✅ File uploaded successfully!")

                    # Show data preview
                    with st.expander("📊 Data Preview", expanded=True):
                        st.dataframe(df.head(10), use_container_width=True)

                        # Data summary
                        summary = get_data_summary(df)
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Rows", summary['shape'][0])
                        with col_b:
                            st.metric("Columns", summary['shape'][1])
                        with col_c:
                            st.metric("Memory", f"{summary['memory_usage']/1024:.1f} KB")

        elif data_input == "🎯 Use Example Dataset":
            example_type = st.selectbox(
                "Select example dataset:",
                ["Weather Model", "Stock Market", "Customer Behavior", "Gene Expression"]
            )

            if example_type == "Weather Model":
                df = create_example_markov_data()
                st.info("🌤️ Using weather transition example data")
            elif example_type == "Stock Market":
                df = create_stock_example_data()
                st.info("📈 Using stock market trend example data")
            elif example_type == "Customer Behavior":
                df = create_customer_example_data()
                st.info("🛒 Using customer behavior example data")
            else:
                df = create_gene_example_data()
                st.info("🧬 Using gene expression example data")

            if df is not None:
                with st.expander("📊 Dataset Preview", expanded=True):
                    st.dataframe(df, use_container_width=True)

        elif data_input == "✏️ Manual Input":
            st.markdown("**Create your own transition data:**")

            # Manual input interface
            num_states = st.number_input("Number of states:", min_value=2, max_value=10, value=3)
            state_names = []

            for i in range(num_states):
                state_name = st.text_input(f"State {i+1} name:", value=f"State_{i+1}", key=f"state_{i}")
                state_names.append(state_name)

            if st.button("Generate Sample Transitions"):
                df = create_manual_data(state_names)
                st.success("✅ Sample data generated!")
                st.dataframe(df, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Analysis configuration panel
        st.markdown("""
        <div class="simulation-controls">
            <h3>⚙️ Analysis Settings</h3>
        </div>
        """, unsafe_allow_html=True)

        # Advanced options
        show_advanced = st.checkbox("🔧 Advanced Options", value=False)

        if show_advanced:
            st.markdown("**Computation Settings:**")
            precision = st.slider("Numerical Precision", 3, 10, 6)
            max_iterations = st.number_input("Max Iterations", 100, 10000, 1000)
            tolerance = st.number_input("Convergence Tolerance", 1e-10, 1e-3, 1e-6, format="%.2e")

            st.markdown("**Visualization Options:**")
            color_scheme = st.selectbox("Color Scheme", ["Viridis", "Plasma", "Blues", "Reds"])
            show_probabilities = st.checkbox("Show Transition Probabilities", True)
            highlight_threshold = st.slider("Highlight Threshold", 0.0, 1.0, 0.1)

    return df


def create_stock_example_data():
    """Create stock market example data."""
    import numpy as np
    np.random.seed(42)

    states = ['Bull', 'Bear', 'Sideways']
    transitions = []

    # Generate realistic stock market transitions
    for _ in range(100):
        current = np.random.choice(states, p=[0.4, 0.3, 0.3])
        if current == 'Bull':
            next_state = np.random.choice(states, p=[0.7, 0.2, 0.1])
        elif current == 'Bear':
            next_state = np.random.choice(states, p=[0.3, 0.6, 0.1])
        else:  # Sideways
            next_state = np.random.choice(states, p=[0.3, 0.3, 0.4])

        transitions.append({'current_state': current, 'next_state': next_state})

    return pd.DataFrame(transitions)


def create_customer_example_data():
    """Create customer behavior example data."""
    import numpy as np
    np.random.seed(42)

    states = ['Browse', 'Cart', 'Checkout', 'Purchase', 'Exit']
    transitions = []

    for _ in range(150):
        current = np.random.choice(states, p=[0.3, 0.2, 0.15, 0.1, 0.25])
        if current == 'Browse':
            next_state = np.random.choice(states, p=[0.4, 0.3, 0.05, 0.05, 0.2])
        elif current == 'Cart':
            next_state = np.random.choice(states, p=[0.2, 0.3, 0.3, 0.1, 0.1])
        elif current == 'Checkout':
            next_state = np.random.choice(states, p=[0.1, 0.2, 0.2, 0.4, 0.1])
        elif current == 'Purchase':
            next_state = np.random.choice(states, p=[0.3, 0.1, 0.05, 0.05, 0.5])
        else:  # Exit
            next_state = np.random.choice(states, p=[0.6, 0.1, 0.05, 0.05, 0.2])

        transitions.append({'current_state': current, 'next_state': next_state})

    return pd.DataFrame(transitions)


def create_gene_example_data():
    """Create gene expression example data."""
    import numpy as np
    np.random.seed(42)

    states = ['Active', 'Inactive', 'Repressed']
    transitions = []

    for _ in range(80):
        current = np.random.choice(states, p=[0.4, 0.4, 0.2])
        if current == 'Active':
            next_state = np.random.choice(states, p=[0.6, 0.3, 0.1])
        elif current == 'Inactive':
            next_state = np.random.choice(states, p=[0.3, 0.5, 0.2])
        else:  # Repressed
            next_state = np.random.choice(states, p=[0.1, 0.7, 0.2])

        transitions.append({'current_state': current, 'next_state': next_state})

    return pd.DataFrame(transitions)


def create_manual_data(state_names):
    """Create manual transition data."""
    import numpy as np
    np.random.seed(42)

    transitions = []
    for _ in range(50):
        current = np.random.choice(state_names)
        next_state = np.random.choice(state_names)
        transitions.append({'current_state': current, 'next_state': next_state})

    return pd.DataFrame(transitions)


def show_markov_analysis_page():
    """Display the enhanced Markov Chain analysis page."""
    # Enhanced header
    create_analysis_header()

    # Advanced data input
    df = create_advanced_data_input()

    # Data input section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("Data Input")

    data_input = st.radio("Choose data source:", ["Upload CSV", "Use Example Data"])
    df = None

    if data_input == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            df = load_csv_data(uploaded_file)
            if df is not None:
                st.success("File uploaded successfully!")
                st.dataframe(df.head())
        else:
            df = None

    elif data_input == "Use Example Data":
        st.info("Using example Markov Chain data")
        df = create_example_markov_data()
        st.dataframe(df)

    st.markdown('</div>', unsafe_allow_html=True)

    # Column mapping and analysis
    if df is not None and not df.empty:
        st.subheader("Column Mapping")

        col1, col2 = st.columns(2)
        with col1:
            current_state_col = st.selectbox(
                "Select current state column:",
                df.columns,
                index=df.columns.get_loc('current_state') if 'current_state' in df.columns else 0
            )
        with col2:
            next_state_col = st.selectbox(
                "Select next state column:",
                df.columns,
                index=df.columns.get_loc('next_state') if 'next_state' in df.columns else min(1, len(df.columns) - 1)
            )

        # Validate data
        is_valid, error_msg = validate_markov_data(df, current_state_col, next_state_col)

        if not is_valid:
            st.error(f"Data validation error: {error_msg}")
            return

        # Analysis button
        if st.button("Analyze Markov Chain", type="primary"):
            with st.spinner("Analyzing..."):
                results = markov_chain_analysis(df, state_col=current_state_col, next_state_col=next_state_col)

            if results:
                display_results(results)
            else:
                st.error("Could not analyze the Markov chain. Please check your data.")


def display_results(results):
    """Display analysis results in organized tabs."""
    st.success("Analysis completed successfully!")

    # Display results in organized tabs
    tabs = st.tabs(["Transition Matrix", "State Diagram", "Steady State", "Passage Times", "Properties"])

    with tabs[0]:
        st.subheader("Transition Matrix")
        st.write("The probability of transitioning from one state to another:")

        # Display formatted matrix
        formatted_matrix = format_matrix_display(results['transition_matrix'])
        st.dataframe(formatted_matrix, use_container_width=True)

        # Heatmap visualization
        try:
            fig = plot_heatmap(results['transition_matrix'], "Transition Probability Heatmap")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")

    with tabs[1]:
        st.subheader("State Transition Diagram")

        # Let user select a state to highlight
        highlight_state = st.selectbox("Highlight state:", ["None"] + results['states'])
        highlight = None if highlight_state == "None" else highlight_state

        # Create and display diagram
        try:
            dot_graph = create_diagram(results['transition_matrix'], highlight)
            st.graphviz_chart(dot_graph)
        except Exception as e:
            st.error(f"Error creating diagram: {e}")

        # Network graph
        st.subheader("Network Graph")
        try:
            fig = plot_network_graph(results['transition_matrix'], "Markov Chain Network")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating network graph: {e}")

    with tabs[2]:
        st.subheader("Steady-State Probabilities")
        st.write("Long-run probability of being in each state:")

        # Display steady state probabilities
        formatted_steady_state = format_matrix_display(results['steady_state_probabilities'])
        st.dataframe(formatted_steady_state, use_container_width=True)

        # Bar chart of steady state probabilities
        try:
            fig = plot_state_probabilities(results['steady_state_probabilities'], "Steady-State Probabilities")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating probability chart: {e}")

    with tabs[3]:
        st.subheader("Average Passage Times")
        st.write("Expected number of steps to reach one state from another:")

        # Display passage times
        formatted_passage_times = format_matrix_display(results['average_passage_times'])
        st.dataframe(formatted_passage_times, use_container_width=True)

        st.subheader("Average Recurrence Times")
        st.write("Expected number of steps to return to each state:")

        # Display recurrence times
        recurrence_df = pd.DataFrame(results['average_recurrence_times'], index=['Recurrence Time']).T
        formatted_recurrence = format_matrix_display(recurrence_df)
        st.dataframe(formatted_recurrence, use_container_width=True)

    with tabs[4]:
        st.subheader("Chain Properties")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**State Information:**")
            st.write(f"Number of states: {len(results['states'])}")
            st.write(f"States: {', '.join(results['states'])}")

            if results['absorbing_states']:
                st.write(f"Absorbing states: {', '.join(results['absorbing_states'])}")
            else:
                st.write("No absorbing states detected")

        with col2:
            st.write("**State Frequencies:**")
            freq_df = pd.DataFrame(list(results['state_frequencies'].items()),
                                 columns=['State', 'Frequency'])
            st.dataframe(freq_df, use_container_width=True)

        # Periods
        st.write("**State Periods:**")
        periods_df = pd.DataFrame(list(results['periods'].items()),
                                columns=['State', 'Period'])
        st.dataframe(periods_df, use_container_width=True)

        # Absorption information
        if results['absorption_steps']:
            st.write("**Expected Steps to Absorption:**")
            absorption_df = pd.DataFrame(list(results['absorption_steps'].items()),
                                       columns=['State', 'Expected Steps'])
            st.dataframe(absorption_df, use_container_width=True)
