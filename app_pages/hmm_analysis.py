"""
Advanced Hidden Markov Model Analysis Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.hidden_markov import hidden_markov_model_analysis
from visualization.charts import plot_heatmap, create_plotly_heatmap
from visualization.diagrams import create_hmm_diagram
from utils.data_processing import (
    load_csv_data, validate_hmm_data, create_example_hmm_data,
    format_matrix_display, get_data_summary
)


def create_hmm_header():
    """Create an enhanced header for the HMM analysis page."""
    st.markdown("""
    <div class="dashboard-card fade-in-up">
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">🔍 Hidden Markov Model Analysis</h1>
            <p style="font-size: 1.1rem; color: #718096; margin-bottom: 1rem;">
                Uncover hidden patterns in sequential data and decode the invisible states driving observable events
            </p>
            <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.4rem 0.8rem; border-radius: 15px; font-size: 0.9rem;">🎭 State Estimation</span>
                <span style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); color: white; padding: 0.4rem 0.8rem; border-radius: 15px; font-size: 0.9rem;">📈 Forward Algorithm</span>
                <span style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 0.4rem 0.8rem; border-radius: 15px; font-size: 0.9rem;">🎯 Viterbi Decoding</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_hmm_data_input():
    """Create advanced data input for HMM analysis."""
    st.markdown("## 📁 **Data Input & Configuration**")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)

        data_input = st.radio(
            "Choose your data source:",
            ["📁 Upload CSV File", "🎯 Use Example Dataset", "🎲 Generate Synthetic Data"],
            horizontal=True
        )

        df = None

        if data_input == "📁 Upload CSV File":
            uploaded_file = st.file_uploader(
                "Upload your CSV file",
                type="csv",
                help="Upload a CSV file with hidden states and observed events"
            )
            if uploaded_file is not None:
                df = load_csv_data(uploaded_file)
                if df is not None:
                    st.success("✅ File uploaded successfully!")

                    with st.expander("📊 Data Preview", expanded=True):
                        st.dataframe(df.head(10), use_container_width=True)

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
                ["Weather-Activity", "Speech Recognition", "Bioinformatics", "Financial Markets"]
            )

            if example_type == "Weather-Activity":
                df = create_example_hmm_data()
                st.info("🌤️ Using weather-activity HMM example")
            elif example_type == "Speech Recognition":
                df = create_speech_hmm_data()
                st.info("🎤 Using speech recognition HMM example")
            elif example_type == "Bioinformatics":
                df = create_bio_hmm_data()
                st.info("🧬 Using bioinformatics HMM example")
            else:
                df = create_financial_hmm_data()
                st.info("💰 Using financial markets HMM example")

            if df is not None:
                with st.expander("📊 Dataset Preview", expanded=True):
                    st.dataframe(df, use_container_width=True)

        elif data_input == "🎲 Generate Synthetic Data":
            st.markdown("**Generate synthetic HMM data:**")

            num_hidden_states = st.number_input("Number of hidden states:", min_value=2, max_value=8, value=3)
            num_observations = st.number_input("Number of observation types:", min_value=2, max_value=10, value=4)
            sequence_length = st.number_input("Sequence length:", min_value=50, max_value=1000, value=200)

            if st.button("🎲 Generate Synthetic Data"):
                df = generate_synthetic_hmm_data(num_hidden_states, num_observations, sequence_length)
                st.success("✅ Synthetic data generated!")
                st.dataframe(df.head(20), use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="simulation-controls">
            <h3>⚙️ HMM Settings</h3>
        </div>
        """, unsafe_allow_html=True)

        # Algorithm selection
        algorithm_options = st.multiselect(
            "Select algorithms to run:",
            ["Forward Algorithm", "Viterbi Algorithm", "Baum-Welch Learning"],
            default=["Forward Algorithm", "Viterbi Algorithm"]
        )

        # Advanced options
        show_advanced = st.checkbox("🔧 Advanced Options", value=False)

        if show_advanced:
            st.markdown("**Algorithm Parameters:**")
            max_iterations = st.number_input("Max EM Iterations", 10, 1000, 100)
            convergence_threshold = st.number_input("Convergence Threshold", 1e-8, 1e-2, 1e-6, format="%.2e")

            st.markdown("**Visualization Options:**")
            show_probabilities = st.checkbox("Show Probability Values", True)
            animation_speed = st.slider("Animation Speed", 0.5, 3.0, 1.0)

    return df


def create_speech_hmm_data():
    """Create speech recognition HMM example data."""
    np.random.seed(42)

    # Hidden states: phonemes
    hidden_states = ['Vowel', 'Consonant', 'Silence']
    # Observed events: acoustic features
    observations = ['High_Freq', 'Low_Freq', 'Mid_Freq', 'Noise']

    data = []
    current_state = np.random.choice(hidden_states)

    for _ in range(150):
        # Transition probabilities
        if current_state == 'Vowel':
            next_state = np.random.choice(hidden_states, p=[0.6, 0.3, 0.1])
            observation = np.random.choice(observations, p=[0.7, 0.1, 0.15, 0.05])
        elif current_state == 'Consonant':
            next_state = np.random.choice(hidden_states, p=[0.4, 0.5, 0.1])
            observation = np.random.choice(observations, p=[0.2, 0.6, 0.15, 0.05])
        else:  # Silence
            next_state = np.random.choice(hidden_states, p=[0.3, 0.3, 0.4])
            observation = np.random.choice(observations, p=[0.1, 0.1, 0.1, 0.7])

        data.append({
            'hidden_state': current_state,
            'observed_event': observation
        })
        current_state = next_state

    return pd.DataFrame(data)


def create_bio_hmm_data():
    """Create bioinformatics HMM example data."""
    np.random.seed(42)

    # Hidden states: gene regions
    hidden_states = ['Exon', 'Intron', 'Promoter']
    # Observed events: nucleotides
    observations = ['A', 'T', 'G', 'C']

    data = []
    current_state = np.random.choice(hidden_states)

    for _ in range(200):
        if current_state == 'Exon':
            next_state = np.random.choice(hidden_states, p=[0.7, 0.2, 0.1])
            observation = np.random.choice(observations, p=[0.25, 0.25, 0.25, 0.25])
        elif current_state == 'Intron':
            next_state = np.random.choice(hidden_states, p=[0.3, 0.6, 0.1])
            observation = np.random.choice(observations, p=[0.4, 0.4, 0.1, 0.1])
        else:  # Promoter
            next_state = np.random.choice(hidden_states, p=[0.5, 0.2, 0.3])
            observation = np.random.choice(observations, p=[0.1, 0.1, 0.4, 0.4])

        data.append({
            'hidden_state': current_state,
            'observed_event': observation
        })
        current_state = next_state

    return pd.DataFrame(data)


def create_financial_hmm_data():
    """Create financial markets HMM example data."""
    np.random.seed(42)

    # Hidden states: market regimes
    hidden_states = ['Bull_Market', 'Bear_Market', 'Volatile_Market']
    # Observed events: price movements
    observations = ['Large_Up', 'Small_Up', 'Small_Down', 'Large_Down']

    data = []
    current_state = np.random.choice(hidden_states)

    for _ in range(180):
        if current_state == 'Bull_Market':
            next_state = np.random.choice(hidden_states, p=[0.8, 0.1, 0.1])
            observation = np.random.choice(observations, p=[0.4, 0.4, 0.15, 0.05])
        elif current_state == 'Bear_Market':
            next_state = np.random.choice(hidden_states, p=[0.1, 0.8, 0.1])
            observation = np.random.choice(observations, p=[0.05, 0.15, 0.4, 0.4])
        else:  # Volatile_Market
            next_state = np.random.choice(hidden_states, p=[0.2, 0.2, 0.6])
            observation = np.random.choice(observations, p=[0.25, 0.25, 0.25, 0.25])

        data.append({
            'hidden_state': current_state,
            'observed_event': observation
        })
        current_state = next_state

    return pd.DataFrame(data)


def generate_synthetic_hmm_data(num_hidden, num_obs, length):
    """Generate synthetic HMM data."""
    np.random.seed(42)

    hidden_states = [f"State_{i+1}" for i in range(num_hidden)]
    observations = [f"Obs_{i+1}" for i in range(num_obs)]

    # Generate random transition and emission matrices
    transition_matrix = np.random.dirichlet(np.ones(num_hidden), num_hidden)
    emission_matrix = np.random.dirichlet(np.ones(num_obs), num_hidden)

    data = []
    current_state_idx = np.random.randint(num_hidden)

    for _ in range(length):
        current_state = hidden_states[current_state_idx]
        observation = np.random.choice(observations, p=emission_matrix[current_state_idx])

        data.append({
            'hidden_state': current_state,
            'observed_event': observation
        })

        # Transition to next state
        current_state_idx = np.random.choice(num_hidden, p=transition_matrix[current_state_idx])

    return pd.DataFrame(data)


def show_hmm_analysis_page():
    """Display the enhanced Hidden Markov Model analysis page."""
    # Enhanced header
    create_hmm_header()

    # Advanced data input
    df = create_hmm_data_input()

    if df is not None and not df.empty:
        # Column mapping
        st.markdown("## ⚙️ **Column Mapping & Analysis**")

        col1, col2 = st.columns(2)
        with col1:
            hidden_state_col = st.selectbox(
                "Select hidden state column:",
                df.columns,
                index=df.columns.get_loc('hidden_state') if 'hidden_state' in df.columns else 0
            )
        with col2:
            observed_event_col = st.selectbox(
                "Select observed event column:",
                df.columns,
                index=df.columns.get_loc('observed_event') if 'observed_event' in df.columns else min(1, len(df.columns) - 1)
            )

        # Validate data
        is_valid, error_msg = validate_hmm_data(df, hidden_state_col, observed_event_col)

        if not is_valid:
            st.error(f"❌ Data validation error: {error_msg}")
            return

        # Analysis button
        if st.button("🔍 Analyze Hidden Markov Model", type="primary"):
            with st.spinner("🔄 Running HMM analysis..."):
                results = hidden_markov_model_analysis(df, hidden_state_col=hidden_state_col, observed_event_col=observed_event_col)

            if results:
                display_hmm_results(results, df, observed_event_col)
            else:
                st.error("❌ Could not analyze the Hidden Markov Model. Please check your data.")


def display_hmm_results(results, df, observed_event_col):
    """Display HMM analysis results with advanced visualizations."""
    st.success("✅ HMM Analysis completed successfully!")

    # Create enhanced tabs
    tabs = st.tabs([
        "🎭 Model Parameters",
        "📊 Probability Matrices",
        "🎯 Viterbi Decoding",
        "📈 Forward Algorithm",
        "🔍 Model Insights"
    ])

    with tabs[0]:
        st.markdown("### 🎭 **Hidden Markov Model Parameters**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔄 **Transition Probabilities**")
            st.markdown("*P(next_state | current_state)*")

            # Interactive heatmap
            fig_trans = create_plotly_heatmap(
                results['transition_probabilities'],
                "Hidden State Transitions"
            )
            st.plotly_chart(fig_trans, use_container_width=True)

        with col2:
            st.markdown("#### 📡 **Emission Probabilities**")
            st.markdown("*P(observation | hidden_state)*")

            # Interactive heatmap
            fig_emit = create_plotly_heatmap(
                results['emission_probabilities'],
                "Observation Emissions"
            )
            st.plotly_chart(fig_emit, use_container_width=True)

    with tabs[1]:
        st.markdown("### 📊 **Detailed Probability Analysis**")

        # Transition probabilities table
        st.markdown("#### 🔄 **Transition Matrix**")
        formatted_trans = format_matrix_display(results['transition_probabilities'])
        st.dataframe(formatted_trans, use_container_width=True)

        # Emission probabilities table
        st.markdown("#### 📡 **Emission Matrix**")
        formatted_emit = format_matrix_display(results['emission_probabilities'])
        st.dataframe(formatted_emit, use_container_width=True)

        # Steady state probabilities
        st.markdown("#### ⚖️ **Steady-State Probabilities**")
        formatted_steady = format_matrix_display(results['steady_state_probabilities'])
        st.dataframe(formatted_steady, use_container_width=True)

    with tabs[2]:
        st.markdown("### 🎯 **Viterbi Algorithm - Most Likely Path**")

        # Observation sequence input
        obs_sequence = st.text_input(
            "Enter observation sequence (comma separated):",
            value=",".join(df[observed_event_col].astype(str).tolist()[:10]),
            help="Enter a sequence of observations to decode"
        )
        obs_sequence = [x.strip() for x in obs_sequence.split(",") if x.strip()]

        if obs_sequence and st.button("🎯 Run Viterbi Decoding"):
            try:
                best_path, viterbi_values, backpointers, path_indices = results['viterbi_algorithm'](
                    obs_sequence,
                    results['initial_probs'],
                    results['transition_probabilities'],
                    results['emission_probabilities']
                )

                st.success(f"✅ Most likely hidden state sequence: **{' → '.join(best_path)}**")

                # Visualization of the path
                create_viterbi_visualization(obs_sequence, best_path, results['hidden_states'])

            except Exception as e:
                st.error(f"❌ Error in Viterbi decoding: {e}")

    with tabs[3]:
        st.markdown("### 📈 **Forward Algorithm - Sequence Probability**")

        if obs_sequence:
            try:
                prob, alphas = results['forward_algorithm'](
                    obs_sequence,
                    results['initial_probs'],
                    results['transition_probabilities'],
                    results['emission_probabilities']
                )

                st.success(f"✅ Sequence probability: **{prob:.2e}**")

                # Forward probabilities visualization
                create_forward_visualization(obs_sequence, alphas, results['hidden_states'])

            except Exception as e:
                st.error(f"❌ Error in forward algorithm: {e}")

    with tabs[4]:
        st.markdown("### 🔍 **Model Insights & Statistics**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 **Model Statistics**")
            st.metric("Hidden States", len(results['hidden_states']))
            st.metric("Observation Types", len(results['observed_events']))
            st.metric("Data Points", len(df))

            # Most probable states
            most_probable_state = results['steady_state_probabilities'].idxmax()[0]
            max_prob = results['steady_state_probabilities'].max()[0]
            st.metric("Most Probable State", most_probable_state, f"{max_prob:.3f}")

        with col2:
            st.markdown("#### 🎯 **Observation Probabilities**")
            obs_probs_df = pd.DataFrame(
                list(results['observation_probabilities'].items()),
                columns=['Observation', 'Probability']
            ).sort_values('Probability', ascending=False)

            st.dataframe(obs_probs_df, use_container_width=True)


def create_viterbi_visualization(observations, best_path, hidden_states):
    """Create visualization for Viterbi algorithm results."""
    fig = go.Figure()

    # Create path visualization
    fig.add_trace(go.Scatter(
        x=list(range(len(observations))),
        y=[hidden_states.index(state) for state in best_path],
        mode='lines+markers',
        name='Most Likely Path',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#667eea')
    ))

    fig.update_layout(
        title="Viterbi Algorithm - Most Likely Hidden State Sequence",
        xaxis_title="Time Step",
        yaxis_title="Hidden State",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(hidden_states))),
            ticktext=hidden_states
        ),
        height=400
    )

    # Add observation annotations
    for i, obs in enumerate(observations):
        fig.add_annotation(
            x=i,
            y=-0.5,
            text=obs,
            showarrow=False,
            font=dict(size=10)
        )

    st.plotly_chart(fig, use_container_width=True)


def create_forward_visualization(observations, alphas, hidden_states):
    """Create visualization for forward algorithm results."""
    fig = make_subplots(
        rows=len(hidden_states),
        cols=1,
        subplot_titles=[f"Forward Probability - {state}" for state in hidden_states],
        shared_xaxes=True
    )

    for i, state in enumerate(hidden_states):
        alpha_values = [alpha[i, :] for alpha in alphas]
        alpha_matrix = np.array(alpha_values).T

        fig.add_trace(
            go.Scatter(
                x=list(range(len(observations))),
                y=alpha_matrix[i, :],
                mode='lines+markers',
                name=f'{state}',
                line=dict(width=2),
                marker=dict(size=4)
            ),
            row=i+1, col=1
        )

    fig.update_layout(
        title="Forward Algorithm - State Probabilities Over Time",
        height=150 * len(hidden_states),
        showlegend=False
    )

    fig.update_xaxes(title_text="Time Step", row=len(hidden_states), col=1)

    st.plotly_chart(fig, use_container_width=True)
