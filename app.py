"""
Stochastic Process Analyzer - Main Application

A comprehensive tool for analyzing various stochastic processes including:
- Markov Chains
- Hidden Markov Models
- Queuing Theory
- Process Simulation
"""

import streamlit as st
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.settings import PAGE_CONFIG
from app_pages.home import show_home_page
from app_pages.markov_analysis import show_markov_analysis_page
from app_pages.hmm_analysis import show_hmm_analysis_page
from app_pages.queuing_analysis import show_queuing_analysis_page
from app_pages.simulation import show_simulation_page


def load_css():
    """Load custom CSS styles."""
    try:
        with open('assets/styles.css', 'r', encoding='utf-8') as f:
            css = f.read()
        st.markdown(f"""
            <style>
                {css}
            </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback to inline styles if CSS file not found
        st.markdown("""
            <style>
                .main {
                    padding: 2rem;
                    background-color: #f8f9fa;
                }
                .metric-card {
                    background: #0E1117;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 15px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    border-left: 4px solid #3498db;
                }
                .dashboard-card {
                    background: #0E1117;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }
                .info-text {
                    background-color: #0E1117;
                    border-left: 4px solid #2196f3;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
            </style>
        """, unsafe_allow_html=True)


def create_enhanced_sidebar():
    """Create an enhanced sidebar with better styling and navigation."""


    # Navigation section
    st.sidebar.markdown("""
    <div style="padding: 0 1.5rem; margin-bottom: 2rem;">
        <h3 style="color: #3b82f6; font-size: 1rem; font-weight: 600; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.05em;">
            📊 Analysis Modules
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced radio buttons with icons
    app_mode = st.sidebar.radio(
        "Navigation Menu",  # Proper label for accessibility
        [
            "🏠 Home",
            "🔗 Markov Chain Analysis",
            "🔍 Hidden Markov Model Analysis",
            "🚶‍♂ Queuing Theory Analysis",
            "🎮 Simulation"
        ],
        label_visibility="collapsed"
    )

    # Sidebar statistics
    st.sidebar.markdown("""
    <div style="padding: 1.5rem; margin: 2rem 0.8rem; background: linear-gradient(135deg, rgba(30, 64, 175, 0.15), rgba(124, 58, 237, 0.1)); border-radius: 16px; border: 1px solid rgba(30, 64, 175, 0.3);">
        <h4 style="color: #1e40af; font-size: 1rem; font-weight: 700; margin-bottom: 1.2rem; text-transform: uppercase; letter-spacing: 0.08em; text-align: center;">
            ⚡ Platform Stats
        </h4>
        <div style="display: flex; flex-direction: column; gap: 0.8rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
                <span style="color: #e2e8f0; font-size: 0.9rem; font-weight: 500;">🎯 Algorithms:</span>
                <span style="color: #3b82f6; font-weight: 700; font-size: 1rem;">15+</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
                <span style="color: #e2e8f0; font-size: 0.9rem; font-weight: 500;">📊 Visualizations:</span>
                <span style="color: #0891b2; font-weight: 700; font-size: 1rem;">20+</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0;">
                <span style="color: #e2e8f0; font-size: 0.9rem; font-weight: 500;">🔬 Analysis Types:</span>
                <span style="color: #7c3aed; font-weight: 700; font-size: 1rem;">4</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Help section
    st.sidebar.markdown("""
    <div style="padding: 1.5rem; margin: 2rem 0.8rem; background: linear-gradient(135deg, rgba(8, 145, 178, 0.15), rgba(5, 150, 105, 0.15)); border-radius: 16px; border: 1px solid rgba(8, 145, 178, 0.3); box-shadow: 0 4px 12px rgba(8, 145, 178, 0.1);">
        <h4 style="color: #0891b2; font-size: 1rem; font-weight: 700; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;">
            💡 Pro Tip
        </h4>
        <p style="color: #cbd5e1; font-size: 0.85rem; margin: 0; line-height: 1.5; font-weight: 500;">
            Start with example datasets to explore the platform's capabilities before uploading your own data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    return app_mode


def main():
    """Main application function."""
    # Set page configuration
    st.set_page_config(**PAGE_CONFIG)

    # Load custom CSS
    load_css()

    # Enhanced sidebar navigation
    app_mode = create_enhanced_sidebar()

    # Debug: Show current selection (remove this later)
    # st.write(f"Debug - Selected mode: {app_mode}")

    # Route to appropriate page based on enhanced navigation
    try:
        if app_mode == "🏠 Home":
            show_home_page()
        elif app_mode == "🔗 Markov Chain Analysis":
            show_markov_analysis_page()
        elif app_mode == "🔍 Hidden Markov Model Analysis":
            show_hmm_analysis_page()
        elif app_mode == "🚶‍♂ Queuing Theory Analysis":
            show_queuing_analysis_page()
        elif app_mode == "🎮 Simulation":
            show_simulation_page()
        else:
            # Fallback to home if no match
            show_home_page()
    except Exception as e:
        st.error(f"Error loading page: {e}")
        # Simple fallback home page
        st.title("🚀 Stochastic Process Analyzer")
        st.write("Welcome to the Stochastic Process Analyzer!")


if __name__ == "__main__":
    main()