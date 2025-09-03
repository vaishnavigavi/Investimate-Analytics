"""
UI Utility Functions for Investimate Analytics
Provides consistent formatting and responsive design across all pages
"""

import streamlit as st
import pandas as pd
import numpy as np

def format_currency(value, decimals=0):
    """Format currency values with proper scaling"""
    if pd.isna(value) or value == 0:
        return "€0"
    
    if abs(value) >= 1_000_000:
        return f"€{value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"€{value/1_000:.1f}K"
    else:
        return f"€{value:,.{decimals}f}"

def format_percentage(value, decimals=1):
    """Format percentage values"""
    if pd.isna(value):
        return "0%"
    return f"{value:.{decimals}f}%"

def format_number(value, decimals=1):
    """Format general numbers with proper scaling"""
    if pd.isna(value):
        return "0"
    
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:,.{decimals}f}"

def format_table_numbers(df, columns_config):
    """
    Format table numbers based on column configuration
    
    Args:
        df: DataFrame to format
        columns_config: Dict with column names as keys and format types as values
                       Format types: 'currency', 'percentage', 'number', 'integer'
    """
    formatted_df = df.copy()
    
    for col, format_type in columns_config.items():
        if col in formatted_df.columns:
            if format_type == 'currency':
                formatted_df[col] = formatted_df[col].apply(lambda x: format_currency(x, 2) if pd.notna(x) else "€0")
            elif format_type == 'percentage':
                formatted_df[col] = formatted_df[col].apply(lambda x: format_percentage(x) if pd.notna(x) else "0%")
            elif format_type == 'number':
                formatted_df[col] = formatted_df[col].apply(lambda x: format_number(x, 1) if pd.notna(x) else "0")
            elif format_type == 'integer':
                formatted_df[col] = formatted_df[col].apply(lambda x: format_number(x, 0) if pd.notna(x) else "0")
    
    return formatted_df

def create_responsive_columns(num_columns, gap="small"):
    """
    Create responsive columns with proper spacing
    
    Args:
        num_columns: Number of columns to create
        gap: Gap size between columns ("small", "medium", "large")
    """
    if gap == "small":
        st.markdown("<br>", unsafe_allow_html=True)
    elif gap == "medium":
        st.markdown("<br><br>", unsafe_allow_html=True)
    elif gap == "large":
        st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    return st.columns(num_columns)

def create_metric_container(metric_data, container_class="metric-container"):
    """
    Create a simple metric using Streamlit's native st.metric
    
    Args:
        metric_data: Dict with 'label', 'value', 'delta', 'delta_color', 'help', 'caption'
        container_class: CSS class for styling (not used in native approach)
    """
    # Use Streamlit's native metric component
    delta_value = None
    delta_color = "normal"
    
    if 'delta' in metric_data and metric_data['delta']:
        # Extract numeric value from delta string (e.g., "-6.1%" -> -6.1)
        delta_text = metric_data['delta']
        if '%' in delta_text:
            delta_numeric = float(delta_text.replace('%', '').replace('vs H1', '').strip())
            delta_value = delta_text  # Keep the formatted string with %
        else:
            # If no % suffix, add it
            delta_numeric = float(delta_text)
            delta_value = f"{delta_numeric:+.1f}%"  # Format with + sign and % suffix
        
        # Auto-determine delta color based on numeric value
        if delta_numeric > 0:
            delta_color = "normal"  # Green for positive changes
        elif delta_numeric < 0:
            delta_color = "inverse"  # Red for negative changes
        else:
            delta_color = "off"  # Gray for no change
        
        # Override with explicit color if provided
        delta_color = metric_data.get('delta_color', delta_color)
    
    # Use Streamlit's native metric
    st.metric(
        label=metric_data.get('label', ''),
        value=metric_data.get('value', ''),
        delta=delta_value,
        help=metric_data.get('help', ''),
        delta_color=delta_color
    )
    
    # Add caption if present
    if 'caption' in metric_data and metric_data['caption']:
        st.caption(metric_data['caption'])

def inject_responsive_css():
    """Inject responsive CSS for better mobile and tablet support"""
    st.markdown("""
    <style>
    
    /* Better table formatting */
    .stDataFrame {
        font-size: 0.9rem;
    }
    
    /* Responsive charts */
    .stPlotlyChart {
        width: 100% !important;
    }
    
    /* Better spacing for sections */
    .stSubheader {
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Info box improvements */
    .stInfo {
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    
    /* Success/Error message improvements */
    .stSuccess {
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    
    .stError {
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }
    
    /* Tab improvements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
    
    /* Selectbox improvements */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    /* Number input improvements */
    .stNumberInput > div > div > input {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_dynamic_sidebar():
    """Create a clean sidebar with just the title"""
    with st.sidebar:
        st.title("🏖️ Investimate Analytics")
        st.markdown("---")
        st.markdown("**Use the navigation above to explore different analytics pages.**")

def load_data_with_spinner():
    """
    Standardized data loading function with spinner and error handling
    Returns the loaded data or None if loading fails
    """
    with st.spinner("Loading data..."):
        try:
            from utils import load_data
            data = load_data()
            st.success("Data loaded successfully!")
            return data
        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")
            return None

def create_performance_summary_cards(data, title="Performance Summary"):
    """
    Create responsive performance summary cards
    
    Args:
        data: Dict with performance metrics
        title: Title for the section
    """
    st.subheader(f"📊 {title}")
    
    # Create responsive columns based on number of metrics
    num_metrics = len(data)
    if num_metrics <= 2:
        cols = st.columns(2)
    elif num_metrics <= 4:
        cols = st.columns(4)
    else:
        # If more than 4 metrics, create multiple rows
        remaining_metrics = list(data.items())
        current_row = 0
        
        while remaining_metrics:
            row_metrics = remaining_metrics[:4]
            remaining_metrics = remaining_metrics[4:]
            
            if current_row > 0:
                st.markdown("<br>", unsafe_allow_html=True)
            
            row_cols = st.columns(4)
            for i, (key, metric_data) in enumerate(row_metrics):
                with row_cols[i]:
                    create_metric_container(metric_data)
            
            current_row += 1
        return
    
    # For 4 or fewer metrics, display in single row
    for i, (key, metric_data) in enumerate(data.items()):
        with cols[i]:
            create_metric_container(metric_data)

def create_responsive_chart_container(chart_func, chart_title, chart_description=None):
    """
    Create a responsive chart container with title and description
    
    Args:
        chart_func: Function that creates the chart
        chart_title: Title for the chart
        chart_description: Optional description
    """
    st.subheader(f"📈 {chart_title}")
    
    if chart_description:
        st.info(chart_description)
    
    # Create the chart
    chart_func()
    
    st.markdown("---")

def create_insights_section(insights, title="💡 Key Insights"):
    """
    Create an insights section with formatted insights
    
    Args:
        insights: List of insight strings
        title: Title for the insights section
    """
    st.subheader(title)
    
    for i, insight in enumerate(insights, 1):
        st.info(f"**{i}.** {insight}")
    
    st.markdown("---")

def create_data_table_with_formatting(df, title="Data Table", columns_config=None):
    """
    Create a formatted data table with proper number formatting
    
    Args:
        df: DataFrame to display
        title: Title for the table
        columns_config: Dict with column formatting configuration
    """
    st.subheader(f"📋 {title}")
    
    if columns_config:
        formatted_df = format_table_numbers(df, columns_config)
    else:
        formatted_df = df
    
    st.dataframe(formatted_df, width='stretch')
    
    st.markdown("---")

def create_tabbed_analysis(tabs_config, title="Analysis"):
    """
    Create tabbed analysis sections
    
    Args:
        tabs_config: Dict with tab names as keys and content functions as values
        title: Title for the tabbed section
    """
    st.subheader(f"📊 {title}")
    
    tab_names = list(tabs_config.keys())
    tabs = st.tabs(tab_names)
    
    for i, (tab_name, content_func) in enumerate(tabs_config.items()):
        with tabs[i]:
            content_func()
    
    st.markdown("---")
