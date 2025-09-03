"""
Seasonality Analysis
===================

This page analyzes monthly performance trends and seasonal patterns
for rental properties across different islands.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_data, seasonality
from ui_utils import load_data_with_spinner, format_currency, format_percentage

def main():
    st.set_page_config(
        page_title="Seasonality Analysis",
        page_icon="📅",
        layout="wide"
    )
    
    st.title("📅 Seasonality Analysis")
    st.markdown("""
    **Monthly performance trends and seasonal patterns analysis**
    
    Explore how rental performance varies throughout the year across different islands.
    """)
    
    # Load data
    data = load_data_with_spinner()
    
    # Get seasonality data
    seasonality_data = seasonality(data)
    
    # Metric selection
    st.subheader("📊 Select Metric to Analyze")
    
    col1, col2 = st.columns(2)
    
    with col1:
        metric = st.selectbox(
            "Choose a metric:",
            ['avg_adr', 'avg_occupancy', 'avg_revpar'],
            format_func=lambda x: {
                'avg_adr': 'Average Daily Rate (ADR)',
                'avg_occupancy': 'Average Occupancy Rate',
                'avg_revpar': 'Revenue per Available Room (RevPAR)'
            }[x]
        )
    
    with col2:
        show_table = st.checkbox("Show detailed table", value=True)
    
    # Create visualizations
    st.subheader("📈 Monthly Trends")
    
    # Line chart
    fig = px.line(
        seasonality_data, 
        x='month', 
        y=metric, 
        color='island',
        title=f'Monthly {metric.replace("avg_", "").title()} Trends by Island',
        labels={
            'month': 'Month',
            metric: metric.replace('avg_', '').title(),
            'island': 'Island'
        }
    )
    
    fig.update_layout(
        height=500,
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        yaxis_title=metric.replace('avg_', '').title()
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Bar chart comparison
    st.subheader("📊 Monthly Comparison")
    
    fig_bar = px.bar(
        seasonality_data,
        x='month',
        y=metric,
        color='island',
        barmode='group',
        title=f'Monthly {metric.replace("avg_", "").title()} Comparison',
        labels={
            'month': 'Month',
            metric: metric.replace('avg_', '').title(),
            'island': 'Island'
        }
    )
    
    fig_bar.update_layout(
        height=500,
        xaxis=dict(tickmode='linear', tick0=1, dtick=1)
    )
    
    st.plotly_chart(fig_bar, width='stretch')
    
    # Seasonality insights
    st.subheader("🔍 Seasonality Insights")
    
    # Calculate seasonal patterns
    island_insights = []
    
    for island in seasonality_data['island'].unique():
        island_data = seasonality_data[seasonality_data['island'] == island]
        
        # Peak and low months
        peak_month = island_data.loc[island_data[metric].idxmax(), 'month']
        low_month = island_data.loc[island_data[metric].idxmin(), 'month']
        
        # Seasonal variation
        max_val = island_data[metric].max()
        min_val = island_data[metric].min()
        variation = ((max_val - min_val) / min_val) * 100 if min_val > 0 else 0
        
        island_insights.append({
            'Island': island,
            'Peak Month': peak_month,
            'Low Month': low_month,
            'Peak Value': max_val,
            'Low Value': min_val,
            'Seasonal Variation (%)': round(variation, 1)
        })
    
    insights_df = pd.DataFrame(island_insights)
    
    # Format values based on metric
    if metric == 'avg_adr' or metric == 'avg_revpar':
        insights_df['Peak Value'] = insights_df['Peak Value'].apply(format_currency)
        insights_df['Low Value'] = insights_df['Low Value'].apply(format_currency)
    elif metric == 'avg_occupancy':
        insights_df['Peak Value'] = insights_df['Peak Value'].apply(format_percentage)
        insights_df['Low Value'] = insights_df['Low Value'].apply(format_percentage)
    
    st.dataframe(insights_df, width='stretch', hide_index=True)
    
    # Detailed table
    if show_table:
        st.subheader("📋 Detailed Monthly Data")
        
        # Format the display data
        display_data = seasonality_data.copy()
        

        # Format numeric columns
        if metric == 'avg_adr' or metric == 'avg_revpar':
            display_data[metric] = display_data[metric].apply(format_currency)
        elif metric == 'avg_occupancy':
            display_data[metric] = display_data[metric].apply(format_percentage)
        
        # Rename columns for display
        display_data.columns = ['Island', 'Month', 'ADR', 'Occupancy', 'RevPAR', 'Revenue', 'Month Name']
        
        st.dataframe(display_data, width='stretch', hide_index=True)
    
    # Seasonal recommendations
    st.subheader("💡 Seasonal Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Peak Season Strategy:**
        - Maximize pricing during high-demand months
        - Ensure optimal property condition
        - Focus on premium amenities
        - Implement dynamic pricing
        """)
    
    with col2:
        st.markdown("""
        **Low Season Strategy:**
        - Offer competitive pricing to maintain occupancy
        - Focus on longer stays and local market
        - Consider property maintenance and upgrades
        - Implement promotional campaigns
        """)

if __name__ == "__main__":
    main()
