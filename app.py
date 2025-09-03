"""
Investimate Analytics - Main Dashboard
=====================================

Main Streamlit application for rental property performance analysis.
This is the entry point for the Investimate Analytics dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_data, agg_island_summary
from ui_utils import load_data_with_spinner, format_currency, format_percentage, format_number, create_metric_container, inject_responsive_css, create_dynamic_sidebar

def calculate_island_performance_changes(data):
    """
    Calculate island performance changes by comparing first half vs second half of the year.
    Returns a dictionary with island performance metrics and their changes.
    """
    # Calculate metrics for first half (months 1-6) and second half (months 7-12)
    first_half = data[data['month'].isin([1, 2, 3, 4, 5, 6])]
    second_half = data[data['month'].isin([7, 8, 9, 10, 11, 12])]
    
    island_changes = {}
    
    for island in data['island'].unique():
        if pd.isna(island):
            continue
            
        # First half metrics
        first_half_data = first_half[first_half['island'] == island]
        second_half_data = second_half[second_half['island'] == island]
        
        if len(first_half_data) > 0 and len(second_half_data) > 0:
            # Calculate metrics for both halves
            first_adr = first_half_data['adr'].mean()
            first_occupancy = first_half_data['occupancy'].mean()
            first_revpar = first_half_data['revpar'].mean()
            first_revenue = first_half_data['revenue'].sum()
            
            second_adr = second_half_data['adr'].mean()
            second_occupancy = second_half_data['occupancy'].mean()
            second_revpar = second_half_data['revpar'].mean()
            second_revenue = second_half_data['revenue'].sum()
            
            # Calculate percentage changes
            adr_change = ((second_adr - first_adr) / first_adr * 100) if first_adr > 0 else 0
            occupancy_change = ((second_occupancy - first_occupancy) / first_occupancy * 100) if first_occupancy > 0 else 0
            revpar_change = ((second_revpar - first_revpar) / first_revpar * 100) if first_revpar > 0 else 0
            revenue_change = ((second_revenue - first_revenue) / first_revenue * 100) if first_revenue > 0 else 0
            
            island_changes[island] = {
                'current_adr': second_adr,
                'current_occupancy': second_occupancy,
                'current_revpar': second_revpar,
                'current_revenue': second_revenue,
                'adr_change': adr_change,
                'occupancy_change': occupancy_change,
                'revpar_change': revpar_change,
                'revenue_change': revenue_change
            }
    
    return island_changes

def main():
    st.set_page_config(
        page_title="Investimate Analytics",
        page_icon="📊",
        layout="wide"
    )
    
    # Inject custom CSS and create sidebar
    inject_responsive_css()
    create_dynamic_sidebar()
    
    st.title("📊 Investimate Analytics")
    st.markdown("""
    **Comprehensive rental property performance analysis and investment insights**
    
    Welcome to your intelligent property analytics dashboard. Explore performance metrics, 
    seasonal trends, and AI-powered investment recommendations for your rental portfolio.
    """)
    
    # Load data with spinner
    data = load_data_with_spinner()
    
    # Overall Portfolio KPIs
    st.header("📈 Overall Portfolio KPIs")
    
    # Calculate overall metrics
    total_revenue = data['revenue'].sum()
    avg_adr = data['adr'].mean()
    avg_occupancy = data['occupancy'].mean()
    avg_revpar = data['revpar'].mean()
    total_properties = data['property_id'].nunique()
    total_nights = data['occupancy'].sum()
    
    # Create KPI containers
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_container({
            'label': 'Total Revenue',
            'value': format_currency(total_revenue),
            'delta': None,
            'caption': 'Sum of all monthly revenues from all properties'
        })
    
    with col2:
        create_metric_container({
            'label': 'Average ADR',
            'value': format_currency(avg_adr),
            'delta': None,
            'caption': 'Average Daily Rate across all properties'
        })
    
    with col3:
        create_metric_container({
            'label': 'Average Occupancy',
            'value': format_percentage(avg_occupancy),
            'delta': None,
            'caption': 'Average occupancy rate across all properties'
        })
    
    with col4:
        create_metric_container({
            'label': 'Average RevPAR',
            'value': format_currency(avg_revpar),
            'delta': None,
            'caption': 'Revenue per Available Room (ADR × Occupancy)'
        })
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        create_metric_container({
            'label': 'Total Properties',
            'value': format_number(total_properties),
            'delta': None,
            'caption': 'Number of unique properties in portfolio'
        })
    
    with col6:
        create_metric_container({
            'label': 'Total Nights',
            'value': format_number(total_nights),
            'delta': None,
            'caption': 'Total occupied nights across all properties'
        })
    
    with col7:
        annualized_revenue = total_revenue / total_properties if total_properties > 0 else 0
        create_metric_container({
            'label': 'Revenue per Property',
            'value': format_currency(annualized_revenue),
            'delta': None,
            'caption': 'Average annualized revenue per property'
        })
    
    with col8:
        revenue_per_night = total_revenue / total_nights if total_nights > 0 else 0
        create_metric_container({
            'label': 'Revenue per Night',
            'value': format_currency(revenue_per_night),
            'delta': None,
            'caption': 'Average revenue per occupied night'
        })
    
    # Island KPI Summary
    st.header("🏝️ Island Performance Summary")
    
    island_summary = agg_island_summary(data)
    island_changes = calculate_island_performance_changes(data)
    
    # Create KPI metrics for each island
    for island in island_summary['island'].unique():
        if pd.isna(island):
            continue
            
        st.subheader(f"🏝️ {island}")
        
        # Get island data
        island_data = island_summary[island_summary['island'] == island].iloc[0]
        island_change_data = island_changes.get(island, {})
        
        # Create 4 columns for KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            adr_change = island_change_data.get('adr_change', 0)
            delta_text = f"{adr_change:+.1f}%" if adr_change != 0 else None
            create_metric_container({
                'label': 'Average ADR',
                'value': format_currency(island_data['avg_adr']),
                'delta': delta_text,
                'caption': f'Average Daily Rate for {island}'
            })
        
        with col2:
            occupancy_change = island_change_data.get('occupancy_change', 0)
            delta_text = f"{occupancy_change:+.1f}%" if occupancy_change != 0 else None
            create_metric_container({
                'label': 'Average Occupancy',
                'value': format_percentage(island_data['avg_occupancy']),
                'delta': delta_text,
                'caption': f'Average occupancy rate for {island}'
            })
        
        with col3:
            revpar_change = island_change_data.get('revpar_change', 0)
            delta_text = f"{revpar_change:+.1f}%" if revpar_change != 0 else None
            create_metric_container({
                'label': 'Average RevPAR',
                'value': format_currency(island_data['avg_revpar']),
                'delta': delta_text,
                'caption': f'Revenue per Available Room for {island}'
            })
        
        with col4:
            revenue_change = island_change_data.get('revenue_change', 0)
            delta_text = f"{revenue_change:+.1f}%" if revenue_change != 0 else None
            create_metric_container({
                'label': 'Total Revenue',
                'value': format_currency(island_data['total_revenue']),
                'delta': delta_text,
                'caption': f'Total revenue for {island}'
            })
        
        # Additional metrics row
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            create_metric_container({
                'label': 'Property Count',
                'value': format_number(island_data['property_count']),
                'delta': None,
                'caption': f'Number of properties in {island}'
            })
        
        with col6:
            create_metric_container({
                'label': 'Revenue per Property',
                'value': format_currency(island_data['annualized_revenue_per_property']),
                'delta': None,
                'caption': f'Average revenue per property in {island}'
            })
        
        with col7:
            # Calculate additional metrics
            island_data_filtered = data[data['island'] == island]
            avg_bedrooms = island_data_filtered['bedrooms_x'].mean() if 'bedrooms_x' in island_data_filtered.columns else 0
            create_metric_container({
                'label': 'Avg Bedrooms',
                'value': f"{avg_bedrooms:.1f}",
                'delta': None,
                'caption': f'Average bedrooms per property in {island}'
            })
        
        with col8:
            # Calculate performance score (combination of metrics)
            performance_score = (island_data['avg_occupancy'] * 100 + 
                               (island_data['avg_adr'] / 100) + 
                               (island_data['avg_revpar'] / 10))
            create_metric_container({
                'label': 'Performance Score',
                'value': f"{performance_score:.1f}",
                'delta': None,
                'caption': f'Overall performance score for {island}'
            })
        
        st.markdown("---")  # Separator between islands
    
    # Add explanation about the change calculation
    st.info("""
    **📊 Change Indicators:** The percentage changes shown above compare the second half of the year (July-December) 
    against the first half (January-June) to show seasonal performance trends. 
    - 🟢 **Green**: Positive growth (improvement)
    - 🔴 **Red**: Negative change (decline)
    - ⚪ **Gray**: No significant change
    """)
    
    # Debug: Show actual delta values
    with st.expander("🔍 Debug: Actual Delta Values"):
        st.write("**Island Performance Changes (H1 vs H2):**")
        for island, changes in island_changes.items():
            st.write(f"**{island}:**")
            st.write(f"- ADR Change: {changes['adr_change']:+.1f}%")
            st.write(f"- Occupancy Change: {changes['occupancy_change']:+.1f}%")
            st.write(f"- RevPAR Change: {changes['revpar_change']:+.1f}%")
            st.write(f"- Revenue Change: {changes['revenue_change']:+.1f}%")
            st.write("---")
        
        # Test color logic
        st.write("**Color Logic Test:**")
        test_values = [5.3, -2.1, 0.0, -8.7, 12.5]
        for val in test_values:
            if val > 0:
                color = "🟢 Green (normal)"
            elif val < 0:
                color = "🔴 Red (inverse)"
            else:
                color = "⚪ Gray (off)"
            st.write(f"Value: {val:+.1f}% → {color}")
    
    # Island Comparison Charts
    st.header("📊 Island Performance Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # ADR Comparison
        fig_adr = px.bar(
            island_summary,
            x='island',
            y='avg_adr',
            title='Average ADR by Island',
            labels={'avg_adr': 'Average ADR (€)', 'island': 'Island'},
            color='avg_adr',
            color_continuous_scale='Blues'
        )
        fig_adr.update_layout(height=400)
        st.plotly_chart(fig_adr, width='stretch')
    
    with col2:
        # Occupancy Comparison
        fig_occupancy = px.bar(
            island_summary,
            x='island',
            y='avg_occupancy',
            title='Average Occupancy by Island',
            labels={'avg_occupancy': 'Average Occupancy (%)', 'island': 'Island'},
            color='avg_occupancy',
            color_continuous_scale='Greens'
        )
        fig_occupancy.update_layout(height=400)
        st.plotly_chart(fig_occupancy, width='stretch')
    
    with col3:
        # RevPAR Comparison
        fig_revpar = px.bar(
            island_summary,
            x='island',
            y='avg_revpar',
            title='Average RevPAR by Island',
            labels={'avg_revpar': 'Average RevPAR (€)', 'island': 'Island'},
            color='avg_revpar',
            color_continuous_scale='Reds'
        )
        fig_revpar.update_layout(height=400)
        st.plotly_chart(fig_revpar, width='stretch')
    
    # Quick Insights
    st.header("💡 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Portfolio Overview:**
        - Total portfolio value based on revenue performance
        - Performance metrics across different islands
        - Property count and distribution analysis
        - Revenue efficiency indicators
        """)
    
    with col2:
        st.markdown("""
        **Navigation:**
        - Use the sidebar to explore detailed analyses
        - Each page provides specific insights and visualizations
        - AI Investment Recommendations for data-driven decisions
        - Comprehensive documentation available
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        📊 Investimate Analytics - Intelligent Property Performance Analysis<br>
        <em>Navigate using the sidebar to explore detailed insights and AI recommendations</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()