"""
Bedrooms and Property Types Analysis
====================================

This page analyzes rental performance by bedroom count and property type,
helping identify the most profitable configurations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_data, bedrooms_perf, ptype_perf
from ui_utils import load_data_with_spinner, format_currency, format_percentage

def main():
    st.set_page_config(
        page_title="Bedrooms and Property Types",
        page_icon="🏠",
        layout="wide"
    )
    
    st.title("🏠 Bedrooms and Property Types Analysis")
    st.markdown("""
    **Performance analysis by bedroom count and property type**
    
    Discover which bedroom configurations and property types generate the highest returns.
    """)
    
    # Load data
    data = load_data_with_spinner()
    
    # Create tabs
    tab1, tab2 = st.tabs(["🛏️ Bedrooms Analysis", "🏘️ Property Types Analysis"])
    
    with tab1:
        st.subheader("🛏️ Performance by Bedroom Count")
        
        # Get bedroom performance data
        bedroom_data = bedrooms_perf(data)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best_bedrooms = bedroom_data.loc[bedroom_data['avg_revpar'].idxmax(), 'bedrooms']
            best_revpar = bedroom_data['avg_revpar'].max()
            st.metric(
                "Best Performing",
                f"{best_bedrooms} bedrooms",
                f"€{best_revpar:.0f} RevPAR"
            )
        
        with col2:
            total_properties = bedroom_data['property_count'].sum()
            st.metric("Total Properties", f"{total_properties:,}")
        
        with col3:
            avg_bedrooms = (bedroom_data['bedrooms'] * bedroom_data['property_count']).sum() / bedroom_data['property_count'].sum()
            st.metric("Average Bedrooms", f"{avg_bedrooms:.1f}")
        
        with col4:
            total_revenue = bedroom_data['total_revenue'].sum()
            st.metric("Total Revenue", format_currency(total_revenue))
        
        # Bedroom performance chart
        fig_bedrooms = px.bar(
            bedroom_data,
            x='bedrooms',
            y='avg_revpar',
            title='Average RevPAR by Bedroom Count',
            labels={'bedrooms': 'Number of Bedrooms', 'avg_revpar': 'Average RevPAR (€)'},
            color='avg_revpar',
            color_continuous_scale='Viridis'
        )
        
        fig_bedrooms.update_layout(height=500)
        st.plotly_chart(fig_bedrooms, )
        
        # Detailed bedroom table
        st.subheader("📊 Bedroom Performance Details")
        
        display_bedroom = bedroom_data.copy()
        display_bedroom['avg_adr'] = display_bedroom['avg_adr'].apply(format_currency)
        display_bedroom['avg_occupancy'] = display_bedroom['avg_occupancy'].apply(format_percentage)
        display_bedroom['avg_revpar'] = display_bedroom['avg_revpar'].apply(format_currency)
        display_bedroom['total_revenue'] = display_bedroom['total_revenue'].apply(format_currency)
        display_bedroom['revenue_per_property'] = display_bedroom['revenue_per_property'].apply(format_currency)
        
        display_bedroom.columns = [
            'Bedrooms', 'Avg ADR', 'Avg Occupancy', 'Avg RevPAR', 
            'Total Revenue', 'Property Count', 'Revenue per Property'
        ]
        
        st.dataframe(display_bedroom, , hide_index=True)
    
    with tab2:
        st.subheader("🏘️ Performance by Property Type")
        
        # Get property type performance data
        type_data = ptype_perf(data)
        
        # Filter to top property types (at least 10 properties)
        type_data_filtered = type_data[type_data['property_count'] >= 10].head(12)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best_type = type_data_filtered.loc[type_data_filtered['avg_revpar'].idxmax(), 'property_type']
            best_revpar = type_data_filtered['avg_revpar'].max()
            st.metric(
                "Best Performing Type",
                best_type[:20] + "..." if len(best_type) > 20 else best_type,
                f"€{best_revpar:.0f} RevPAR"
            )
        
        with col2:
            total_types = len(type_data_filtered)
            st.metric("Property Types", total_types)
        
        with col3:
            total_properties = type_data_filtered['property_count'].sum()
            st.metric("Total Properties", f"{total_properties:,}")
        
        with col4:
            total_revenue = type_data_filtered['total_revenue'].sum()
            st.metric("Total Revenue", format_currency(total_revenue))
        
        # Property type performance charts
        st.subheader("📈 Property Type Performance")
        
        # Create three separate charts for better readability
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # ADR Chart
            fig_adr = px.bar(
                type_data_filtered.sort_values('avg_adr', ascending=True),
                x='avg_adr',
                y='property_type',
                orientation='h',
                title='Average ADR by Property Type',
                labels={'avg_adr': 'Average ADR (€)', 'property_type': 'Property Type'},
                color='avg_adr',
                color_continuous_scale='Blues'
            )
            fig_adr.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_adr, )
        
        with col2:
            # Occupancy Chart
            fig_occ = px.bar(
                type_data_filtered.sort_values('avg_occupancy', ascending=True),
                x='avg_occupancy',
                y='property_type',
                orientation='h',
                title='Average Occupancy by Property Type',
                labels={'avg_occupancy': 'Average Occupancy (%)', 'property_type': 'Property Type'},
                color='avg_occupancy',
                color_continuous_scale='Greens'
            )
            fig_occ.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_occ, )
        
        with col3:
            # RevPAR Chart
            fig_revpar = px.bar(
                type_data_filtered.sort_values('avg_revpar', ascending=True),
                x='avg_revpar',
                y='property_type',
                orientation='h',
                title='Average RevPAR by Property Type',
                labels={'avg_revpar': 'Average RevPAR (€)', 'property_type': 'Property Type'},
                color='avg_revpar',
                color_continuous_scale='Reds'
            )
            fig_revpar.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_revpar, )
        
        # Detailed property type table
        st.subheader("📊 Property Type Performance Details")
        
        display_type = type_data_filtered.copy()
        display_type['avg_adr'] = display_type['avg_adr'].apply(format_currency)
        display_type['avg_occupancy'] = display_type['avg_occupancy'].apply(format_percentage)
        display_type['avg_revpar'] = display_type['avg_revpar'].apply(format_currency)
        display_type['total_revenue'] = display_type['total_revenue'].apply(format_currency)
        display_type['revenue_per_property'] = display_type['revenue_per_property'].apply(format_currency)
        
        display_type.columns = [
            'Property Type', 'Avg ADR', 'Avg Occupancy', 'Avg RevPAR',
            'Total Revenue', 'Property Count', 'Revenue per Property'
        ]
        
        st.dataframe(display_type, , hide_index=True)
    
    # Insights and recommendations
    st.subheader("💡 Key Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Bedroom Configuration Insights:**
        - Focus on bedroom counts with highest RevPAR
        - Consider market demand for different sizes
        - Balance between capacity and pricing power
        - Monitor occupancy vs. rate trade-offs
        """)
    
    with col2:
        st.markdown("""
        **Property Type Insights:**
        - Invest in high-performing property types
        - Consider market saturation for popular types
        - Evaluate amenity requirements by type
        - Monitor competitive positioning
        """)

if __name__ == "__main__":
    main()
