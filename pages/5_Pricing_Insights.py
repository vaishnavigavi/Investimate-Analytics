"""
Pricing Insights and Optimization
=================================

This page provides price elasticity analysis and optimization recommendations
to help maximize revenue through strategic pricing.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_data, price_elasticity_simple
from ui_utils import load_data_with_spinner, format_currency, format_percentage

def main():
    st.set_page_config(
        page_title="Pricing Insights",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Pricing Insights and Optimization")
    st.markdown("""
    **Price elasticity analysis and revenue optimization strategies**
    
    Discover optimal pricing strategies based on demand elasticity and market conditions.
    """)
    
    # Load data
    data = load_data_with_spinner()
    
    # Get price elasticity data
    elasticity_data = price_elasticity_simple(data)
    
    # Summary metrics
    st.subheader("📊 Price Elasticity Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_elasticity = elasticity_data['elasticity'].mean()
        st.metric(
            "Average Elasticity",
            f"{avg_elasticity:.3f}",
            help="Negative values indicate price sensitivity"
        )
    
    with col2:
        elastic_months = len(elasticity_data[elasticity_data['elasticity'] < -0.1])
        st.metric("Elastic Months", elastic_months)
    
    with col3:
        inelastic_months = len(elasticity_data[elasticity_data['elasticity'] > -0.1])
        st.metric("Inelastic Months", inelastic_months)
    
    with col4:
        total_months = len(elasticity_data)
        st.metric("Total Months Analyzed", total_months)
    
    # Price elasticity analysis
    st.subheader("📈 Price Elasticity Analysis")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Elasticity Overview", "Optimal Pricing", "Revenue Impact"])
    
    with tab1:
        # Elasticity by month and island
        fig_elasticity = px.bar(
            elasticity_data,
            x='month',
            y='elasticity',
            color='island',
            title='Price Elasticity by Month and Island',
            labels={'elasticity': 'Price Elasticity', 'month': 'Month', 'island': 'Island'},
            barmode='group'
        )
        fig_elasticity.update_layout(height=500)
        st.plotly_chart(fig_elasticity, width='stretch')
        
        # Elasticity distribution
        fig_dist = px.histogram(
            elasticity_data,
            x='elasticity',
            nbins=20,
            title='Distribution of Price Elasticity',
            labels={'elasticity': 'Price Elasticity', 'count': 'Frequency'}
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, width='stretch')
    
    with tab2:
        # Optimal pricing recommendations
        st.subheader("🎯 Optimal Pricing Recommendations")
        
        # Island selection for detailed analysis
        selected_island = st.selectbox(
            "Select Island for Detailed Analysis:",
            elasticity_data['island'].unique()
        )
        
        island_data = elasticity_data[elasticity_data['island'] == selected_island]
        
        if len(island_data) > 0:
            # Optimal ADR chart
            fig_optimal = px.line(
                island_data,
                x='month',
                y='optimal_adr',
                title=f'Optimal ADR Recommendations - {selected_island}',
                labels={'optimal_adr': 'Optimal ADR (€)', 'month': 'Month'},
                markers=True
            )
            fig_optimal.update_layout(height=500)
            st.plotly_chart(fig_optimal, width='stretch')
            
            # Current vs Optimal pricing
            # Get current average ADR for comparison
            current_adr = data[data['island'] == selected_island].groupby('month')['adr'].mean().reset_index()
            current_adr.columns = ['month', 'current_adr']
            
            comparison_data = island_data.merge(current_adr, on='month', how='left')
            
            fig_comparison = go.Figure()
            
            fig_comparison.add_trace(go.Scatter(
                x=comparison_data['month'],
                y=comparison_data['current_adr'],
                mode='lines+markers',
                name='Current ADR',
                line=dict(color='blue')
            ))
            
            fig_comparison.add_trace(go.Scatter(
                x=comparison_data['month'],
                y=comparison_data['optimal_adr'],
                mode='lines+markers',
                name='Optimal ADR',
                line=dict(color='red')
            ))
            
            fig_comparison.update_layout(
                title=f'Current vs Optimal ADR - {selected_island}',
                xaxis_title='Month',
                yaxis_title='ADR (€)',
                height=500
            )
            
            st.plotly_chart(fig_comparison, width='stretch')
            
            # Pricing recommendations table
            st.subheader("📋 Monthly Pricing Recommendations")
            
            display_recommendations = comparison_data.copy()
            display_recommendations['current_adr'] = display_recommendations['current_adr'].apply(format_currency)
            display_recommendations['optimal_adr'] = display_recommendations['optimal_adr'].apply(format_currency)
            display_recommendations['elasticity'] = display_recommendations['elasticity'].round(3)
            
            # Calculate potential revenue impact
            display_recommendations['price_change_pct'] = (
                (display_recommendations['optimal_adr'] - display_recommendations['current_adr']) / 
                display_recommendations['current_adr'] * 100
            ).round(1)
            
            display_recommendations.columns = [
                'Island', 'Month', 'Elasticity', 'Current ADR', 'Optimal ADR', 'Price Change %'
            ]
            
            st.dataframe(display_recommendations, width='stretch', hide_index=True)
    
    with tab3:
        # Revenue impact analysis
        st.subheader("💵 Revenue Impact Analysis")
        
        # Calculate potential revenue impact
        revenue_impact = []
        
        for _, row in elasticity_data.iterrows():
            island = row['island']
            month = row['month']
            current_adr = data[(data['island'] == island) & (data['month'] == month)]['adr'].mean()
            optimal_adr = row['optimal_adr']
            elasticity = row['elasticity']
            
            # Estimate occupancy change based on elasticity
            price_change_pct = (optimal_adr - current_adr) / current_adr
            occupancy_change_pct = elasticity * price_change_pct
            
            # Get current occupancy
            current_occupancy = data[(data['island'] == island) & (data['month'] == month)]['occupancy'].mean()
            new_occupancy = current_occupancy * (1 + occupancy_change_pct)
            
            # Calculate revenue impact
            current_revpar = current_adr * current_occupancy
            new_revpar = optimal_adr * new_occupancy
            revenue_impact_pct = (new_revpar - current_revpar) / current_revpar * 100
            
            revenue_impact.append({
                'island': island,
                'month': month,
                'current_adr': current_adr,
                'optimal_adr': optimal_adr,
                'current_occupancy': current_occupancy,
                'new_occupancy': new_occupancy,
                'revenue_impact_pct': revenue_impact_pct
            })
        
        revenue_df = pd.DataFrame(revenue_impact)
        
        if len(revenue_df) > 0:
            # Revenue impact by month
            fig_revenue = px.bar(
                revenue_df,
                x='month',
                y='revenue_impact_pct',
                color='island',
                title='Potential Revenue Impact by Month',
                labels={'revenue_impact_pct': 'Revenue Impact (%)', 'month': 'Month', 'island': 'Island'},
                barmode='group'
            )
            fig_revenue.update_layout(height=500)
            st.plotly_chart(fig_revenue, width='stretch')
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_impact = revenue_df['revenue_impact_pct'].mean()
                st.metric("Average Revenue Impact", f"{avg_impact:.1f}%")
            
            with col2:
                positive_impact = len(revenue_df[revenue_df['revenue_impact_pct'] > 0])
                st.metric("Months with Positive Impact", positive_impact)
            
            with col3:
                max_impact = revenue_df['revenue_impact_pct'].max()
                st.metric("Maximum Revenue Impact", f"{max_impact:.1f}%")
    
    # Detailed elasticity table
    st.subheader("📊 Detailed Price Elasticity Data")
    
    # Format the display data
    display_elasticity = elasticity_data.copy()
    display_elasticity['elasticity'] = display_elasticity['elasticity'].round(3)
    display_elasticity['optimal_adr'] = display_elasticity['optimal_adr'].apply(format_currency)
    
    display_elasticity.columns = ['Island', 'Month', 'Elasticity', 'Optimal ADR']
    
    st.dataframe(display_elasticity, width='stretch', hide_index=True)
    
    # Pricing strategy recommendations
    st.subheader("💡 Pricing Strategy Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Elastic Demand (Elasticity < -0.1):**
        - Lower prices to increase occupancy
        - Focus on volume over rate
        - Implement promotional pricing
        - Monitor competitive pricing closely
        """)
    
    with col2:
        st.markdown("""
        **Inelastic Demand (Elasticity > -0.1):**
        - Increase prices to maximize revenue
        - Focus on rate over occupancy
        - Premium positioning strategies
        - Limited-time offers less effective
        """)
    
    # Methodology explanation
    st.subheader("🔬 Methodology")
    
    st.markdown("""
    **Price Elasticity Analysis:**
    
    The price elasticity analysis uses a simple linear regression model:
    ```
    Occupancy = α + β × ADR + ε
    ```
    
    Where:
    - **β (Elasticity)** is the price elasticity coefficient
    - **Negative β** indicates price-sensitive demand (elastic)
    - **Positive β** indicates price-insensitive demand (inelastic)
    
    **Optimal Pricing:**
    
    The optimal ADR is calculated to maximize RevPAR (Revenue per Available Room):
    ```
    RevPAR = ADR × Occupancy
    ```
    
    **Note:** Optimal ADR represents the RevPAR-optimal price within the observed range and should be validated with market testing.
    """)

if __name__ == "__main__":
    main()
