"""
Amenities and Management Analysis
=================================

This page analyzes the impact of amenities and management features
on rental property performance and revenue.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_data, uplift_table
from ui_utils import load_data_with_spinner, format_currency, format_percentage

def main():
    st.set_page_config(
        page_title="Amenities and Management",
        page_icon="⭐",
        layout="wide"
    )
    
    st.title("⭐ Amenities and Management Analysis")
    st.markdown("""
    **Impact of amenities and management features on rental performance**
    
    Discover which amenities and management features provide the highest return on investment.
    """)
    
    # Load data
    data = load_data_with_spinner()
    
    # Get uplift analysis
    uplift_data = uplift_table(data)
    
    # Summary metrics
    st.subheader("📊 Amenity Impact Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_amenity = uplift_data.loc[uplift_data['revpar_uplift_pct'].idxmax(), 'feature']
        best_uplift = uplift_data['revpar_uplift_pct'].max()
        st.metric(
            "Highest RevPAR Impact",
            best_amenity.title(),
            f"+{best_uplift:.1f}%"
        )
    
    with col2:
        total_features = len(uplift_data)
        st.metric("Features Analyzed", total_features)
    
    with col3:
        avg_revpar_uplift = uplift_data['revpar_uplift_pct'].mean()
        st.metric("Average RevPAR Uplift", f"+{avg_revpar_uplift:.1f}%")
    
    with col4:
        positive_features = len(uplift_data[uplift_data['revpar_uplift_pct'] > 0])
        st.metric("Positive Impact Features", positive_features)
    
    # Uplift analysis table
    st.subheader("📈 Amenity Uplift Analysis")
    
    # Format the display data
    display_uplift = uplift_data.copy()
    
    # Format currency columns
    currency_cols = ['adr_with', 'adr_without', 'revpar_with', 'revpar_without']
    for col in currency_cols:
        if col in display_uplift.columns:
            display_uplift[col] = display_uplift[col].apply(format_currency)
    
    # Format percentage columns
    percentage_cols = ['occupancy_with', 'occupancy_without', 'adr_uplift_pct', 'occupancy_uplift_pct', 'revpar_uplift_pct']
    for col in percentage_cols:
        if col in display_uplift.columns:
            display_uplift[col] = display_uplift[col].apply(lambda x: f"{x:.1f}%")
    
    # Rename columns for display
    display_uplift.columns = [
        'Feature', 'Properties With', 'Properties Without',
        'ADR With', 'ADR Without', 'ADR Uplift %',
        'Occupancy With', 'Occupancy Without', 'Occupancy Uplift %',
        'RevPAR With', 'RevPAR Without', 'RevPAR Uplift %'
    ]
    
    st.dataframe(display_uplift, , hide_index=True)
    
    # Visualizations
    st.subheader("📊 Amenity Impact Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["RevPAR Impact", "ADR Impact", "Occupancy Impact"])
    
    with tab1:
        # RevPAR uplift chart
        fig_revpar = px.bar(
            uplift_data.sort_values('revpar_uplift_pct', ascending=True),
            x='revpar_uplift_pct',
            y='feature',
            orientation='h',
            title='RevPAR Uplift by Amenity',
            labels={'revpar_uplift_pct': 'RevPAR Uplift (%)', 'feature': 'Amenity'},
            color='revpar_uplift_pct',
            color_continuous_scale='RdYlGn'
        )
        fig_revpar.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_revpar, )
    
    with tab2:
        # ADR uplift chart
        fig_adr = px.bar(
            uplift_data.sort_values('adr_uplift_pct', ascending=True),
            x='adr_uplift_pct',
            y='feature',
            orientation='h',
            title='ADR Uplift by Amenity',
            labels={'adr_uplift_pct': 'ADR Uplift (%)', 'feature': 'Amenity'},
            color='adr_uplift_pct',
            color_continuous_scale='Blues'
        )
        fig_adr.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_adr, )
    
    with tab3:
        # Occupancy uplift chart
        fig_occ = px.bar(
            uplift_data.sort_values('occupancy_uplift_pct', ascending=True),
            x='occupancy_uplift_pct',
            y='feature',
            orientation='h',
            title='Occupancy Uplift by Amenity',
            labels={'occupancy_uplift_pct': 'Occupancy Uplift (%)', 'feature': 'Amenity'},
            color='occupancy_uplift_pct',
            color_continuous_scale='Greens'
        )
        fig_occ.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_occ, )
    
    # Island-specific analysis
    st.subheader("🏝️ Island-Specific Amenity Impact")
    
    # Create island-specific uplift analysis
    island_analysis = []
    
    for island in data['island'].unique():
        island_data = data[data['island'] == island]
        island_uplift = uplift_table(island_data)
        
        for _, row in island_uplift.iterrows():
            island_analysis.append({
                'island': island,
                'feature': row['feature'],
                'revpar_uplift_pct': row['revpar_uplift_pct'],
                'adr_uplift_pct': row['adr_uplift_pct'],
                'occupancy_uplift_pct': row['occupancy_uplift_pct']
            })
    
    island_df = pd.DataFrame(island_analysis)
    
    if len(island_df) > 0:
        # RevPAR uplift by island
        fig_island = px.bar(
            island_df,
            x='feature',
            y='revpar_uplift_pct',
            color='island',
            title='RevPAR Uplift by Amenity and Island',
            labels={'revpar_uplift_pct': 'RevPAR Uplift (%)', 'feature': 'Amenity'},
            barmode='group'
        )
        fig_island.update_layout(height=500)
        st.plotly_chart(fig_island, )
    
    # ROI Analysis
    st.subheader("💰 Return on Investment Analysis")
    
    # Calculate ROI for each amenity
    roi_analysis = []
    
    for _, row in uplift_data.iterrows():
        feature = row['feature']
        
        # Estimate cost (simplified)
        cost_estimates = {
            'pool': 50000,  # Pool installation
            'wifi': 2000,   # WiFi setup
            'sea_view': 0,  # Natural feature
            'instant_book': 0,  # Platform feature
            'professionally_managed': 0  # Service feature
        }
        
        estimated_cost = cost_estimates.get(feature, 10000)  # Default cost
        
        # Calculate annual revenue uplift
        properties_with = row['properties_with']
        avg_revpar_with = row['revpar_with']
        avg_revpar_without = row['revpar_without']
        revpar_uplift = avg_revpar_with - avg_revpar_without
        
        # Annual revenue uplift (assuming 12 months)
        annual_uplift = revpar_uplift * 12 * properties_with
        
        # ROI calculation
        roi = (annual_uplift / estimated_cost) * 100 if estimated_cost > 0 else float('inf')
        
        roi_analysis.append({
            'feature': feature,
            'estimated_cost': estimated_cost,
            'annual_revenue_uplift': annual_uplift,
            'roi_percentage': roi,
            'payback_period_months': (estimated_cost / (annual_uplift / 12)) if annual_uplift > 0 else float('inf')
        })
    
    roi_df = pd.DataFrame(roi_analysis)
    
    # Format ROI data
    display_roi = roi_df.copy()
    display_roi['estimated_cost'] = display_roi['estimated_cost'].apply(format_currency)
    display_roi['annual_revenue_uplift'] = display_roi['annual_revenue_uplift'].apply(format_currency)
    display_roi['roi_percentage'] = display_roi['roi_percentage'].apply(lambda x: f"{x:.1f}%" if x != float('inf') else "∞")
    display_roi['payback_period_months'] = display_roi['payback_period_months'].apply(lambda x: f"{x:.1f}" if x != float('inf') else "∞")
    
    display_roi.columns = [
        'Feature', 'Estimated Cost', 'Annual Revenue Uplift',
        'ROI Percentage', 'Payback Period (Months)'
    ]
    
    st.dataframe(display_roi, , hide_index=True)
    
    # Recommendations
    st.subheader("💡 Amenity Investment Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **High-Impact Amenities:**
        - Focus on amenities with highest RevPAR uplift
        - Consider cost vs. benefit ratio
        - Prioritize features with quick payback periods
        - Monitor market demand for specific amenities
        """)
    
    with col2:
        st.markdown("""
        **Management Features:**
        - Professional management often shows positive impact
        - Instant booking can improve occupancy
        - Consider amenity combinations for maximum effect
        - Regular performance monitoring is essential
        """)

if __name__ == "__main__":
    main()
