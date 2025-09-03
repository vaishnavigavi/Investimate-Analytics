import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import load_data, calculate_competitive_intelligence, analyze_price_tier_performance, analyze_management_performance

def main():
    st.set_page_config(page_title="Competitive Intelligence - Investimate Analytics", layout="wide")
    
    st.title("🏆 Competitive Intelligence Dashboard")
    st.markdown("**Market positioning, percentile rankings, and competitive analysis**")
    
    # Load data
    if 'data' not in st.session_state:
        from ui_utils import load_data_with_spinner
        st.session_state.data = load_data_with_spinner()
        if st.session_state.data is None:
            return
    
    data = st.session_state.data
    
    # Calculate competitive intelligence
    with st.spinner("Analyzing competitive landscape..."):
        competitive_data = calculate_competitive_intelligence(data)
        price_tier_analysis = analyze_price_tier_performance(data)
        management_analysis = analyze_management_performance(data)
    
    st.markdown("---")
    
    # Competitive Overview
    st.subheader("📊 Competitive Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        market_leaders = competitive_data[competitive_data['competitive_position'] == 'Market Leader']
        st.metric(
            "Market Leaders",
            f"{len(market_leaders):,}",
            f"{len(market_leaders)/len(competitive_data)*100:.1f}% of portfolio"
        )
    
    with col2:
        strong_performers = competitive_data[competitive_data['competitive_position'] == 'Strong']
        st.metric(
            "Strong Performers",
            f"{len(strong_performers):,}",
            f"{len(strong_performers)/len(competitive_data)*100:.1f}% of portfolio"
        )
    
    with col3:
        avg_performers = competitive_data[competitive_data['competitive_position'] == 'Average']
        st.metric(
            "Average Performers",
            f"{len(avg_performers):,}",
            f"{len(avg_performers)/len(competitive_data)*100:.1f}% of portfolio"
        )
    
    with col4:
        underperformers = competitive_data[competitive_data['competitive_position'] == 'Underperformer']
        st.metric(
            "Underperformers",
            f"{len(underperformers):,}",
            f"{len(underperformers)/len(competitive_data)*100:.1f}% of portfolio"
        )
    
    # Competitive Position Distribution
    st.markdown("---")
    st.subheader("🎯 Competitive Position Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Competitive position distribution
        position_dist = competitive_data['competitive_position'].value_counts()
        
        fig = px.pie(
            values=position_dist.values,
            names=position_dist.index,
            title="Competitive Position Distribution",
            color_discrete_sequence=['#2E8B57', '#32CD32', '#FFD700', '#FF6347']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Competitive score distribution
        fig = px.histogram(
            competitive_data,
            x='competitive_score',
            nbins=20,
            title="Competitive Score Distribution",
            color='island'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Percentile Rankings
    st.markdown("---")
    st.subheader("📈 Percentile Rankings Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ADR percentile distribution
        fig = px.histogram(
            competitive_data,
            x='adr_percentile',
            nbins=20,
            title="ADR Percentile Distribution",
            color='island'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Occupancy percentile distribution
        fig = px.histogram(
            competitive_data,
            x='occupancy_percentile',
            nbins=20,
            title="Occupancy Percentile Distribution",
            color='island'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # RevPAR and Revenue percentiles
    col1, col2 = st.columns(2)
    
    with col1:
        # RevPAR percentile distribution
        fig = px.histogram(
            competitive_data,
            x='revpar_percentile',
            nbins=20,
            title="RevPAR Percentile Distribution",
            color='island'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Revenue percentile distribution
        fig = px.histogram(
            competitive_data,
            x='revenue_percentile',
            nbins=20,
            title="Revenue Percentile Distribution",
            color='island'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Market Share Analysis
    st.markdown("---")
    st.subheader("📊 Market Share Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market share by property type
        type_market_share = competitive_data.groupby('property_type')['market_share_pct'].sum().sort_values(ascending=False)
        
        fig = px.bar(
            x=type_market_share.index,
            y=type_market_share.values,
            title="Market Share by Property Type",
            color=type_market_share.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Market share distribution
        fig = px.histogram(
            competitive_data,
            x='market_share_pct',
            nbins=20,
            title="Market Share Distribution",
            color='island'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Top Performers by Category
    st.markdown("---")
    st.subheader("🏆 Top Performers by Category")
    
    # Create tabs for different categories
    tab1, tab2, tab3, tab4 = st.tabs(["Market Leaders", "Top ADR", "Top Occupancy", "Top RevPAR"])
    
    with tab1:
        market_leaders = competitive_data[competitive_data['competitive_position'] == 'Market Leader'].nlargest(10, 'competitive_score')
        display_leaders = market_leaders[['property_id', 'island', 'property_type', 'competitive_score', 'adr_percentile', 'occupancy_percentile', 'revpar_percentile']].copy()
        display_leaders['competitive_score'] = display_leaders['competitive_score'].round(1)
        display_leaders['adr_percentile'] = display_leaders['adr_percentile'].round(1)
        display_leaders['occupancy_percentile'] = display_leaders['occupancy_percentile'].round(1)
        display_leaders['revpar_percentile'] = display_leaders['revpar_percentile'].round(1)
        st.dataframe(display_leaders, width='stretch')
    
    with tab2:
        top_adr = competitive_data.nlargest(10, 'adr_percentile')
        display_adr = top_adr[['property_id', 'island', 'property_type', 'adr', 'adr_percentile', 'competitive_score']].copy()
        display_adr['adr'] = '€' + display_adr['adr'].round(2).astype(str)
        display_adr['adr_percentile'] = display_adr['adr_percentile'].round(1)
        display_adr['competitive_score'] = display_adr['competitive_score'].round(1)
        st.dataframe(display_adr, width='stretch')
    
    with tab3:
        top_occupancy = competitive_data.nlargest(10, 'occupancy_percentile')
        display_occupancy = top_occupancy[['property_id', 'island', 'property_type', 'occupancy', 'occupancy_percentile', 'competitive_score']].copy()
        display_occupancy['occupancy'] = (display_occupancy['occupancy'] * 100).round(1).astype(str) + '%'
        display_occupancy['occupancy_percentile'] = display_occupancy['occupancy_percentile'].round(1)
        display_occupancy['competitive_score'] = display_occupancy['competitive_score'].round(1)
        st.dataframe(display_occupancy, width='stretch')
    
    with tab4:
        top_revpar = competitive_data.nlargest(10, 'revpar_percentile')
        display_revpar = top_revpar[['property_id', 'island', 'property_type', 'revpar', 'revpar_percentile', 'competitive_score']].copy()
        display_revpar['revpar'] = '€' + display_revpar['revpar'].round(2).astype(str)
        display_revpar['revpar_percentile'] = display_revpar['revpar_percentile'].round(1)
        display_revpar['competitive_score'] = display_revpar['competitive_score'].round(1)
        st.dataframe(display_revpar, width='stretch')
    
    # Price Tier Performance Analysis
    st.markdown("---")
    st.subheader("💰 Price Tier Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price tier performance by island
        fig = px.bar(
            price_tier_analysis,
            x='price_tier',
            y='revpar',
            color='island',
            title="Average RevPAR by Price Tier",
            barmode='group'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Market share by price tier
        fig = px.bar(
            price_tier_analysis,
            x='price_tier',
            y='market_share_pct',
            color='island',
            title="Market Share by Price Tier",
            barmode='group'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Price tier efficiency analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Occupancy efficiency by price tier
        fig = px.bar(
            price_tier_analysis,
            x='price_tier',
            y='occupancy_efficiency',
            color='island',
            title="Occupancy Efficiency by Price Tier",
            barmode='group'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # RevPAR efficiency by price tier
        fig = px.bar(
            price_tier_analysis,
            x='price_tier',
            y='revpar_efficiency',
            color='island',
            title="RevPAR Efficiency by Price Tier",
            barmode='group'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Management Performance Analysis
    st.markdown("---")
    st.subheader("👥 Management Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Management type performance comparison
        fig = px.bar(
            management_analysis,
            x='management_type',
            y='revpar',
            color='island',
            title="Average RevPAR by Management Type",
            barmode='group'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Management type occupancy comparison
        fig = px.bar(
            management_analysis,
            x='management_type',
            y='occupancy',
            color='island',
            title="Average Occupancy by Management Type",
            barmode='group'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Management performance table
    st.subheader("📋 Management Performance Summary")
    display_mgmt = management_analysis.copy()
    display_mgmt['revpar'] = '€' + display_mgmt['revpar'].round(2).astype(str)
    display_mgmt['adr'] = '€' + display_mgmt['adr'].round(2).astype(str)
    display_mgmt['occupancy'] = (display_mgmt['occupancy'] * 100).round(1).astype(str) + '%'
    display_mgmt['revenue_per_property'] = '€' + display_mgmt['revenue_per_property'].round(0).astype(str)
    
    st.dataframe(
        display_mgmt[['island', 'management_type', 'property_id', 'revpar', 'occupancy', 'adr', 'revenue_per_property']],
        width='stretch'
    )
    
    # Competitive Positioning Matrix
    st.markdown("---")
    st.subheader("🎯 Competitive Positioning Matrix")
    
    # Create competitive positioning scatter plot
    fig = px.scatter(
        competitive_data,
        x='adr_percentile',
        y='occupancy_percentile',
        color='competitive_position',
        size='revpar_percentile',
        hover_data=['property_id', 'property_type', 'island', 'competitive_score'],
        title="Competitive Positioning Matrix (ADR vs Occupancy Percentiles)",
        color_discrete_map={
            'Market Leader': '#2E8B57',
            'Strong': '#32CD32',
            'Average': '#FFD700',
            'Underperformer': '#FF6347'
        }
    )
    
    # Add quadrant lines
    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=75, y=75, text="High ADR<br>High Occupancy", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=25, y=75, text="Low ADR<br>High Occupancy", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=75, y=25, text="High ADR<br>Low Occupancy", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=25, y=25, text="Low ADR<br>Low Occupancy", showarrow=False, font=dict(size=12))
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, width='stretch')
    
    # Key Insights
    st.markdown("---")
    st.subheader("💡 Competitive Intelligence Insights")
    
    # Calculate insights
    market_leaders_count = len(competitive_data[competitive_data['competitive_position'] == 'Market Leader'])
    strong_performers_count = len(competitive_data[competitive_data['competitive_position'] == 'Strong'])
    avg_competitive_score = competitive_data['competitive_score'].mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **🏆 Market Leadership:**
        - {market_leaders_count:,} properties are market leaders
        - {strong_performers_count:,} properties are strong performers
        - Average competitive score: {avg_competitive_score:.1f}/100
        - Top performing property type: {competitive_data[competitive_data['competitive_position'] == 'Market Leader']['property_type'].mode().iloc[0] if market_leaders_count > 0 else 'N/A'}
        """)
    
    with col2:
        st.info(f"""
        **📊 Market Share Insights:**
        - Highest market share property type: {competitive_data.groupby('property_type')['market_share_pct'].sum().idxmax()}
        - Average market share per property: {competitive_data['market_share_pct'].mean():.3f}%
        - Properties with >1% market share: {len(competitive_data[competitive_data['market_share_pct'] > 1]):,}
        - Market concentration: {competitive_data['market_share_pct'].std():.3f}% standard deviation
        """)
    
    # Strategic recommendations
    st.success(f"""
    **🎯 Strategic Recommendations:**
    
    1. **Market Leadership:** {market_leaders_count:,} properties are market leaders - replicate their strategies across the portfolio
    
    2. **Performance Improvement:** {len(competitive_data[competitive_data['competitive_position'] == 'Underperformer']):,} underperformers need immediate attention
    
    3. **Price Optimization:** Properties in the "High ADR, Low Occupancy" quadrant need pricing strategy review
    
    4. **Market Share Growth:** Focus on expanding market share in high-performing property types
    
    5. **Management Strategy:** {management_analysis[management_analysis['management_type'] == 'Professional']['revpar'].mean():.2f} vs {management_analysis[management_analysis['management_type'] == 'Individual']['revpar'].mean():.2f} RevPAR difference between professional and individual management
    """)

if __name__ == "__main__":
    main()
