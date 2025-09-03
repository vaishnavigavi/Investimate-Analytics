import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, calculate_advanced_metrics, calculate_revenue_concentration, calculate_seasonal_volatility

def main():
    st.set_page_config(page_title="Advanced Metrics - Investimate Analytics", layout="wide")
    
    st.title("📊 Advanced Performance Metrics Dashboard")
    st.markdown("**Deep dive into revenue efficiency, volatility analysis, and performance optimization**")
    
    # Load data
    if 'data' not in st.session_state:
        from ui_utils import load_data_with_spinner
        st.session_state.data = load_data_with_spinner()
        if st.session_state.data is None:
            return
    
    data = st.session_state.data
    
    # Calculate advanced metrics
    with st.spinner("Calculating advanced metrics..."):
        advanced_metrics = calculate_advanced_metrics(data)
        concentration_analysis = calculate_revenue_concentration(data)
        volatility_metrics = calculate_seasonal_volatility(data)
    
    st.markdown("---")
    
    # Revenue Efficiency Analysis
    st.subheader("💰 Revenue Efficiency Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Avg Revenue per Bedroom",
            f"€{advanced_metrics['revenue_per_bedroom'].mean():.0f}",
            delta=f"€{advanced_metrics['revenue_per_bedroom'].mean() - advanced_metrics['revenue_per_bedroom'].median():.0f}"
        )
    
    with col2:
        st.metric(
            "Avg Revenue per Bathroom",
            f"€{advanced_metrics['revenue_per_bathroom'].mean():.0f}",
            delta=f"€{advanced_metrics['revenue_per_bathroom'].mean() - advanced_metrics['revenue_per_bathroom'].median():.0f}"
        )
    
    with col3:
        st.metric(
            "Avg ADR per Bedroom",
            f"€{advanced_metrics['adr_per_bedroom'].mean():.2f}",
            delta=f"€{advanced_metrics['adr_per_bedroom'].mean() - advanced_metrics['adr_per_bedroom'].median():.2f}"
        )
    
    with col4:
        st.metric(
            "Avg Performance Score",
            f"{advanced_metrics['performance_score'].mean():.1f}/100",
            delta=f"{advanced_metrics['performance_score'].mean() - advanced_metrics['performance_score'].median():.1f}"
        )
    
    # Revenue efficiency charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue per bedroom by property type
        fig = px.box(
            advanced_metrics,
            x='property_type',
            y='revenue_per_bedroom',
            title="Revenue per Bedroom by Property Type",
            color='island'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # ADR per bedroom by property type
        fig = px.box(
            advanced_metrics,
            x='property_type',
            y='adr_per_bedroom',
            title="ADR per Bedroom by Property Type",
            color='island'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')
    
    # Performance efficiency analysis
    st.markdown("---")
    st.subheader("⚡ Performance Efficiency Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Occupancy efficiency distribution
        fig = px.histogram(
            advanced_metrics,
            x='occupancy_efficiency',
            nbins=20,
            title="Occupancy Efficiency Distribution",
            color='island'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # RevPAR efficiency distribution
        fig = px.histogram(
            advanced_metrics,
            x='revpar_efficiency',
            nbins=20,
            title="RevPAR Efficiency Distribution",
            color='island'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Performance score analysis
    st.markdown("---")
    st.subheader("🏆 Performance Score Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance score by property type
        fig = px.box(
            advanced_metrics,
            x='property_type',
            y='performance_score',
            title="Performance Score by Property Type",
            color='island'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Performance score vs revenue correlation
        fig = px.scatter(
            advanced_metrics,
            x='performance_score',
            y='total_revenue',
            color='island',
            size='bedrooms',
            title="Performance Score vs Total Revenue",
            hover_data=['property_type', 'price_tier']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Top and bottom performers
    st.markdown("---")
    st.subheader("🎯 Top & Bottom Performers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Performers")
        top_performers = advanced_metrics.nlargest(10, 'performance_score')[
            ['property_id', 'island', 'property_type', 'bedrooms', 'performance_score', 'total_revenue', 'revenue_per_bedroom']
        ]
        top_performers['performance_score'] = top_performers['performance_score'].round(1)
        top_performers['total_revenue'] = '€' + top_performers['total_revenue'].round(0).astype(str)
        top_performers['revenue_per_bedroom'] = '€' + top_performers['revenue_per_bedroom'].round(0).astype(str)
        st.dataframe(top_performers, width='stretch')
    
    with col2:
        st.subheader("📉 Bottom 10 Performers")
        bottom_performers = advanced_metrics.nsmallest(10, 'performance_score')[
            ['property_id', 'island', 'property_type', 'bedrooms', 'performance_score', 'total_revenue', 'revenue_per_bedroom']
        ]
        bottom_performers['performance_score'] = bottom_performers['performance_score'].round(1)
        bottom_performers['total_revenue'] = '€' + bottom_performers['total_revenue'].round(0).astype(str)
        bottom_performers['revenue_per_bedroom'] = '€' + bottom_performers['revenue_per_bedroom'].round(0).astype(str)
        st.dataframe(bottom_performers, width='stretch')
    
    # Revenue Concentration Analysis
    st.markdown("---")
    st.subheader("📊 Revenue Concentration Analysis (Pareto Principle)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Top 10% Properties",
            f"{concentration_analysis['top_10_properties']:,}",
            f"{concentration_analysis['top_10_revenue_pct']:.1f}% of revenue"
        )
    
    with col2:
        st.metric(
            "Top 20% Properties",
            f"{concentration_analysis['top_20_properties']:,}",
            f"{concentration_analysis['top_20_revenue_pct']:.1f}% of revenue"
        )
    
    with col3:
        st.metric(
            "Pareto 80/20 Split",
            f"{concentration_analysis['pareto_20_properties']:,} properties",
            f"generate 80% of revenue"
        )
    
    with col4:
        st.metric(
            "Gini Coefficient",
            f"{concentration_analysis['gini_coefficient']:.3f}",
            "Inequality measure"
        )
    
    # Revenue concentration visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Pareto chart
        property_revenue = data.groupby('property_id')['revenue'].sum().sort_values(ascending=False)
        cumulative_revenue = property_revenue.cumsum()
        cumulative_percentage = (cumulative_revenue / property_revenue.sum() * 100)
        
        fig = go.Figure()
        
        # Add cumulative revenue line
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cumulative_percentage) + 1)),
            y=cumulative_percentage,
            mode='lines',
            name='Cumulative Revenue %',
            line=dict(color='blue', width=2)
        ))
        
        # Add 80% line
        fig.add_hline(y=80, line_dash="dash", line_color="red", 
                     annotation_text="80% Revenue Line")
        
        fig.update_layout(
            title="Pareto Chart - Revenue Concentration",
            xaxis_title="Properties (Ranked by Revenue)",
            yaxis_title="Cumulative Revenue %",
            height=400
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Revenue distribution by property type
        type_revenue = data.groupby('property_type')['revenue'].sum().sort_values(ascending=False)
        
        fig = px.pie(
            values=type_revenue.values,
            names=type_revenue.index,
            title="Revenue Distribution by Property Type"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Volatility Analysis
    st.markdown("---")
    st.subheader("📈 Performance Volatility Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Avg ADR Volatility",
            f"{volatility_metrics['adr_cv'].mean():.3f}",
            "Coefficient of Variation"
        )
    
    with col2:
        st.metric(
            "Avg Occupancy Volatility",
            f"{volatility_metrics['occupancy_cv'].mean():.3f}",
            "Coefficient of Variation"
        )
    
    with col3:
        st.metric(
            "Avg RevPAR Volatility",
            f"{volatility_metrics['revpar_cv'].mean():.3f}",
            "Coefficient of Variation"
        )
    
    # Volatility analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Volatility by property type
        fig = px.box(
            volatility_metrics,
            x='property_type',
            y='revpar_cv',
            title="RevPAR Volatility by Property Type",
            color='island'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Volatility levels distribution
        volatility_dist = volatility_metrics['volatility_level'].value_counts()
        
        fig = px.pie(
            values=volatility_dist.values,
            names=volatility_dist.index,
            title="Property Volatility Levels Distribution"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Volatility vs Performance correlation
    st.markdown("---")
    st.subheader("🔍 Volatility vs Performance Correlation")
    
    # Merge volatility with performance data
    volatility_performance = volatility_metrics.merge(
        advanced_metrics[['property_id', 'performance_score', 'total_revenue']], 
        on='property_id', 
        how='left'
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Volatility vs Performance Score
        fig = px.scatter(
            volatility_performance,
            x='revpar_cv',
            y='performance_score',
            color='island',
            size='total_revenue',
            title="Volatility vs Performance Score",
            hover_data=['property_type', 'volatility_level']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Volatility vs Revenue
        fig = px.scatter(
            volatility_performance,
            x='revpar_cv',
            y='total_revenue',
            color='island',
            size='performance_score',
            title="Volatility vs Total Revenue",
            hover_data=['property_type', 'volatility_level']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Key Insights
    st.markdown("---")
    st.subheader("💡 Key Insights & Recommendations")
    
    # Calculate insights
    high_performers = advanced_metrics[advanced_metrics['performance_score'] > advanced_metrics['performance_score'].quantile(0.8)]
    low_volatility = volatility_metrics[volatility_metrics['volatility_level'] == 'Low']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **🏆 High Performers Analysis:**
        - {len(high_performers):,} properties in top 20% performance
        - Average revenue per bedroom: €{high_performers['revenue_per_bedroom'].mean():.0f}
        - Most common property type: {high_performers['property_type'].mode().iloc[0] if len(high_performers) > 0 else 'N/A'}
        - Average performance score: {high_performers['performance_score'].mean():.1f}/100
        """)
    
    with col2:
        st.info(f"""
        **📈 Volatility Insights:**
        - {len(low_volatility):,} properties have low volatility
        - Low volatility properties average {low_volatility['avg_revpar'].mean():.2f} RevPAR
        - Volatility level distribution: {volatility_metrics['volatility_level'].value_counts().to_dict()}
        - Most stable property type: {low_volatility['property_type'].mode().iloc[0] if len(low_volatility) > 0 else 'N/A'}
        """)
    
    # Recommendations
    st.success(f"""
    **🎯 Strategic Recommendations:**
    
    1. **Revenue Optimization:** Focus on properties with high revenue per bedroom (€{advanced_metrics['revenue_per_bedroom'].quantile(0.8):.0f}+)
    
    2. **Volatility Management:** {len(volatility_metrics[volatility_metrics['volatility_level'] == 'High']):,} properties have high volatility - consider dynamic pricing strategies
    
    3. **Performance Improvement:** {len(advanced_metrics[advanced_metrics['performance_score'] < 50]):,} properties score below 50 - prioritize optimization efforts
    
    4. **Concentration Risk:** Top 20% of properties generate {concentration_analysis['top_20_revenue_pct']:.1f}% of revenue - diversify portfolio
    """)

if __name__ == "__main__":
    main()
