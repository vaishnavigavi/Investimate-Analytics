import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from utils import load_data, calculate_advanced_metrics, calculate_competitive_intelligence

def main():
    st.set_page_config(page_title="Impressive Visualizations - Investimate Analytics", layout="wide")
    
    st.title("🎨 Impressive Visualizations & Advanced Analytics")
    st.markdown("**3D scatter plots, correlation matrices, Sankey diagrams, and advanced visualizations**")
    
    # Load data
    if 'data' not in st.session_state:
        from ui_utils import load_data_with_spinner
        st.session_state.data = load_data_with_spinner()
        if st.session_state.data is None:
            return
    
    data = st.session_state.data
    
    # Calculate advanced metrics
    with st.spinner("Preparing advanced visualizations..."):
        advanced_metrics = calculate_advanced_metrics(data)
        competitive_data = calculate_competitive_intelligence(data)
    
    st.markdown("---")
    
    # 3D Scatter Plots
    st.subheader("🌐 3D Scatter Plots")
    
    # Create tabs for different 3D visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["ADR vs Occupancy vs RevPAR", "Performance vs Volatility", "Revenue vs Efficiency", "Competitive Positioning"])
    
    with tab1:
        # 3D scatter: ADR vs Occupancy vs RevPAR
        fig = px.scatter_3d(
            advanced_metrics,
            x='avg_adr',
            y='avg_occupancy',
            z='avg_revpar',
            color='island',
            size='total_revenue',
            hover_data=['property_type', 'bedrooms', 'performance_score'],
            title="3D Analysis: ADR vs Occupancy vs RevPAR",
            labels={
                'avg_adr': 'Average ADR (€)',
                'avg_occupancy': 'Average Occupancy (%)',
                'avg_revpar': 'Average RevPAR (€)'
            }
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, width='stretch')
    
    with tab2:
        # 3D scatter: Performance vs Volatility
        fig = px.scatter_3d(
            advanced_metrics,
            x='performance_score',
            y='adr_volatility',
            z='revpar_volatility',
            color='island',
            size='total_revenue',
            hover_data=['property_type', 'bedrooms', 'avg_revpar'],
            title="3D Analysis: Performance Score vs Volatility",
            labels={
                'performance_score': 'Performance Score',
                'adr_volatility': 'ADR Volatility',
                'revpar_volatility': 'RevPAR Volatility'
            }
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, width='stretch')
    
    with tab3:
        # 3D scatter: Revenue vs Efficiency
        fig = px.scatter_3d(
            advanced_metrics,
            x='total_revenue',
            y='occupancy_efficiency',
            z='revpar_efficiency',
            color='island',
            size='bedrooms',
            hover_data=['property_type', 'performance_score', 'revenue_per_bedroom'],
            title="3D Analysis: Revenue vs Efficiency Metrics",
            labels={
                'total_revenue': 'Total Revenue (€)',
                'occupancy_efficiency': 'Occupancy Efficiency',
                'revpar_efficiency': 'RevPAR Efficiency'
            }
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, width='stretch')
    
    with tab4:
        # 3D scatter: Competitive Positioning
        fig = px.scatter_3d(
            competitive_data,
            x='adr_percentile',
            y='occupancy_percentile',
            z='revpar_percentile',
            color='competitive_position',
            size='revenue_percentile',
            hover_data=['property_type', 'island', 'competitive_score'],
            title="3D Competitive Positioning Matrix",
            labels={
                'adr_percentile': 'ADR Percentile',
                'occupancy_percentile': 'Occupancy Percentile',
                'revpar_percentile': 'RevPAR Percentile'
            }
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, width='stretch')
    
    # Correlation Matrices
    st.markdown("---")
    st.subheader("🔗 Correlation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select metrics for correlation analysis
        correlation_metrics = st.multiselect(
            "Select metrics for correlation analysis",
            options=['avg_adr', 'avg_occupancy', 'avg_revpar', 'total_revenue', 'performance_score', 
                    'revenue_per_bedroom', 'revenue_per_bathroom', 'adr_per_bedroom', 'occupancy_efficiency', 'revpar_efficiency'],
            default=['avg_adr', 'avg_occupancy', 'avg_revpar', 'total_revenue', 'performance_score']
        )
    
    with col2:
        # Correlation method
        correlation_method = st.selectbox(
            "Correlation method",
            options=['pearson', 'spearman', 'kendall'],
            index=0
        )
    
    if len(correlation_metrics) > 1:
        # Calculate correlation matrix
        corr_data = advanced_metrics[correlation_metrics].corr(method=correlation_method)
        
        # Create correlation heatmap
        fig = px.imshow(
            corr_data,
            text_auto=True,
            aspect="auto",
            title=f"Correlation Matrix ({correlation_method.title()})",
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
        
        # Correlation insights
        st.subheader("📊 Correlation Insights")
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_data.columns)):
            for j in range(i+1, len(corr_data.columns)):
                corr_pairs.append({
                    'metric1': corr_data.columns[i],
                    'metric2': corr_data.columns[j],
                    'correlation': corr_data.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['abs_correlation'] = abs(corr_df['correlation'])
        top_correlations = corr_df.nlargest(5, 'abs_correlation')
        
        st.dataframe(
            top_correlations[['metric1', 'metric2', 'correlation']].round(3),
            width='stretch'
        )
    
    # Sankey Diagrams
    st.markdown("---")
    st.subheader("🌊 Sankey Diagrams - Market Flow Analysis")
    
    # Create Sankey diagram for property type flow
    property_type_flow = data.groupby(['island', 'property_type']).size().reset_index(name='count')
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=list(property_type_flow['island'].unique()) + list(property_type_flow['property_type'].unique()),
            color="blue"
        ),
        link=dict(
            source=[list(property_type_flow['island'].unique()).index(island) for island in property_type_flow['island']],
            target=[len(property_type_flow['island'].unique()) + list(property_type_flow['property_type'].unique()).index(ptype) for ptype in property_type_flow['property_type']],
            value=property_type_flow['count']
        )
    )])
    
    fig.update_layout(title_text="Property Type Distribution Flow", font_size=10, height=500)
    st.plotly_chart(fig, width='stretch')
    
    # Advanced Performance Visualizations
    st.markdown("---")
    st.subheader("📈 Advanced Performance Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Performance Heatmap", "Box Plots", "Violin Plots", "Ridge Plots"])
    
    with tab1:
        # Performance heatmap by property type and island
        heatmap_data = data.groupby(['island', 'property_type'])['revpar'].mean().unstack(fill_value=0)
        
        fig = px.imshow(
            heatmap_data,
            text_auto=True,
            aspect="auto",
            title="RevPAR Heatmap by Island and Property Type",
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    with tab2:
        # Box plots for performance metrics
        metrics_to_plot = st.multiselect(
            "Select metrics for box plots",
            options=['adr', 'occupancy', 'revpar', 'revenue'],
            default=['adr', 'occupancy', 'revpar']
        )
        
        if metrics_to_plot:
            fig = make_subplots(
                rows=1, cols=len(metrics_to_plot),
                subplot_titles=metrics_to_plot
            )
            
            for i, metric in enumerate(metrics_to_plot):
                for island in data['island'].unique():
                    island_data = data[data['island'] == island][metric]
                    fig.add_trace(
                        go.Box(y=island_data, name=island, showlegend=(i==0)),
                        row=1, col=i+1
                    )
            
            fig.update_layout(height=400, title="Performance Metrics Distribution by Island")
            st.plotly_chart(fig, width='stretch')
    
    with tab3:
        # Violin plots for performance distribution
        fig = px.violin(
            data,
            x='island',
            y='revpar',
            color='island',
            box=True,
            title="RevPAR Distribution by Island (Violin Plot)"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    with tab4:
        # Ridge plots (density plots)
        fig = go.Figure()
        
        for island in data['island'].unique():
            island_data = data[data['island'] == island]['revpar']
            fig.add_trace(go.Histogram(
                x=island_data,
                name=island,
                opacity=0.7,
                histnorm='probability density'
            ))
        
        fig.update_layout(
            title="RevPAR Density Distribution by Island",
            xaxis_title="RevPAR",
            yaxis_title="Density",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, width='stretch')
    
    # Interactive Time Series
    st.markdown("---")
    st.subheader("⏰ Interactive Time Series Analysis")
    
    # Monthly performance trends
    monthly_trends = data.groupby(['island', 'month']).agg({
        'adr': 'mean',
        'occupancy': 'mean',
        'revpar': 'mean',
        'revenue': 'sum'
    }).reset_index()
    
    # Add month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_trends['month_name'] = monthly_trends['month'].map(lambda x: month_names[x-1])
    
    # Create animated time series
    fig = px.line(
        monthly_trends,
        x='month_name',
        y='revpar',
        color='island',
        title="Monthly RevPAR Trends",
        markers=True
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, width='stretch')
    
    # Multi-metric time series
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['ADR', 'Occupancy', 'RevPAR', 'Revenue'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    for island in monthly_trends['island'].unique():
        island_data = monthly_trends[monthly_trends['island'] == island]
        
        fig.add_trace(
            go.Scatter(x=island_data['month_name'], y=island_data['adr'], name=f'{island} ADR'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=island_data['month_name'], y=island_data['occupancy'], name=f'{island} Occupancy'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=island_data['month_name'], y=island_data['revpar'], name=f'{island} RevPAR'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=island_data['month_name'], y=island_data['revenue'], name=f'{island} Revenue'),
            row=2, col=2
        )
    
    fig.update_layout(height=600, title="Multi-Metric Time Series Analysis")
    st.plotly_chart(fig, width='stretch')
    
    # Advanced Statistical Visualizations
    st.markdown("---")
    st.subheader("📊 Advanced Statistical Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Q-Q plots for normality testing
        st.subheader("📈 Q-Q Plots (Normality Testing)")
        
        metric_for_qq = st.selectbox(
            "Select metric for Q-Q plot",
            options=['adr', 'occupancy', 'revpar', 'revenue'],
            index=2
        )
        
        # Create Q-Q plot
        from scipy import stats
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        stats.probplot(data[metric_for_qq], dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot for {metric_for_qq.upper()}")
        st.pyplot(fig)
    
    with col2:
        # Distribution comparison
        st.subheader("📊 Distribution Comparison")
        
        metric_for_dist = st.selectbox(
            "Select metric for distribution comparison",
            options=['adr', 'occupancy', 'revpar', 'revenue'],
            index=2
        )
        
        fig = go.Figure()
        
        for island in data['island'].unique():
            island_data = data[data['island'] == island][metric_for_dist]
            fig.add_trace(go.Histogram(
                x=island_data,
                name=island,
                opacity=0.7,
                histnorm='probability density'
            ))
        
        fig.update_layout(
            title=f"{metric_for_dist.upper()} Distribution Comparison",
            xaxis_title=metric_for_dist.upper(),
            yaxis_title="Density",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, width='stretch')
    
    # Key Insights
    st.markdown("---")
    st.subheader("💡 Visualization Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **🎨 Visualization Highlights:**
        - 3D scatter plots reveal complex relationships between ADR, occupancy, and RevPAR
        - Correlation analysis shows strongest relationships between performance metrics
        - Sankey diagrams illustrate market flow and property type distribution
        - Time series analysis reveals seasonal patterns and trends
        """)
    
    with col2:
        st.info(f"""
        **📊 Statistical Insights:**
        - Performance distributions vary significantly between islands
        - Property types show distinct performance clusters in 3D space
        - Competitive positioning reveals clear market segments
        - Volatility patterns indicate different risk profiles
        """)
    
    # Recommendations
    st.success(f"""
    **🎯 Visualization-Driven Recommendations:**
    
    1. **3D Analysis:** Use 3D scatter plots to identify optimal performance clusters for investment focus
    
    2. **Correlation Insights:** Leverage strong correlations between metrics for predictive modeling
    
    3. **Market Flow:** Use Sankey diagrams to understand property type distribution and market dynamics
    
    4. **Time Series:** Implement seasonal adjustments based on time series analysis
    
    5. **Statistical Modeling:** Use distribution analysis for risk assessment and portfolio optimization
    """)

if __name__ == "__main__":
    main()
