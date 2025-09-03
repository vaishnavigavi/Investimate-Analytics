import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, get_property_locations, calculate_advanced_metrics

def main():
    st.set_page_config(page_title="Interactive Maps - Investimate Analytics", layout="wide")
    
    st.title("🗺️ Interactive Property Performance Maps")
    st.markdown("**Explore property locations with performance heatmaps and advanced filtering**")
    
    # Load data
    if 'data' not in st.session_state:
        from ui_utils import load_data_with_spinner
        st.session_state.data = load_data_with_spinner()
        if st.session_state.data is None:
            return
    
    data = st.session_state.data
    
    # Get property locations
    with st.spinner("Processing property locations..."):
        property_locations = get_property_locations(data)
        advanced_metrics = calculate_advanced_metrics(data)
    
    # Merge location data with advanced metrics
    map_data = property_locations.merge(
        advanced_metrics[['property_id', 'performance_score', 'revenue_per_bedroom', 'revenue_per_bathroom', 'adr_per_bedroom']], 
        on='property_id', 
        how='left'
    )
    
    st.markdown("---")
    
    # Sidebar filters
    st.sidebar.header("🗺️ Map Filters")
    
    # Island filter
    selected_islands = st.sidebar.multiselect(
        "Select Islands",
        options=map_data['island'].unique(),
        default=map_data['island'].unique()
    )
    
    # Property type filter
    selected_types = st.sidebar.multiselect(
        "Select Property Types",
        options=map_data['property_type'].unique(),
        default=map_data['property_type'].unique()
    )
    
    # Price tier filter
    selected_tiers = st.sidebar.multiselect(
        "Select Price Tiers",
        options=map_data['price_tier'].unique(),
        default=map_data['price_tier'].unique()
    )
    
    # Performance metric for coloring
    color_metric = st.sidebar.selectbox(
        "Color Properties By",
        options=['revpar', 'occupancy', 'adr', 'performance_score', 'revenue_per_bedroom'],
        format_func=lambda x: {
            'revpar': 'Average RevPAR',
            'occupancy': 'Average Occupancy',
            'adr': 'Average ADR',
            'performance_score': 'Performance Score',
            'revenue_per_bedroom': 'Revenue per Bedroom'
        }[x]
    )
    
    # Performance range filter
    min_performance = st.sidebar.slider(
        f"Minimum {color_metric.replace('_', ' ').title()}",
        min_value=float(map_data[color_metric].min()),
        max_value=float(map_data[color_metric].max()),
        value=float(map_data[color_metric].min())
    )
    
    # Apply filters
    filtered_data = map_data[
        (map_data['island'].isin(selected_islands)) &
        (map_data['property_type'].isin(selected_types)) &
        (map_data['price_tier'].isin(selected_tiers)) &
        (map_data[color_metric] >= min_performance)
    ]
    
    st.sidebar.metric("Properties Shown", len(filtered_data))
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📍 Interactive Property Map")
        
        if len(filtered_data) > 0:
            # Create map
            center_lat = filtered_data['latitude'].mean()
            center_lon = filtered_data['longitude'].mean()
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=11,
                tiles='OpenStreetMap'
            )
            
            # Add performance heatmap layer
            from folium.plugins import HeatMap
            
            heat_data = [[row['latitude'], row['longitude'], row[color_metric]] 
                        for idx, row in filtered_data.iterrows()]
            
            HeatMap(heat_data, name='Performance Heatmap').add_to(m)
            
            # Add property markers
            for idx, row in filtered_data.iterrows():
                # Color based on performance
                if color_metric == 'revpar':
                    color = 'green' if row[color_metric] > filtered_data[color_metric].quantile(0.7) else 'orange' if row[color_metric] > filtered_data[color_metric].quantile(0.3) else 'red'
                elif color_metric == 'occupancy':
                    color = 'green' if row[color_metric] > 0.7 else 'orange' if row[color_metric] > 0.4 else 'red'
                elif color_metric == 'adr':
                    color = 'blue' if row[color_metric] > filtered_data[color_metric].quantile(0.7) else 'lightblue' if row[color_metric] > filtered_data[color_metric].quantile(0.3) else 'gray'
                else:
                    color = 'green' if row[color_metric] > filtered_data[color_metric].quantile(0.7) else 'orange' if row[color_metric] > filtered_data[color_metric].quantile(0.3) else 'red'
                
                # Create popup text
                popup_text = f"""
                <b>Property ID:</b> {row['property_id']}<br>
                <b>Type:</b> {row['property_type']}<br>
                <b>Bedrooms:</b> {row['bedrooms']}<br>
                <b>Price Tier:</b> {row['price_tier']}<br>
                <b>ADR:</b> €{row['adr']:.2f}<br>
                <b>Occupancy:</b> {row['occupancy']:.1%}<br>
                <b>RevPAR:</b> €{row['revpar']:.2f}<br>
                <b>Performance Score:</b> {row.get('performance_score', 'N/A'):.1f}/100
                """
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    popup=folium.Popup(popup_text, max_width=300),
                    color=color,
                    fill=True,
                    fillOpacity=0.7
                ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Display map
            st_folium(m, width=700, height=500)
            
        else:
            st.warning("No properties match the selected filters.")
    
    with col2:
        st.subheader("📊 Performance Summary")
        
        if len(filtered_data) > 0:
            # Performance metrics
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                st.metric(
                    "Avg RevPAR",
                    f"€{filtered_data['revpar'].mean():.2f}",
                    delta=f"€{filtered_data['revpar'].mean() - map_data['revpar'].mean():.2f}"
                )
                
                st.metric(
                    "Avg Occupancy",
                    f"{filtered_data['occupancy'].mean():.1%}",
                    delta=f"{(filtered_data['occupancy'].mean() - map_data['occupancy'].mean()):.1%}"
                )
            
            with col2_2:
                st.metric(
                    "Avg ADR",
                    f"€{filtered_data['adr'].mean():.2f}",
                    delta=f"€{filtered_data['adr'].mean() - map_data['adr'].mean():.2f}"
                )
                
                st.metric(
                    "Properties",
                    len(filtered_data),
                    delta=f"{len(filtered_data) - len(map_data)}"
                )
            
            # Performance distribution
            st.subheader("📈 Performance Distribution")
            
            fig = px.histogram(
                filtered_data,
                x=color_metric,
                nbins=20,
                title=f"Distribution of {color_metric.replace('_', ' ').title()}",
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, )
            
            # Top performers
            st.subheader("🏆 Top Performers")
            top_performers = filtered_data.nlargest(5, color_metric)[
                ['property_id', 'property_type', 'bedrooms', color_metric]
            ]
            st.dataframe(top_performers, )
    
    # Hex grid analysis
    st.markdown("---")
    st.subheader("🔍 Hex Grid Performance Analysis")
    
    if len(filtered_data) > 0:
        # Analyze performance by hex grid
        hex_performance = filtered_data.groupby('hex_id_res7').agg({
            'revpar': 'mean',
            'occupancy': 'mean',
            'adr': 'mean',
            'property_id': 'count',
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        
        hex_performance = hex_performance.rename(columns={'property_id': 'property_count'})
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Hex grid performance table
            st.subheader("📋 Grid Performance Summary")
            display_hex = hex_performance.copy()
            display_hex['revpar'] = '€' + display_hex['revpar'].round(2).astype(str)
            display_hex['adr'] = '€' + display_hex['adr'].round(2).astype(str)
            display_hex['occupancy'] = (display_hex['occupancy'] * 100).round(1).astype(str) + '%'
            
            st.dataframe(
                display_hex[['hex_id_res7', 'property_count', 'revpar', 'occupancy', 'adr']],
                
            )
        
        with col4:
            # Hex grid performance chart
            st.subheader("📊 Grid Performance Comparison")
            
            fig = px.scatter(
                hex_performance,
                x='occupancy',
                y='revpar',
                size='property_count',
                hover_data=['hex_id_res7', 'adr'],
                title="RevPAR vs Occupancy by Grid",
                color='adr',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, )
    
    # Property type performance by location
    st.markdown("---")
    st.subheader("🏠 Property Type Performance by Location")
    
    if len(filtered_data) > 0:
        # Property type performance by island
        type_performance = filtered_data.groupby(['island', 'property_type']).agg({
            'revpar': 'mean',
            'occupancy': 'mean',
            'adr': 'mean',
            'property_id': 'count'
        }).reset_index()
        
        type_performance = type_performance.rename(columns={'property_id': 'property_count'})
        
        # Create grouped bar chart
        fig = px.bar(
            type_performance,
            x='property_type',
            y='revpar',
            color='island',
            title="Average RevPAR by Property Type and Island",
            barmode='group'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, )
        
        # Property type performance table
        st.subheader("📋 Detailed Property Type Performance")
        display_type = type_performance.copy()
        display_type['revpar'] = '€' + display_type['revpar'].round(2).astype(str)
        display_type['adr'] = '€' + display_type['adr'].round(2).astype(str)
        display_type['occupancy'] = (display_type['occupancy'] * 100).round(1).astype(str) + '%'
        
        st.dataframe(
            display_type[['island', 'property_type', 'property_count', 'revpar', 'occupancy', 'adr']],
            
        )

if __name__ == "__main__":
    main()
