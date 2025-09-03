"""
AI-Powered Investment Recommendations
====================================

This page provides intelligent investment recommendations using advanced machine learning
models to analyze property performance and suggest optimal investment strategies.

Features:
- Property scoring and ranking
- Buy/Sell/Hold recommendations
- Investment opportunity identification
- Portfolio optimization suggestions
- Future performance predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_data
from ui_utils import load_data_with_spinner, format_currency, format_percentage, format_number
from ai_investment_engine import create_investment_dashboard_data

def main():
    st.set_page_config(
        page_title="AI Investment Recommendations",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI-Powered Investment Recommendations")
    st.markdown("""
    **Intelligent property investment analysis powered by machine learning**
    
    This dashboard uses advanced AI models to analyze your property portfolio and provide 
    data-driven investment recommendations, risk assessments, and optimization strategies.
    """)
    
    # Add refresh button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Refresh AI Analysis", help="Click to re-run the AI analysis with fresh data"):
            if 'ai_investment_data' in st.session_state:
                del st.session_state.ai_investment_data
            st.rerun()
    
    # Load data with spinner
    with st.spinner("Loading data..."):
        data = load_data_with_spinner()
    
    # Check if AI analysis is already cached
    if 'ai_investment_data' not in st.session_state:
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🤖 Initializing AI Investment Engine...")
            progress_bar.progress(10)
            
            status_text.text("🔧 Preparing features for AI models...")
            progress_bar.progress(25)
            
            status_text.text("🤖 Training AI models...")
            progress_bar.progress(50)
            
            status_text.text("📊 Calculating property investment scores...")
            progress_bar.progress(75)
            
            status_text.text("💡 Generating investment recommendations...")
            progress_bar.progress(90)
            
            # Run AI investment analysis and cache it
            st.session_state.ai_investment_data = create_investment_dashboard_data(data)
            
            progress_bar.progress(100)
            status_text.text("✅ AI analysis complete!")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ AI analysis complete! Results are now cached for faster loading.")
            
        except Exception as e:
            st.error(f"❌ Error running AI analysis: {str(e)}")
            st.info("Please try refreshing the page or contact support if the issue persists.")
            return
    
    else:
        st.info("📊 Using cached AI analysis results. Click 'Refresh AI Analysis' to update.")
    
    # Use cached data
    investment_data = st.session_state.ai_investment_data
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Property Scores & Rankings", 
        "💡 Investment Recommendations",
        "🎯 Investment Opportunities", 
        "⚡ Portfolio Optimization",
        "🔮 Future Predictions"
    ])
    
    with tab1:
        st.header("📊 Property Investment Scores & Rankings")
        st.markdown("""
        **AI-calculated investment scores based on performance, stability, amenities, and location.**
        Higher scores indicate better investment potential.
        """)
        
        # Property scores table
        scores_df = investment_data['property_scores'].copy()
        
        # Format scores for display
        display_scores = scores_df.copy()
        for col in ['revenue_score', 'adr_score', 'occupancy_score', 'stability_score', 
                   'amenity_score', 'location_score', 'overall_score']:
            display_scores[col] = display_scores[col].round(1)
        
        # Create score distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall score distribution
            fig_scores = px.histogram(
                scores_df, 
                x='overall_score', 
                nbins=20,
                title="Investment Score Distribution",
                labels={'overall_score': 'Overall Investment Score', 'count': 'Number of Properties'}
            )
            fig_scores.update_layout(height=400)
            st.plotly_chart(fig_scores, )
        
        with col2:
            # Score by investment grade
            grade_counts = scores_df['investment_grade'].value_counts().sort_index()
            fig_grades = px.bar(
                x=grade_counts.index, 
                y=grade_counts.values,
                title="Properties by Investment Grade",
                labels={'x': 'Investment Grade', 'y': 'Number of Properties'}
            )
            fig_grades.update_layout(height=400)
            st.plotly_chart(fig_grades, )
        
        # Top performers
        st.subheader("🏆 Top Investment Properties")
        top_properties = scores_df.head(10)[
            ['property_id', 'island', 'bedrooms', 'investment_grade', 'overall_score', 
             'risk_level', 'revenue_score', 'adr_score', 'occupancy_score']
        ]
        
        # Format the display
        top_properties_display = top_properties.copy()
        top_properties_display.columns = [
            'Property ID', 'Island', 'Bedrooms', 'Grade', 'Overall Score',
            'Risk Level', 'Revenue Score', 'ADR Score', 'Occupancy Score'
        ]
        
        st.dataframe(
            top_properties_display,
            ,
            hide_index=True
        )
        
        # Score breakdown
        st.subheader("📈 Score Breakdown Analysis")
        
        # Create radar chart for top property
        if len(scores_df) > 0:
            top_property = scores_df.iloc[0]
            
            categories = ['Revenue', 'ADR', 'Occupancy', 'Stability', 'Amenities', 'Location']
            values = [
                top_property['revenue_score'],
                top_property['adr_score'], 
                top_property['occupancy_score'],
                top_property['stability_score'],
                top_property['amenity_score'],
                top_property['location_score']
            ]
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f"Property {top_property['property_id']}",
                line_color='blue'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title=f"Score Breakdown - Top Property {top_property['property_id']}",
                height=500
            )
            
            st.plotly_chart(fig_radar, )
    
    with tab2:
        st.header("💡 AI Investment Recommendations")
        st.markdown("""
        **Buy/Sell/Hold recommendations based on AI analysis of property performance, risk, and market conditions.**
        """)
        
        recommendations_df = investment_data['recommendations'].copy()
        
        # Recommendation summary
        rec_summary = recommendations_df['recommendation'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🟢 BUY Recommendations",
                rec_summary.get('BUY', 0),
                help="Properties recommended for acquisition or keeping"
            )
        
        with col2:
            st.metric(
                "🟡 HOLD Recommendations", 
                rec_summary.get('HOLD', 0),
                help="Properties to monitor and potentially optimize"
            )
        
        with col3:
            st.metric(
                "🔴 SELL Recommendations",
                rec_summary.get('SELL', 0),
                help="Properties recommended for divestment"
            )
        
        with col4:
            high_confidence = len(recommendations_df[recommendations_df['confidence'] == 'High'])
            st.metric(
                "🎯 High Confidence",
                high_confidence,
                help="Recommendations with high confidence level"
            )
        
        # Recommendation breakdown by island
        st.subheader("🏝️ Recommendations by Island")
        
        island_recs = recommendations_df.groupby(['island', 'recommendation']).size().unstack(fill_value=0)
        
        fig_island = px.bar(
            island_recs.reset_index(),
            x='island',
            y=['BUY', 'HOLD', 'SELL'],
            title="Investment Recommendations by Island",
            barmode='group'
        )
        fig_island.update_layout(height=400)
        st.plotly_chart(fig_island, )
        
        # Detailed recommendations table
        st.subheader("📋 Detailed Recommendations")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_recommendation = st.selectbox(
                "Filter by Recommendation",
                ['All'] + list(recommendations_df['recommendation'].unique())
            )
        
        with col2:
            selected_confidence = st.selectbox(
                "Filter by Confidence",
                ['All'] + list(recommendations_df['confidence'].unique())
            )
        
        with col3:
            selected_island = st.selectbox(
                "Filter by Island",
                ['All'] + list(recommendations_df['island'].unique())
            )
        
        # Apply filters
        filtered_recs = recommendations_df.copy()
        
        if selected_recommendation != 'All':
            filtered_recs = filtered_recs[filtered_recs['recommendation'] == selected_recommendation]
        
        if selected_confidence != 'All':
            filtered_recs = filtered_recs[filtered_recs['confidence'] == selected_confidence]
        
        if selected_island != 'All':
            filtered_recs = filtered_recs[filtered_recs['island'] == selected_island]
        
        # Display filtered recommendations
        display_recs = filtered_recs[[
            'property_id', 'island', 'investment_grade', 'recommendation', 
            'confidence', 'overall_score', 'risk_level', 'reasoning'
        ]].copy()
        
        display_recs.columns = [
            'Property ID', 'Island', 'Grade', 'Recommendation',
            'Confidence', 'Score', 'Risk', 'Reasoning'
        ]
        
        st.dataframe(
            display_recs,
            ,
            hide_index=True
        )
    
    with tab3:
        st.header("🎯 Investment Opportunities")
        st.markdown("""
        **Identified market opportunities for portfolio expansion and optimization.**
        """)
        
        opportunities_df = investment_data['opportunities'].copy()
        
        if len(opportunities_df) > 0:
            # Opportunity summary
            opp_summary = opportunities_df['opportunity_type'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🏠 Property Type Opportunities",
                    opp_summary.get('Property Type', 0)
                )
            
            with col2:
                st.metric(
                    "📍 Location Opportunities", 
                    opp_summary.get('Location', 0)
                )
            
            with col3:
                st.metric(
                    "⭐ Amenity Opportunities",
                    opp_summary.get('Amenities', 0)
                )
            
            # Priority opportunities
            st.subheader("🔥 High Priority Opportunities")
            high_priority = opportunities_df[opportunities_df['priority'] == 'High']
            
            if len(high_priority) > 0:
                for _, opp in high_priority.iterrows():
                    with st.expander(f"🎯 {opp['description']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Type:** {opp['opportunity_type']}")
                            st.write(f"**Reasoning:** {opp['reasoning']}")
                            st.write(f"**Potential ROI:** {opp['potential_roi']}")
                        
                        with col2:
                            st.write(f"**Risk Level:** {opp['risk_level']}")
                            st.write(f"**Priority:** {opp['priority']}")
            
            # All opportunities table
            st.subheader("📊 All Investment Opportunities")
            
            display_opps = opportunities_df[[
                'opportunity_type', 'description', 'reasoning', 
                'potential_roi', 'risk_level', 'priority'
            ]].copy()
            
            display_opps.columns = [
                'Type', 'Description', 'Reasoning',
                'Potential ROI', 'Risk Level', 'Priority'
            ]
            
            st.dataframe(
                display_opps,
                ,
                hide_index=True
            )
        else:
            st.info("No specific investment opportunities identified in the current dataset.")
    
    with tab4:
        st.header("⚡ Portfolio Optimization")
        st.markdown("""
        **AI-generated suggestions for optimizing your current property portfolio.**
        """)
        
        portfolio_suggestions = investment_data['portfolio_suggestions']
        
        if len(portfolio_suggestions) > 0:
            # Portfolio metrics
            total_properties = len(investment_data['recommendations'])
            buy_count = len(investment_data['recommendations'][
                investment_data['recommendations']['recommendation'] == 'BUY'
            ])
            sell_count = len(investment_data['recommendations'][
                investment_data['recommendations']['recommendation'] == 'SELL'
            ])
            avg_score = investment_data['recommendations']['overall_score'].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Properties", total_properties)
            
            with col2:
                st.metric("BUY Recommendations", buy_count)
            
            with col3:
                st.metric("SELL Recommendations", sell_count)
            
            with col4:
                st.metric("Average Score", f"{avg_score:.1f}")
            
            # Optimization suggestions
            st.subheader("💡 Portfolio Optimization Suggestions")
            
            for _, suggestion in portfolio_suggestions.iterrows():
                with st.expander(f"📋 {suggestion['category']}: {suggestion['suggestion']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Impact:** {suggestion['impact']}")
                    
                    with col2:
                        st.write(f"**Timeline:** {suggestion['timeline']}")
                    
                    with col3:
                        st.write(f"**Reasoning:** {suggestion['reasoning']}")
            
            # Risk analysis
            st.subheader("⚠️ Risk Analysis")
            
            risk_distribution = investment_data['recommendations']['risk_level'].value_counts()
            
            fig_risk = px.pie(
                values=risk_distribution.values,
                names=risk_distribution.index,
                title="Portfolio Risk Distribution"
            )
            fig_risk.update_layout(height=400)
            st.plotly_chart(fig_risk, )
            
        else:
            st.info("No specific portfolio optimization suggestions available.")
    
    with tab5:
        st.header("🔮 Future Performance Predictions")
        st.markdown("""
        **AI-powered predictions of future property performance based on historical data and trends.**
        """)
        
        # Property selection for prediction
        st.subheader("Select Property for Prediction")
        
        property_options = investment_data['property_scores']['property_id'].tolist()
        selected_property = st.selectbox(
            "Choose a property to predict future performance:",
            property_options
        )
        
        if selected_property:
            # Get prediction
            engine = investment_data['engine']
            predictions = engine.predict_future_performance(selected_property, months_ahead=12)
            
            if predictions is not None and len(predictions) > 0:
                # Display predictions
                st.subheader(f"📈 12-Month Performance Forecast - Property {selected_property}")
                
                # Create prediction charts
                fig_pred = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Revenue Forecast', 'ADR Forecast', 'Occupancy Forecast', 'RevPAR Forecast'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Revenue prediction
                fig_pred.add_trace(
                    go.Scatter(x=predictions['month'], y=predictions['revenue'], 
                              mode='lines+markers', name='Revenue', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # ADR prediction
                fig_pred.add_trace(
                    go.Scatter(x=predictions['month'], y=predictions['adr'], 
                              mode='lines+markers', name='ADR', line=dict(color='green')),
                    row=1, col=2
                )
                
                # Occupancy prediction
                fig_pred.add_trace(
                    go.Scatter(x=predictions['month'], y=predictions['occupancy'], 
                              mode='lines+markers', name='Occupancy', line=dict(color='orange')),
                    row=2, col=1
                )
                
                # RevPAR prediction
                fig_pred.add_trace(
                    go.Scatter(x=predictions['month'], y=predictions['revpar'], 
                              mode='lines+markers', name='RevPAR', line=dict(color='red')),
                    row=2, col=2
                )
                
                fig_pred.update_layout(height=600, showlegend=False, title_text="Performance Predictions")
                fig_pred.update_xaxes(title_text="Month")
                fig_pred.update_yaxes(title_text="Value")
                
                st.plotly_chart(fig_pred, )
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_revenue = predictions['revenue'].mean()
                    st.metric("Avg Predicted Revenue", format_currency(avg_revenue))
                
                with col2:
                    avg_adr = predictions['adr'].mean()
                    st.metric("Avg Predicted ADR", format_currency(avg_adr))
                
                with col3:
                    avg_occupancy = predictions['occupancy'].mean()
                    st.metric("Avg Predicted Occupancy", format_percentage(avg_occupancy))
                
                with col4:
                    avg_revpar = predictions['revpar'].mean()
                    st.metric("Avg Predicted RevPAR", format_currency(avg_revpar))
                
                # Prediction table
                st.subheader("📊 Monthly Predictions")
                
                display_predictions = predictions.copy()
                display_predictions['revenue'] = display_predictions['revenue'].apply(format_currency)
                display_predictions['adr'] = display_predictions['adr'].apply(format_currency)
                display_predictions['occupancy'] = display_predictions['occupancy'].apply(format_percentage)
                display_predictions['revpar'] = display_predictions['revpar'].apply(format_currency)
                
                display_predictions.columns = ['Month', 'Revenue', 'ADR', 'Occupancy', 'RevPAR']
                
                st.dataframe(
                    display_predictions,
                    ,
                    hide_index=True
                )
                
            else:
                st.warning(f"Unable to generate predictions for property {selected_property}. This may be due to insufficient historical data.")
        
        # AI Model Performance
        st.subheader("🤖 AI Model Performance")
        
        st.info("""
        **Model Information:**
        - **Algorithm:** Random Forest, Gradient Boosting, Linear Regression
        - **Features:** Property characteristics, amenities, seasonality, market conditions
        - **Validation:** Cross-validation with R² scoring
        - **Data:** Historical performance data from your portfolio
        
        **Note:** Predictions are based on historical patterns and may not account for 
        external factors like market changes, economic conditions, or property-specific events.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        🤖 AI Investment Recommendations powered by machine learning<br>
        <em>Always consult with financial advisors before making investment decisions</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
