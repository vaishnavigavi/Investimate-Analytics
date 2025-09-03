import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from utils import load_data, price_elasticity_simple, calculate_seasonal_volatility

def main():
    st.set_page_config(page_title="Predictive Analytics - Investimate Analytics", layout="wide")
    
    st.title("📈 Predictive Analytics & Forecasting")
    st.markdown("**Machine learning models, seasonal forecasting, and price optimization**")
    
    # Load data
    if 'data' not in st.session_state:
        from ui_utils import load_data_with_spinner
        st.session_state.data = load_data_with_spinner()
        if st.session_state.data is None:
            return
    
    data = st.session_state.data
    
    # Calculate predictive analytics
    with st.spinner("Building predictive models..."):
        elasticity_results = price_elasticity_simple(data)
        volatility_metrics = calculate_seasonal_volatility(data)
    
    st.markdown("---")
    
    # Price Elasticity Analysis
    st.subheader("💰 Price Elasticity Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_elasticity = elasticity_results['elasticity'].mean()
        st.metric(
            "Average Elasticity",
            f"{avg_elasticity:.3f}",
            "Price sensitivity"
        )
    
    with col2:
        avg_improvement = elasticity_results['revpar_improvement_pct'].mean()
        st.metric(
            "Avg RevPAR Improvement",
            f"{avg_improvement:.1f}%",
            "Potential optimization"
        )
    
    with col3:
        total_improvement = elasticity_results['revpar_improvement'].sum()
        st.metric(
            "Total Revenue Potential",
            f"€{total_improvement:,.0f}",
            "Annual improvement"
        )
    
    with col4:
        high_elasticity = len(elasticity_results[elasticity_results['elasticity'] > 0.5])
        st.metric(
            "High Elasticity Periods",
            f"{high_elasticity}",
            "Price-sensitive months"
        )
    
    # Price elasticity visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Elasticity by month
        fig = px.bar(
            elasticity_results.groupby('month')['elasticity'].mean().reset_index(),
            x='month',
            y='elasticity',
            title="Average Price Elasticity by Month",
            color='elasticity',
            color_continuous_scale='RdYlBu_r'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, )
    
    with col2:
        # RevPAR improvement potential
        fig = px.bar(
            elasticity_results.groupby('month')['revpar_improvement_pct'].mean().reset_index(),
            x='month',
            y='revpar_improvement_pct',
            title="Average RevPAR Improvement Potential by Month",
            color='revpar_improvement_pct',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, )
    
    # Optimal pricing recommendations
    st.markdown("---")
    st.subheader("🎯 Optimal Pricing Recommendations")
    
    # Island selection for detailed analysis
    selected_island = st.selectbox(
        "Select Island for Detailed Analysis",
        options=elasticity_results['island'].unique(),
        index=0
    )
    
    island_elasticity = elasticity_results[elasticity_results['island'] == selected_island]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Current vs Optimal ADR
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=island_elasticity['month'],
            y=island_elasticity['current_avg_adr'],
            mode='lines+markers',
            name='Current ADR',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=island_elasticity['month'],
            y=island_elasticity['optimal_adr'],
            mode='lines+markers',
            name='Optimal ADR',
            line=dict(color='red', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title=f"Current vs Optimal ADR - {selected_island}",
            xaxis_title="Month",
            yaxis_title="ADR (€)",
            height=400
        )
        st.plotly_chart(fig, )
    
    with col2:
        # RevPAR improvement potential
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=island_elasticity['month'],
            y=island_elasticity['revpar_improvement_pct'],
            name='RevPAR Improvement %',
            marker_color='green'
        ))
        
        fig.update_layout(
            title=f"RevPAR Improvement Potential - {selected_island}",
            xaxis_title="Month",
            yaxis_title="Improvement %",
            height=400
        )
        st.plotly_chart(fig, )
    
    # Detailed elasticity results table
    st.subheader("📋 Detailed Elasticity Analysis")
    display_elasticity = island_elasticity.copy()
    display_elasticity['current_avg_adr'] = '€' + display_elasticity['current_avg_adr'].round(2).astype(str)
    display_elasticity['optimal_adr'] = '€' + display_elasticity['optimal_adr'].round(2).astype(str)
    display_elasticity['current_avg_revpar'] = '€' + display_elasticity['current_avg_revpar'].round(2).astype(str)
    display_elasticity['optimal_revpar'] = '€' + display_elasticity['optimal_revpar'].round(2).astype(str)
    display_elasticity['revpar_improvement'] = '€' + display_elasticity['revpar_improvement'].round(2).astype(str)
    display_elasticity['revpar_improvement_pct'] = display_elasticity['revpar_improvement_pct'].round(1).astype(str) + '%'
    display_elasticity['elasticity'] = display_elasticity['elasticity'].round(3).astype(str)
    
    st.dataframe(
        display_elasticity[['month', 'data_points', 'current_avg_adr', 'optimal_adr', 'current_avg_revpar', 'optimal_revpar', 'revpar_improvement', 'revpar_improvement_pct', 'elasticity']],
        
    )
    
    # Seasonal Forecasting
    st.markdown("---")
    st.subheader("📅 Seasonal Forecasting Models")
    
    # Prepare data for forecasting
    monthly_data = data.groupby(['island', 'month']).agg({
        'adr': 'mean',
        'occupancy': 'mean',
        'revpar': 'mean',
        'revenue': 'sum'
    }).reset_index()
    
    # Create forecasting models
    def create_forecast_model(data, target_col, island_name):
        """Create a simple seasonal forecasting model."""
        island_data = data[data['island'] == island_name].sort_values('month')
        
        if len(island_data) < 6:  # Need minimum data points
            return None, None, None
        
        # Features: month, month^2 (seasonality), lagged values
        X = island_data[['month']].copy()
        X['month_squared'] = X['month'] ** 2
        X['sin_month'] = np.sin(2 * np.pi * X['month'] / 12)
        X['cos_month'] = np.cos(2 * np.pi * X['month'] / 12)
        
        # Add lagged values if available
        if len(island_data) > 1:
            X['lag_1'] = island_data[target_col].shift(1).fillna(island_data[target_col].mean())
        
        y = island_data[target_col]
        
        # Remove NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 3:
            return None, None, None
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return model, y_pred, {'mse': mse, 'r2': r2}
    
    # Forecasting for different metrics
    forecast_metrics = ['adr', 'occupancy', 'revpar']
    forecast_results = {}
    
    for metric in forecast_metrics:
        forecast_results[metric] = {}
        for island in monthly_data['island'].unique():
            model, predictions, metrics = create_forecast_model(monthly_data, metric, island)
            if model is not None:
                forecast_results[metric][island] = {
                    'model': model,
                    'predictions': predictions,
                    'metrics': metrics
                }
    
    # Display forecasting results
    col1, col2 = st.columns(2)
    
    with col1:
        # Model performance summary
        st.subheader("📊 Model Performance Summary")
        
        performance_data = []
        for metric in forecast_metrics:
            for island in monthly_data['island'].unique():
                if metric in forecast_results and island in forecast_results[metric]:
                    metrics = forecast_results[metric][island]['metrics']
                    performance_data.append({
                        'Metric': metric.upper(),
                        'Island': island,
                        'R² Score': metrics['r2'],
                        'RMSE': np.sqrt(metrics['mse'])
                    })
        
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            performance_df['R² Score'] = performance_df['R² Score'].round(3)
            performance_df['RMSE'] = performance_df['RMSE'].round(3)
            st.dataframe(performance_df, )
    
    with col2:
        # Forecasting visualization
        st.subheader("📈 Forecasting Visualization")
        
        selected_metric = st.selectbox(
            "Select Metric for Forecasting",
            options=forecast_metrics,
            index=2
        )
        
        if selected_metric in forecast_results:
            fig = go.Figure()
            
            for island in monthly_data['island'].unique():
                if island in forecast_results[selected_metric]:
                    island_data = monthly_data[monthly_data['island'] == island].sort_values('month')
                    predictions = forecast_results[selected_metric][island]['predictions']
                    
                    # Actual values
                    fig.add_trace(go.Scatter(
                        x=island_data['month'],
                        y=island_data[selected_metric],
                        mode='lines+markers',
                        name=f'{island} Actual',
                        line=dict(width=3)
                    ))
                    
                    # Predictions
                    fig.add_trace(go.Scatter(
                        x=island_data['month'],
                        y=predictions,
                        mode='lines+markers',
                        name=f'{island} Predicted',
                        line=dict(dash='dash', width=2)
                    ))
            
            fig.update_layout(
                title=f"{selected_metric.upper()} Forecasting",
                xaxis_title="Month",
                yaxis_title=selected_metric.upper(),
                height=400
            )
            st.plotly_chart(fig, )
    
    # Machine Learning Models
    st.markdown("---")
    st.subheader("🤖 Machine Learning Models")
    
    # Prepare data for ML models
    ml_data = data.groupby('property_id').agg({
        'adr': 'mean',
        'occupancy': 'mean',
        'revpar': 'mean',
        'revenue': 'sum',
        'bedrooms_x': 'first',
        'bathrooms_x': 'first',
        'island': 'first',
        'property_type': 'first',
        'price_tier': 'first',
        'professionally_managed': 'first'
    }).reset_index()
    
    ml_data = ml_data.rename(columns={'bedrooms_x': 'bedrooms', 'bathrooms_x': 'bathrooms'})
    
    # Feature engineering
    ml_data['island_encoded'] = ml_data['island'].map({'Mykonos': 1, 'Paros': 0})
    ml_data['professionally_managed_encoded'] = ml_data['professionally_managed'].fillna(False).astype(int)
    
    # Property type encoding
    property_type_encoding = {ptype: i for i, ptype in enumerate(ml_data['property_type'].dropna().unique())}
    ml_data['property_type_encoded'] = ml_data['property_type'].map(property_type_encoding).fillna(0)
    
    # Price tier encoding
    price_tier_encoding = {tier: i for i, tier in enumerate(ml_data['price_tier'].dropna().unique())}
    ml_data['price_tier_encoded'] = ml_data['price_tier'].map(price_tier_encoding).fillna(0)
    
    # Select features and target
    feature_cols = ['bedrooms', 'bathrooms', 'island_encoded', 'property_type_encoded', 'price_tier_encoded', 'professionally_managed_encoded']
    target_col = 'revpar'
    
    X = ml_data[feature_cols].fillna(0)
    y = ml_data[target_col]
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    model_results = {}
    
    for name, model in models.items():
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_results[name] = {
            'model': model,
            'predictions': y_pred,
            'mse': mse,
            'r2': r2
        }
    
    # Display ML results
    col1, col2 = st.columns(2)
    
    with col1:
        # Model comparison
        st.subheader("🏆 Model Performance Comparison")
        
        comparison_data = []
        for name, results in model_results.items():
            comparison_data.append({
                'Model': name,
                'R² Score': results['r2'],
                'RMSE': np.sqrt(results['mse'])
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df['R² Score'] = comparison_df['R² Score'].round(3)
        comparison_df['RMSE'] = comparison_df['RMSE'].round(3)
        st.dataframe(comparison_df, )
    
    with col2:
        # Feature importance (for Random Forest)
        if 'Random Forest' in model_results:
            st.subheader("🔍 Feature Importance")
            
            feature_importance = model_results['Random Forest']['model'].feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance (Random Forest)"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, )
        else:
            st.subheader("🔍 Feature Importance")
            st.info("Feature importance is only available for Random Forest model.")
    
    # Prediction vs Actual
    st.subheader("📊 Prediction vs Actual")
    
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
    best_predictions = model_results[best_model_name]['predictions']
    
    fig = px.scatter(
        x=y_test,
        y=best_predictions,
        title=f"Prediction vs Actual ({best_model_name})",
        labels={'x': 'Actual RevPAR', 'y': 'Predicted RevPAR'}
    )
    
    # Add perfect prediction line
    min_val = min(y_test.min(), best_predictions.min())
    max_val = max(y_test.max(), best_predictions.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, )
    
    # Key Insights
    st.markdown("---")
    st.subheader("💡 Predictive Analytics Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **💰 Price Optimization:**
        - Average elasticity: {avg_elasticity:.3f}
        - Potential RevPAR improvement: {avg_improvement:.1f}%
        - Total revenue potential: €{total_improvement:,.0f}
        - High elasticity periods: {high_elasticity} months
        """)
    
    with col2:
        st.info(f"""
        **🤖 Machine Learning:**
        - Best model: {best_model_name}
        - R² Score: {model_results[best_model_name]['r2']:.3f}
        - RMSE: {np.sqrt(model_results[best_model_name]['mse']):.2f}
        - Most important feature: {feature_cols[np.argmax(feature_importance)] if 'Random Forest' in model_results else 'N/A'}
        """)
    
    # Recommendations
    st.success(f"""
    **🎯 Predictive Analytics Recommendations:**
    
    1. **Price Optimization:** Implement dynamic pricing based on elasticity analysis - potential {avg_improvement:.1f}% RevPAR improvement
    
    2. **Seasonal Strategy:** Focus on high-elasticity months for maximum pricing impact
    
    3. **ML Predictions:** Use {best_model_name} model for RevPAR forecasting with {model_results[best_model_name]['r2']:.1%} accuracy
    
    4. **Feature Focus:** Prioritize {feature_cols[np.argmax(model_results['Random Forest']['model'].feature_importances_)] if 'Random Forest' in model_results else 'property characteristics'} for performance optimization
    
    5. **Revenue Growth:** Implement recommended pricing strategies to capture €{total_improvement:,.0f} in additional annual revenue
    """)

if __name__ == "__main__":
    main()
