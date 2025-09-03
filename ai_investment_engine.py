"""
AI-Powered Investment Recommendation Engine
==========================================

This module provides intelligent investment recommendations for rental properties
based on advanced machine learning models and comprehensive performance analysis.

Features:
- Property scoring and ranking
- Buy/Sell/Hold recommendations
- Investment opportunity identification
- Portfolio optimization suggestions
- Risk assessment and ROI predictions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class InvestmentRecommendationEngine:
    """
    AI-powered investment recommendation engine for rental properties.
    
    This class analyzes property performance data and provides intelligent
    recommendations for investment decisions.
    """
    
    def __init__(self, data):
        """
        Initialize the investment recommendation engine.
        
        Args:
            data (pd.DataFrame): Combined property performance and details data
        """
        self.data = data.copy()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.property_scores = None
        self.recommendations = None
        
        # Prepare data for ML models
        self._prepare_features()
        self._train_models()
        
    def _prepare_features(self):
        """Prepare features for machine learning models."""
        print("🔧 Preparing features for AI models...")
        
        # Create investment-relevant features
        self.data['revenue_per_bedroom'] = self.data['revenue'] / self.data['bedrooms_x'].clip(1)
        
        # Handle bathrooms column - it might be missing or have different name
        if 'bathrooms' in self.data.columns:
            self.data['revenue_per_bathroom'] = self.data['revenue'] / self.data['bathrooms'].clip(1)
        else:
            # If bathrooms column is missing, use bedrooms as proxy
            self.data['revenue_per_bathroom'] = self.data['revenue'] / self.data['bedrooms_x'].clip(1)
        
        self.data['adr_per_bedroom'] = self.data['adr'] / self.data['bedrooms_x'].clip(1)
        self.data['occupancy_efficiency'] = self.data['occupancy'] * self.data['adr']
        self.data['profit_margin'] = (self.data['revenue'] - self.data['adr'] * 0.3) / self.data['revenue']  # Assuming 30% costs
        
        # Seasonal features
        self.data['is_peak_season'] = self.data['month'].isin([6, 7, 8, 9]).astype(int)
        self.data['is_shoulder_season'] = self.data['month'].isin([4, 5, 10]).astype(int)
        self.data['is_low_season'] = self.data['month'].isin([11, 12, 1, 2, 3]).astype(int)
        
        # Property quality features
        self.data['has_premium_amenities'] = (
            self.data['has_pool'].astype(int) + 
            self.data['has_sea_view'].astype(int) + 
            self.data['has_professionally_managed'].astype(int)
        )
        
        # Market position features
        self.data['is_mykonos'] = (self.data['island'] == 'Mykonos').astype(int)
        
        # Performance volatility
        property_volatility = self.data.groupby('property_id').agg({
            'adr': 'std',
            'occupancy': 'std',
            'revenue': 'std'
        }).fillna(0)
        
        property_volatility.columns = ['adr_volatility', 'occupancy_volatility', 'revenue_volatility']
        self.data = self.data.merge(property_volatility, on='property_id', how='left')
        
        # Fill missing values
        self.data = self.data.fillna(0)
        
        print(f"✅ Features prepared. Dataset shape: {self.data.shape}")
        
    def _train_models(self):
        """Train machine learning models for different predictions."""
        print("🤖 Training AI models...")
        
        # Prepare target variables
        targets = {
            'revenue': 'revenue',
            'adr': 'adr', 
            'occupancy': 'occupancy',
            'revpar': 'revpar'
        }
        
        # Feature columns for ML
        feature_cols = [
            'bedrooms_x', 'is_mykonos', 'month',
            'has_pool', 'has_wifi', 'has_sea_view', 'has_instant_book', 
            'has_professionally_managed', 'is_peak_season', 'is_shoulder_season',
            'is_low_season', 'has_premium_amenities', 'adr_volatility',
            'occupancy_volatility', 'revenue_volatility'
        ]
        
        # Add bathrooms if available
        if 'bathrooms' in self.data.columns:
            feature_cols.insert(1, 'bathrooms')
        
        X = self.data[feature_cols].fillna(0)
        
        # Train models for each target
        for target_name, target_col in targets.items():
            y = self.data[target_col].fillna(0)
            
            # Remove outliers
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (y >= lower_bound) & (y <= upper_bound)
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) > 100:  # Ensure we have enough data
                # Train multiple models (reduced complexity for faster training)
                models = {
                    'rf': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
                    'gb': GradientBoostingRegressor(n_estimators=50, random_state=42),
                    'lr': LinearRegression()
                }
                
                best_model = None
                best_score = -np.inf
                
                for model_name, model in models.items():
                    try:
                        # Cross-validation score (reduced folds for faster training)
                        scores = cross_val_score(model, X_clean, y_clean, cv=2, scoring='r2')
                        avg_score = scores.mean()
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            best_model = model
                            
                    except Exception as e:
                        print(f"⚠️ Error training {model_name} for {target_name}: {e}")
                        continue
                
                if best_model is not None:
                    best_model.fit(X_clean, y_clean)
                    self.models[target_name] = best_model
                    print(f"✅ {target_name} model trained (R² = {best_score:.3f})")
        
        print(f"🎯 Trained {len(self.models)} AI models successfully!")
        
    def calculate_property_scores(self):
        """Calculate comprehensive property investment scores."""
        print("📊 Calculating property investment scores...")
        
        # Group by property to get property-level metrics
        agg_dict = {
            'revenue': ['mean', 'sum', 'std'],
            'adr': ['mean', 'std'],
            'occupancy': ['mean', 'std'],
            'revpar': ['mean', 'std'],
            'bedrooms_x': 'first',
            'island': 'first',
            'has_pool': 'first',
            'has_wifi': 'first',
            'has_sea_view': 'first',
            'has_instant_book': 'first',
            'has_professionally_managed': 'first',
            'property_type': 'first'
        }
        
        # Add bathrooms if available
        if 'bathrooms' in self.data.columns:
            agg_dict['bathrooms'] = 'first'
        
        property_metrics = self.data.groupby('property_id').agg(agg_dict).round(2)
        
        # Flatten column names
        property_metrics.columns = ['_'.join(col).strip() for col in property_metrics.columns]
        property_metrics = property_metrics.reset_index()
        
        # Calculate investment scores
        scores = []
        
        for _, prop in property_metrics.iterrows():
            score_data = {
                'property_id': prop['property_id'],
                'island': prop['island_first'],
                'bedrooms': prop['bedrooms_x_first'],
                'property_type': prop['property_type_first']
            }
            
            # Add bathrooms if available
            if 'bathrooms_first' in prop:
                score_data['bathrooms'] = prop['bathrooms_first']
            else:
                score_data['bathrooms'] = prop['bedrooms_x_first']  # Use bedrooms as proxy
            
            # Revenue Score (0-100)
            revenue_score = min(100, max(0, (prop['revenue_mean'] / 5000) * 100))
            score_data['revenue_score'] = revenue_score
            
            # ADR Score (0-100)
            adr_score = min(100, max(0, (prop['adr_mean'] / 500) * 100))
            score_data['adr_score'] = adr_score
            
            # Occupancy Score (0-100)
            occupancy_score = prop['occupancy_mean'] * 100
            score_data['occupancy_score'] = occupancy_score
            
            # Stability Score (0-100) - Lower volatility = higher score
            revenue_volatility = prop['revenue_std'] / prop['revenue_mean'] if prop['revenue_mean'] > 0 else 1
            stability_score = max(0, 100 - (revenue_volatility * 100))
            score_data['stability_score'] = stability_score
            
            # Amenity Score (0-100)
            amenity_score = (
                prop['has_pool_first'] * 25 +
                prop['has_wifi_first'] * 15 +
                prop['has_sea_view_first'] * 30 +
                prop['has_instant_book_first'] * 15 +
                prop['has_professionally_managed_first'] * 15
            )
            score_data['amenity_score'] = amenity_score
            
            # Location Score (Mykonos premium)
            location_score = 100 if prop['island_first'] == 'Mykonos' else 70
            score_data['location_score'] = location_score
            
            # Overall Investment Score (weighted average)
            overall_score = (
                revenue_score * 0.25 +
                adr_score * 0.20 +
                occupancy_score * 0.20 +
                stability_score * 0.15 +
                amenity_score * 0.10 +
                location_score * 0.10
            )
            score_data['overall_score'] = overall_score
            
            # Risk Level
            if revenue_volatility < 0.2:
                risk_level = 'Low'
            elif revenue_volatility < 0.4:
                risk_level = 'Medium'
            else:
                risk_level = 'High'
            score_data['risk_level'] = risk_level
            
            # Investment Grade
            if overall_score >= 80:
                grade = 'A+'
            elif overall_score >= 70:
                grade = 'A'
            elif overall_score >= 60:
                grade = 'B+'
            elif overall_score >= 50:
                grade = 'B'
            else:
                grade = 'C'
            score_data['investment_grade'] = grade
            
            scores.append(score_data)
        
        self.property_scores = pd.DataFrame(scores)
        self.property_scores = self.property_scores.sort_values('overall_score', ascending=False)
        
        print(f"✅ Calculated scores for {len(self.property_scores)} properties")
        return self.property_scores
        
    def generate_recommendations(self):
        """Generate buy/sell/hold recommendations for each property."""
        print("💡 Generating AI investment recommendations...")
        
        if self.property_scores is None:
            self.calculate_property_scores()
        
        recommendations = []
        
        for _, prop in self.property_scores.iterrows():
            rec_data = {
                'property_id': prop['property_id'],
                'island': prop['island'],
                'investment_grade': prop['investment_grade'],
                'overall_score': prop['overall_score'],
                'risk_level': prop['risk_level']
            }
            
            # Generate recommendation based on score and risk
            score = prop['overall_score']
            risk = prop['risk_level']
            
            if score >= 75 and risk == 'Low':
                recommendation = 'BUY'
                confidence = 'High'
                reasoning = 'Excellent performance with low risk. Strong investment opportunity.'
            elif score >= 70 and risk in ['Low', 'Medium']:
                recommendation = 'BUY'
                confidence = 'Medium'
                reasoning = 'Good performance with acceptable risk. Solid investment choice.'
            elif score >= 60:
                recommendation = 'HOLD'
                confidence = 'Medium'
                reasoning = 'Average performance. Monitor closely for improvement opportunities.'
            elif score >= 50:
                recommendation = 'HOLD'
                confidence = 'Low'
                reasoning = 'Below average performance. Consider optimization or divestment.'
            else:
                recommendation = 'SELL'
                confidence = 'High'
                reasoning = 'Poor performance. Strong candidate for divestment.'
            
            # Adjust for risk
            if risk == 'High' and recommendation == 'BUY':
                recommendation = 'HOLD'
                reasoning += ' High risk profile requires careful consideration.'
            elif risk == 'High' and recommendation == 'HOLD':
                recommendation = 'SELL'
                reasoning += ' High risk combined with poor performance suggests divestment.'
            
            rec_data['recommendation'] = recommendation
            rec_data['confidence'] = confidence
            rec_data['reasoning'] = reasoning
            
            # Add specific optimization suggestions
            suggestions = []
            if prop['revenue_score'] < 60:
                suggestions.append('Optimize pricing strategy')
            if prop['occupancy_score'] < 60:
                suggestions.append('Improve marketing and booking channels')
            if prop['amenity_score'] < 50:
                suggestions.append('Consider adding premium amenities')
            if prop['stability_score'] < 60:
                suggestions.append('Focus on reducing revenue volatility')
            
            rec_data['optimization_suggestions'] = '; '.join(suggestions) if suggestions else 'No major optimizations needed'
            
            recommendations.append(rec_data)
        
        self.recommendations = pd.DataFrame(recommendations)
        print(f"✅ Generated recommendations for {len(self.recommendations)} properties")
        return self.recommendations
        
    def identify_investment_opportunities(self):
        """Identify new investment opportunities in the market."""
        print("🔍 Identifying investment opportunities...")
        
        # Analyze market gaps and opportunities
        opportunities = []
        
        # High-performing property types
        type_performance = self.data.groupby('property_type').agg({
            'revpar': 'mean',
            'occupancy': 'mean',
            'adr': 'mean',
            'property_id': 'nunique'
        }).round(2)
        
        type_performance = type_performance[type_performance['property_id'] >= 5]  # At least 5 properties
        type_performance = type_performance.sort_values('revpar', ascending=False)
        
        # Top opportunities by property type
        for prop_type, metrics in type_performance.head(5).iterrows():
            opportunities.append({
                'opportunity_type': 'Property Type',
                'description': f'Invest in {prop_type} properties',
                'reasoning': f'High RevPAR (€{metrics["revpar"]:.0f}) with {metrics["occupancy"]:.1%} occupancy',
                'potential_roi': f'{(metrics["revpar"] / 30):.1%} monthly ROI',
                'risk_level': 'Medium',
                'priority': 'High' if metrics['revpar'] > 100 else 'Medium'
            })
        
        # Location opportunities
        island_performance = self.data.groupby('island').agg({
            'revpar': 'mean',
            'occupancy': 'mean',
            'adr': 'mean'
        }).round(2)
        
        for island, metrics in island_performance.iterrows():
            opportunities.append({
                'opportunity_type': 'Location',
                'description': f'Expand portfolio in {island}',
                'reasoning': f'Strong market with €{metrics["revpar"]:.0f} RevPAR',
                'potential_roi': f'{(metrics["revpar"] / 30):.1%} monthly ROI',
                'risk_level': 'Low' if island == 'Mykonos' else 'Medium',
                'priority': 'High' if metrics['revpar'] > 80 else 'Medium'
            })
        
        # Amenity opportunities
        amenity_impact = self.data.groupby(['has_pool', 'has_sea_view']).agg({
            'revpar': 'mean',
            'property_id': 'nunique'
        }).round(2)
        
        # Find amenity combinations with high performance
        high_perf_amenities = amenity_impact[
            (amenity_impact['revpar'] > amenity_impact['revpar'].quantile(0.8)) &
            (amenity_impact['property_id'] >= 3)
        ]
        
        for (has_pool, has_sea_view), metrics in high_perf_amenities.iterrows():
            amenities = []
            if has_pool:
                amenities.append('Pool')
            if has_sea_view:
                amenities.append('Sea View')
            
            opportunities.append({
                'opportunity_type': 'Amenities',
                'description': f'Properties with {", ".join(amenities)}',
                'reasoning': f'Premium amenities drive €{metrics["revpar"]:.0f} RevPAR',
                'potential_roi': f'{(metrics["revpar"] / 30):.1%} monthly ROI',
                'risk_level': 'Low',
                'priority': 'High'
            })
        
        return pd.DataFrame(opportunities)
        
    def get_portfolio_optimization_suggestions(self):
        """Get suggestions for optimizing the current portfolio."""
        print("⚡ Generating portfolio optimization suggestions...")
        
        if self.recommendations is None:
            self.generate_recommendations()
        
        # Portfolio analysis
        total_properties = len(self.recommendations)
        buy_recommendations = len(self.recommendations[self.recommendations['recommendation'] == 'BUY'])
        sell_recommendations = len(self.recommendations[self.recommendations['recommendation'] == 'SELL'])
        hold_recommendations = len(self.recommendations[self.recommendations['recommendation'] == 'HOLD'])
        
        # Calculate portfolio metrics
        avg_score = self.recommendations['overall_score'].mean()
        high_risk_properties = len(self.recommendations[self.recommendations['risk_level'] == 'High'])
        
        suggestions = []
        
        # Portfolio composition suggestions
        if sell_recommendations > total_properties * 0.2:
            suggestions.append({
                'category': 'Portfolio Cleanup',
                'suggestion': f'Consider divesting {sell_recommendations} underperforming properties',
                'impact': 'High',
                'timeline': '3-6 months',
                'reasoning': f'{sell_recommendations} properties have SELL recommendations, indicating poor performance'
            })
        
        if buy_recommendations > 0:
            suggestions.append({
                'category': 'Portfolio Expansion',
                'suggestion': f'Acquire {buy_recommendations} high-performing properties',
                'impact': 'High',
                'timeline': '6-12 months',
                'reasoning': f'{buy_recommendations} properties show strong investment potential'
            })
        
        # Risk management
        if high_risk_properties > total_properties * 0.3:
            suggestions.append({
                'category': 'Risk Management',
                'suggestion': 'Reduce portfolio risk by divesting high-risk properties',
                'impact': 'Medium',
                'timeline': '6-12 months',
                'reasoning': f'{high_risk_properties} properties have high risk profiles'
            })
        
        # Performance optimization
        if avg_score < 70:
            suggestions.append({
                'category': 'Performance Optimization',
                'suggestion': 'Focus on improving underperforming properties',
                'impact': 'High',
                'timeline': '3-6 months',
                'reasoning': f'Average portfolio score of {avg_score:.1f} indicates room for improvement'
            })
        
        # Diversification
        island_distribution = self.recommendations['island'].value_counts()
        if len(island_distribution) == 1:
            suggestions.append({
                'category': 'Diversification',
                'suggestion': 'Consider diversifying across multiple locations',
                'impact': 'Medium',
                'timeline': '12+ months',
                'reasoning': 'Portfolio is concentrated in a single market'
            })
        
        return pd.DataFrame(suggestions)
        
    def predict_future_performance(self, property_id, months_ahead=12):
        """Predict future performance for a specific property."""
        if not self.models:
            return None
        
        # Get property data
        prop_data = self.data[self.data['property_id'] == property_id]
        if prop_data.empty:
            return None
        
        # Prepare features for prediction
        feature_cols = [
            'bedrooms_x', 'is_mykonos', 'month',
            'has_pool', 'has_wifi', 'has_sea_view', 'has_instant_book', 
            'has_professionally_managed', 'is_peak_season', 'is_shoulder_season',
            'is_low_season', 'has_premium_amenities', 'adr_volatility',
            'occupancy_volatility', 'revenue_volatility'
        ]
        
        # Add bathrooms if available
        if 'bathrooms' in self.data.columns:
            feature_cols.insert(1, 'bathrooms')
        
        predictions = []
        
        for month in range(1, months_ahead + 1):
            # Create feature vector for this month
            features = prop_data[feature_cols].iloc[0].copy()
            features['month'] = month
            features['is_peak_season'] = 1 if month in [6, 7, 8, 9] else 0
            features['is_shoulder_season'] = 1 if month in [4, 5, 10] else 0
            features['is_low_season'] = 1 if month in [11, 12, 1, 2, 3] else 0
            
            features_df = pd.DataFrame([features])
            
            # Make predictions
            month_predictions = {'month': month}
            for target, model in self.models.items():
                try:
                    pred = model.predict(features_df)[0]
                    month_predictions[target] = max(0, pred)  # Ensure non-negative
                except:
                    month_predictions[target] = 0
            
            predictions.append(month_predictions)
        
        return pd.DataFrame(predictions)

def create_investment_dashboard_data(data):
    """
    Create comprehensive investment dashboard data.
    
    Args:
        data (pd.DataFrame): Combined property performance and details data
        
    Returns:
        dict: Dictionary containing all investment analysis results
    """
    print("🚀 Initializing AI Investment Recommendation Engine...")
    
    # Initialize the engine
    engine = InvestmentRecommendationEngine(data)
    
    # Generate all analyses
    results = {
        'property_scores': engine.calculate_property_scores(),
        'recommendations': engine.generate_recommendations(),
        'opportunities': engine.identify_investment_opportunities(),
        'portfolio_suggestions': engine.get_portfolio_optimization_suggestions(),
        'engine': engine  # Keep engine for future predictions
    }
    
    print("✅ AI Investment Analysis Complete!")
    return results
