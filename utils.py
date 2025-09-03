import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
import json

def load_data() -> pd.DataFrame:
    """
    Load and merge all CSV files into a single dataset.
    
    Returns:
        Merged DataFrame with performance and property data
    """
    # Load performance data
    mykonos_perf = pd.read_csv('mykonos_monthly_performance.csv')
    paros_perf = pd.read_csv('paros_monthly_performance.csv')
    
    # Load property data
    mykonos_prop = pd.read_csv('mykonos_property_details.csv')
    paros_prop = pd.read_csv('paros_property_details.csv')
    
    # Combine performance data
    performance_data = pd.concat([mykonos_perf, paros_perf], ignore_index=True)
    
    # Combine property data
    property_data = pd.concat([mykonos_prop, paros_prop], ignore_index=True)
    
    # Clean and process data
    performance_data = _clean_performance_data(performance_data)
    property_data = _clean_property_data(property_data)
    
    # Merge performance with property details
    # Handle potential column conflicts by renaming adm_3_id in property data
    if 'adm_3_id' in property_data.columns and 'adm_3_id' in performance_data.columns:
        property_data = property_data.rename(columns={'adm_3_id': 'adm_3_id_prop'})
    
    merged_data = performance_data.merge(property_data, on='property_id', how='left')
    
    # Normalize occupancy to 0-1 scale
    merged_data['occupancy'] = merged_data['occupancy'].clip(0, 1)
    
    # Parse year+month to datetime
    merged_data['date'] = pd.to_datetime(merged_data[['year', 'month']].assign(day=1))
    
    # Extract amenity flags
    merged_data = _extract_amenity_flags(merged_data)
    
    # Create island column for all data
    island_mapping = {
        'GRC.1.2.21_1': 'Mykonos',
        'GRC.1.2.24_1': 'Paros'
    }
    merged_data['island'] = merged_data['adm_3_id'].map(island_mapping)
    
    return merged_data

def _clean_performance_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and process performance data."""
    # Remove rows with missing essential data
    df = df.dropna(subset=['property_id', 'year', 'month', 'revenue'])
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['occupancy', 'adr', 'revenue', 'revpar', 'fetched_revenue']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove outliers (revenue > 3 standard deviations from mean)
    if 'revenue' in df.columns:
        mean_revenue = df['revenue'].mean()
        std_revenue = df['revenue'].std()
        df = df[df['revenue'] <= mean_revenue + 3 * std_revenue]
    
    return df

def _clean_property_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and process property data."""
    # Remove rows with missing property_id
    df = df.dropna(subset=['property_id'])
    
    # Clean numeric columns
    numeric_columns = ['bedrooms', 'bathrooms', 'rating', 'cleaning_fee', 'accommodates']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean text columns
    text_columns = ['property_type', 'title', 'city']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', np.nan)
    
    return df

def _extract_amenity_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Extract amenity flags from text fields."""
    # Combine title and amenities_title for analysis
    text_fields = df['title'].fillna('') + ' ' + df['amenities_title'].fillna('')
    text_fields = text_fields.str.lower()
    
    # Define amenity keywords
    amenity_keywords = {
        'pool': ['pool', 'swimming pool', 'private pool', 'shared pool'],
        'wifi': ['wifi', 'internet', 'wireless'],
        'sea_view': ['sea view', 'ocean view', 'beach view', 'waterfront'],
        'instant_book': ['instant book', 'instant booking'],
        'professionally_managed': ['professionally managed', 'professional management']
    }
    
    # Extract amenity flags
    for amenity, keywords in amenity_keywords.items():
        df[f'has_{amenity}'] = text_fields.str.contains('|'.join(keywords), case=False, na=False).astype(int)
    
    # Handle instant_book and professionally_managed from existing columns
    if 'instant_book' in df.columns:
        df['has_instant_book'] = df['instant_book'].astype(str).str.contains('True', case=False, na=False).astype(int)
    
    if 'professionally_managed' in df.columns:
        df['has_professionally_managed'] = df['professionally_managed'].astype(str).str.contains('True', case=False, na=False).astype(int)
    
    return df

def agg_island_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ADR, Occupancy, RevPAR, and annualized revenue per property by island.
    
    Returns:
        DataFrame with island-level summary statistics
    """
    # Island column is already created in load_data()
    
    # Calculate summary statistics by island
    summary = df.groupby('island').agg({
        'adr': 'mean',
        'occupancy': 'mean',
        'revpar': 'mean',
        'revenue': 'sum',
        'property_id': 'nunique'
    }).reset_index()
    
    # Calculate annualized revenue per property
    summary['annualized_revenue_per_property'] = summary['revenue'] / summary['property_id']
    
    # Rename columns for clarity
    summary.columns = ['island', 'avg_adr', 'avg_occupancy', 'avg_revpar', 'total_revenue', 'property_count', 'annualized_revenue_per_property']
    
    return summary

def seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze monthly trends by island.
    
    Returns:
        DataFrame with monthly performance by island
    """
    # Island column is already created in load_data()
    
    # Calculate monthly trends by island
    monthly_trends = df.groupby(['island', 'month']).agg({
        'adr': 'mean',
        'occupancy': 'mean',
        'revpar': 'mean',
        'revenue': 'sum'
    }).reset_index()
    
    # Rename columns to match expected format
    monthly_trends = monthly_trends.rename(columns={
        'adr': 'avg_adr',
        'occupancy': 'avg_occupancy',
        'revpar': 'avg_revpar'
    })
    
    # Add month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_trends['month_name'] = monthly_trends['month'].map(lambda x: month_names[x-1])
    
    return monthly_trends

def bedrooms_perf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance grouped by bedroom count.
    
    Returns:
        DataFrame with performance metrics by bedroom count
    """
    # Group by bedrooms and calculate performance metrics
    bedroom_performance = df.groupby('bedrooms_x').agg({
        'adr': 'mean',
        'occupancy': 'mean',
        'revpar': 'mean',
        'revenue': 'sum',
        'property_id': 'nunique'
    }).reset_index()
    
    # Calculate revenue per property
    bedroom_performance['revenue_per_property'] = bedroom_performance['revenue'] / bedroom_performance['property_id']
    
    # Rename columns
    bedroom_performance.columns = ['bedrooms', 'avg_adr', 'avg_occupancy', 'avg_revpar', 'total_revenue', 'property_count', 'revenue_per_property']
    
    return bedroom_performance

def ptype_perf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance grouped by property type.
    
    Returns:
        DataFrame with performance metrics by property type
    """
    # Group by property type and calculate performance metrics
    type_performance = df.groupby('property_type').agg({
        'adr': 'mean',
        'occupancy': 'mean',
        'revpar': 'mean',
        'revenue': 'sum',
        'property_id': 'nunique'
    }).reset_index()
    
    # Calculate revenue per property
    type_performance['revenue_per_property'] = type_performance['revenue'] / type_performance['property_id']
    
    # Rename columns
    type_performance.columns = ['property_type', 'avg_adr', 'avg_occupancy', 'avg_revpar', 'total_revenue', 'property_count', 'revenue_per_property']
    
    return type_performance

def uplift_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare properties with vs without features and calculate ADR/Occ/RevPAR uplift.
    
    Returns:
        DataFrame with uplift analysis for each feature
    """
    # Get amenity columns
    amenity_cols = [col for col in df.columns if col.startswith('has_')]
    
    uplift_results = []
    
    for amenity in amenity_cols:
        # Calculate metrics for properties with and without the amenity
        with_amenity = df[df[amenity] == 1]
        without_amenity = df[df[amenity] == 0]
        
        if len(with_amenity) > 0 and len(without_amenity) > 0:
            # Calculate averages
            with_adr = with_amenity['adr'].mean()
            without_adr = without_amenity['adr'].mean()
            with_occ = with_amenity['occupancy'].mean()
            without_occ = without_amenity['occupancy'].mean()
            with_revpar = with_amenity['revpar'].mean()
            without_revpar = without_amenity['revpar'].mean()
            
            # Calculate uplift percentages
            adr_uplift = ((with_adr - without_adr) / without_adr * 100) if without_adr > 0 else 0
            occ_uplift = ((with_occ - without_occ) / without_occ * 100) if without_occ > 0 else 0
            revpar_uplift = ((with_revpar - without_revpar) / without_revpar * 100) if without_revpar > 0 else 0
            
            uplift_results.append({
                'feature': amenity.replace('has_', ''),
                'properties_with': len(with_amenity),
                'properties_without': len(without_amenity),
                'adr_with': with_adr,
                'adr_without': without_adr,
                'adr_uplift_pct': adr_uplift,
                'occupancy_with': with_occ,
                'occupancy_without': without_occ,
                'occupancy_uplift_pct': occ_uplift,
                'revpar_with': with_revpar,
                'revpar_without': without_revpar,
                'revpar_uplift_pct': revpar_uplift
            })
    
    return pd.DataFrame(uplift_results)

def price_elasticity_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a simple linear model per island×month for occupancy ~ ADR, and suggest RevPAR-optimal ADR.
    
    Returns:
        DataFrame with elasticity analysis and optimal ADR suggestions
    """
    # Island column is already created in load_data()
    
    elasticity_results = []
    
    # Analyze by island and month
    for island in df['island'].unique():
        for month in df['month'].unique():
            subset = df[(df['island'] == island) & (df['month'] == month)]
            
            if len(subset) > 10:  # Need sufficient data points
                # Fit simple linear regression: occupancy ~ ADR
                X = subset['adr'].values
                y = subset['occupancy'].values
                
                # Remove any NaN values
                mask = ~(np.isnan(X) | np.isnan(y))
                X = X[mask]
                y = y[mask]
                
                if len(X) > 5:  # Minimum data points for regression
                    # Simple linear regression
                    X_mean = np.mean(X)
                    y_mean = np.mean(y)
                    
                    # Calculate slope and intercept
                    numerator = np.sum((X - X_mean) * (y - y_mean))
                    denominator = np.sum((X - X_mean) ** 2)
                    
                    if denominator != 0:
                        slope = numerator / denominator
                        intercept = y_mean - slope * X_mean
                        
                        # Calculate elasticity (slope * mean_adr / mean_occupancy)
                        mean_adr = np.mean(X)
                        mean_occupancy = np.mean(y)
                        elasticity = slope * mean_adr / mean_occupancy if mean_occupancy > 0 else 0
                        
                        # Find RevPAR-optimal ADR
                        # RevPAR = ADR * Occupancy = ADR * (slope * ADR + intercept)
                        # dRevPAR/dADR = slope * ADR + intercept + ADR * slope = 2 * slope * ADR + intercept
                        # Set to 0: 2 * slope * ADR + intercept = 0
                        # ADR_optimal = -intercept / (2 * slope) if slope != 0
                        
                        if slope != 0:
                            optimal_adr = -intercept / (2 * slope)
                            optimal_occupancy = slope * optimal_adr + intercept
                            optimal_revpar = optimal_adr * optimal_occupancy
                        else:
                            optimal_adr = mean_adr
                            optimal_occupancy = mean_occupancy
                            optimal_revpar = mean_adr * mean_occupancy
                        
                        # Current average metrics
                        current_revpar = mean_adr * mean_occupancy
                        revpar_improvement = optimal_revpar - current_revpar
                        
                        elasticity_results.append({
                            'island': island,
                            'month': month,
                            'data_points': len(X),
                            'current_avg_adr': mean_adr,
                            'current_avg_occupancy': mean_occupancy,
                            'current_avg_revpar': current_revpar,
                            'elasticity': elasticity,
                            'slope': slope,
                            'intercept': intercept,
                            'optimal_adr': optimal_adr,
                            'optimal_occupancy': optimal_occupancy,
                            'optimal_revpar': optimal_revpar,
                            'revpar_improvement': revpar_improvement,
                            'revpar_improvement_pct': (revpar_improvement / current_revpar * 100) if current_revpar > 0 else 0
                        })
    
    return pd.DataFrame(elasticity_results)

# =============================================================================
# ADVANCED ANALYTICS FUNCTIONS
# =============================================================================

def get_property_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get property locations with performance metrics for mapping.
    
    Returns:
        DataFrame with property locations and performance data
    """
    # Get unique properties with their locations and average performance
    property_locations = df.groupby('property_id').agg({
        'latitude': 'first',
        'longitude': 'first',
        'city': 'first',
        'property_type': 'first',
        'bedrooms_x': 'first',
        'bathrooms_x': 'first',
        'price_tier': 'first',
        'professionally_managed': 'first',
        'adr': 'mean',
        'occupancy': 'mean',
        'revpar': 'mean',
        'revenue': 'sum',
        'island': 'first',
        'hex_id_res7': 'first'
    }).reset_index()
    
    # Clean up column names
    property_locations = property_locations.rename(columns={
        'bedrooms_x': 'bedrooms',
        'bathrooms_x': 'bathrooms'
    })
    
    # Remove properties without coordinates
    property_locations = property_locations.dropna(subset=['latitude', 'longitude'])
    
    return property_locations

def calculate_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate advanced performance metrics including revenue per bedroom/bathroom,
    performance volatility, and efficiency metrics.
    
    Returns:
        DataFrame with advanced metrics by property
    """
    # Calculate property-level metrics
    property_metrics = df.groupby('property_id').agg({
        'adr': ['mean', 'std'],
        'occupancy': ['mean', 'std'],
        'revpar': ['mean', 'std'],
        'revenue': 'sum',
        'bedrooms_x': 'first',
        'bathrooms_x': 'first',
        'island': 'first',
        'property_type': 'first',
        'price_tier': 'first'
    }).reset_index()
    
    # Flatten column names
    property_metrics.columns = [
        'property_id', 'avg_adr', 'adr_volatility', 'avg_occupancy', 
        'occupancy_volatility', 'avg_revpar', 'revpar_volatility', 
        'total_revenue', 'bedrooms', 'bathrooms', 'island', 
        'property_type', 'price_tier'
    ]
    
    # Calculate advanced metrics
    property_metrics['revenue_per_bedroom'] = property_metrics['total_revenue'] / property_metrics['bedrooms'].replace(0, 1)
    property_metrics['revenue_per_bathroom'] = property_metrics['total_revenue'] / property_metrics['bathrooms'].replace(0, 1)
    property_metrics['adr_per_bedroom'] = property_metrics['avg_adr'] / property_metrics['bedrooms'].replace(0, 1)
    
    # Calculate efficiency metrics
    property_metrics['occupancy_efficiency'] = property_metrics['avg_occupancy'] / property_metrics['avg_occupancy'].quantile(0.9)
    property_metrics['revpar_efficiency'] = property_metrics['avg_revpar'] / property_metrics['avg_revpar'].quantile(0.9)
    
    # Calculate performance score (0-100)
    property_metrics['performance_score'] = (
        (property_metrics['avg_revpar'] / property_metrics['avg_revpar'].max() * 40) +
        (property_metrics['avg_occupancy'] / property_metrics['avg_occupancy'].max() * 30) +
        (property_metrics['total_revenue'] / property_metrics['total_revenue'].max() * 30)
    ) * 100
    
    return property_metrics

def calculate_competitive_intelligence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate competitive intelligence metrics including percentile rankings,
    market share, and competitive positioning.
    
    Returns:
        DataFrame with competitive intelligence metrics
    """
    # Calculate property-level performance
    property_performance = df.groupby('property_id').agg({
        'adr': 'mean',
        'occupancy': 'mean',
        'revpar': 'mean',
        'revenue': 'sum',
        'island': 'first',
        'property_type': 'first',
        'price_tier': 'first',
        'bedrooms_x': 'first'
    }).reset_index()
    
    property_performance = property_performance.rename(columns={'bedrooms_x': 'bedrooms'})
    
    # Calculate percentiles for each metric
    for metric in ['adr', 'occupancy', 'revpar', 'revenue']:
        property_performance[f'{metric}_percentile'] = property_performance[metric].rank(pct=True) * 100
    
    # Calculate market share by property type
    type_totals = property_performance.groupby('property_type')['revenue'].sum()
    property_performance['market_share_pct'] = property_performance.apply(
        lambda row: (row['revenue'] / type_totals[row['property_type']] * 100) if row['property_type'] in type_totals else 0, 
        axis=1
    )
    
    # Calculate competitive positioning score
    property_performance['competitive_score'] = (
        property_performance['adr_percentile'] * 0.3 +
        property_performance['occupancy_percentile'] * 0.3 +
        property_performance['revpar_percentile'] * 0.4
    )
    
    # Categorize competitive position
    property_performance['competitive_position'] = pd.cut(
        property_performance['competitive_score'],
        bins=[0, 25, 50, 75, 100],
        labels=['Underperformer', 'Average', 'Strong', 'Market Leader']
    )
    
    return property_performance

def calculate_revenue_concentration(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate revenue concentration analysis (80/20 rule, Pareto analysis).
    
    Returns:
        Dictionary with concentration analysis results
    """
    # Calculate total revenue by property
    property_revenue = df.groupby('property_id')['revenue'].sum().sort_values(ascending=False)
    
    # Calculate cumulative percentages
    total_revenue = property_revenue.sum()
    cumulative_revenue = property_revenue.cumsum()
    cumulative_percentage = (cumulative_revenue / total_revenue * 100)
    
    # Find 80/20 split
    pareto_80_properties = len(cumulative_percentage[cumulative_percentage <= 80])
    pareto_20_properties = len(property_revenue) - pareto_80_properties
    
    # Calculate concentration metrics
    top_10_pct_properties = int(len(property_revenue) * 0.1)
    top_10_revenue = property_revenue.head(top_10_pct_properties).sum()
    top_10_revenue_pct = (top_10_revenue / total_revenue * 100)
    
    top_20_pct_properties = int(len(property_revenue) * 0.2)
    top_20_revenue = property_revenue.head(top_20_pct_properties).sum()
    top_20_revenue_pct = (top_20_revenue / total_revenue * 100)
    
    return {
        'total_properties': len(property_revenue),
        'total_revenue': total_revenue,
        'pareto_80_properties': pareto_80_properties,
        'pareto_20_properties': pareto_20_properties,
        'top_10_properties': top_10_pct_properties,
        'top_10_revenue_pct': top_10_revenue_pct,
        'top_20_properties': top_20_pct_properties,
        'top_20_revenue_pct': top_20_revenue_pct,
        'concentration_ratio': top_20_revenue_pct,
        'gini_coefficient': _calculate_gini_coefficient(property_revenue.values)
    }

def _calculate_gini_coefficient(values: np.ndarray) -> float:
    """Calculate Gini coefficient for inequality measurement."""
    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    # Calculate Gini coefficient
    cumsum = np.cumsum(sorted_values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

def analyze_price_tier_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance by price tier (luxury, midscale, budget, economy).
    
    Returns:
        DataFrame with price tier performance analysis
    """
    # Calculate performance by price tier
    tier_performance = df.groupby(['island', 'price_tier']).agg({
        'adr': 'mean',
        'occupancy': 'mean',
        'revpar': 'mean',
        'revenue': 'sum',
        'property_id': 'nunique'
    }).reset_index()
    
    # Calculate additional metrics
    tier_performance['revenue_per_property'] = tier_performance['revenue'] / tier_performance['property_id']
    tier_performance['market_share_pct'] = tier_performance.groupby('island')['revenue'].transform(
        lambda x: x / x.sum() * 100
    )
    
    # Calculate performance efficiency
    tier_performance['occupancy_efficiency'] = tier_performance.groupby('island')['occupancy'].transform(
        lambda x: x / x.max()
    )
    tier_performance['revpar_efficiency'] = tier_performance.groupby('island')['revpar'].transform(
        lambda x: x / x.max()
    )
    
    return tier_performance

def analyze_management_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance differences between professional and individual management.
    
    Returns:
        DataFrame with management type performance analysis
    """
    # Calculate performance by management type
    mgmt_performance = df.groupby(['island', 'professionally_managed']).agg({
        'adr': 'mean',
        'occupancy': 'mean',
        'revpar': 'mean',
        'revenue': 'sum',
        'property_id': 'nunique'
    }).reset_index()
    
    # Calculate additional metrics
    mgmt_performance['revenue_per_property'] = mgmt_performance['revenue'] / mgmt_performance['property_id']
    
    # Calculate performance differences
    mgmt_performance['management_type'] = mgmt_performance['professionally_managed'].map({
        True: 'Professional',
        False: 'Individual'
    })
    
    return mgmt_performance

def calculate_seasonal_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate seasonal volatility and predictability metrics.
    
    Returns:
        DataFrame with seasonal volatility analysis
    """
    # Calculate monthly performance by property
    monthly_performance = df.groupby(['property_id', 'month']).agg({
        'adr': 'mean',
        'occupancy': 'mean',
        'revpar': 'mean',
        'island': 'first',
        'property_type': 'first'
    }).reset_index()
    
    # Calculate volatility metrics by property
    volatility_metrics = monthly_performance.groupby('property_id').agg({
        'adr': ['mean', 'std', 'min', 'max'],
        'occupancy': ['mean', 'std', 'min', 'max'],
        'revpar': ['mean', 'std', 'min', 'max'],
        'island': 'first',
        'property_type': 'first'
    }).reset_index()
    
    # Flatten column names
    volatility_metrics.columns = [
        'property_id', 'avg_adr', 'adr_std', 'adr_min', 'adr_max',
        'avg_occupancy', 'occupancy_std', 'occupancy_min', 'occupancy_max',
        'avg_revpar', 'revpar_std', 'revpar_min', 'revpar_max',
        'island', 'property_type'
    ]
    
    # Calculate coefficient of variation (volatility)
    volatility_metrics['adr_cv'] = volatility_metrics['adr_std'] / volatility_metrics['avg_adr']
    volatility_metrics['occupancy_cv'] = volatility_metrics['occupancy_std'] / volatility_metrics['avg_occupancy']
    volatility_metrics['revpar_cv'] = volatility_metrics['revpar_std'] / volatility_metrics['avg_revpar']
    
    # Calculate seasonal range
    volatility_metrics['adr_range'] = volatility_metrics['adr_max'] - volatility_metrics['adr_min']
    volatility_metrics['occupancy_range'] = volatility_metrics['occupancy_max'] - volatility_metrics['occupancy_min']
    volatility_metrics['revpar_range'] = volatility_metrics['revpar_max'] - volatility_metrics['revpar_min']
    
    # Categorize volatility levels
    volatility_metrics['volatility_level'] = pd.cut(
        volatility_metrics['revpar_cv'],
        bins=[0, 0.2, 0.4, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    return volatility_metrics

def extract_amenities_from_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract detailed amenities from text fields and create binary flags.
    
    Returns:
        DataFrame with extracted amenity flags
    """
    # Combine title and amenities_title
    text_fields = df['title'].fillna('') + ' ' + df['amenities_title'].fillna('')
    text_fields = text_fields.str.lower()
    
    # Define comprehensive amenity keywords
    amenity_keywords = {
        'pool': ['pool', 'swimming pool', 'private pool', 'shared pool', 'infinity pool'],
        'wifi': ['wifi', 'internet', 'wireless', 'wireless_internet'],
        'sea_view': ['sea view', 'ocean view', 'beach view', 'waterfront', 'sea front'],
        'parking': ['parking', 'free parking', 'garage', 'car park'],
        'kitchen': ['kitchen', 'fully equipped kitchen', 'kitchenette'],
        'air_conditioning': ['air conditioning', 'aircon', 'a/c', 'ac'],
        'tv': ['tv', 'television', 'smart tv', 'flat screen'],
        'washer': ['washer', 'washing machine', 'laundry'],
        'heating': ['heating', 'central heating', 'radiator'],
        'elevator': ['elevator', 'lift'],
        'balcony': ['balcony', 'terrace', 'patio'],
        'garden': ['garden', 'yard', 'outdoor space'],
        'pets_allowed': ['pets allowed', 'pet friendly', 'dogs allowed'],
        'family_friendly': ['family friendly', 'children', 'kids'],
        'fireplace': ['fireplace', 'indoor fireplace', 'wood burning'],
        'hot_tub': ['hot tub', 'jacuzzi', 'spa'],
        'gym': ['gym', 'fitness', 'exercise'],
        'bbq': ['bbq', 'barbecue', 'grill']
    }
    
    # Extract amenity flags
    for amenity, keywords in amenity_keywords.items():
        df[f'has_{amenity}'] = text_fields.str.contains('|'.join(keywords), case=False, na=False).astype(int)
    
    return df
