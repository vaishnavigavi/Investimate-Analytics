# Investimate Analytics - Complete Documentation

## 📊 Project Overview

**Investimate Analytics** is a comprehensive Streamlit dashboard for analyzing short-term rental performance data from Mykonos and Paros islands. The application provides detailed insights into property performance, seasonal trends, amenity impacts, and pricing optimization.

## 📁 Data Structure & Columns

### Raw Data Files
1. **mykonos_monthly_performance.csv** - Monthly performance metrics for Mykonos properties
2. **mykonos_property_details.csv** - Property characteristics for Mykonos
3. **paros_monthly_performance.csv** - Monthly performance metrics for Paros properties  
4. **paros_property_details.csv** - Property characteristics for Paros

### Performance Data Columns (Monthly Performance CSVs)
| Column | Data Type | Description | Sample Values |
|--------|-----------|-------------|---------------|
| `property_id` | Integer | Unique identifier for each property | 1001, 1002, 1003 |
| `year` | Integer | Year of the data | 2023, 2024 |
| `month` | Integer | Month (1-12) | 1, 2, 3, ..., 12 |
| `adr` | Float | Average Daily Rate (€) | 150.50, 200.75 |
| `occupancy` | Float | Occupancy rate (0-1 scale) | 0.65, 0.80 |
| `revenue` | Float | Monthly revenue (€) | 2500.00, 3200.50 |
| `revpar` | Float | Revenue per Available Room (€) | 97.83, 160.60 |
| `adm_3_id` | Integer | Administrative region ID | 1 (Mykonos), 2 (Paros) |

### Property Details Columns (Property Details CSVs)
| Column | Data Type | Description | Sample Values |
|--------|-----------|-------------|---------------|
| `property_id` | Integer | Unique identifier for each property | 1001, 1002, 1003 |
| `property_type` | String | Type of property | "Villa", "Apartment", "House" |
| `bedrooms` | Integer | Number of bedrooms | 1, 2, 3, 4, 5 |
| `bathrooms` | Integer | Number of bathrooms | 1, 2, 3 |
| `max_guests` | Integer | Maximum number of guests | 2, 4, 6, 8 |
| `amenities` | String | Text description of amenities | "Pool, WiFi, Sea View" |
| `instant_book` | Boolean | Instant booking availability | True, False |
| `professionally_managed` | Boolean | Professional management status | True, False |
| `adm_3_id` | Integer | Administrative region ID | 1 (Mykonos), 2 (Paros) |

## 🔄 Data Processing Pipeline

### Step 1: Data Loading (`load_data()` function)

```python
def load_data():
    # Load all CSV files
    mykonos_perf = pd.read_csv('mykonos_monthly_performance.csv')
    mykonos_prop = pd.read_csv('mykonos_property_details.csv')
    paros_perf = pd.read_csv('paros_monthly_performance.csv')
    paros_prop = pd.read_csv('paros_property_details.csv')
    
    # Combine performance data
    performance_data = pd.concat([mykonos_perf, paros_perf], ignore_index=True)
    
    # Combine property data
    property_data = pd.concat([mykonos_prop, paros_prop], ignore_index=True)
    
    # Rename conflicting column
    property_data = property_data.rename(columns={'adm_3_id': 'adm_3_id_prop'})
    
    # Merge performance and property data
    merged_data = performance_data.merge(property_data, on='property_id', how='left')
    
    # Create island column
    merged_data['island'] = merged_data['adm_3_id'].map({1: 'Mykonos', 2: 'Paros'})
    
    return merged_data
```

### Step 2: Data Cleaning

#### Performance Data Cleaning (`_clean_performance_data()`)
- **Remove null values**: Drop rows with missing critical data
- **Type conversion**: Convert columns to appropriate data types
- **Outlier removal**: Remove extreme values using IQR method
- **Occupancy normalization**: Ensure occupancy is between 0-1

#### Property Data Cleaning (`_clean_property_data()`)
- **Remove null values**: Drop rows with missing property details
- **Type conversion**: Convert numeric columns to proper types
- **Text cleaning**: Standardize property type names

### Step 3: Feature Engineering

#### Amenity Flag Extraction (`_extract_amenity_flags()`)
```python
def _extract_amenity_flags(df):
    # Extract boolean flags from text descriptions
    df['pool'] = df['amenities'].str.contains('pool', case=False, na=False)
    df['wifi'] = df['amenities'].str.contains('wifi', case=False, na=False)
    df['sea_view'] = df['amenities'].str.contains('sea view', case=False, na=False)
    
    # Use existing boolean columns
    df['instant_book'] = df['instant_book'].fillna(False)
    df['professionally_managed'] = df['professionally_managed'].fillna(False)
    
    return df
```

## 📈 Key Performance Indicators (KPIs) - Formulas & Calculations

### 1. Average Daily Rate (ADR)
**Formula**: `ADR = Total Revenue ÷ Occupied Nights`
**Calculation**: `adr = revenue / (occupancy * days_in_month)`
**Business Meaning**: Average price charged per occupied night

### 2. Occupancy Rate
**Formula**: `Occupancy = Occupied Nights ÷ Available Nights × 100`
**Calculation**: `occupancy = occupied_nights / available_nights`
**Business Meaning**: Percentage of available nights that were booked

### 3. Revenue per Available Room (RevPAR)
**Formula**: `RevPAR = ADR × Occupancy Rate`
**Calculation**: `revpar = adr × occupancy`
**Business Meaning**: Revenue generated per available room, regardless of occupancy

### 4. Monthly Revenue
**Formula**: `Revenue = ADR × Occupied Nights`
**Calculation**: `revenue = adr × occupancy × days_in_month`
**Business Meaning**: Total revenue generated in a month

### 5. Annualized Revenue per Property
**Formula**: `Annualized Revenue = (Total Revenue ÷ Property Count) × 12`
**Calculation**: `annualized_revenue = (total_revenue / property_count) × 12`
**Business Meaning**: Average annual revenue per property

## 🔧 Core Functions Explained

### 1. `agg_island_summary(df)`
**Purpose**: Calculate island-level performance metrics
**Input**: Merged dataset with performance and property data
**Output**: DataFrame with island-level KPIs

```python
def agg_island_summary(df):
    summary = df.groupby('island').agg({
        'adr': 'mean',                    # Average ADR per island
        'occupancy': 'mean',              # Average occupancy per island
        'revpar': 'mean',                 # Average RevPAR per island
        'revenue': 'sum',                 # Total revenue per island
        'property_id': 'nunique'          # Property count per island
    }).reset_index()
    
    # Calculate annualized revenue per property
    summary['annualized_revenue_per_property'] = (
        summary['revenue'] / summary['property_id'] * 12
    )
    
    return summary
```

### 2. `seasonality(df)`
**Purpose**: Analyze monthly performance trends by island
**Input**: Merged dataset
**Output**: DataFrame with monthly averages by island

```python
def seasonality(df):
    monthly_data = df.groupby(['island', 'month']).agg({
        'adr': 'mean',
        'occupancy': 'mean', 
        'revpar': 'mean',
        'revenue': 'sum'
    }).reset_index()
    
    # Add month names
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    monthly_data['month_name'] = monthly_data['month'].map(month_names)
    
    return monthly_data
```

### 3. `bedrooms_perf(df)`
**Purpose**: Analyze performance by bedroom count
**Input**: Merged dataset
**Output**: DataFrame with performance metrics by bedroom count

```python
def bedrooms_perf(df):
    bedroom_performance = df.groupby('bedrooms_x').agg({
        'adr': 'mean',
        'occupancy': 'mean',
        'revpar': 'mean',
        'revenue': 'sum',
        'property_id': 'nunique'
    }).reset_index()
    
    bedroom_performance.columns = [
        'bedrooms', 'avg_adr', 'avg_occupancy', 'avg_revpar', 
        'total_revenue', 'property_count'
    ]
    
    return bedroom_performance
```

### 4. `ptype_perf(df)`
**Purpose**: Analyze performance by property type
**Input**: Merged dataset
**Output**: DataFrame with performance metrics by property type

```python
def ptype_perf(df):
    type_performance = df.groupby('property_type').agg({
        'adr': 'mean',
        'occupancy': 'mean',
        'revpar': 'mean',
        'revenue': 'sum',
        'property_id': 'nunique'
    }).reset_index()
    
    type_performance.columns = [
        'property_type', 'avg_adr', 'avg_occupancy', 'avg_revpar',
        'total_revenue', 'property_count'
    ]
    
    return type_performance
```

### 5. `uplift_table(df)`
**Purpose**: Calculate the impact of amenities on performance
**Input**: Merged dataset with amenity flags
**Output**: DataFrame showing performance uplift for each amenity

```python
def uplift_table(df):
    features = ['pool', 'wifi', 'sea_view', 'instant_book', 'professionally_managed']
    results = []
    
    for feature in features:
        # Properties with feature
        with_feature = df[df[feature] == True]
        # Properties without feature
        without_feature = df[df[feature] == False]
        
        if len(with_feature) > 0 and len(without_feature) > 0:
            # Calculate metrics
            with_adr = with_feature['adr'].mean()
            with_occ = with_feature['occupancy'].mean()
            with_revpar = with_feature['revpar'].mean()
            
            without_adr = without_feature['adr'].mean()
            without_occ = without_feature['occupancy'].mean()
            without_revpar = without_feature['revpar'].mean()
            
            # Calculate uplifts
            adr_uplift = ((with_adr - without_adr) / without_adr) * 100
            occ_uplift = ((with_occ - without_occ) / without_occ) * 100
            revpar_uplift = ((with_revpar - without_revpar) / without_revpar) * 100
            
            results.append({
                'feature': feature,
                'properties_with': len(with_feature),
                'properties_without': len(without_feature),
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
    
    return pd.DataFrame(results)
```

### 6. `price_elasticity_simple(df)`
**Purpose**: Calculate price elasticity and suggest optimal pricing
**Input**: Merged dataset
**Output**: DataFrame with elasticity analysis and pricing recommendations

```python
def price_elasticity_simple(df):
    results = []
    
    for island in df['island'].unique():
        island_data = df[df['island'] == island]
        
        for month in island_data['month'].unique():
            month_data = island_data[island_data['month'] == month]
            
            if len(month_data) > 10:  # Minimum data points
                # Simple linear regression: occupancy ~ ADR
                X = month_data['adr'].values.reshape(-1, 1)
                y = month_data['occupancy'].values
                
                # Fit linear model
                X_with_intercept = np.column_stack([np.ones(len(X)), X])
                coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                beta = coeffs[1]  # Slope coefficient
                
                # Calculate price elasticity
                mean_adr = month_data['adr'].mean()
                mean_occupancy = month_data['occupancy'].mean()
                elasticity = (beta * mean_adr) / mean_occupancy
                
                # Suggest optimal ADR (maximize RevPAR)
                # RevPAR = ADR × Occupancy = ADR × (intercept + beta × ADR)
                # dRevPAR/dADR = intercept + 2 × beta × ADR = 0
                # Optimal ADR = -intercept / (2 × beta)
                intercept = coeffs[0]
                if beta != 0:
                    optimal_adr = -intercept / (2 * beta)
                    optimal_adr = max(optimal_adr, month_data['adr'].min())
                    optimal_adr = min(optimal_adr, month_data['adr'].max())
                else:
                    optimal_adr = mean_adr
                
                # Calculate potential improvement
                optimal_occupancy = intercept + beta * optimal_adr
                optimal_occupancy = max(0, min(1, optimal_occupancy))
                optimal_revpar = optimal_adr * optimal_occupancy
                current_revpar = mean_adr * mean_occupancy
                improvement = ((optimal_revpar - current_revpar) / current_revpar) * 100
                
                results.append({
                    'island': island,
                    'month': month,
                    'beta': beta,
                    'elasticity': elasticity,
                    'current_adr': mean_adr,
                    'suggested_adr': optimal_adr,
                    'current_revpar': current_revpar,
                    'optimal_revpar': optimal_revpar,
                    'revpar_improvement_pct': improvement
                })
    
    return pd.DataFrame(results)
```

## 📊 Visualization Methods

### 1. Main Dashboard (`app.py`)

#### Portfolio KPIs
- **Method**: Streamlit `st.metric()` with delta indicators
- **Visualization**: 5-column layout with color-coded change indicators
- **Data Source**: `agg_island_summary()` function

#### Island Comparison Charts
- **Method**: Plotly Express bar charts
- **Visualization**: Side-by-side comparison of ADR, Occupancy, RevPAR
- **Data Source**: `agg_island_summary()` function

```python
# ADR Comparison Chart
fig_adr = px.bar(island_summary, x='island', y='avg_adr', 
                 title='Average Daily Rate by Island',
                 color='island', color_discrete_map={'Mykonos': '#1f77b4', 'Paros': '#ff7f0e'})
```

### 2. Overview Page (`pages/1_Overview.py`)

#### Headline KPIs
- **Method**: Streamlit `st.metric()` with period-over-period changes
- **Visualization**: 5-column layout per island
- **Data Source**: `agg_island_summary()` function

#### Performance Comparison Charts
- **Method**: Matplotlib subplots
- **Visualization**: Side-by-side bar charts for ADR, Occupancy, RevPAR
- **Data Source**: `agg_island_summary()` function

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
# ADR Chart
axes[0].bar(islands, adr_values, color=['#1f77b4', '#ff7f0e'])
axes[0].set_title('Average Daily Rate (€)')
```

### 3. Seasonality Page (`pages/2_Seasonality.py`)

#### Monthly Trends
- **Method**: Matplotlib line plots
- **Visualization**: Line charts showing monthly trends by island
- **Data Source**: `seasonality()` function

```python
for island in islands:
    island_data = seasonal_data[seasonal_data['island'] == island]
    plt.plot(island_data['month'], island_data[metric_column], 
             marker='o', label=island, linewidth=2)
```

### 4. Bedrooms & Types Page (`pages/3_Bedrooms_and_Types.py`)

#### Bedroom Performance
- **Method**: Matplotlib line plots
- **Visualization**: Line chart showing RevPAR by bedroom count
- **Data Source**: `bedrooms_perf()` function

#### Property Type Performance
- **Method**: Matplotlib horizontal bar charts
- **Visualization**: Separate charts for ADR, Occupancy, RevPAR
- **Data Source**: `ptype_perf()` function

```python
# Filter top property types
top_types = type_data[type_data['property_count'] >= 10].nlargest(12, 'avg_revpar')

# Create separate charts
fig_adr, ax_adr = plt.subplots(figsize=(10, 8))
ax_adr.barh(top_types['property_type'], top_types['avg_adr'])
ax_adr.set_title('Average Daily Rate by Property Type')
```

### 5. Amenities & Management Page (`pages/4_Amenities_and_Management.py`)

#### Amenity Impact Analysis
- **Method**: Matplotlib bar charts
- **Visualization**: Bar charts showing RevPAR uplift by feature
- **Data Source**: `uplift_table()` function

```python
# RevPAR Uplift Chart
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(uplift_data['feature'], uplift_data['revpar_uplift_pct'])
ax.set_title('RevPAR Uplift by Feature')
```

### 6. Pricing Insights Page (`pages/5_Pricing_Insights.py`)

#### Price Elasticity Analysis
- **Method**: Matplotlib dual charts
- **Visualization**: Current vs Optimal ADR, Potential RevPAR Improvement
- **Data Source**: `price_elasticity_simple()` function

```python
# Current vs Optimal ADR
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(elasticity_data['month'], elasticity_data['current_adr'], 
         'o-', label='Current ADR', linewidth=2)
ax1.plot(elasticity_data['month'], elasticity_data['suggested_adr'], 
         's-', label='Optimal ADR', linewidth=2)
```

## 🎨 UI Enhancements

### KPI Box Styling
Each KPI metric is displayed in a bordered box using Streamlit's `st.metric()` component with:
- **Border**: Automatic border from Streamlit metric component
- **Color coding**: Green for positive changes, red for negative changes
- **Icons**: Emoji icons for visual appeal
- **Tooltips**: Hover explanations for each metric

### Responsive Layout
- **Columns**: Uses `st.columns()` for responsive grid layout
- **Container width**: Uses `width='stretch'` for full-width components
- **Mobile friendly**: Responsive design that adapts to screen size

## 📱 Page-by-Page Breakdown with Code Explanations

### 1. Main Dashboard (`app.py`)

#### **Page Structure & Functions**

```python
def main():
    """Main dashboard function that orchestrates the entire page"""
    # Page configuration
    st.set_page_config(page_title="Investimate Analytics", layout="wide")
    
    # Title and description
    st.title("🏝️ Investimate Analytics")
    st.markdown("Comprehensive short-term rental performance analysis for Mykonos and Paros")
```

#### **Data Loading Function**
```python
# Load data with error handling
if 'data' in st.session_state:
    data = st.session_state.data  # Use cached data
else:
    with st.spinner("Loading data..."):
        data = load_data()  # Load from CSV files
        st.session_state.data = data  # Cache for other pages
```

**Explanation**: 
- Uses Streamlit session state to cache data across pages
- Shows loading spinner during data processing
- Calls `load_data()` from utils.py to merge and clean all CSV files

#### **KPI Explanation Section**
```python
# Add custom CSS for KPI boxes
st.markdown("""
<style>
.metric-container {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    background-color: #f8f9fa;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)
```

**Explanation**: 
- Injects custom CSS to style KPI boxes
- Creates bordered containers with hover effects
- Uses `unsafe_allow_html=True` to render HTML/CSS

#### **Portfolio KPI Calculation**
```python
# Calculate overall portfolio metrics
total_revenue = data['revenue'].sum()
avg_adr = data['adr'].mean()
avg_occupancy = data['occupancy'].mean()
avg_revpar = data['revpar'].mean()
total_properties = data['property_id'].nunique()

# Calculate period changes (H1 vs H2)
first_half = data[data['month'].isin([1, 2, 3, 4, 5, 6])]
second_half = data[data['month'].isin([7, 8, 9, 10, 11, 12])]

portfolio_adr_change = ((second_half['adr'].mean() - first_half['adr'].mean()) / first_half['adr'].mean()) * 100
```

**Explanation**:
- Uses pandas aggregation methods (`sum()`, `mean()`, `nunique()`)
- Filters data by month ranges for period-over-period comparison
- Calculates percentage change: `((new - old) / old) * 100`

#### **KPI Display with Bordered Containers**
```python
with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(
        label="💰 Total Portfolio Revenue",
        value=f"€{total_revenue:,.0f}",
        help="Total revenue across all properties and islands"
    )
    st.caption("Sum of all monthly revenues from all properties")
    st.markdown('</div>', unsafe_allow_html=True)
```

**Explanation**:
- Wraps each metric in custom HTML div with CSS class
- `st.metric()` creates the metric display with delta indicators
- `st.caption()` adds explanatory text below the metric
- Uses f-string formatting with comma separators for large numbers

#### **Island Comparison Charts**
```python
# Create Plotly bar charts
fig_adr = px.bar(island_summary, x='island', y='avg_adr', 
                 title='Average Daily Rate by Island',
                 color='island', 
                 color_discrete_map={'Mykonos': '#1f77b4', 'Paros': '#ff7f0e'})

fig_adr.update_layout(showlegend=False, height=400)
st.plotly_chart(fig_adr, width='stretch')
```

**Explanation**:
- Uses Plotly Express (`px.bar`) for interactive charts
- Maps specific colors to each island for consistency
- `update_layout()` customizes chart appearance
- `width='stretch'` makes chart responsive

### 2. Overview Page (`pages/1_Overview.py`)

#### **Page Structure**
```python
def main():
    st.set_page_config(page_title="Overview - Investimate Analytics", layout="wide")
    st.title("📊 Portfolio Overview")
    st.markdown("Comprehensive performance analysis across both islands")
```

#### **Data Loading with Session State**
```python
# Load data from session state or load fresh
if 'data' in st.session_state:
    data = st.session_state.data
else:
    with st.spinner("Loading data..."):
        data = load_data()
        st.session_state.data = data
```

**Explanation**: Same pattern as main dashboard - uses cached data for performance

#### **Period Change Calculation Function**
```python
def calculate_period_changes(data):
    """Calculate changes between first half (Jan-Jun) and second half (Jul-Dec) of year"""
    first_half = data[data['month'].isin([1, 2, 3, 4, 5, 6])]
    second_half = data[data['month'].isin([7, 8, 9, 10, 11, 12])]
    
    changes = {}
    for island in ['Mykonos', 'Paros']:
        island_first = first_half[first_half['island'] == island]
        island_second = second_half[second_half['island'] == island]
        
        if len(island_first) > 0 and len(island_second) > 0:
            changes[island] = {
                'adr_change': ((island_second['adr'].mean() - island_first['adr'].mean()) / island_first['adr'].mean()) * 100,
                'occupancy_change': ((island_second['occupancy'].mean() - island_first['occupancy'].mean()) / island_first['occupancy'].mean()) * 100,
                'revpar_change': ((island_second['revpar'].mean() - island_first['revpar'].mean()) / island_first['revpar'].mean()) * 100
            }
        else:
            changes[island] = {'adr_change': 0, 'occupancy_change': 0, 'revpar_change': 0}
    
    return changes
```

**Explanation**:
- Nested function that calculates period-over-period changes
- Filters data by month ranges and island
- Calculates percentage change for each metric
- Returns dictionary with change percentages for each island

#### **KPI Display with Change Indicators**
```python
for _, island_data in island_summary.iterrows():
    island_name = island_data['island']
    changes = period_changes[island_name]
    
    st.markdown(f"### {island_name}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        adr_change = changes['adr_change']
        delta_color = "normal" if adr_change == 0 else ("inverse" if adr_change < 0 else "normal")
        change_text = "increased" if adr_change > 0 else "decreased" if adr_change < 0 else "unchanged"
        st.metric(
            label="📊 Average Daily Rate (ADR)",
            value=f"€{island_data['avg_adr']:.2f}",
            delta=f"{adr_change:+.1f}% vs H1",
            delta_color=delta_color,
            help="Average price per occupied night"
        )
        st.caption(f"ADR {change_text} by {abs(adr_change):.1f}% from first half to second half")
        st.markdown('</div>', unsafe_allow_html=True)
```

**Explanation**:
- Iterates through each island in the summary data
- Uses `iterrows()` to get both index and row data
- Determines delta color based on change direction
- Creates descriptive text for the change
- Uses conditional formatting for positive/negative changes

#### **Matplotlib Chart Creation**
```python
import matplotlib.pyplot as plt

# Create comparison charts
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

islands = island_summary['island'].tolist()
adr_values = island_summary['avg_adr'].tolist()
occupancy_values = [x * 100 for x in island_summary['avg_occupancy'].tolist()]
revpar_values = island_summary['avg_revpar'].tolist()

# ADR Chart
axes[0].bar(islands, adr_values, color=['#1f77b4', '#ff7f0e'])
axes[0].set_title('Average Daily Rate (€)')
axes[0].set_ylabel('ADR (€)')

# Occupancy Chart
axes[1].bar(islands, occupancy_values, color=['#1f77b4', '#ff7f0e'])
axes[1].set_title('Average Occupancy Rate (%)')
axes[1].set_ylabel('Occupancy (%)')

# RevPAR Chart
axes[2].bar(islands, revpar_values, color=['#1f77b4', '#ff7f0e'])
axes[2].set_title('Revenue per Available Room (€)')
axes[2].set_ylabel('RevPAR (€)')

plt.tight_layout()
st.pyplot(fig)
```

**Explanation**:
- Creates subplot layout with 1 row, 3 columns
- Extracts data from DataFrame using `.tolist()`
- Converts occupancy from decimal to percentage
- Uses consistent color scheme for both islands
- `plt.tight_layout()` prevents label overlap
- `st.pyplot()` displays the matplotlib figure in Streamlit

### 3. Seasonality Page (`pages/2_Seasonality.py`)

#### **Metric Selection Interface**
```python
# Metric selection dropdown
metric_options = {
    'Average Daily Rate (€)': 'avg_adr',
    'Average Occupancy Rate (%)': 'avg_occupancy', 
    'Average RevPAR (€)': 'avg_revpar'
}

selected_metric = st.selectbox(
    "Select a metric to analyze:",
    options=list(metric_options.keys()),
    index=0
)

metric_column = metric_options[selected_metric]
```

**Explanation**:
- Creates dictionary mapping display names to column names
- `st.selectbox()` creates dropdown interface
- Maps user selection to actual DataFrame column name

#### **Data Processing and Formatting**
```python
# Get seasonal data
seasonal_data = seasonality(data)

# Format data for display
display_data = seasonal_data.copy()

# Format columns based on selected metric
if metric_column != 'avg_adr':
    display_data['avg_adr'] = '€' + display_data['avg_adr'].round(2).astype(str)

if metric_column != 'avg_occupancy':
    display_data['avg_occupancy'] = (display_data['avg_occupancy'] * 100).round(1).astype(str) + '%'

if metric_column != 'avg_revpar':
    display_data['avg_revpar'] = '€' + display_data['avg_revpar'].round(2).astype(str)
```

**Explanation**:
- Calls `seasonality()` function from utils.py
- Creates copy to avoid modifying original data
- Conditionally formats columns only if they're not the selected metric
- Prevents double-formatting of the selected metric

#### **Line Chart Creation**
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

islands = seasonal_data['island'].unique()

for island in islands:
    island_data = seasonal_data[seasonal_data['island'] == island]
    values = island_data[metric_column]
    
    plt.plot(island_data['month'], values, 
             marker='o', label=island, linewidth=2, markersize=6)

plt.title(f'{selected_metric} - Monthly Trends by Island')
plt.xlabel('Month')
plt.ylabel(selected_metric)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

st.pyplot(fig)
```

**Explanation**:
- Creates single subplot for line chart
- Iterates through each island to create separate lines
- Uses `marker='o'` for data point visibility
- `linewidth=2` makes lines more prominent
- `plt.grid()` adds background grid for readability
- `plt.xticks()` customizes month labels

### 4. Bedrooms & Types Page (`pages/3_Bedrooms_and_Types.py`)

#### **Tabbed Interface**
```python
tab1, tab2 = st.tabs(["🏠 Bedrooms", "🏘️ Property Types"])

with tab1:
    # Bedroom analysis content
    
with tab2:
    # Property type analysis content
```

**Explanation**: `st.tabs()` creates tabbed interface for organizing different analyses

#### **Bedroom Performance Analysis**
```python
with tab1:
    st.subheader("Performance by Bedroom Count")
    
    # Get bedroom performance data
    bedroom_performance = bedrooms_perf(data)
    
    # Display table
    st.dataframe(bedroom_performance, use_container_width=True)
    
    # Create line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group data by island for plotting
    island_bedroom_data = data.groupby(['island', 'bedrooms_x'])['revpar'].mean().reset_index()
    
    for island in island_bedroom_data['island'].unique():
        island_data = island_bedroom_data[island_bedroom_data['island'] == island]
        plt.plot(island_data['bedrooms_x'], island_data['revpar'], 
                 marker='o', label=island, linewidth=2)
    
    plt.title('RevPAR by Bedroom Count')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('RevPAR (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)
```

**Explanation**:
- Calls `bedrooms_perf()` function for aggregated data
- Creates line chart by grouping raw data by island and bedroom count
- Uses `bedrooms_x` column (from merged data) instead of `bedrooms`
- Plots separate lines for each island

#### **Property Type Analysis with Filtering**
```python
with tab2:
    st.subheader("Performance by Property Type")
    
    # Get property type performance data
    type_performance = ptype_perf(data)
    
    # Filter to show top property types
    top_types = type_performance[
        (type_performance['property_count'] >= 10) & 
        (type_performance['property_count'] <= 12)
    ].nlargest(12, 'avg_revpar')
    
    # Create three separate charts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_adr, ax_adr = plt.subplots(figsize=(8, 10))
        ax_adr.barh(top_types['property_type'], top_types['avg_adr'])
        ax_adr.set_title('Average Daily Rate by Property Type')
        ax_adr.set_xlabel('ADR (€)')
        st.pyplot(fig_adr)
    
    with col2:
        fig_occ, ax_occ = plt.subplots(figsize=(8, 10))
        ax_occ.barh(top_types['property_type'], top_types['avg_occupancy'] * 100)
        ax_occ.set_title('Average Occupancy Rate by Property Type')
        ax_occ.set_xlabel('Occupancy (%)')
        st.pyplot(fig_occ)
    
    with col3:
        fig_revpar, ax_revpar = plt.subplots(figsize=(8, 10))
        ax_revpar.barh(top_types['property_type'], top_types['avg_revpar'])
        ax_revpar.set_title('Average RevPAR by Property Type')
        ax_revpar.set_xlabel('RevPAR (€)')
        st.pyplot(fig_revpar)
```

**Explanation**:
- Filters property types by minimum count (10) and maximum count (12)
- Uses `nlargest()` to get top 12 by RevPAR
- Creates three separate charts in columns using `st.columns(3)`
- Uses horizontal bar charts (`barh`) for better readability of long property type names
- Each chart focuses on one metric (ADR, Occupancy, RevPAR)

### 5. Amenities & Management Page (`pages/4_Amenities_and_Management.py`)

#### **Uplift Analysis**
```python
def main():
    st.set_page_config(page_title="Amenities & Management - Investimate Analytics", layout="wide")
    st.title("🎯 Amenities & Management Impact Analysis")
    
    # Load data
    if 'data' in st.session_state:
        data = st.session_state.data
    else:
        with st.spinner("Loading data..."):
            data = load_data()
            st.session_state.data = data
    
    # Get uplift analysis
    uplift_data = uplift_table(data)
    
    # Display uplift table
    st.subheader("📊 Feature Impact Analysis")
    st.dataframe(uplift_data, use_container_width=True)
```

**Explanation**: Calls `uplift_table()` function to analyze amenity impact on performance

#### **Visualization by Island**
```python
# Create charts for each island
islands = data['island'].unique()

for island in islands:
    st.subheader(f"🏝️ {island} - RevPAR Uplift by Feature")
    
    # Filter data for this island
    island_data = data[data['island'] == island]
    island_uplift = uplift_table(island_data)
    
    if len(island_uplift) > 0:
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(island_uplift['feature'], island_uplift['revpar_uplift_pct'])
        ax.set_title(f'RevPAR Uplift by Feature - {island}')
        ax.set_xlabel('Feature')
        ax.set_ylabel('RevPAR Uplift (%)')
        ax.tick_params(axis='x', rotation=45)
        
        # Color bars based on positive/negative uplift
        for i, bar in enumerate(bars):
            if island_uplift.iloc[i]['revpar_uplift_pct'] > 0:
                bar.set_color('#2ecc71')  # Green for positive
            else:
                bar.set_color('#e74c3c')  # Red for negative
        
        plt.tight_layout()
        st.pyplot(fig)
```

**Explanation**:
- Iterates through each island for separate analysis
- Calls `uplift_table()` for each island's data
- Creates bar chart with color coding (green for positive, red for negative)
- Uses `plt.tight_layout()` to prevent label overlap

### 6. Pricing Insights Page (`pages/5_Pricing_Insights.py`)

#### **Price Elasticity Analysis**
```python
def main():
    st.set_page_config(page_title="Pricing Insights - Investimate Analytics", layout="wide")
    st.title("💰 Pricing Insights & Optimization")
    
    # Load data
    if 'data' in st.session_state:
        data = st.session_state.data
    else:
        with st.spinner("Loading data..."):
            data = load_data()
            st.session_state.data = data
    
    # Get price elasticity analysis
    elasticity_data = price_elasticity_simple(data)
    
    # Display results table
    st.subheader("📊 Price Elasticity Analysis Results")
    st.dataframe(elasticity_data, use_container_width=True)
```

**Explanation**: Calls `price_elasticity_simple()` function for pricing optimization analysis

#### **Island Selection and Visualization**
```python
# Island selection
islands = elasticity_data['island'].unique()
selected_island = st.selectbox("Select an island for detailed analysis:", islands)

# Filter data for selected island
island_elasticity = elasticity_data[elasticity_data['island'] == selected_island]

if len(island_elasticity) > 0:
    # Create dual charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Current vs Optimal ADR
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(island_elasticity['month'], island_elasticity['current_adr'], 
                 'o-', label='Current ADR', linewidth=2, markersize=6)
        ax1.plot(island_elasticity['month'], island_elasticity['suggested_adr'], 
                 's-', label='Optimal ADR', linewidth=2, markersize=6)
        ax1.set_title(f'Current vs Optimal ADR - {selected_island}')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('ADR (€)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
    
    with col2:
        # Potential RevPAR Improvement
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bars = ax2.bar(island_elasticity['month'], island_elasticity['revpar_improvement_pct'])
        ax2.set_title(f'Potential RevPAR Improvement - {selected_island}')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('RevPAR Improvement (%)')
        
        # Color bars based on improvement
        for i, bar in enumerate(bars):
            if island_elasticity.iloc[i]['revpar_improvement_pct'] > 0:
                bar.set_color('#2ecc71')  # Green for positive
            else:
                bar.set_color('#e74c3c')  # Red for negative
        
        st.pyplot(fig2)
```

**Explanation**:
- Creates dropdown for island selection
- Filters elasticity data for selected island
- Creates two side-by-side charts using `st.columns(2)`
- First chart shows current vs optimal ADR with different markers
- Second chart shows potential improvement with color-coded bars

#### **Key Insights Generation**
```python
# Generate key insights
st.subheader("💡 Key Insights & Recommendations")

# Calculate summary statistics
total_improvement = island_elasticity['revpar_improvement_pct'].sum()
avg_improvement = island_elasticity['revpar_improvement_pct'].mean()
best_month = island_elasticity.loc[island_elasticity['revpar_improvement_pct'].idxmax()]

st.markdown(f"""
**For {selected_island}:**
- **Total Potential Improvement**: {total_improvement:.1f}% across all months
- **Average Monthly Improvement**: {avg_improvement:.1f}%
- **Best Optimization Opportunity**: Month {best_month['month']} with {best_month['revpar_improvement_pct']:.1f}% improvement
- **Recommended ADR for Best Month**: €{best_month['suggested_adr']:.2f} (vs current €{best_month['current_adr']:.2f})
""")
```

**Explanation**:
- Calculates summary statistics from elasticity data
- Uses `idxmax()` to find the month with highest improvement
- Formats insights using f-strings and markdown
- Provides actionable recommendations based on analysis

## 🔍 Data Quality & Validation

### Data Validation Steps
1. **Null Value Handling**: Systematic removal of incomplete records
2. **Outlier Detection**: IQR-based outlier removal for performance metrics
3. **Type Validation**: Ensuring correct data types for calculations
4. **Range Validation**: Occupancy rates between 0-1, positive revenue values

### Business Logic Validation
1. **RevPAR Consistency**: RevPAR = ADR × Occupancy validation
2. **Revenue Consistency**: Revenue = ADR × Occupancy × Days validation
3. **Island Mapping**: Correct mapping of adm_3_id to island names
4. **Property Matching**: Successful merge of performance and property data

## 🚀 Technical Architecture

### Dependencies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations and linear regression
- **Matplotlib**: Static plotting and visualization
- **Plotly**: Interactive plotting and charts

### File Structure
```
investimate/
├── app.py                          # Main dashboard
├── pages/
│   ├── 1_Overview.py              # Portfolio overview
│   ├── 2_Seasonality.py           # Seasonal analysis
│   ├── 3_Bedrooms_and_Types.py    # Property characteristics
│   ├── 4_Amenities_and_Management.py # Amenity impact
│   └── 5_Pricing_Insights.py      # Pricing optimization
├── utils.py                       # Core functions
├── requirements.txt               # Dependencies
├── README.md                      # Setup instructions
└── DOCUMENTATION.md               # This documentation
```

### Performance Optimizations
1. **Session State**: Data loaded once and stored in session state
2. **Efficient Grouping**: Optimized pandas groupby operations
3. **Caching**: Streamlit's built-in caching for data loading
4. **Lazy Loading**: Data loaded only when needed

## 📈 Business Applications

### Investment Decision Support
- **Property Selection**: Identify high-performing property types and locations
- **Amenity ROI**: Quantify the impact of specific amenities
- **Pricing Strategy**: Data-driven pricing recommendations
- **Seasonal Planning**: Optimize operations for seasonal variations

### Performance Monitoring
- **KPI Tracking**: Monitor key performance indicators over time
- **Benchmarking**: Compare performance across islands and property types
- **Trend Analysis**: Identify performance trends and patterns
- **Alert System**: Visual indicators for performance changes

### Strategic Planning
- **Market Analysis**: Understand market dynamics and opportunities
- **Competitive Positioning**: Benchmark against market performance
- **Resource Allocation**: Optimize investment in amenities and improvements
- **Risk Management**: Identify and mitigate performance risks

This documentation provides a complete understanding of the Investimate Analytics system, from data structure to business applications, ensuring transparency and reproducibility of all analyses and visualizations.
