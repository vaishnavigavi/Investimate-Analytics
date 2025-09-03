import streamlit as st
import pandas as pd
import numpy as np
from utils import load_data, agg_island_summary, seasonality, bedrooms_perf, ptype_perf, uplift_table, price_elasticity_simple

def main():
    st.set_page_config(page_title="Documentation - Investimate Analytics", layout="wide")
    
    st.title("📚 Complete Documentation")
    st.markdown("**Comprehensive guide to Investimate Analytics - Your short-term rental performance dashboard**")
    
    # Current Navigation Structure
    st.info("""
    **📱 Current Navigation Structure:**
    
    **Core Analytics:** Seasonality → Bedrooms & Types → Amenities & Management → Pricing Insights
    
    **Advanced Analytics:** Interactive Maps → Advanced Metrics → Competitive Intelligence → Impressive Visualizations → Predictive Analytics → AI Investment Recommendations
    
    **Documentation:** Complete technical guide (this page)
    """)
    
    st.markdown("---")
    
    # Load data
    if 'data' in st.session_state:
        data = st.session_state.data
    else:
        from ui_utils import load_data_with_spinner
        data = load_data_with_spinner()
        if data is None:
            return
        st.session_state.data = data
    
    # Table of Contents
    st.subheader("📋 Table of Contents")
    
    toc_col1, toc_col2, toc_col3 = st.columns(3)
    
    with toc_col1:
        st.markdown("""
        **📊 Data & Structure**
        - [Data Files](#data-files)
        - [Column Definitions](#column-definitions)
        - [Data Processing](#data-processing)
        """)
    
    with toc_col2:
        st.markdown("""
        **📈 Formulas & Calculations**
        - [KPI Formulas](#kpi-formulas)
        - [Core Functions](#core-functions)
        - [Visualization Methods](#visualization-methods)
        - [AI Investment Engine](#ai-investment-recommendations)
        """)
    
    with toc_col3:
        st.markdown("""
        **🎨 UI & Features**
        - [Page Breakdown](#page-breakdown)
        - [Business Applications](#business-applications)
        - [Technical Architecture](#technical-architecture)
        """)
    
    st.markdown("---")
    
    # Data Files Section
    st.subheader("📁 Data Files")
    
    st.info("""
    **Raw Data Files:**
    1. **mykonos_monthly_performance.csv** - Monthly performance metrics for Mykonos properties
    2. **mykonos_property_details.csv** - Property characteristics for Mykonos
    3. **paros_monthly_performance.csv** - Monthly performance metrics for Paros properties  
    4. **paros_property_details.csv** - Property characteristics for Paros
    """)
    
    # Show data summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        st.metric("Unique Properties", f"{data['property_id'].nunique():,}")
    with col3:
        st.metric("Date Range", f"{data['year'].min()}-{data['year'].max()}")
    with col4:
        st.metric("Islands", f"{data['island'].nunique()}")
    
    st.markdown("---")
    
    # Column Definitions
    st.subheader("📊 Column Definitions")
    
    tab1, tab2 = st.tabs(["Performance Data", "Property Details"])
    
    with tab1:
        st.markdown("""
        **Performance Data Columns (Monthly Performance CSVs)**
        
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
        """)
    
    with tab2:
        st.markdown("""
        **Property Details Columns (Property Details CSVs)**
        
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
        """)
    
    st.markdown("---")
    
    # KPI Formulas
    st.subheader("📈 KPI Formulas & Calculations")
    
    formula_col1, formula_col2 = st.columns(2)
    
    with formula_col1:
        st.markdown("""
        **1. Average Daily Rate (ADR)**
        ```
        ADR = Total Revenue ÷ Occupied Nights
        ADR = Revenue ÷ (Occupancy × Days in Month)
        ```
        **Business Meaning**: Average price charged per occupied night
        
        **2. Occupancy Rate**
        ```
        Occupancy = Occupied Nights ÷ Available Nights × 100
        Occupancy = Occupied Nights ÷ (Days in Month × 1) × 100
        ```
        **Business Meaning**: Percentage of available nights that were booked
        """)
    
    with formula_col2:
        st.markdown("""
        **3. Revenue per Available Room (RevPAR)**
        ```
        RevPAR = ADR × Occupancy Rate
        RevPAR = ADR × (Occupied Nights ÷ Available Nights)
        ```
        **Business Meaning**: Revenue generated per available room
        
        **4. Monthly Revenue**
        ```
        Revenue = ADR × Occupied Nights
        Revenue = ADR × Occupancy × Days in Month
        ```
        **Business Meaning**: Total revenue generated in a month
        """)
    
    st.markdown("---")
    
    # Core Functions
    st.subheader("🔧 Core Functions Explained")
    
    func_tabs = st.tabs(["agg_island_summary", "seasonality", "bedrooms_perf", "ptype_perf", "uplift_table", "price_elasticity_simple"])
    
    with func_tabs[0]:
        st.markdown("""
        **`agg_island_summary(df)`**
        
        **Purpose**: Calculate island-level performance metrics
        
        **Input**: Merged dataset with performance and property data
        
        **Output**: DataFrame with island-level KPIs
        
        **Process**:
        1. Group data by island
        2. Calculate mean ADR, occupancy, RevPAR
        3. Sum total revenue per island
        4. Count unique properties per island
        5. Calculate annualized revenue per property
        """)
        
        # Show example output
        island_summary = agg_island_summary(data)
        st.dataframe(island_summary, )
    
    with func_tabs[1]:
        st.markdown("""
        **`seasonality(df)`**
        
        **Purpose**: Analyze monthly performance trends by island
        
        **Input**: Merged dataset
        
        **Output**: DataFrame with monthly averages by island
        
        **Process**:
        1. Group data by island and month
        2. Calculate monthly averages for ADR, occupancy, RevPAR
        3. Sum monthly revenue per island
        4. Add month names for better readability
        """)
        
        # Show example output
        seasonal_data = seasonality(data)
        st.dataframe(seasonal_data.head(10), )
    
    with func_tabs[2]:
        st.markdown("""
        **`bedrooms_perf(df)`**
        
        **Purpose**: Analyze performance by bedroom count
        
        **Input**: Merged dataset
        
        **Output**: DataFrame with performance metrics by bedroom count
        
        **Process**:
        1. Group data by bedroom count
        2. Calculate average ADR, occupancy, RevPAR
        3. Sum total revenue by bedroom count
        4. Count properties by bedroom count
        """)
        
        # Show example output
        bedroom_data = bedrooms_perf(data)
        st.dataframe(bedroom_data, )
    
    with func_tabs[3]:
        st.markdown("""
        **`ptype_perf(df)`**
        
        **Purpose**: Analyze performance by property type
        
        **Input**: Merged dataset
        
        **Output**: DataFrame with performance metrics by property type
        
        **Process**:
        1. Group data by property type
        2. Calculate average ADR, occupancy, RevPAR
        3. Sum total revenue by property type
        4. Count properties by property type
        """)
        
        # Show example output
        type_data = ptype_perf(data)
        st.dataframe(type_data.head(10), )
    
    with func_tabs[4]:
        st.markdown("""
        **`uplift_table(df)`**
        
        **Purpose**: Calculate the impact of amenities on performance
        
        **Input**: Merged dataset with amenity flags
        
        **Output**: DataFrame showing performance uplift for each amenity
        
        **Process**:
        1. For each amenity feature (pool, wifi, sea_view, etc.)
        2. Split data into properties with/without feature
        3. Calculate average metrics for each group
        4. Calculate percentage uplift: ((with - without) / without) × 100
        5. Return comprehensive uplift analysis
        """)
        
        # Show example output
        uplift_data = uplift_table(data)
        st.dataframe(uplift_data, )
    
    with func_tabs[5]:
        st.markdown("""
        **`price_elasticity_simple(df)`**
        
        **Purpose**: Calculate price elasticity and suggest optimal pricing
        
        **Input**: Merged dataset
        
        **Output**: DataFrame with elasticity analysis and pricing recommendations
        
        **Process**:
        1. For each island and month combination
        2. Fit linear regression: occupancy ~ ADR
        3. Calculate price elasticity: (beta × mean_ADR) / mean_occupancy
        4. Find optimal ADR to maximize RevPAR
        5. Calculate potential revenue improvement
        """)
        
        # Show example output
        elasticity_data = price_elasticity_simple(data)
        st.dataframe(elasticity_data.head(10), )
    
    st.markdown("---")
    
    # Visualization Methods
    st.subheader("📊 Visualization Methods")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("""
        **Main Dashboard (`app.py`)**
        - **Portfolio KPIs**: Streamlit `st.metric()` with delta indicators
        - **Island Charts**: Plotly Express bar charts
        - **Layout**: 5-column responsive grid
        
        **Seasonality Page (`pages/2_Seasonality.py`)**
        - **Monthly Trends**: Matplotlib line plots
        - **Interactive Selection**: Dropdown for metric selection
        - **Data Table**: Detailed monthly performance
        """)
    
    with viz_col2:
        st.markdown("""
        **Seasonality Page (`pages/2_Seasonality.py`)**
        - **Monthly Trends**: Matplotlib line plots
        - **Interactive Selection**: Dropdown for metric selection
        - **Data Table**: Detailed monthly performance
        
        **Bedrooms & Types Page (`pages/3_Bedrooms_and_Types.py`)**
        - **Tabbed Interface**: Separate tabs for different analyses
        - **Line Charts**: RevPAR by bedroom count
        - **Bar Charts**: Performance by property type
        """)
    
    st.markdown("---")
    
    # Page Breakdown
    st.subheader("📱 Page-by-Page Breakdown")
    
    page_tabs = st.tabs(["Main Dashboard", "Core Analytics", "Advanced Analytics", "Documentation"])
    
    with page_tabs[0]:
        st.markdown("""
        **Main Dashboard (`app.py`)**
        - **Portfolio KPIs**: Overall performance with change indicators
        - **Island KPIs**: Individual island performance with comparisons
        - **Interactive Charts**: Plotly charts for island comparisons
        - **Standardized Data Loading**: Centralized data handling
        - **Responsive Design**: Mobile-friendly layout
        """)
    
    with page_tabs[1]:
        st.markdown("""
        **Core Analytics Pages:**
        
        **📅 Seasonality (`pages/2_Seasonality.py`)**
        - Monthly trend analysis by island
        - Interactive metric selection (ADR, Occupancy, RevPAR)
        - Line charts showing seasonal patterns
        
        **🛏️ Bedrooms & Types (`pages/3_Bedrooms_and_Types.py`)**
        - Performance by bedroom count
        - Property type analysis
        - Separate tabs for different analyses
        
        **🏊 Amenities & Management (`pages/4_Amenities_and_Management.py`)**
        - Feature impact analysis (pool, wifi, sea view, etc.)
        - Uplift calculations for amenities
        - Management type performance comparison
        
        **💰 Pricing Insights (`pages/5_Pricing_Insights.py`)**
        - Price elasticity analysis
        - Optimal ADR recommendations
        - Revenue optimization strategies
        """)
    
    with page_tabs[2]:
        st.markdown("""
        **Advanced Analytics Pages:**
        
        **🗺️ Interactive Maps (`pages/7_Interactive_Maps.py`)**
        - Geographic property performance visualization
        - Performance heatmaps by location
        - Interactive filtering and clustering
        
        **📊 Advanced Metrics (`pages/8_Advanced_Metrics.py`)**
        - Revenue efficiency analysis
        - Performance volatility metrics
        - Revenue concentration analysis
        
        **🏆 Competitive Intelligence (`pages/9_Competitive_Intelligence.py`)**
        - Market positioning analysis
        - Performance percentile rankings
        - Price tier and management performance comparison
        
        **🎨 Impressive Visualizations (`pages/10_Impressive_Visualizations.py`)**
        - 3D scatter plots and correlation matrices
        - Advanced statistical visualizations
        - Interactive data exploration
        
        **📈 Predictive Analytics (`pages/11_Predictive_Analytics.py`)**
        - Machine learning models for forecasting
        - Seasonal demand prediction
        - Revenue optimization recommendations
        
        **🤖 AI Investment Recommendations (`pages/13_AI_Investment_Recommendations.py`)**
        - **Property Investment Scores**: 6-dimensional scoring system (Revenue, ADR, Occupancy, Stability, Amenities, Location)
        - **Investment Grades**: A+ to C grading system with color-coded badges
        - **Risk Assessment**: Low/Medium/High risk categorization with explanations
        - **Buy/Sell/Hold Recommendations**: Actionable investment advice with reasoning
        - **Portfolio Optimization**: Strategic suggestions for portfolio improvement
        - **Future Performance Prediction**: 12-month forecasting with confidence intervals
        - **Investment Opportunities**: Identified high-potential properties
        - **Interactive Filters**: Filter by island, property type, investment grade, and risk level
        - **Export Functionality**: Download recommendations and analysis results
        
        **🔧 AI Engine Technical Features:**
        - **Machine Learning Models**: RandomForestRegressor, GradientBoostingRegressor, LinearRegression
        - **Feature Engineering**: 79 engineered features from raw data
        - **Model Performance**: R² scores for validation (Revenue: 0.62, ADR: 0.83, Occupancy: 0.37)
        - **Caching Strategy**: Session state caching to prevent re-computation
        - **Error Handling**: Graceful handling of missing data (e.g., bathrooms column)
        - **Progress Indicators**: Visual feedback during model training
        - **Scalability**: Handles datasets with 50K+ records efficiently
        """)
    
    with page_tabs[3]:
        st.markdown("""
        **Documentation Page (`pages/14_Documentation.py`)**
        - **Complete Technical Documentation**: This comprehensive guide
        - **Data Structure**: Detailed explanation of datasets and columns
        - **Formula Reference**: All calculations and methodologies
        - **Function Documentation**: Code-level explanations
        - **Business Applications**: Real-world use cases
        """)
    

    
    st.markdown("---")
    
    # Recent Improvements
    st.subheader("🔄 Recent Improvements & Consolidations")
    
    st.success("""
    **Streamlined Architecture (Latest Update):**
    
    ✅ **Removed Redundancies**: Eliminated duplicate pages and content
    ✅ **Standardized Data Loading**: Centralized data handling across all pages
    ✅ **Cleaner Navigation**: Streamlined sidebar without redundant sections
    ✅ **Optimized Performance**: Faster loading with reduced code duplication
    ✅ **Better Maintainability**: Single source of truth for utilities and functions
    
    **Key Changes:**
    - Removed redundant `1_Overview.py` (functionality moved to main `app.py`)
    - Removed duplicate `12_Deep_Dive_Analyses.py` (consolidated into `9_Competitive_Intelligence.py`)
    - Standardized data loading with `load_data_with_spinner()` function
    - Cleaned up sidebar navigation and removed redundant sections
    - Consolidated documentation to single comprehensive page
    """)
    
    st.markdown("---")
    
    # Business Applications
    st.subheader("🚀 Business Applications")
    
    app_col1, app_col2 = st.columns(2)
    
    with app_col1:
        st.markdown("""
        **Investment Decision Support**
        - **Property Selection**: Identify high-performing property types and locations
        - **Amenity ROI**: Quantify the impact of specific amenities
        - **Pricing Strategy**: Data-driven pricing recommendations
        - **Seasonal Planning**: Optimize operations for seasonal variations
        """)
    
    with app_col2:
        st.markdown("""
        **Performance Monitoring**
        - **KPI Tracking**: Monitor key performance indicators over time
        - **Benchmarking**: Compare performance across islands and property types
        - **Trend Analysis**: Identify performance trends and patterns
        - **Alert System**: Visual indicators for performance changes
        """)
    
    st.markdown("---")
    
    # Technical Architecture
    st.subheader("🔧 Technical Architecture")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        **Dependencies**
        - **Streamlit**: Web application framework
        - **Pandas**: Data manipulation and analysis
        - **NumPy**: Numerical computations and linear regression
        - **Matplotlib**: Static plotting and visualization
        - **Plotly**: Interactive plotting and charts
        """)
    
    with tech_col2:
        st.markdown("""
        **Performance Optimizations**
        - **Session State**: Data loaded once and stored in session state
        - **Efficient Grouping**: Optimized pandas groupby operations
        - **Caching**: Streamlit's built-in caching for data loading
        - **Lazy Loading**: Data loaded only when needed
        """)
    
    st.markdown("---")
    
    # Data Quality & Validation
    st.subheader("🔍 Data Quality & Validation")
    
    st.markdown("""
    **Data Validation Steps**
    1. **Null Value Handling**: Systematic removal of incomplete records
    2. **Outlier Detection**: IQR-based outlier removal for performance metrics
    3. **Type Validation**: Ensuring correct data types for calculations
    4. **Range Validation**: Occupancy rates between 0-1, positive revenue values
    
    **Business Logic Validation**
    1. **RevPAR Consistency**: RevPAR = ADR × Occupancy validation
    2. **Revenue Consistency**: Revenue = ADR × Occupancy × Days validation
    3. **Island Mapping**: Correct mapping of adm_3_id to island names
    4. **Property Matching**: Successful merge of performance and property data
    """)
    
    st.markdown("---")
    
    # AI Investment Engine Section
    st.subheader("🤖 AI Investment Recommendations")
    
    st.markdown("""
    **The AI Investment Recommendations page (`pages/13_AI_Investment_Recommendations.py`) provides intelligent investment analysis using machine learning models.**
    
    ### 🎯 Key Features
    
    **Property Investment Scoring:**
    - **6-Dimensional Scoring**: Revenue, ADR, Occupancy, Stability, Amenities, Location
    - **Investment Grades**: A+ to C grading system with color-coded badges
    - **Risk Assessment**: Low/Medium/High risk categorization with detailed explanations
    
    **Investment Recommendations:**
    - **Buy/Sell/Hold Advice**: Actionable investment recommendations with reasoning
    - **Portfolio Optimization**: Strategic suggestions for portfolio improvement
    - **Future Performance Prediction**: 12-month forecasting with confidence intervals
    - **Investment Opportunities**: Identified high-potential properties
    
    **Interactive Dashboard:**
    - **Executive Summary**: Key insights and recommendations at a glance
    - **Property Scoring Matrix**: Visual representation of all properties with scores
    - **Investment Grade Distribution**: Pie charts showing grade distribution
    - **Risk Analysis**: Risk distribution and mitigation strategies
    - **Top Opportunities**: Highlighted high-potential investments
    - **Interactive Filters**: Filter by island, property type, investment grade, and risk level
    - **Export Functionality**: Download recommendations and analysis results
    
    ### 🔧 Technical Implementation
    
    **Machine Learning Models:**
    - **RandomForestRegressor**: For revenue and RevPAR prediction
    - **GradientBoostingRegressor**: For ADR and occupancy prediction
    - **LinearRegression**: Baseline model for comparison
    - **Cross-validation**: Model performance evaluation
    
    **Performance Metrics:**
    - **Revenue Model**: R² = 0.62 (62% variance explained)
    - **ADR Model**: R² = 0.83 (83% variance explained)
    - **Occupancy Model**: R² = 0.37 (37% variance explained)
    - **RevPAR Model**: R² = 0.62 (62% variance explained)
    
    **Technical Features:**
    - **Feature Engineering**: 79 engineered features from raw data
    - **Caching Strategy**: Session state caching to prevent re-computation
    - **Error Handling**: Graceful handling of missing data (e.g., bathrooms column)
    - **Progress Indicators**: Visual feedback during model training
    - **Scalability**: Handles datasets with 50K+ records efficiently
    
    ### 📊 Business Value
    
    **Investment Decision Support:**
    - Identify high-performing properties for acquisition
    - Optimize existing portfolio performance
    - Risk assessment and mitigation strategies
    - Data-driven investment recommendations
    
    **Portfolio Management:**
    - Strategic portfolio rebalancing suggestions
    - Performance benchmarking against market
    - Future performance forecasting
    - Investment opportunity identification
    """)
    
    st.markdown("---")
    
    # Code Function Explanations
    st.subheader("💻 Python Code Function Explanations")
    
    code_tabs = st.tabs(["Main Dashboard", "Overview Page", "Seasonality Page", "Bedrooms & Types", "Amenities Page", "Pricing Page"])
    
    with code_tabs[0]:
        st.markdown("""
        **Main Dashboard (`app.py`) - Key Functions:**
        
        ```python
        # Data Loading with Session State
        if 'data' in st.session_state:
            data = st.session_state.data  # Use cached data
        else:
            with st.spinner("Loading data..."):
                data = load_data()  # Load from CSV files
                st.session_state.data = data  # Cache for other pages
        ```
        
        **Explanation**: Uses Streamlit session state to cache data across pages, shows loading spinner during data processing.
        
        ```python
        # Portfolio KPI Calculation
        total_revenue = data['revenue'].sum()
        avg_adr = data['adr'].mean()
        portfolio_adr_change = ((second_half['adr'].mean() - first_half['adr'].mean()) / first_half['adr'].mean()) * 100
        ```
        
        **Explanation**: Uses pandas aggregation methods, filters data by month ranges for period-over-period comparison, calculates percentage change.
        
        ```python
        # KPI Display with Bordered Containers
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(label="💰 Total Portfolio Revenue", value=f"€{total_revenue:,.0f}")
        st.caption("Sum of all monthly revenues from all properties")
        st.markdown('</div>', unsafe_allow_html=True)
        ```
        
        **Explanation**: Wraps each metric in custom HTML div with CSS class, creates metric display with delta indicators, adds explanatory text.
        """)
    
    with code_tabs[1]:
        st.markdown("""
        **Main Dashboard (`app.py`) - Key Functions:**
        
        ```python
        # Period Change Calculation Function
        def calculate_period_changes(data):
            first_half = data[data['month'].isin([1, 2, 3, 4, 5, 6])]
            second_half = data[data['month'].isin([7, 8, 9, 10, 11, 12])]
            
            changes = {}
            for island in ['Mykonos', 'Paros']:
                island_first = first_half[first_half['island'] == island]
                island_second = second_half[second_half['island'] == island]
                
                changes[island] = {
                    'adr_change': ((island_second['adr'].mean() - island_first['adr'].mean()) / island_first['adr'].mean()) * 100
                }
            return changes
        ```
        
        **Explanation**: Nested function that calculates period-over-period changes, filters data by month ranges and island, returns dictionary with change percentages.
        
        ```python
        # Matplotlib Chart Creation
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        axes[0].bar(islands, adr_values, color=['#1f77b4', '#ff7f0e'])
        axes[0].set_title('Average Daily Rate (€)')
        plt.tight_layout()
        st.pyplot(fig)
        ```
        
        **Explanation**: Creates subplot layout with 1 row, 3 columns, uses consistent color scheme, prevents label overlap with tight_layout.
        """)
    
    with code_tabs[2]:
        st.markdown("""
        **Seasonality Page (`pages/2_Seasonality.py`) - Key Functions:**
        
        ```python
        # Metric Selection Interface
        metric_options = {
            'Average Daily Rate (€)': 'avg_adr',
            'Average Occupancy Rate (%)': 'avg_occupancy', 
            'Average RevPAR (€)': 'avg_revpar'
        }
        selected_metric = st.selectbox("Select a metric to analyze:", options=list(metric_options.keys()))
        metric_column = metric_options[selected_metric]
        ```
        
        **Explanation**: Creates dictionary mapping display names to column names, creates dropdown interface, maps user selection to DataFrame column.
        
        ```python
        # Line Chart Creation
        for island in islands:
            island_data = seasonal_data[seasonal_data['island'] == island]
            values = island_data[metric_column]
            plt.plot(island_data['month'], values, marker='o', label=island, linewidth=2)
        
        plt.title(f'{selected_metric} - Monthly Trends by Island')
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ```
        
        **Explanation**: Iterates through each island to create separate lines, uses markers for data point visibility, customizes month labels.
        """)
    
    with code_tabs[3]:
        st.markdown("""
        **Bedrooms & Types Page (`pages/3_Bedrooms_and_Types.py`) - Key Functions:**
        
        ```python
        # Tabbed Interface
        tab1, tab2 = st.tabs(["🏠 Bedrooms", "🏘️ Property Types"])
        
        with tab1:
            # Bedroom analysis content
        with tab2:
            # Property type analysis content
        ```
        
        **Explanation**: Creates tabbed interface for organizing different analyses.
        
        ```python
        # Property Type Analysis with Filtering
        top_types = type_performance[
            (type_performance['property_count'] >= 10) & 
            (type_performance['property_count'] <= 12)
        ].nlargest(12, 'avg_revpar')
        
        # Create three separate charts
        col1, col2, col3 = st.columns(3)
        with col1:
            ax_adr.barh(top_types['property_type'], top_types['avg_adr'])
        ```
        
        **Explanation**: Filters property types by count range, uses nlargest() to get top performers, creates separate charts in columns, uses horizontal bar charts for readability.
        """)
    
    with code_tabs[4]:
        st.markdown("""
        **Amenities & Management Page (`pages/4_Amenities_and_Management.py`) - Key Functions:**
        
        ```python
        # Uplift Analysis
        uplift_data = uplift_table(data)
        st.dataframe(uplift_data, )
        ```
        
        **Explanation**: Calls uplift_table() function to analyze amenity impact on performance.
        
        ```python
        # Visualization by Island
        for island in islands:
            island_data = data[data['island'] == island]
            island_uplift = uplift_table(island_data)
            
            bars = ax.bar(island_uplift['feature'], island_uplift['revpar_uplift_pct'])
            
            # Color bars based on positive/negative uplift
            for i, bar in enumerate(bars):
                if island_uplift.iloc[i]['revpar_uplift_pct'] > 0:
                    bar.set_color('#2ecc71')  # Green for positive
                else:
                    bar.set_color('#e74c3c')  # Red for negative
        ```
        
        **Explanation**: Iterates through each island for separate analysis, creates bar chart with color coding, uses conditional coloring for positive/negative values.
        """)
    
    with code_tabs[5]:
        st.markdown("""
        **Pricing Insights Page (`pages/5_Pricing_Insights.py`) - Key Functions:**
        
        ```python
        # Island Selection and Visualization
        islands = elasticity_data['island'].unique()
        selected_island = st.selectbox("Select an island for detailed analysis:", islands)
        island_elasticity = elasticity_data[elasticity_data['island'] == selected_island]
        ```
        
        **Explanation**: Creates dropdown for island selection, filters elasticity data for selected island.
        
        ```python
        # Dual Charts Creation
        col1, col2 = st.columns(2)
        
        with col1:
            # Current vs Optimal ADR
            ax1.plot(island_elasticity['month'], island_elasticity['current_adr'], 'o-', label='Current ADR')
            ax1.plot(island_elasticity['month'], island_elasticity['suggested_adr'], 's-', label='Optimal ADR')
        
        with col2:
            # Potential RevPAR Improvement
            bars = ax2.bar(island_elasticity['month'], island_elasticity['revpar_improvement_pct'])
            for i, bar in enumerate(bars):
                if island_elasticity.iloc[i]['revpar_improvement_pct'] > 0:
                    bar.set_color('#2ecc71')  # Green for positive
        ```
        
        **Explanation**: Creates two side-by-side charts, shows current vs optimal ADR with different markers, displays potential improvement with color-coded bars.
        
        ```python
        # Key Insights Generation
        total_improvement = island_elasticity['revpar_improvement_pct'].sum()
        best_month = island_elasticity.loc[island_elasticity['revpar_improvement_pct'].idxmax()]
        
        st.markdown(f'''
        **For {selected_island}:**
        - **Total Potential Improvement**: {total_improvement:.1f}% across all months
        - **Best Optimization Opportunity**: Month {best_month['month']} with {best_month['revpar_improvement_pct']:.1f}% improvement
        ''')
        ```
        
        **Explanation**: Calculates summary statistics, uses idxmax() to find best month, formats insights using f-strings and markdown.
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    ---
    **📚 Complete Documentation for Investimate Analytics**
    
    This documentation provides a comprehensive understanding of the system, from data structure to business applications, ensuring transparency and reproducibility of all analyses and visualizations.
    
    **Code Functions Explained**: Each page's Python functions are documented with explanations of their purpose, parameters, and implementation details.
    
    For technical support or questions, refer to the source code in the respective files.
    """)

if __name__ == "__main__":
    main()
