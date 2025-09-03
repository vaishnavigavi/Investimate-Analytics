# 🏝️ Investimate Analytics

**Advanced Short-Term Rental Performance Dashboard**

A comprehensive Streamlit application for analyzing short-term rental property performance data from Mykonos and Paros islands. Features AI-powered investment recommendations, interactive visualizations, and advanced analytics.

## 🚀 Features

### 📊 Core Analytics
- **Portfolio Overview**: KPI dashboard with performance metrics and trend indicators
- **Seasonality Analysis**: Monthly performance trends and seasonal patterns
- **Property Analysis**: Performance by bedroom count and property type
- **Amenity Impact**: ROI analysis of property features and amenities
- **Pricing Insights**: Price elasticity analysis and optimization recommendations

### 🤖 Advanced Analytics
- **AI Investment Engine**: Machine learning-powered property scoring and recommendations
- **Interactive Maps**: Geographic performance visualization with heatmaps
- **Advanced Metrics**: Revenue concentration, volatility analysis, and efficiency metrics
- **Competitive Intelligence**: Market positioning and benchmarking
- **Predictive Analytics**: ML models for forecasting performance
- **3D Visualizations**: Advanced charts and interactive plots

### 🎯 AI-Powered Features
- **Property Scoring**: 6-dimensional investment scoring system
- **Investment Grades**: A+ to C grading with risk assessment
- **Buy/Sell/Hold Recommendations**: Data-driven investment decisions
- **Portfolio Optimization**: AI-suggested portfolio improvements
- **Performance Forecasting**: Future revenue and occupancy predictions

## 📁 Project Structure

```
investimate/
├── app.py                                    # Main Streamlit application
├── pages/                                    # Individual analysis pages
│   ├── 2_Seasonality.py                     # Monthly trends analysis
│   ├── 3_Bedrooms_and_Types.py              # Property type performance
│   ├── 4_Amenities_and_Management.py        # Amenity impact analysis
│   ├── 5_Pricing_Insights.py                # Price elasticity & optimization
│   ├── 7_Interactive_Maps.py                # Geographic visualizations
│   ├── 8_Advanced_Metrics.py                # Advanced performance metrics
│   ├── 9_Competitive_Intelligence.py        # Market analysis
│   ├── 10_Impressive_Visualizations.py      # 3D charts and advanced plots
│   ├── 11_Predictive_Analytics.py           # ML forecasting models
│   ├── 13_AI_Investment_Recommendations.py  # AI investment engine
│   └── 14_Documentation.py                  # Complete technical documentation
├── utils.py                                  # Core data processing functions
├── ui_utils.py                               # UI utilities and formatting
├── ai_investment_engine.py                   # AI recommendation engine
├── requirements.txt                          # Python dependencies
├── DOCUMENTATION.md                          # Comprehensive documentation
└── data/                                     # CSV data files
    ├── mykonos_monthly_performance.csv
    ├── mykonos_property_details.csv
    ├── paros_monthly_performance.csv
    └── paros_property_details.csv
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd investimate
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - Local URL: http://localhost:8501
   - The app will automatically open in your default browser

## 📊 Data Overview

### Dataset Information
- **Total Records**: 53,419 monthly performance records
- **Unique Properties**: 9,648 properties
- **Date Range**: 2024 (full year)
- **Islands**: Mykonos and Paros
- **Data Sources**: Monthly performance + property details

### Key Metrics
- **ADR (Average Daily Rate)**: Revenue per occupied night
- **Occupancy Rate**: Percentage of available nights booked
- **RevPAR (Revenue per Available Room)**: ADR × Occupancy
- **Revenue**: Total monthly revenue per property

## 🎯 Key Features by Page

### Main Dashboard (`app.py`)
- Portfolio KPIs with trend indicators
- Island performance comparison
- Interactive charts and metrics

### Seasonality Analysis (`pages/2_Seasonality.py`)
- Monthly performance trends
- Seasonal pattern identification
- Peak/low season analysis

### Property Analysis (`pages/3_Bedrooms_and_Types.py`)
- Performance by bedroom count
- Property type analysis
- Revenue optimization insights

### Amenity Impact (`pages/4_Amenities_and_Management.py`)
- ROI analysis of property features
- Uplift calculations for amenities
- Investment recommendations

### Pricing Insights (`pages/5_Pricing_Insights.py`)
- Price elasticity analysis
- Optimal pricing recommendations
- Revenue impact modeling

### AI Investment Engine (`pages/13_AI_Investment_Recommendations.py`)
- Property scoring (6 dimensions)
- Investment grades (A+ to C)
- Buy/Sell/Hold recommendations
- Portfolio optimization
- Performance forecasting

## 🔧 Technical Architecture

### Dependencies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning models
- **Folium**: Interactive maps
- **Matplotlib**: Static plotting

### Performance Optimizations
- **Session State Caching**: Data loaded once and cached
- **Efficient Grouping**: Optimized pandas operations
- **Lazy Loading**: Data loaded only when needed
- **ML Model Caching**: AI models cached for performance

## 📈 Business Applications

### Investment Decision Support
- Identify high-performing property types and locations
- Quantify amenity ROI and impact
- Data-driven pricing strategies
- Seasonal operation optimization

### Performance Monitoring
- KPI tracking over time
- Benchmarking across islands and property types
- Trend analysis and pattern identification
- Visual performance indicators

### Market Intelligence
- Competitive positioning analysis
- Market share insights
- Price tier performance
- Management efficiency metrics

## 🤖 AI Investment Engine

### Scoring System
1. **Revenue Performance** (25%): Historical revenue trends
2. **Occupancy Efficiency** (20%): Booking rate optimization
3. **Pricing Power** (20%): ADR performance vs market
4. **Location Value** (15%): Geographic performance
5. **Property Quality** (10%): Amenities and features
6. **Market Position** (10%): Competitive standing

### Investment Grades
- **A+**: Exceptional investment opportunity
- **A**: Strong investment with high potential
- **B+**: Good investment with solid returns
- **B**: Moderate investment opportunity
- **C+**: Fair investment with limited upside
- **C**: Poor investment opportunity

## 📚 Documentation

For complete technical documentation, including:
- Data structure and column definitions
- Formula explanations and calculations
- Function documentation and code examples
- Business applications and use cases
- Troubleshooting and technical support

Visit the **Documentation** page in the application or see `DOCUMENTATION.md`.

## 🚀 Getting Started

1. **Run the application** using the installation steps above
2. **Explore the Main Dashboard** for portfolio overview
3. **Navigate through pages** using the sidebar
4. **Try the AI Investment Engine** for property recommendations
5. **Review the Documentation** for detailed explanations

## 📞 Support

For technical support or questions:
- Review the comprehensive documentation in the app
- Check the `DOCUMENTATION.md` file
- Examine the source code for implementation details

## 🔄 Recent Updates

- ✅ **AI Investment Engine**: Advanced ML-powered recommendations
- ✅ **Interactive Maps**: Geographic performance visualization
- ✅ **Advanced Analytics**: 3D visualizations and predictive models
- ✅ **UI/UX Improvements**: Responsive design and better formatting
- ✅ **Performance Optimizations**: Caching and efficient data processing


