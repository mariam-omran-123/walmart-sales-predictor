"""
Walmart Store Sales Prediction Web App
Save as: walmart_sales_app.py
Run with: streamlit run walmart_sales_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Walmart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #0071ce 0%, #004f9a 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #0071ce 0%, #004f9a 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_sample_data():
    """Create sample Walmart sales data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='W')
    n = len(dates)
    
    stores = np.random.choice(['Store_1', 'Store_2', 'Store_3', 'Store_4', 'Store_5'], n)
    departments = np.random.choice(['Electronics', 'Grocery', 'Clothing', 'Home', 'Sports'], n)
    
    # Create realistic sales patterns
    base_sales = np.random.uniform(5000, 50000, n)
    
    # Add seasonality (higher sales in Q4)
    month = pd.DatetimeIndex(dates).month
    seasonal_factor = 1 + 0.3 * np.sin((month - 1) * np.pi / 6)
    
    # Add holiday effect
    is_holiday = np.random.choice([0, 1], n, p=[0.85, 0.15])
    holiday_boost = 1 + (0.5 * is_holiday)
    
    # Add temperature effect
    temperature = 50 + 30 * np.sin((month - 1) * np.pi / 6) + np.random.normal(0, 5, n)
    temp_factor = 1 + ((temperature - 50) / 100)
    
    # Add fuel price effect (inverse relationship)
    fuel_price = 2.5 + np.random.normal(0, 0.5, n)
    fuel_factor = 1 - ((fuel_price - 2.5) / 10)
    
    # Add CPI (Consumer Price Index)
    cpi = 200 + np.random.normal(0, 10, n)
    
    # Add unemployment rate
    unemployment = 5 + np.random.normal(0, 1, n)
    
    # Calculate final sales
    weekly_sales = base_sales * seasonal_factor * holiday_boost * temp_factor * fuel_factor
    weekly_sales = np.maximum(weekly_sales, 1000)  # Minimum sales threshold
    
    df = pd.DataFrame({
        'Date': dates,
        'Store': stores,
        'Department': departments,
        'Weekly_Sales': weekly_sales,
        'IsHoliday': is_holiday,
        'Temperature': temperature,
        'Fuel_Price': fuel_price,
        'CPI': cpi,
        'Unemployment': unemployment
    })
    
    return df

def preprocess_data(df):
    """Preprocess the data for modeling"""
    df = df.copy()
    
    # Extract date features
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['Quarter'] = df['Date'].dt.quarter
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'Date':
            if col not in st.session_state.encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                st.session_state.encoders[col] = le
            else:
                le = st.session_state.encoders[col]
                # Handle unseen labels
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col].astype(str))
    
    return df

def train_model(df, model_type='Random Forest'):
    """Train the sales prediction model"""
    # Prepare data
    df_processed = preprocess_data(df)
    
    # Define features and target
    exclude_cols = ['Date', 'Weekly_Sales']
    feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
    
    X = df_processed[feature_cols]
    y = df_processed['Weekly_Sales']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    metrics = {
        'train': {
            'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'mae': mean_absolute_error(y_train, train_pred),
            'r2': r2_score(y_train, train_pred)
        },
        'test': {
            'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'mae': mean_absolute_error(y_test, test_pred),
            'r2': r2_score(y_test, test_pred)
        }
    }
    
    return model, metrics, feature_cols

def predict_sales(model, input_data, feature_cols):
    """Make sales prediction"""
    input_processed = preprocess_data(input_data)
    X = input_processed[feature_cols]
    prediction = model.predict(X)
    return prediction[0]

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #0071ce;'>
            🛒 Walmart Store Sales Prediction
        </h1>
        <p style='text-align: center; font-size: 1.2rem; color: #666;'>
            AI-Powered Sales Forecasting System
        </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("---")
        
        st.header("⚙️ Configuration")
        
        # Navigation
        page = st.radio(
            "Navigate to:",
            ["📊 Data Overview", "🤖 Train Model", "🔮 Make Predictions", "📈 Analytics"]
        )
        
        st.markdown("---")
        
        # Data source
        st.subheader("Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Use Sample Data", "Upload CSV"]
        )
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your sales data", type=['csv'])
            if uploaded_file:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success("✅ Data loaded successfully!")
        else:
            if st.button("Generate Sample Data"):
                st.session_state.data = create_sample_data()
                st.success("✅ Sample data generated!")
        
        st.markdown("---")
        st.markdown("""
            ### 📋 Expected Columns:
            - Date
            - Store
            - Department
            - Weekly_Sales
            - IsHoliday
            - Temperature
            - Fuel_Price
            - CPI
            - Unemployment
        """)
    
    # Main content based on selected page
    if st.session_state.data is not None:
        df = st.session_state.data
        
        if page == "📊 Data Overview":
            show_data_overview(df)
        elif page == "🤖 Train Model":
            show_train_model(df)
        elif page == "🔮 Make Predictions":
            show_predictions(df)
        else:
            show_analytics(df)
    else:
        st.info("👈 Please load or generate data from the sidebar to get started!")
        
        # Show welcome message
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div style='text-align: center; padding: 3rem;'>
                    <h2>Welcome to Walmart Sales Predictor! 🎯</h2>
                    <p style='font-size: 1.1rem; color: #666;'>
                        This app helps you forecast store sales using machine learning.
                    </p>
                    <br>
                    <p><strong>Features:</strong></p>
                    <ul style='text-align: left; display: inline-block;'>
                        <li>📊 Interactive data exploration</li>
                        <li>🤖 Multiple ML models</li>
                        <li>🔮 Real-time predictions</li>
                        <li>📈 Advanced analytics</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

def show_data_overview(df):
    """Display data overview and statistics"""
    st.header("📊 Data Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3>{len(df):,}</h3>
                <p>Total Records</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3>${df['Weekly_Sales'].mean():,.0f}</h3>
                <p>Avg Weekly Sales</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h3>{df['Store'].nunique()}</h3>
                <p>Unique Stores</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <h3>{df['Department'].nunique()}</h3>
                <p>Departments</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data preview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📊 Data Info")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Date Range:** {df['Date'].min()} to {df['Date'].max()}")
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
    
    # Statistics
    st.markdown("---")
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Visualizations
    st.markdown("---")
    st.subheader("📊 Data Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Sales Trends", "Distribution", "Correlations"])
    
    with tab1:
        # Sales over time
        df['Date'] = pd.to_datetime(df['Date'])
        sales_by_date = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
        
        fig = px.line(sales_by_date, x='Date', y='Weekly_Sales',
                     title='Total Weekly Sales Over Time')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sales by store
        sales_by_store = df.groupby('Store')['Weekly_Sales'].mean().reset_index()
        fig2 = px.bar(sales_by_store, x='Store', y='Weekly_Sales',
                     title='Average Sales by Store')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # Distribution of sales
        fig = px.histogram(df, x='Weekly_Sales', nbins=50,
                          title='Distribution of Weekly Sales')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plot by department
        fig2 = px.box(df, x='Department', y='Weekly_Sales',
                     title='Sales Distribution by Department')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                       title='Feature Correlation Heatmap')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

def show_train_model(df):
    """Model training interface"""
    st.header("🤖 Train Sales Prediction Model")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Configuration")
        
        model_type = st.selectbox(
            "Select Model Type",
            ["Random Forest", "Gradient Boosting", "Linear Regression"]
        )
        
        st.info(f"""
        **{model_type}** selected
        
        - Random Forest: Ensemble of decision trees, robust to outliers
        - Gradient Boosting: Sequential ensemble, high accuracy
        - Linear Regression: Simple, fast, interpretable
        """)
        
        if st.button("🚀 Train Model", key="train"):
            with st.spinner("Training model... This may take a moment..."):
                try:
                    model, metrics, features = train_model(df, model_type)
                    st.session_state.model = model
                    st.session_state.feature_cols = features
                    st.session_state.trained = True
                    st.session_state.metrics = metrics
                    st.success("✅ Model trained successfully!")
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
    
    with col2:
        if st.session_state.trained and st.session_state.metrics:
            st.subheader("📊 Model Performance")
            
            metrics = st.session_state.metrics
            
            # Training metrics
            st.markdown("**Training Set:**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("RMSE", f"${metrics['train']['rmse']:,.0f}")
            with col_b:
                st.metric("MAE", f"${metrics['train']['mae']:,.0f}")
            with col_c:
                st.metric("R² Score", f"{metrics['train']['r2']:.3f}")
            
            # Test metrics
            st.markdown("**Test Set:**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("RMSE", f"${metrics['test']['rmse']:,.0f}")
            with col_b:
                st.metric("MAE", f"${metrics['test']['mae']:,.0f}")
            with col_c:
                st.metric("R² Score", f"{metrics['test']['r2']:.3f}")
            
            # Performance interpretation
            r2_test = metrics['test']['r2']
            if r2_test > 0.8:
                st.success("🎯 Excellent model performance!")
            elif r2_test > 0.6:
                st.info("👍 Good model performance!")
            else:
                st.warning("⚠️ Model could be improved. Try different features or model types.")
        else:
            st.info("👈 Train a model to see performance metrics")
    
    # Save/Load model
    st.markdown("---")
    st.subheader("💾 Save/Load Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.trained:
            if st.button("💾 Save Model"):
                model_data = {
                    'model': st.session_state.model,
                    'features': st.session_state.feature_cols,
                    'encoders': st.session_state.encoders
                }
                pickle_data = pickle.dumps(model_data)
                st.download_button(
                    label="📥 Download Model",
                    data=pickle_data,
                    file_name="walmart_sales_model.pkl",
                    mime="application/octet-stream"
                )
    
    with col2:
        uploaded_model = st.file_uploader("Upload saved model", type=['pkl'])
        if uploaded_model:
            try:
                model_data = pickle.load(uploaded_model)
                st.session_state.model = model_data['model']
                st.session_state.feature_cols = model_data['features']
                st.session_state.encoders = model_data['encoders']
                st.session_state.trained = True
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")

def show_predictions(df):
    """Make predictions interface"""
    st.header("🔮 Make Sales Predictions")
    
    if not st.session_state.trained:
        st.warning("⚠️ Please train a model first in the 'Train Model' section!")
        return
    
    st.subheader("Enter Store Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_input = st.date_input("Date", datetime.now())
        store = st.selectbox("Store", df['Store'].unique())
        department = st.selectbox("Department", df['Department'].unique())
    
    with col2:
        is_holiday = st.checkbox("Is Holiday?")
        temperature = st.slider("Temperature (°F)", 0, 100, 70)
        fuel_price = st.slider("Fuel Price ($)", 2.0, 4.0, 3.0, 0.1)
    
    with col3:
        cpi = st.number_input("Consumer Price Index", value=200.0)
        unemployment = st.slider("Unemployment Rate (%)", 3.0, 10.0, 5.0, 0.1)
    
    st.markdown("---")
    
    if st.button("🎯 Predict Sales", key="predict"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Date': [pd.to_datetime(date_input)],
            'Store': [store],
            'Department': [department],
            'IsHoliday': [1 if is_holiday else 0],
            'Temperature': [temperature],
            'Fuel_Price': [fuel_price],
            'CPI': [cpi],
            'Unemployment': [unemployment],
            'Weekly_Sales': [0]  # Placeholder
        })
        
        try:
            prediction = predict_sales(
                st.session_state.model, 
                input_data, 
                st.session_state.feature_cols
            )
            
            # Display prediction
            st.markdown(f"""
                <div class="prediction-box">
                    <h2 style='margin:0;'>Predicted Weekly Sales</h2>
                    <h1 style='margin: 1rem 0; font-size: 3rem;'>${prediction:,.2f}</h1>
                    <p>Based on the provided inputs</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_sales = df['Weekly_Sales'].mean()
                diff = prediction - avg_sales
                pct = (diff / avg_sales) * 100
                st.metric(
                    "vs. Average Sales",
                    f"${diff:,.0f}",
                    f"{pct:+.1f}%"
                )
            
            with col2:
                store_avg = df[df['Store'] == store]['Weekly_Sales'].mean()
                diff_store = prediction - store_avg
                pct_store = (diff_store / store_avg) * 100
                st.metric(
                    f"vs. {store} Average",
                    f"${diff_store:,.0f}",
                    f"{pct_store:+.1f}%"
                )
            
            with col3:
                dept_avg = df[df['Department'] == department]['Weekly_Sales'].mean()
                diff_dept = prediction - dept_avg
                pct_dept = (diff_dept / dept_avg) * 100
                st.metric(
                    f"vs. {department} Average",
                    f"${diff_dept:,.0f}",
                    f"{pct_dept:+.1f}%"
                )
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

def show_analytics(df):
    """Advanced analytics and insights"""
    st.header("📈 Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Seasonal Patterns", "Store Performance", "Department Analysis"])
    
    with tab1:
        st.subheader("Seasonal Sales Patterns")
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        
        # Monthly trends
        monthly_sales = df.groupby('Month')['Weekly_Sales'].mean().reset_index()
        fig = px.line(monthly_sales, x='Month', y='Weekly_Sales',
                     title='Average Sales by Month', markers=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Holiday vs non-holiday
        holiday_sales = df.groupby('IsHoliday')['Weekly_Sales'].mean().reset_index()
        holiday_sales['IsHoliday'] = holiday_sales['IsHoliday'].map({0: 'Non-Holiday', 1: 'Holiday'})
        fig2 = px.bar(holiday_sales, x='IsHoliday', y='Weekly_Sales',
                     title='Holiday vs Non-Holiday Sales', color='IsHoliday')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Store Performance Comparison")
        
        store_perf = df.groupby('Store').agg({
            'Weekly_Sales': ['mean', 'sum', 'std']
        }).reset_index()
        store_perf.columns = ['Store', 'Avg_Sales', 'Total_Sales', 'Std_Dev']
        
        fig = px.bar(store_perf, x='Store', y='Avg_Sales',
                    title='Average Sales by Store', color='Avg_Sales')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(store_perf, use_container_width=True)
    
    with tab3:
        st.subheader("Department Analysis")
        
        dept_perf = df.groupby('Department').agg({
            'Weekly_Sales': ['mean', 'sum']
        }).reset_index()
        dept_perf.columns = ['Department', 'Avg_Sales', 'Total_Sales']
        
        fig = px.pie(dept_perf, values='Total_Sales', names='Department',
                    title='Total Sales Distribution by Department')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(dept_perf, use_container_width=True)

if __name__ == "__main__":
    main()
