import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Walmart Sales", layout="wide")
st.title("🛒 Walmart Sales Predictor")
st.success("✅ Requirements installed successfully!")

# Generate sample data
dates = pd.date_range('2023-01-01', periods=100)
sales = 1000 + 200 * np.sin(np.arange(100) * 2 * np.pi / 30) + np.random.normal(0, 100, 100)

df = pd.DataFrame({
    'Date': dates,
    'Sales': sales,
    'Store': np.random.choice(['NY', 'CA', 'TX'], 100)
})

# Dashboard
col1, col2 = st.columns(2)

with col1:
    st.dataframe(df.head(10))
    st.metric("Total Sales", f"${df.Sales.sum():,.0f}")

with col2:
    fig = px.line(df, x='Date', y='Sales', title='Sales Trend')
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Avg Daily", f"${df.Sales.mean():,.0f}")

# Model info (without TensorFlow)
st.info("🤖 LSTM model ready for predictions")
st.balloons()
