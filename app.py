import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(page_title="Boston Housing Price Predictor", layout="wide")

# Define column types based on dataset schema
float_cols = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
int_cols = ['CHAS', 'RAD']

# Load the dataset and model
@st.cache_data
def load_data():
    data = pd.read_csv('data/Boston House Prices.csv')
    # Ensure correct dtypes
    for col in float_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')
    for col in int_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').astype('int64')
    return data

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load data and model
try:
    data = load_data()
    model = load_model()
except FileNotFoundError as e:
    st.error(f"File not found: {e}. Please ensure 'data/Boston House Prices.csv' and 'model.pkl' are in the correct paths.")
    st.stop()

# Title and Description
st.title("Boston Housing Price Predictor")
st.markdown("""
This application predicts house prices in Boston using a Random Forest model trained on the Boston Housing Prices dataset. 
Explore the dataset, visualize relationships, and make predictions based on input features.
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Section", ["Data Exploration", "Visualizations", "Model Prediction", "Model Performance"])

# Data Exploration Section
if page == "Data Exploration":
    st.header("Data Exploration")
    
    # Dataset Overview
    st.subheader("Dataset Overview")
    st.write(f"Shape: {data.shape}")
    st.write("Columns:", list(data.columns))
    st.write("Data Types:")
    st.write(data.dtypes)
    
    # Sample Data
    st.subheader("Sample Data")
    n_rows = st.slider("Select number of rows to display", 5, 50, 5)
    st.dataframe(data.head(n_rows).astype({col: 'float64' for col in float_cols}).astype({col: 'int64' for col in int_cols}))
    
    # Interactive Filtering
    st.subheader("Filter Data")
    columns = data.columns.tolist()
    selected_column = st.selectbox("Select column to filter", columns)
    if data[selected_column].dtype in ['int64', 'float64']:
        min_val, max_val = st.slider(
            f"Select range for {selected_column}",
            float(data[selected_column].min()),
            float(data[selected_column].max()),
            (float(data[selected_column].min()), float(data[selected_column].max()))
        )
        filtered_data = data[(data[selected_column] >= min_val) & (data[selected_column] <= max_val)]
    else:
        unique_values = data[selected_column].unique()
        selected_values = st.multiselect(f"Select values for {selected_column}", unique_values, default=unique_values[:2])
        filtered_data = data[data[selected_column].isin(selected_values)]
    # Ensure filtered_data has consistent types
    filtered_data = filtered_data.astype({col: 'float64' for col in float_cols}).astype({col: 'int64' for col in int_cols})
    st.dataframe(filtered_data)

# Visualization Section
elif page == "Visualizations":
    st.header("Visualizations")
    
    # Plot 1: Correlation Matrix
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)
    
    # Plot 2: House Price Distribution
    st.subheader("House Price Distribution")
    fig = px.histogram(data, x='MEDV', nbins=30, title="Distribution of House Prices (MEDV)")
    st.plotly_chart(fig)
    
    # Plot 3: Scatter Plot with Interactive Feature Selection
    st.subheader("Scatter Plot: Feature vs House Price")
    feature = st.selectbox("Select feature to plot against MEDV", [col for col in data.columns if col != 'MEDV'])
    fig = px.scatter(data, x=feature, y='MEDV', color=feature, title=f"{feature} vs House Price")
    st.plotly_chart(fig)

# Model Prediction Section
elif page == "Model Prediction":
    st.header("Model Prediction")
    
    st.subheader("Input Features for Prediction")
    inputs = {}
    cols = st.columns(3)
    for i, feature in enumerate(data.drop('MEDV', axis=1).columns):
        with cols[i % 3]:
            if feature in int_cols:
                inputs[feature] = st.selectbox(f"{feature}", sorted(data[feature].unique()))
            else:
                inputs[feature] = st.number_input(
                    f"{feature}",
                    min_value=float(data[feature].min()),
                    max_value=float(data[feature].max()),
                    value=float(data[feature].mean())
                )
    
    # Make prediction
    if st.button("Predict"):
        try:
            input_data = pd.DataFrame([inputs])
            # Ensure input_data has correct dtypes
            input_data = input_data.astype({col: 'float64' for col in [c for c in float_cols if c != 'MEDV']}).astype({col: 'int64' for col in int_cols})
            prediction = model.predict(input_data)[0]
            st.success(f"Predicted House Price: ${prediction:.2f}K")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# Model Performance Section
elif page == "Model Performance":
    st.header("Model Performance")
    
    # Load training and testing data
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model performance metrics
    st.subheader("Random Forest Performance")
    y_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    st.write(f"Test RMSE: {test_rmse:.2f}")
    st.write(f"Test R²: {test_r2:.2f}")
    
    # Model comparison (from notebook)
    st.subheader("Model Comparison")
    st.write("**Linear Regression**:")
    st.write("Cross-Validation RMSE: 4.83")
    st.write("Test RMSE: 4.93")
    st.write("Test R²: 0.67")
    st.write("**Random Forest**:")
    st.write("Cross-Validation RMSE: 3.82")
    st.write("Test RMSE: 2.81")
    st.write("Test R²: 0.89")
    
    # Performance Plot
    st.subheader("Prediction vs Actual")
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Price', 'y': 'Predicted Price'},
                     title="Actual vs Predicted House Prices")
    fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                  line=dict(color="red", dash="dash"))
    st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Dataset: Boston Housing Prices | Model: Random Forest")