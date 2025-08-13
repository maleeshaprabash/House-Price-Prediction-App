# Boston Housing Price Prediction Project

## Overview
This project develops a machine learning model to predict house prices in Boston using the Boston Housing Prices dataset. It includes a Jupyter Notebook for data exploration, preprocessing, and model training, and a Streamlit application for interactive data visualization and prediction. The project compares Linear Regression and Random Forest models, with the latter selected for its superior performance (Test RMSE: 2.81, Test R²: 0.89). A comprehensive report in LaTeX format documents the methodology, results, and challenges.

## Project Structure
- **data/Boston House Prices.csv**: The dataset used for training and evaluation.
- **model_training.ipynb**: Jupyter Notebook containing data exploration, visualization, preprocessing, model training, and evaluation.
- **app.py**: Streamlit application for interactive data exploration, visualization, and prediction.
- **model.pkl**: Saved Random Forest model for use in the Streamlit app.
- **House_Price_Prediction_Report.tex**: LaTeX report summarizing the project, including methodology, results, and challenges.
- **requirements.txt**: List of Python dependencies required to run the project.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure the following packages are included:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn
   - streamlit
   - plotly
   - joblib
   - pickle

4. **Verify Dataset and Model**:
   - Ensure `data/Boston House Prices.csv` is in the `data/` directory.
   - Ensure `model.pkl` is in the root directory.

## Usage
### Running the Jupyter Notebook
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `model_training.ipynb` and run the cells to explore the dataset, train models, and evaluate performance.

### Running the Streamlit Application
1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Access the app in your browser at `http://localhost:8501`.
3. Navigate through the sections:
   - **Data Exploration**: View dataset details and filter data interactively.
   - **Visualizations**: Explore correlations, price distributions, and scatter plots.
   - **Model Prediction**: Input feature values to predict house prices.
   - **Model Performance**: Review model metrics and actual vs. predicted plots.

### Compiling the LaTeX Report
1. Ensure a LaTeX distribution (e.g., TeX Live) is installed.
2. Compile `House_Price_Prediction_Report.tex`:
   ```bash
   pdflatex House_Price_Prediction_Report.tex
   ```
3. View the generated PDF for a detailed project summary.

## Dataset
The Boston Housing Prices dataset contains 506 observations with 14 features, including:
- **CRIM**: Per capita crime rate
- **RM**: Average number of rooms per dwelling
- **LSTAT**: Percentage of lower-status population
- **MEDV**: Median house value (target variable, in $1000s)

## Model
- **Random Forest**: Selected as the best model with Test RMSE of 2.81 and Test R² of 0.89.
- **Linear Regression**: Evaluated for comparison, with Test RMSE of 4.93 and Test R² of 0.67.

## Challenges
- **Data Path Issues**: Absolute paths in the notebook caused compatibility issues; resolved with relative paths in the Streamlit app.
- **Feature Scaling**: Not implemented for Linear Regression, potentially limiting its performance.
- **Outlier Handling**: Potential outliers (e.g., high CRIM values) were not addressed.
- **Visualization Compatibility**: Ensuring Plotly and Matplotlib work seamlessly in Streamlit required careful configuration.
- **Model Interpretability**: Random Forest's complexity makes it less interpretable than Linear Regression.

## Future Improvements
- Implement feature scaling for Linear Regression.
- Add outlier detection and treatment.
- Include feature importance visualizations for better interpretability.
- Deploy the Streamlit app on a cloud platform for broader access.