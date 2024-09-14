import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Injecting CSS for custom background color
st.markdown(
    """
    <style>
    .main {
        background-color: #D2E0FB
    }
    </style>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');

    .stMarkdown h2, .stMarkdown h3, .stMarkdown p {
        color: black; /* Main text color */
        font-family: "Quicksand", system-ui;
    }
    /* Apply custom font to the title */

    /* Header style */
    header {
        background-color: #227B94;
        padding: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    header h1 {
        font-family: 'Libre Baskerville', sans-serif;
        font-size: 2.3em;
        color: white;
        text-shadow: 1px 1px 2px #000000;
        margin: 0;
    }
    .stButton > button {
        background-color: #7695FF;
        border-radius: 12px;
        padding: 10px 24px;
        color: white;
        font-size: 16px;
        font-family: 'Quicksand', sans-serif;
        font-weight: bold;
    }

    /* Hover effect for buttons */
    .stButton > button:hover {
        background-color: #125B9A;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(f"""
<header>
    <h1>Critical Ratio</h1>
</header>
""", unsafe_allow_html=True)

# Upload dataset
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# Function to load and display dataset
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    else:
        st.write("Please upload a CSV file to proceed.")
        return None

data = load_data(uploaded_file)

# Display dataset
if data is not None:
    st.subheader('Dataset Overview')
    st.write(data.head())

    # Select features and target
    st.sidebar.subheader("Model Parameters")
    features = st.sidebar.multiselect("Select the feature(s) for regression:", data.columns[:-1])
    target = st.sidebar.selectbox("Select the target variable:", data.columns)

    if features and target:
        # Splitting the dataset
        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions and model performance
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display model performance
        st.subheader("Model Performance")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R-squared (R²): {r2}")

        # Take input for prediction
        st.sidebar.subheader("Make Predictions")
        input_data = []
        for feature in features:
            val = st.sidebar.number_input(f"Input value for {feature}:", value=float(X[feature].mean()))
            input_data.append(val)

        if st.sidebar.button("Predict"):
            input_array = np.array(input_data).reshape(1, -1)
            prediction = model.predict(input_array)
            st.subheader("Prediction")
            st.write(f"Predicted value for {target}: {prediction[0]}")

if st.button("Home"):
    st.switch_page("Home.py")
