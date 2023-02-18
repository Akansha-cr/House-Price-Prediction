import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Set the title and subtitle of the web app
st.title('House Price Prediction System')
st.write('Upload a CSV file to generate house price predictions:')

# Create a file upload control for the CSV file
uploaded_file = st.file_uploader("Choose a file", type="csv")

# If the file is uploaded and not empty
if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Display the raw data
    st.write('Raw Data:')
    st.write(df)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    X = df.drop('price', axis=1)
    y = df['price']
    model.fit(X, y)

    # Get the input data from the user
    st.write('Enter the values for the following features:')
    inputs = {}
    for feature in X.columns:
        value = st.number_input(feature)
        inputs[feature] = value
    input_df = pd.DataFrame([inputs])

    # Make a prediction and display the result
    prediction = model.predict(input_df)
    st.write('Prediction:')
    st.write(prediction[0])
