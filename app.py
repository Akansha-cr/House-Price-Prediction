import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv('housing.csv')

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Define UI elements
st.title("House Price Prediction System")
st.header("Enter House Details or Upload a CSV file")

# Create a file uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Load data if a file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Build model
    model = LinearRegression()
    model.fit(df[['latitude', 'longitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']], df['price'])

    # Predict house prices
    prices = model.predict(df[['latitude', 'longitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']])
    
    # Add predicted prices to the DataFrame
    df['predicted_price'] = prices

    # Show predicted prices
    st.subheader("Predicted House Prices")
    st.write(df)
else:
    # Get user input
    total_bedrooms = st.number_input("No. of Bedrooms")
    total_rooms = st.number_input("No. of rooms")
    latitude = st.number_input("Latitude")
    longitude = st.number_input("Longitude")
    housing_median_age = st.number_input("housing_median_age")
    population = st.number_input("population")
    households = st.number_input("households")
    median_income = st.number_input("median income")

    # Build model
    model = LinearRegression()
    model.fit([[latitude, longitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income]], [0])

    # Predict house price
    price = model.predict([[latitude, longitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income]])

    # Show predicted price
    st.subheader("Predicted House Price")
    st.write(price[0])
