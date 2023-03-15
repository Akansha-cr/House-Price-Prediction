import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('housing.csv')

# Build model
model = LinearRegression()
#model.fit(df[['latitude', 'longitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','price']], df['price'])

# Define UI elements
st.title("House Price Prediction System")
st.header("Enter House Details")
total_bedrooms = st.number_input("No. of Bedrooms")
total_rooms = st.number_input("No. of rooms")
latitude = st.number_input("Latitude")
longitude = st.number_input("Longitude")
housing_median_age = st.number_input("housing_median_age")
population = st.number_input("population")
households = st.number_input("households")
median_income = st.number_input("median income")

# Add file uploader
csv_file = st.file_uploader("Upload CSV", type=['csv'])

if csv_file is not None:
    # Read uploaded CSV file
    uploaded_data = pd.read_csv(csv_file)

    # Generate predictions for uploaded data
    predictions = model.predict(uploaded_data[['latitude', 'longitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'price']])

    # Show predicted prices
    st.subheader("Predicted House Prices")
    st.write(predictions)

# Predict house price for entered data
#price = model.predict([[housing_median_age,total_rooms,total_bedrooms,population,households,median_income,price,latitude,longitude]])

# Show predicted price for entered data
#st.subheader("Predicted House Price")
#st.write(price[0])
