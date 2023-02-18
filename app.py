import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('housing.csv')

# Build model
model = LinearRegression()
model.fit(df[['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long']], df['price'])

# Define UI elements
st.title("House Price Prediction System")
st.header("Enter House Details")
sqft_living = st.number_input("Living Area (in sqft)")
bedrooms = st.number_input("No. of Bedrooms")
bathrooms = st.number_input("No. of Bathrooms")
floors = st.number_input("No. of Floors")
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.selectbox("View", [0, 1, 2, 3, 4])
condition = st.selectbox("Condition", [1, 2, 3, 4, 5])
grade = st.selectbox("Grade", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
sqft_above = st.number_input("Area Above Ground (in sqft)")
sqft_basement = st.number_input("Basement Area (in sqft)")
yr_built = st.number_input("Year Built")
yr_renovated = st.number_input("Year Renovated")
zipcode = st.number_input("Zipcode")
lat = st.number_input("Latitude")
long = st.number_input("Longitude")

# Predict house price
price = model.predict([[sqft_living, bedrooms, bathrooms, floors, waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, lat, long]])

# Show predicted price
st.subheader("Predicted House Price")
st.write(price[0])
