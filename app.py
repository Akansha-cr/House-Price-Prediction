import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load data and impute missing values
@st.cache(allow_output_mutation=True)
def load_data(file):
    data = pd.read_csv(file)
    imputer = SimpleImputer(strategy='median')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data

# Build model
@st.cache(allow_output_mutation=True)
def build_model(data):
    model = LinearRegression()
    model.fit(data[['latitude', 'longitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','price']], data['price'])
    return model

# Define UI elements
st.title("House Price Prediction System")
st.header("Upload Data")
file = st.file_uploader("Choose a CSV file", type="csv")

# Load data and build model
if file is not None:
    data = load_data(file)
    model = build_model(data)
    st.success("Data uploaded and model built successfully!")

    # Show uploaded data and predicted prices
    st.subheader("Predicted House Prices")
    predictions = model.predict(data[['latitude', 'longitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','price']])
    data['predicted_price'] = predictions
    st.write(data)

else:
    st.warning("Please upload a CSV file.")
