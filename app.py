import streamlit as st
import pandas as pd
import joblib
import numpy as np


@st.cache_resource
def load_model():
    return joblib.load("airbnb_price_model (1).pkl")


model = load_model()

# Page config
st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide"
)


st.title("🏠 Airbnb Price Prediction")
st.write(
    "Predict Airbnb listing price using XGBoost Regression"
)


st.divider()


# User Inputs

# User Inputs

col1, col2 = st.columns(2)


with col1:

    host_id = st.number_input(
        "Host ID",
        value=12345
    )

    neighbourhood_group = st.selectbox(
        "Neighbourhood Group",
        [
            "Manhattan",
            "Brooklyn",
            "Queens",
            "Bronx",
            "Staten Island"
        ]
    )

    neighbourhood = st.text_input(
        "Neighbourhood",
        "Midtown"
    )

    latitude = st.number_input(
        "Latitude",
        value=40.75
    )

    longitude = st.number_input(
        "Longitude",
        value=-73.98
    )

    room_type = st.selectbox(
        "Room Type",
        [
            "Entire home/apt",
            "Private room",
            "Shared room"
        ]
    )


with col2:

    minimum_nights = st.number_input(
        "Minimum Nights",
        min_value=1,
        value=3
    )

    number_of_reviews = st.number_input(
        "Number of Reviews",
        min_value=0,
        value=20
    )

    reviews_per_month = st.number_input(
        "Reviews per Month",
        min_value=0.0,
        value=2.0
    )

    host_listings = st.number_input(
        "Calculated Host Listings Count",
        min_value=1,
        value=1
    )

    availability_365 = st.number_input(
        "Availability (365 Days)",
        min_value=0,
        max_value=365,
        value=200
    )


# Feature Engineering (same as training)

host_experience = host_listings


review_intensity = (
    number_of_reviews /
    (availability_365 + 1)
)


if minimum_nights <= 3:

    stay_category = "Short Stay"

elif minimum_nights <= 30:

    stay_category = "Medium Stay"

else:

    stay_category = "Long Stay"



# Final dataframe for model

input_data = pd.DataFrame({

    "host_id": [host_id],

    "neighbourhood_group": [neighbourhood_group],

    "neighbourhood": [neighbourhood],

    "latitude": [latitude],

    "longitude": [longitude],

    "room_type": [room_type],

    "minimum_nights": [minimum_nights],

    "number_of_reviews": [number_of_reviews],

    "reviews_per_month": [reviews_per_month],

    "calculated_host_listings_count": [host_listings],

    "availability_365": [availability_365],

    "host_experience": [host_experience],

    "review_intensity": [review_intensity],

    "stay_category": [stay_category]

})

st.divider()

if st.button("Predict Price"):

    required_columns = list(
        model.named_steps["preprocessor"].feature_names_in_
    )


    input_data = input_data.reindex(
        columns=required_columns,
        fill_value=0
    )


    prediction = model.predict(
        input_data
    )


    price = round(
        prediction[0],
        2
    )


    st.success(
        f"Estimated Airbnb Price: ${price} per night"
    )
