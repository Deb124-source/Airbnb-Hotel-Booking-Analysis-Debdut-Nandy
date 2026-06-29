import streamlit as st
import pandas as pd
import joblib
import os


# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():

    model_path = os.path.join(
        os.path.dirname(__file__),
        "airbnb_price_model.pkl"
    )

    return joblib.load(model_path)


model = load_model()



# -----------------------------
# Title
# -----------------------------

st.title("🏠 Airbnb Price Prediction")

st.write(
    "Enter listing details to estimate price"
)



# -----------------------------
# Inputs
# -----------------------------


col1, col2 = st.columns(2)



with col1:


    host_id = st.number_input(
        "Host ID",
        value=12345
    )


    neighbourhood_group = st.text_input(
        "Neighbourhood Group",
        "Manhattan"
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


    room_type = st.text_input(
        "Room Type",
        "Entire home/apt"
    )



with col2:


    minimum_nights = st.number_input(
        "Minimum Nights",
        value=3
    )


    number_of_reviews = st.number_input(
        "Number of Reviews",
        value=20
    )


    reviews_per_month = st.number_input(
        "Reviews Per Month",
        value=2.0
    )


    calculated_host_listings_count = st.number_input(
        "Host Listings Count",
        value=1
    )


    availability_365 = st.number_input(
        "Availability 365",
        value=200
    )



# -----------------------------
# Feature Engineering
# -----------------------------


host_experience = calculated_host_listings_count


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



# -----------------------------
# Input DataFrame
# -----------------------------


input_data = pd.DataFrame({

    "host_id":[host_id],

    "neighbourhood_group":[
        neighbourhood_group
    ],

    "neighbourhood":[
        neighbourhood
    ],

    "latitude":[
        latitude
    ],

    "longitude":[
        longitude
    ],

    "room_type":[
        room_type
    ],

    "minimum_nights":[
        minimum_nights
    ],

    "number_of_reviews":[
        number_of_reviews
    ],

    "reviews_per_month":[
        reviews_per_month
    ],

    "calculated_host_listings_count":[
        calculated_host_listings_count
    ],

    "availability_365":[
        availability_365
    ],

    "host_experience":[
        host_experience
    ],

    "review_intensity":[
        review_intensity
    ],

    "stay_category":[
        stay_category
    ]

})



# -----------------------------
# Prediction
# -----------------------------


if st.button("Predict Price"):


    # Match training columns

    required_columns = (
        model
        .named_steps["preprocessor"]
        .feature_names_in_
    )


    input_data = input_data[
        required_columns
    ]


    # Force same datatype behavior

    for col in input_data.columns:

        if input_data[col].dtype == "object":

            input_data[col] = (
                input_data[col]
                .astype(str)
            )


    prediction = model.predict(
        input_data
    )


    st.success(
        f"Predicted Airbnb Price: ${round(prediction[0],2)}"
    )
