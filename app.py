import streamlit as st
import pandas as pd
import joblib
import os


# -----------------------------
# Page Setup
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
        "airbnb_price_model (1).pkl"
    )

    return joblib.load(model_path)



model = load_model()



# -----------------------------
# Title
# -----------------------------

st.title("🏠 Airbnb Price Prediction")

st.write(
    "Predict Airbnb listing price using your trained ML pipeline"
)


st.divider()



# -----------------------------
# Inputs
# -----------------------------

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


    neighbourhood = st.selectbox(
        "Neighbourhood",
        [
            "Williamsburg",
            "Harlem",
            "Bushwick",
            "Midtown",
            "Upper West Side"
        ]
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
        "Reviews Per Month",
        min_value=0.0,
        value=2.0
    )


    calculated_host_listings_count = st.number_input(
        "Host Listings Count",
        min_value=1,
        value=1
    )


    availability_365 = st.number_input(
        "Availability 365",
        min_value=0,
        max_value=365,
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
# Create Dataframe
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


    # Exact training columns

    required_columns = (
        model
        .named_steps["preprocessor"]
        .feature_names_in_
    )


    input_data = input_data[
        required_columns
    ]



    # Datatype correction


    numeric_columns = [

        "host_id",
        "latitude",
        "longitude",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "host_experience",
        "review_intensity"

    ]



    for col in numeric_columns:

        input_data[col] = pd.to_numeric(
            input_data[col]
        )



    categorical_columns = [

        "neighbourhood_group",
        "neighbourhood",
        "room_type",
        "stay_category"

    ]



    for col in categorical_columns:

        input_data[col] = (
            input_data[col]
            .astype(str)
        )

    st.write("INPUT DATA")
    st.write(input_data)


    preprocessor = model.named_steps["preprocessor"]

    for name, transformer, columns in preprocessor.transformers_:

        st.write("Transformer:", name)
        st.write("Columns:", columns)

        if hasattr(transformer, "named_steps"):

             for step_name, step in transformer.named_steps.items():

             st.write("Step:", step_name)

             if hasattr(step, "categories_"):

                 st.write(
                    "Categories:",
                    step.categories_
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
