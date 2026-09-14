import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Custom UI Styling
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #f8f9fc 0%,
        #eef2f7 50%,
        #ffffff 100%
    );
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}


/* --------------------------------------------------
   Hero
   -------------------------------------------------- */

.hero {
    background: linear-gradient(
        135deg,
        #ff5a5f 0%,
        #ff385c 50%,
        #d90429 100%
    );

    padding: 35px 40px;
    border-radius: 20px;
    margin-bottom: 30px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.12);
}

.hero-title {
    color: white;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: rgba(255,255,255,0.92);
    font-size: 18px;
    line-height: 1.5;
}


/* --------------------------------------------------
   Section headings
   -------------------------------------------------- */

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}


/* --------------------------------------------------
   Information cards
   -------------------------------------------------- */

.info-card {
    background: white;
    padding: 22px;
    border-radius: 15px;

    border: 1px solid #e6e9ef;

    box-shadow:
        0 5px 18px rgba(0,0,0,0.06);

    min-height: 130px;
}

.info-title {
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 8px;
}

.info-text {
    color: #5f6368;
    font-size: 14px;
    line-height: 1.6;
}


/* --------------------------------------------------
   Prediction card
   -------------------------------------------------- */

.prediction-card {
    background: white;

    border-radius: 20px;
    padding: 30px;

    text-align: center;

    margin-top: 25px;
    margin-bottom: 25px;

    border: 2px solid #ff385c;

    box-shadow:
        0 10px 30px rgba(255,56,92,0.15);
}

.prediction-label {
    color: #666;
    font-size: 16px;
    margin-bottom: 5px;
}

.prediction-price {
    color: #ff385c;
    font-size: 42px;
    font-weight: 800;
}

.prediction-note {
    color: #777;
    font-size: 13px;
    margin-top: 8px;
}


/* --------------------------------------------------
   Buttons
   -------------------------------------------------- */

.stButton > button {
    width: 100%;
    border-radius: 12px;

    height: 3.2em;

    font-size: 17px;
    font-weight: 700;
}


/* --------------------------------------------------
   Footer
   -------------------------------------------------- */

.footer {
    text-align: center;
    color: #777;

    font-size: 13px;

    margin-top: 40px;
    padding-top: 20px;
    padding-bottom: 10px;

    border-top: 1px solid #e6e9ef;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load("airbnb_price_model.pkl")


# --------------------------------------------------
# Hero section
# --------------------------------------------------

st.title("🏠 Airbnb Price Predictor")

st.caption(
    "Estimate the expected nightly price of a New York City "
    "Airbnb listing using a trained machine learning model."
)

# --------------------------------------------------
# Location Information
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📍 Location</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

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

with col2:

    neighbourhood = st.selectbox(
        "Neighbourhood",
        [
            'Allerton',
            'Arden Heights',
            'Arrochar',
            'Arverne',
            'Astoria',
            'Bath Beach',
            'Battery Park City',
            'Bay Ridge',
            'Bay Terrace',
            'Bay Terrace, Staten Island',
            'Baychester',
            'Bayside',
            'Bayswater',
            'Bedford-Stuyvesant',
            'Belle Harbor',
            'Bellerose',
            'Belmont',
            'Bensonhurst',
            'Bergen Beach',
            'Boerum Hill',
            'Borough Park',
            'Breezy Point',
            'Briarwood',
            'Brighton Beach',
            'Bronxdale',
            'Brooklyn Heights',
            'Brownsville',
            "Bull's Head",
            'Bushwick',
            'Cambria Heights',
            'Canarsie',
            'Carroll Gardens',
            'Castle Hill',
            'Castleton Corners',
            'Chelsea',
            'Chinatown',
            'City Island',
            'Civic Center',
            'Claremont Village',
            'Clason Point',
            'Clifton',
            'Clinton Hill',
            'Co-op City',
            'Cobble Hill',
            'College Point',
            'Columbia St',
            'Concord',
            'Concourse',
            'Concourse Village',
            'Coney Island',
            'Corona',
            'Crown Heights',
            'Cypress Hills',
            'DUMBO',
            'Ditmars Steinway',
            'Dongan Hills',
            'Douglaston',
            'Downtown Brooklyn',
            'Dyker Heights',
            'East Elmhurst',
            'East Flatbush',
            'East Harlem',
            'East Morrisania',
            'East New York',
            'East Village',
            'Eastchester',
            'Edenwald',
            'Edgemere',
            'Elmhurst',
            'Eltingville',
            'Emerson Hill',
            'Far Rockaway',
            'Fieldston',
            'Financial District',
            'Flatbush',
            'Flatiron District',
            'Flatlands',
            'Flushing',
            'Fordham',
            'Forest Hills',
            'Fort Greene',
            'Fort Hamilton',
            'Fort Wadsworth',
            'Fresh Meadows',
            'Glendale',
            'Gowanus',
            'Gramercy',
            'Graniteville',
            'Grant City',
            'Gravesend',
            'Great Kills',
            'Greenpoint',
            'Greenwich Village',
            'Grymes Hill',
            'Harlem',
            "Hell's Kitchen",
            'Highbridge',
            'Hollis',
            'Holliswood',
            'Howard Beach',
            'Howland Hook',
            'Huguenot',
            'Hunts Point',
            'Inwood',
            'Jackson Heights',
            'Jamaica',
            'Jamaica Estates',
            'Jamaica Hills',
            'Kensington',
            'Kew Gardens',
            'Kew Gardens Hills',
            'Kingsbridge',
            'Kips Bay',
            'Laurelton',
            'Lighthouse Hill',
            'Little Italy',
            'Little Neck',
            'Long Island City',
            'Longwood',
            'Lower East Side',
            'Manhattan Beach',
            'Marble Hill',
            'Mariners Harbor',
            'Maspeth',
            'Melrose',
            'Middle Village',
            'Midland Beach',
            'Midtown',
            'Midwood',
            'Mill Basin',
            'Morningside Heights',
            'Morris Heights',
            'Morris Park',
            'Morrisania',
            'Mott Haven',
            'Mount Eden',
            'Mount Hope',
            'Murray Hill',
            'Navy Yard',
            'Neponsit',
            'New Brighton',
            'New Dorp',
            'New Dorp Beach',
            'New Springville',
            'NoHo',
            'Nolita',
            'North Riverdale',
            'Norwood',
            'Oakwood',
            'Olinville',
            'Ozone Park',
            'Park Slope',
            'Parkchester',
            'Pelham Bay',
            'Pelham Gardens',
            'Port Morris',
            'Port Richmond',
            "Prince's Bay",
            'Prospect Heights',
            'Prospect-Lefferts Gardens',
            'Queens Village',
            'Randall Manor',
            'Red Hook',
            'Rego Park',
            'Richmond Hill',
            'Richmondtown',
            'Ridgewood',
            'Riverdale',
            'Rockaway Beach',
            'Roosevelt Island',
            'Rosebank',
            'Rossville',
            'Schuylerville',
            'Sea Gate',
            'Sheepshead Bay',
            'Shore Acres',
            'Silver Lake',
            'SoHo',
            'Soundview',
            'South Beach',
            'South Ozone Park',
            'South Slope',
            'Springfield Gardens',
            'Spuyten Duyvil',
            'St. Albans',
            'St. George',
            'Stapleton',
            'Stuyvesant Town',
            'Sunnyside',
            'Sunset Park',
            'Theater District',
            'Throgs Neck',
            'Todt Hill',
            'Tompkinsville',
            'Tottenville',
            'Tremont',
            'Tribeca',
            'Two Bridges',
            'Unionport',
            'University Heights',
            'Upper East Side',
            'Upper West Side',
            'Van Nest',
            'Vinegar Hill',
            'Wakefield',
            'Washington Heights',
            'West Brighton',
            'West Farms',
            'West Village',
            'Westchester Square',
            'Westerleigh',
            'Whitestone',
            'Williamsbridge',
            'Williamsburg',
            'Willowbrook',
            'Windsor Terrace',
            'Woodhaven',
            'Woodlawn',
            'Woodrow',
            'Woodside'
        ]
    )


col1, col2 = st.columns(2)

with col1:

    latitude = st.number_input(
        "Latitude",
        min_value=40.49,
        max_value=40.92,
        value=40.72,
        format="%.5f"
    )

with col2:

    longitude = st.number_input(
        "Longitude",
        min_value=-74.25,
        max_value=-73.70,
        value=-73.99,
        format="%.5f"
    )


# --------------------------------------------------
# Property Information
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🏡 Property Details</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)

with col1:

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
        max_value=45,
        value=3
    )

with col3:

    availability_365 = st.number_input(
        "Availability (365 days)",
        min_value=0,
        max_value=365,
        value=100
    )


# --------------------------------------------------
# Listing Activity
# --------------------------------------------------

st.markdown(
    '<div class="section-title">⭐ Listing Activity</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    number_of_reviews = st.number_input(
        "Number of Reviews",
        min_value=0,
        max_value=629,
        value=10
    )

with col2:

    reviews_per_month = st.number_input(
        "Reviews per Month",
        min_value=0.0,
        max_value=58.5,
        value=1.0
    )

with col3:

    calculated_host_listings_count = st.number_input(
        "Host Listings Count",
        min_value=1,
        max_value=327,
        value=1
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔮 Predict Nightly Price",
    use_container_width=True
):

    # Apply the same transformations used during training

    minimum_nights_capped = min(
        minimum_nights,
        45
    )

    minimum_nights_log = np.log1p(
        minimum_nights_capped
    )

    number_of_reviews_log = np.log1p(
        number_of_reviews
    )

    reviews_per_month_log = np.log1p(
        reviews_per_month
    )

    calculated_host_listings_count_log = np.log1p(
        calculated_host_listings_count
    )

    # Create input dataframe

    input_data = pd.DataFrame({
        "latitude": [latitude],
        "longitude": [longitude],
        "minimum_nights_log": [minimum_nights_log],
        "number_of_reviews_log": [number_of_reviews_log],
        "reviews_per_month_log": [reviews_per_month_log],
        "calculated_host_listings_count_log": [
            calculated_host_listings_count_log
        ],
        "availability_365": [availability_365],
        "neighbourhood_group": [neighbourhood_group],
        "neighbourhood": [neighbourhood],
        "room_type": [room_type]
    })

    # Predict log(price)

    predicted_log_price = model.predict(
        input_data
    )[0]

    # Convert back to original price

    predicted_price = np.expm1(
        predicted_log_price
    )

    
    # Display prediction

st.success(
    f"Estimated Nightly Price: ${predicted_price:,.2f}"
)

st.caption(
    "Estimated using the trained Random Forest model"
)

# --------------------------------------------------
# How It Works
# --------------------------------------------------

st.markdown(
    '<div class="section-title">💡 How It Works</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        """
        **1️⃣ Enter Listing Details**

        Provide the location, room type, availability,
        reviews and other listing characteristics.
        """
    )

with col2:
    st.info(
        """
        **2️⃣ Machine Learning**

        The trained Random Forest regression model
        processes the submitted listing information.
        """
    )

with col3:
    st.info(
        """
        **3️⃣ Get an Estimate**

        The model returns an estimated nightly Airbnb
        price based on the supplied characteristics.
        """
    )

# --------------------------------------------------
# Model Information
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📊 About the Model</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", "Random Forest")

with col2:
    st.metric("MAE", "$55.28")

with col3:
    st.metric("RMSE", "$180.17")

with col4:
    st.metric("R²", "0.1892")


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.info(
    "The predicted price is an estimate based on historical "
    "Airbnb listing data. Actual market prices may differ."
)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("""
<div class="footer">

    Airbnb Price Predictor • Machine Learning Project

</div>
""", unsafe_allow_html=True)