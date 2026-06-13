
import pandas as pd
import numpy as np
import streamlit as st
import joblib

st.set_page_config(layout='wide', page_title='NYC Airbnb Price Deployment')

html_title = """<h1 style="color:white;text-align:center;"> NYC Airbnb Price Prediction </h1>"""
st.markdown(html_title, unsafe_allow_html=True)

st.image('/Users/abdullahshaker/Downloads/New_York_City_.png')

df = pd.read_csv('cleaned_df.csv')
st.dataframe(df.head())

col1, col2 = st.columns(2)

with col1:
    neighbourhood_group = st.selectbox('Borough (Neighbourhood Group)', df['neighbourhood_group'].unique())
    room_type = st.selectbox('Room Type', df['room_type'].unique())
    minimum_nights = st.slider('Minimum Nights', min_value=1, max_value=30, step=1)

with col2:
    availability_365 = st.slider('Availability 365 (Days)', min_value=0, max_value=365, step=1)
    number_of_reviews = st.text_input('Number of Reviews', '10')

    keywords = st.multiselect(
        'Listing Special Keywords (Optional)', 
        ['luxury', 'view', 'penthouse', 'spacious', 'central park', 'loft', 'studio', 'modern', 'vintage']
    )

ml_model = joblib.load('airbnb_nyc_model.pkl')

if st.button('Predict Apartment Price', use_container_width=True):

    # 1. تحويل الكلمات السحرية لـ 1 و 0
    has_luxury = 1 if 'luxury' in keywords else 0
    has_view = 1 if 'view' in keywords else 0
    has_penthouse = 1 if 'penthouse' in keywords else 0
    has_spacious = 1 if 'spacious' in keywords else 0
    has_central_park = 1 if 'central park' in keywords else 0
    has_loft = 1 if 'loft' in keywords else 0
    has_studio = 1 if 'studio' in keywords else 0
    has_modern = 1 if 'modern' in keywords else 0
    has_vintage = 1 if 'vintage' in keywords else 0

    # 2. بناء الداتا فريم بنفس الترتيب اللي طﻠعلك بالظبططططط
    new_data = pd.DataFrame({
        'neighbourhood_group': [neighbourhood_group],
        'neighbourhood': ['Williamsburg'],  # قيمة افتراضية لأشهر حي عشان الموديل يشتغل
        'room_type': [room_type],
        'minimum_nights': [minimum_nights],
        'number_of_reviews': [int(number_of_reviews)],
        'reviews_per_month': [1.5],         # قيمة افتراضية
        'calculated_host_listings_count': [1], # قيمة افتراضية
        'availability_365': [availability_365],
        'dist_to_times_square': [5.0],      # مسافة افتراضية بالكيلومتر
        'dist_to_central_park': [6.0],      # مسافة افتراضية بالكيلومتر
        'dist_to_jfk': [20.0],              # مسافة افتراضية بالكيلومتر
        'location_cluster': [0],            # الكلاستر الافتراضي
        'has_luxury': [has_luxury],
        'has_view': [has_view],
        'has_penthouse': [has_penthouse],
        'has_spacious': [has_spacious],
        'has_central_park': [has_central_park],
        'has_loft': [has_loft],
        'has_studio': [has_studio],
        'has_modern': [has_modern],
        'has_vintage': [has_vintage]
    })

    try:
        prediction_log = ml_model.predict(new_data)
        prediction_real = np.expm1(prediction_log)[0]

        st.success(f'Apartment Price per Night: **${round(prediction_real, 2)}**')
    except Exception as e:
        st.error(f"Error Details: {e}")
