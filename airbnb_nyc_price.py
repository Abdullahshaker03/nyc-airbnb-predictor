
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px

st.set_page_config(layout='wide', page_title='NYC Airbnb Price Deployment')

html_title = """<h1 style="color:white;text-align:center;"> NYC Airbnb Price Prediction </h1>"""
st.markdown(html_title, unsafe_allow_html=True)

st.image('New_York_City_.png')

df = pd.read_csv('cleaned_df.csv')

tab1, tab2 = st.tabs(["Data Analysis", "Price Prediction"])

with tab1:
    st.dataframe(df.head())

    room_counts = df['room_type'].value_counts().reset_index()
    room_counts.columns = ['Room Type', 'Count']
    fig1 = px.pie(data_frame=room_counts, names='Room Type', values='Count', title='Room Type Distribution in NYC', color_discrete_sequence=['#4A90E2', '#50E3C2', '#F5A623'])
    st.plotly_chart(fig1, use_container_width=True)

    borough_prices = df.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=True).reset_index()
    borough_prices['price'] = borough_prices['price'].round(1)
    fig2 = px.bar(data_frame=borough_prices, x='neighbourhood_group', y='price', title='Average Price per Night by Borough', text_auto=True, color='neighbourhood_group', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig2, use_container_width=True)

    hist_df = df[df['price'] < 1000]
    fig3 = px.histogram(data_frame=hist_df, x='room_type', y='price', title='Price Distribution by Room Type', text_auto=True, color='room_type', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.markdown("### 📍 Location & Property Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        neighbourhood_group = st.selectbox('Borough', df['neighbourhood_group'].unique())
        room_type = st.selectbox('Room Type', df['room_type'].unique())

    with col2:
        filtered_neighbourhoods = df[df['neighbourhood_group'] == neighbourhood_group]['neighbourhood'].unique()
        neighbourhood = st.selectbox('Neighbourhood', filtered_neighbourhoods)
        location_cluster = st.selectbox('Location Cluster', sorted(df['location_cluster'].unique()))

    with col3:
        keywords = st.multiselect(
            'Special Keywords', 
            ['luxury', 'view', 'penthouse', 'spacious', 'central park', 'loft', 'studio', 'modern', 'vintage']
        )

    st.markdown("### 📏 Distances to Landmarks (km)")
    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        dist_to_times_square = st.number_input('Distance to Times Square', min_value=0.0, value=5.0, step=0.5)
    with col_d2:
        dist_to_central_park = st.number_input('Distance to Central Park', min_value=0.0, value=6.0, step=0.5)
    with col_d3:
        dist_to_jfk = st.number_input('Distance to JFK Airport', min_value=0.0, value=20.0, step=0.5)

    st.markdown("### 📅 Booking Details")
    col_b1, col_b2, col_b3 = st.columns(3)

    with col_b1:
        minimum_nights = st.slider('Minimum Nights', min_value=1, max_value=30, step=1)
    with col_b2:
        availability_365 = st.slider('Availability 365', min_value=0, max_value=365, step=1)
    with col_b3:
        number_of_reviews = st.number_input('Total Reviews', min_value=0, value=10, step=1)

    ml_model = joblib.load('airbnb_nyc_model.pkl')

    if st.button('Predict Apartment Price', use_container_width=True):

        has_luxury = 1 if 'luxury' in keywords else 0
        has_view = 1 if 'view' in keywords else 0
        has_penthouse = 1 if 'penthouse' in keywords else 0
        has_spacious = 1 if 'spacious' in keywords else 0
        has_central_park = 1 if 'central park' in keywords else 0
        has_loft = 1 if 'loft' in keywords else 0
        has_studio = 1 if 'studio' in keywords else 0
        has_modern = 1 if 'modern' in keywords else 0
        has_vintage = 1 if 'vintage' in keywords else 0

        new_data = pd.DataFrame({
            'neighbourhood_group': [neighbourhood_group],
            'neighbourhood': [neighbourhood],  
            'room_type': [room_type],
            'minimum_nights': [minimum_nights],
            'number_of_reviews': [int(number_of_reviews)],
            'reviews_per_month': [1.5],        
            'calculated_host_listings_count': [1], 
            'availability_365': [availability_365],
            'dist_to_times_square': [float(dist_to_times_square)],      
            'dist_to_central_park': [float(dist_to_central_park)],      
            'dist_to_jfk': [float(dist_to_jfk)],             
            'location_cluster': [int(location_cluster)],     
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
