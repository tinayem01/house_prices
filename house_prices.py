import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

model = joblib.load('rf_house_prices_v1.pkl')

st.set_page_config(
    page_title = 'House Price Prediction',
    layout='centered',
    initial_sidebar_state='expanded'
)

image = Image.open('house.jpg')
st.image(image,use_column_width=True)



def predict_price(size, beds, bath, location, names, title_deeds, borehole,garage, swimming_pool):
    values = np.array([[size, beds, bath, location, names, title_deeds, borehole, garage, swimming_pool]])
    prediction = model.predict(pd.DataFrame(values,columns = ['size', 'beds', 'bath', 'location', 'names', 'title_deeds', 'borehole','garage', 'swimming_pool']))
    return prediction

def main():

    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">House Pricing ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    size = st.number_input('Area in Square metres',value = 1000)
    beds = st.number_input('Number of Bedrooms',value=3)
    bath = st.number_input('Number of Bathrooms',value=1)
    location = st.text_input('Choose location among west,central,north,east,high density,south','west')
    names = st.text_input('Area where house is located eg Avondale','Avondale')
    title_deeds = st.text_input('Type of Title deeds eg Full Title Deeds,Cession,Sectional Title Deeds,Cession of Trust or missing if you dont know','Full Title Deeds')
    borehole = st.text_input('Is there is a borehole or not?. Valid options are Yes or no','no')
    garage = st.text_input('Is there a garage? Enter either no or Yes','no')
    swimming_pool = st.text_input('Is there a swimming pool? Enter either Yes or no','no')


    if st.button('Predict Price',):
        prediction = predict_price(size, beds, bath, location, names, title_deeds, borehole,garage, swimming_pool)
        st.success(f'The price is ${int(np.expm1(prediction[0]))}')


if __name__ == '__main__':
    main()