import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from PIL import Image

st.set_page_config(layout='wide')

with open('linear_model.pkl','rb') as f:
    lm2=pickle.load(f)
    
def load_feature_importance(file_path):
    return pd.read_excel(file_path)

final_fi=load_feature_importance('feature_importance.xlsx')

# Load and resize sidebar image (pic1)
image_sidebar = Image.open("pic1.png")
image_sidebar_resized = image_sidebar.resize((150, 120))  # Width=150, Height=120
st.sidebar.image(image_sidebar_resized, use_container_width=False)
st.sidebar.header("Vechicle Features")

def user_input():
    horsepower=st.sidebar.number_input(('Horsepower (NO)'),min_value=0,max_value=1000, step=1,value=300)
    torque=st.sidebar.number_input(("Torque (NO)"),min_value=0,max_value=1500,step=1,value=400)
    
    make=st.sidebar.selectbox("Make",['Aston Martin', 'Audi', 'BMW', 'Bentley', 'Ford', 'Mercedes-Benz', 'Nissan'])
    body_size=st.sidebar.selectbox('Body Size', ['Compact', 'Large', 'Midsize'])
    body_style=st.sidebar.selectbox('Body Style', ['Cargo Minivan', 'Cargo Van', 'Convertible', 'Convertible SUV', 'Coupe', 'Hatchback', 'Passenger Minivan', 'Passenger Van', 'Pickup Truck', 'SUV', 'Sedan', 'Wagon'])
    engine_aspiration = st.sidebar.selectbox('Engine Aspiration', ['Electric Motor', 'Naturally Aspirated', 'Supercharged', 'Turbocharged', 'Twin-Turbo', 'Twincharged'])
    drivetrain=st.sidebar.selectbox('Drivetrain', ['4WD', 'AWD', 'FWD', 'RWD'])
    transmission = st.sidebar.selectbox('Transmission', ['automatic', 'manual'])
    
    
    user_data={
        'Horsepower_no':horsepower,
        'Torque_no':torque,
        f'Make_{make}':1,
        f'Body Size_{body_size}':1,
        f'Engine Aspiration_{engine_aspiration}':1,
        f'Drivetrain_{drivetrain}':1,
        f'Transmission_{transmission}':1,
    }
    
    return user_data

# Load and resize banner image (pic2)
image_banner = Image.open("pic2.png")
image_banner_resized = image_banner.resize((900, 400))  # Width=900, Height=250 (landscape banner)
st.image(image_banner_resized, use_container_width=False)


st.markdown("<h1 style='text-align:center;'>Vechile Price Prediction App</h1>",unsafe_allow_html=True)

left_col,right_col=st.columns(2)

with left_col:
    st.header("Feature Importance")
    final_fi_sorted=final_fi.sort_values(by='Feature Importance Score',ascending=True)

    #Creating interactive bar chart with plotly
    fig=px.bar(final_fi_sorted,x='Feature Importance Score',
            y='Variable',orientation='h',
            title='Feature Importance',
            labels={"Feature Importance Score":"Importance","Variable":'Feature'},
            text='Feature Importance Score',
            color_discrete_sequence=['#48a3b4']
    )

    fig.update_layout(
        xaxis_title="Feature Importance Score",
        yaxis_title="Variable",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig,use_container_width=True)

with right_col:
    st.header("Predict Vehicle Price")
    
     # User inputs from sidebar
    user_data = user_input()

    # Transform the input into the required format
    def prepare_input(data, feature_list):
        input_data = {feature: data.get(feature, 0) for feature in feature_list}
        return np.array([list(input_data.values())])

    # Feature list (same order as used during model training)
    features = [
        'Horsepower_No', 'Torque_No', 'Make_Aston Martin', 'Make_Audi', 'Make_BMW', 'Make_Bentley',
        'Make_Ford', 'Make_Mercedes-Benz', 'Make_Nissan', 'Body Size_Compact', 'Body Size_Large',
        'Body Size_Midsize', 'Body Style_Cargo Minivan', 'Body Style_Cargo Van', 
        'Body Style_Convertible', 'Body Style_Convertible SUV', 'Body Style_Coupe', 
        'Body Style_Hatchback', 'Body Style_Passenger Minivan', 'Body Style_Passenger Van',
        'Body Style_Pickup Truck', 'Body Style_SUV', 'Body Style_Sedan', 'Body Style_Wagon',
        'Engine Aspiration_Electric Motor', 'Engine Aspiration_Naturally Aspirated',
        'Engine Aspiration_Supercharged', 'Engine Aspiration_Turbocharged',
        'Engine Aspiration_Twin-Turbo', 'Engine Aspiration_Twincharged', 
        'Drivetrain_4WD', 'Drivetrain_AWD', 'Drivetrain_FWD', 'Drivetrain_RWD', 
        'Transmission_automatic', 'Transmission_manual'
    ]

    # Predict button
    if st.button("Predict"):
        input_array = prepare_input(user_data, features)
        prediction = lm2.predict(input_array)
        st.subheader("Predicted Price")
        st.write(f"${prediction[0]:,.2f}")
 # Footer
st.markdown("---", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center;'>
        <strong>Made by <a href='https://www.linkedin.com/in/rajat-kumar-sahu1/' target='_blank'>Rajat Kumar Sahu</a></strong><br>
        📧 <a href='mailto:rajatks1997@gmail.com'>rajatks1997@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True
    
