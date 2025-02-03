import streamlit as st
import pandas as pd
import pickle
#import xgboost

# Set page config
st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧️",
    layout="centered"
)

# Initialize session state variables
if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 25.0
if 'dew_point' not in st.session_state:
    st.session_state['dew_point'] = 20.0
if 'humidity' not in st.session_state:
    st.session_state['humidity'] = 50.0
if 'pressure' not in st.session_state:
    st.session_state['pressure'] = 1013.0
if 'visibility' not in st.session_state:
    st.session_state['visibility'] = 10.0
if 'wind' not in st.session_state:
    st.session_state['wind'] = 5.0

def load_model():
    try:
        with open('random_forest_model.pkl', 'rb') as file:
            model_dict = pickle.load(file)
        return model_dict
    except FileNotFoundError:
        st.error("Model file 'dumped.pkl' not found!")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_rainfall():
    model_dict = load_model()
    if model_dict is not None:
        try:
            # Extract model and scaler from the dictionary
            model = model_dict['model']
            scaler = model_dict['scaler']
            
            # Prepare input data
            input_data = pd.DataFrame({
                'tempavg': [st.session_state.temperature],
                'DPavg': [st.session_state.dew_point],
                'humidity avg': [st.session_state.humidity],
                'SLPavg': [st.session_state.pressure],
                'visibilityavg': [st.session_state.visibility],
                'windavg': [st.session_state.wind]
            })
            
            # Scale the input data
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            return prediction[0]
            
        except KeyError as e:
            st.error(f"Model dictionary is missing required components: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

def main():
    st.title('🌧️ Rainfall Prediction App')
    st.write('Enter weather parameters to predict rainfall')
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.temperature = st.number_input(
                'Average Temperature (°C)',
                min_value=-50.0,
                max_value=50.0,
                value=st.session_state.temperature,
                step=0.1,
                key='temp_input'
            )
            
            st.session_state.dew_point = st.number_input(
                'Average Dew Point (°C)',
                min_value=-50.0,
                max_value=50.0,
                value=st.session_state.dew_point,
                step=0.1,
                key='dew_input'
            )
            
            st.session_state.humidity = st.number_input(
                'Average Humidity (%)',
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.humidity,
                step=0.1,
                key='humidity_input'
            )
            
        with col2:
            st.session_state.pressure = st.number_input(
                'Average Sea Level Pressure (hPa)',
                min_value=900.0,
                max_value=1100.0,
                value=st.session_state.pressure,
                step=0.1,
                key='pressure_input'
            )
            
            st.session_state.visibility = st.number_input(
                'Average Visibility (km)',
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.visibility,
                step=0.1,
                key='visibility_input'
            )
            
            st.session_state.wind = st.number_input(
                'Average Wind Speed (km/h)',
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.wind,
                step=0.1,
                key='wind_input'
            )

    if st.button('Predict Rainfall', key='predict_button'):
        with st.spinner('Calculating prediction...'):
            prediction = predict_rainfall()
            if prediction is not None:
                st.success(f'Predicted Rainfall: {prediction:.2f} mm')
                
                # Display input parameters summary
                st.subheader('Input Parameters Summary')
                input_data = pd.DataFrame({
                    'Parameter': ['Temperature', 'Dew Point', 'Humidity', 'Pressure', 'Visibility', 'Wind'],
                    'Value': [
                        st.session_state.temperature,
                        st.session_state.dew_point,
                        st.session_state.humidity,
                        st.session_state.pressure,
                        st.session_state.visibility,
                        st.session_state.wind
                    ]
                })
                st.bar_chart(input_data.set_index('Parameter'))

if __name__ == '__main__':
    main()
