# pip install streamlit
# streamlit run app.py
import numpy as np
import pandas as pd
import requests
import streamlit as st
from matplotlib import pyplot as plt
from data_loader import DataLoader
from main import energy_forecasting

# Streamlit UI
st.title('Energy_Consumption_Forecasting App')
st.markdown('This is a basic app to explore the relation between the total electricity demand and the weather temperature in **Germany**. ')

with st.expander("Resources:"):
    st.markdown("Hourly data from 22.11.2024 to 15.02.2025")
    st.markdown("The data used in this app is from the following sources:")
    st.markdown("1. [Power Generation and Consumption Dataset](https://www.agora-energiewende.org/data-tools/agorameter/chart/today/power_generation/22.11.2024/15.02.2025/hourly)")
    st.markdown("2. [Open Meteo Dataset](https://open-meteo.com/en/docs)")

#Load data
data_loader = DataLoader(pgc_df='../Datas/power_generation_and_consumption.csv',
                         om_path='../Datas/open-meteo-51.50N10.50E309m.csv')
df = data_loader.load_data()

with st.expander('Energy Demand and Temperature Over Time'):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['Timestamp'], df['Total electricity demand'], color='blue', label='Total electricity demand')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Total electricity demand', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.bar(df['Timestamp'], df['temperature_2m (°C)'], color='orange', alpha=0.6, label='Temperature (°C)')
    ax2.set_ylabel('Temperature (°C)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    plt.title('Energy Demand and Temperature Over Time')
    fig.tight_layout()
    st.pyplot(fig)

# Checkbox to show data
st.markdown('## Do you want to try?')
st.text('Total electricity demand (desired output)')
with st.expander("#### Option for Energy usage:"):
    col1, col2 = st.columns(2)
    with col1:
        if not st.checkbox('Biomass', True):
            df = df.drop(columns=['Biomass'])
        if not st.checkbox('Grid emission factor', True):
            df = df.drop(columns=['Grid emission factor'])
        if not st.checkbox('Hard Coal', True):
            df = df.drop(columns=['Hard Coal'])
        if not st.checkbox('Hydro', True):
            df = df.drop(columns=['Hydro'])
        if not st.checkbox('Lignite', True):
            df = df.drop(columns=['Lignite'])
        if not st.checkbox('Natural Gas', True):
            df = df.drop(columns=['Natural Gas'])
        if not st.checkbox('Other', True):
            df = df.drop(columns=['Other'])
        if not st.checkbox('Pumped storage generation', True):
            df = df.drop(columns=['Pumped storage generation'])
        if not st.checkbox('Solar', True):
            df = df.drop(columns=['Solar'])
    with col2:
        if not st.checkbox('Conventional', True):
            df = df.drop(columns=['Conventional'])
        if not st.checkbox('Total grid emissions', True):
            df = df.drop(columns=['Total grid emissions'])
        if not st.checkbox('Wind offshore', True):
            df = df.drop(columns=['Wind offshore'])
        if not st.checkbox('Wind onshore', True):
            df = df.drop(columns=['Wind onshore'])
        if not st.checkbox('Total Renewables', True):
            df = df.drop(columns=['Total Renewables'])
        if not st.checkbox('Total Conventional', True):
            df = df.drop(columns=['Total Conventional'])
        if not st.checkbox('Renewable share', True):
            df = df.drop(columns=['Renewable share'])
        if not st.checkbox('Conventional share', True):
            df = df.drop(columns=['Conventional share'])
st.write("")
with st.expander("#### Option for weather:"):
    if not st.checkbox('temperature_2m (°C)', True):
        df = df.drop(columns=['temperature_2m (°C)'])
    if not st.checkbox('relative_humidity_2m (%)', True):
        df = df.drop(columns=['relative_humidity_2m (%)'])
    if not st.checkbox('rain (mm)', True):
        df = df.drop(columns=['rain (mm)'])
    if not st.checkbox('weather_code (wmo code)', True):
        df = df.drop(columns=['weather_code (wmo code)'])

st.write("")
if st.button('Show me!'):
    st.session_state['rf_model'] = energy_forecasting(df)

if 'rf_model' in st.session_state:
    rf_model = st.session_state['rf_model']
    if st.button('Try forcasting'):
        url = 'https://api.open-meteo.com/v1/forecast?latitude=51.5&longitude=10.5&hourly=temperature_2m,relative_humidity_2m,rain,weather_code'
        response = requests.get(url)
        data = response.json()
        hourly_data = data["hourly"]
        om_forcast = pd.DataFrame(hourly_data)
        om_forcast['Timestamp'] = pd.to_datetime(om_forcast['time'], errors='coerce')
        om_forcast = om_forcast.drop(columns=['time'])

        # Ensure the forecast data has the same columns as the training data
        required_columns = df.columns
        for col in required_columns:
            if col not in om_forcast.columns:
                om_forcast[col] = np.nan
        forcast_df = om_forcast[required_columns].drop(columns=['Total electricity demand', 'Timestamp'])

        forcast = rf_model.forcast(forcast_df)
        om_forcast['Total electricity demand forecast'] = forcast
        st.dataframe(om_forcast[['Timestamp', 'temperature_2m', 'Total electricity demand forecast']])

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(om_forcast['Timestamp'], om_forcast['Total electricity demand forecast'], color='blue', label='Total electricity demand forecast')
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('Total electricity demand forecast', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        # ax1.set_ylim(56, 59)
        ax2 = ax1.twinx()
        ax2.plot(om_forcast['Timestamp'], om_forcast['temperature_2m'], color='orange', alpha=0.6, label='temperature_2m')
        ax2.set_ylabel('temperature_2m', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        plt.title('Total Electricity Demand Forecast Over Time')
        fig.tight_layout()
        st.pyplot(fig)