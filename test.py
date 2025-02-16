# import pandas as pd
# import requests
#
# # Fetch data from the API
# url = 'https://api.open-meteo.com/v1/forecast?latitude=51.5&longitude=10.5&hourly=temperature_2m,relative_humidity_2m,rain,weather_code'
# response = requests.get(url)
# data = response.json()
# hourly_data = data["hourly"]
# om_forcast = pd.DataFrame(hourly_data)
#
# # Display the DataFrame
#
# om_forcast['Timestamp'] = pd.to_datetime(om_forcast['time'], errors='coerce')
# om_forcast = om_forcast.drop(columns=['time'])
#
# # Ensure the forecast data has the same columns as the training data
# required_columns = ['temperature_2m (°C)']
# for col in required_columns:
#     if col not in om_forcast.columns:
#         om_forcast[col] = 0  # or some default value
#
# # Drop any columns not used in the model
# om_forcast = om_forcast[required_columns]
# print(om_forcast.columns)
# mae, mse, r2, y_pred = rf_model.evaluate(om_forcast)
# st.write(f'Random Forest MAE: {mae}')
# st.write(f'Random Forest MSE: {mse}')
# st.write(f'Random Forest R2: {r2}')

# predicted_value = rf_model.predict(single_data)