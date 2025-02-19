import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import DateFrame 22.11.2024 to 15.02.2025
# https://www.agora-energiewende.org/data-tools/agorameter/chart/today/power_generation/01.03.2024/15.02.2025/hourly
pgc_df = pd.read_csv("Datas/power_generation_and_consumption.csv")
pgc_df['Timestamp'] = pd.to_datetime(pgc_df['date_id'], errors='coerce')
pgc_df = pgc_df.drop(columns=['date_id', 'date_id.1'])
print(pgc_df.columns)

# Import DateFrame open-meteo Historical data Frankfurt (22/11/2024–15/02/2025)
# https://open-meteo.com/en/docs#hourly=temperature_2m,precipitation_probability
om_df = pd.read_csv("Datas/open-meteo-51.50N10.50E309m.csv")
om_df['Timestamp'] = pd.to_datetime(om_df['time'], errors='coerce')
om_df = om_df.drop(columns=['time'])
print(om_df.columns)

# unify dataframes
df = pd.merge(pgc_df, om_df, how='left', on='Timestamp',)
print(df.columns)
# print(df.head())

# check for missing value and clean data
print("Null values from dataset\n", df.isnull().sum())

# Encode categorical variables using one-hot encoding
# df_encoded = pd.get_dummies(df, drop_first=True)

# Select target variable (Total Bill) and features
X = df.drop(columns=['Total electricity demand', 'Timestamp'])
y = df['Total electricity demand']

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO Random Forest Regression (Non-linear Model)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict using Random Forest model
y_pred_rf = rf_regressor.predict(X_test)

# Evaluate Random Forest model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)
print(f'Random Forest MAE: {mae_rf}')
print(f'Random Forest MSE: {mse_rf}')
print(f'Random Forest R2: {r2}')

# Sample Prediction
single_data = X_test.iloc[0:1]
predicted_value = rf_regressor.predict(single_data)
print(f"Predicted Value: {predicted_value[0]:.2f}")
print(f"Actual Value: {y_test.iloc[0]:.2f}")

# Random Forest visualization
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_rf)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Random Forest: Actual vs Predicted Tip')
plt.xlabel('Actual Tip')
plt.ylabel('Predicted Tip')
plt.tight_layout()
plt.show()

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

# Drop any extra columns not used in the model
forcast_df = om_forcast[required_columns].drop(columns=['Total electricity demand', 'Timestamp'])

forcast = rf_regressor.predict(forcast_df)
print(forcast)
om_forcast['Total electricity demand forecast GWh'] = forcast
print(om_forcast.columns)

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(om_forcast['Timestamp'], om_forcast['Total electricity demand forecast GWh'], color='blue',
         label='Total electricity demand forecast GWh')
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Total electricity demand forecast GWh', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.plot(om_forcast['Timestamp'], om_forcast['temperature_2m'], color='orange', alpha=0.6, label='temperature_2m')
ax2.set_ylabel('temperature_2m', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
plt.title('Total Electricity Demand Forecast Next Week')
fig.tight_layout()
plt.show()
