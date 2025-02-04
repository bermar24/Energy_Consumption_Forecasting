import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import DateFrame electricity_market_dataset Frankfurt
emd_df = pd.read_csv("electricity_market_dataset.csv")
emd_df['Timestamp'] = pd.to_datetime(emd_df['Timestamp'], errors='coerce')
emd_df = emd_df.drop(columns=['Renewable_Investment_Costs',
       'Fossil_Fuel_Costs', 'Electricity_Export_Prices', 'Market_Elasticity',
       'Subsidies', 'Energy_Production_By_Solar', 'Energy_Production_By_Wind',
       'Energy_Production_By_Coal', 'Energy_Storage_Capacity', 'GHG_Emissions',
       'Renewable_Penetration_Rate', 'Regulatory_Policies',
       'Energy_Access_Data', 'LCOE', 'ROI', 'Net_Present_Value',
       'Population_Growth', 'Optimal_Energy_Mix', 'Electricity_Price_Forecast',
       'Project_Risk_Analysis', 'Investment_Feasibility'])
# print(emd_df.columns)

# Import DateFrame open-meteo Historical data Frankfurt
om_df = pd.read_csv("open-meteo-50.09N8.65E117m.csv")
om_df['Timestamp'] = pd.to_datetime(om_df['time'], errors='coerce')
om_df = om_df.drop(columns=['time'])
# print(om_df.columns)

# unify dataframes
df = pd.merge(emd_df, om_df, how='left', on='Timestamp',)
print(df.columns)
print(df.head())

# check for missing value and clean data
# print("Null values from dataset\n", df.isnull().sum())

# Encode categorical variables using one-hot encoding
# df_encoded = pd.get_dummies(df, drop_first=True)

# Select target variable (Total Bill) and features
X = df.drop(columns=['GDP_Growth_Rate', 'Timestamp'])
y = df['GDP_Growth_Rate']

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

plt.figure(figsize=(8,6))
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_rf)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Random Forest: Actual vs Predicted Tip')
plt.xlabel('Actual Tip')
plt.ylabel('Predicted Tip')
plt.tight_layout()
plt.show()
