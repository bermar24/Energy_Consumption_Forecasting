import pandas as pd
from app.data_loader import DataLoader
from app.model import RandomForestModel

def main():
    # Load data
    data_loader = DataLoader(emd_path='../data/electricity_market_dataset.csv', om_path='../data/open-meteo-50.09N8.65E117m.csv')
    df = data_loader.load_data()

    # Prepare features and target
    X = df.drop(columns=['GDP_Growth_Rate', 'Timestamp'])
    y = df['GDP_Growth_Rate']

    # Train model
    rf_model = RandomForestModel()
    rf_model.train(X, y)

    # Evaluate model
    mae, mse, r2 = rf_model.evaluate()
    print(f'Random Forest MAE: {mae}')
    print(f'Random Forest MSE: {mse}')
    print(f'Random Forest R2: {r2}')

    # Sample prediction
    single_data = X.iloc[0:1]
    predicted_value = rf_model.predict(single_data)
    print(f"Predicted Value: {predicted_value[0]:.2f}")
    print(f"Actual Value: {y.iloc[0]:.2f}")

if __name__ == "__main__":
    main()