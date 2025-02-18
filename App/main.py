import streamlit as st
from matplotlib import pyplot as plt
from model import RandomForestModel

def energy_forecasting(df):
    X = df.drop(columns=['Total electricity demand', 'Timestamp'])
    y = df['Total electricity demand']

    rf_model = RandomForestModel()
    y_test = rf_model.train(X, y)

    mae, mse, r2, y_pred = rf_model.evaluate()
    st.write(f'Random Forest MAE: {mae}')
    st.write(f'Random Forest MSE: {mse}')
    st.write(f'Random Forest R2: {r2}')

    single_data = X.iloc[0:1]
    predicted_value = rf_model.forcast(single_data)
    st.write(f"Predicted Value: {predicted_value[0]:.2f}")
    st.write(f"Actual Value: {y.iloc[0]:.2f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.title('Random Forest: Actual vs Predicted Tip')
    plt.xlabel('Actual Tip')
    plt.ylabel('Predicted Tip')
    plt.tight_layout()
    st.pyplot(plt)

    return rf_model