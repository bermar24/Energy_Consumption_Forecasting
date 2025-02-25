# [Energy Consumption Forecasting](https://github.com/bermar24/Energy_Consumption_Forecasting)

Energy consumption forecasting is an important area of study in many fields, including environmental science, economics, and business. It involves predicting future energy usage patterns based on historical data, weather conditions, and other relevant factors. Accurate forecasting of energy demand can help utility companies optimize their resources, improve efficiency, reduce costs, and ensure a reliable supply of energy. 

In this project, the focus is on predicting energy consumption for a specific region (e.g., an entire city) based on historical usage data, weather patterns, time of day, and possibly other external factors like holidays or special events.

## Libraries and Dependencies

- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation and analysis.
- **Requests**: For making HTTP requests to fetch data.
- **Streamlit**: For creating the web application interface.
- **Matplotlib**: For plotting and visualizations.
- **Scikit-learn**: For machine learning models and evaluation metrics.

## Project Structure

- `App/app.py`: Main application file for the Streamlit web app.
- `energy_forecasting.py`: Contains the data processing and machine learning model code.
- `data_loader.py`: Presumably contains the `DataLoader` class for loading datasets.

## Data Sources

- **Power Generation and Consumption Dataset**: Data from Agora Energiewende.
- **Open Meteo Dataset**: Weather data from Open Meteo.
- **Data Range**: 22.11.2024 to 15.02.2025.
- **API**: Open Meteo API for fetching forecast data.

## Running the Project

### Install Dependencies:
```sh
pip install numpy pandas requests streamlit matplotlib scikit-learn
```

### Run the Streamlit App:
```sh
cd App
streamlit run app.py
```

## Key Functionalities

- **Data Loading**: Loads and merges power generation and weather data.
- **Data Visualization**: Plots energy demand and temperature over time.
- **Feature Selection**: Allows users to select features for energy usage and weather.
- **Model Training**: Trains a Random Forest model to forecast energy demand.
- **Forecasting**: Uses the trained model to forecast future energy demand based on weather data.

## Conclusion

- **Not Enough Data** – The dataset may not have been large or detailed enough to allow the model to learn meaningful trends. More historical data, with finer granularity, could improve predictions.​
- **Data Quality and Appropriateness** – The features used may not have been the most relevant for an efficient forecasting model. Additional factors, such as human behavior, economic activity, and grid constraints, could significantly influence energy usage and should be considered.
- **Dependency on Weather Forecasting** – The model's accuracy is highly dependent on the quality of weather forecasts. Any inaccuracies in weather predictions propagate through the energy usage forecast, making the model less reliable.
