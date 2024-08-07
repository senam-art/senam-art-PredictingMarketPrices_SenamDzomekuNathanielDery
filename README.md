
# Rice Price Prediction in Ghana using Nixtla's TimeGPT

## Project Overview

This project aims to predict rice prices in Ghana using economic exogenous features such as exchange rates, maize prices, inflation, and rainfall. After exploring various methods for time series prediction, we found that Nixtla's TimeGPT model provided the most accurate results by reducing biases commonly found in other methods.

## Setup Instructions

### Prerequisites

Before running the project, ensure you have the following Python packages installed:

```bash
pip install xgb
pip install keras-tuner
pip install nixtla
```

### Project Structure

- `rice_price_prediction.ipynb`: The main notebook where the entire workflow is carried out.
- `forecast_combined_df.csv`: The CSV file containing the forecasted rice prices along with exogenous features.
- `data/`: Directory containing input data files for the project.
- `outputs/`: Directory where the outputs, including visualizations and forecasts, are saved.

### Data

The dataset was created from real data in Ghana. It includes historical data on rice prices and exogenous variables such as maize prices, exchange rates, inflation, and rainfall.

## Workflow

1. **Data Preprocessing**:
    - Convert date columns to the appropriate datetime format.
    - Set the date column as the index for easier time series handling.
    - Perform exploratory data analysis (EDA) including visualizations and correlation analysis.

2. **Feature Scaling**:
    - Use `StandardScaler` to scale the features for the models that require it.

3. **Feature Importance**:
    - Evaluate feature importance using RandomForest and XGBoost models.

4. **Nixtla TimeGPT**:
    - Use Nixtla's TimeGPT model to forecast rice prices. 
    - Forecast exogenous features (maize prices and inflation) first, then use them to predict rice prices.
    - Visualize the forecasted prices and important features.

5. **Saving Forecasts**:
    - Save the combined forecast of rice prices and exogenous variables to a CSV file for future reference.

## Results

The TimeGPT model identified the exchange rate as the most critical feature influencing rice prices in Ghana. The model's forecasts have been saved in `forecast_combined_df.csv` for the period from 2023-07 to 2024-07. This is used for predictions in the deployed app.

## Future Work

- **Fine-Tuning**: Future iterations could involve further fine-tuning of the model for more accurate predictions.
- **Accuracy** : Building a larger dataset that captures the granularity which would improve the accurqcy of predictions


## Conclusion

The project successfully demonstrates the power of using advanced models like TimeGPT for accurate time series forecasting, especially in the context of predicting economic variables.

