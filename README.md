
# Rice Price Prediction in Ghana using Nixtla's TimeGPT

## Project Overview

This project aims to predict rice prices in Ghana using economic exogenous features such as exchange rates, maize prices, inflation, and rainfall. After exploring various methods for time series prediction, we found that Nixtla's TimeGPT model provided the most accurate results by reducing biases commonly found in other methods.

***Youtube Link***: https://youtu.be/UuKaaygph7M (demo of the application)
The application can be accessed at https://senam-art-predictingmarketpricessenamdzomekunathanieldery-kbrn.streamlit.app/

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

## Improvements

- **Fine-Tuning**: Future iterations could involve further fine-tuning of the model for more accurate predictions.
- **Accuracy** : Building a larger dataset that captures the granularity which would improve the accurqcy of predictions

To include instructions for running your Streamlit application locally in your README file, you'll want to provide clear steps for setting up the environment, installing dependencies, and starting the app. Based on your imports and the use of `NixtlaClient`, here’s a suggested section to add to your README:

---



## Running the Application Locally

To run this Streamlit application locally, follow these steps:

### Prerequisites

Make sure you have Python 3.8 or later installed. You can download Python from [python.org](https://www.python.org/downloads/).

### Setup

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and Activate a Virtual Environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**

   Ensure you have `pip` installed, then install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   Make sure `requirements.txt` includes the following (add other dependencies as necessary):

   ```plaintext
   streamlit
   pandas
   numpy
   matplotlib
   nixtla
   plotly
   ```


4. **Run the Application:**

   Start the Streamlit app by running:

   ```bash
   streamlit run app.py
   ```

   This will open a new tab in your default web browser with the running application.


