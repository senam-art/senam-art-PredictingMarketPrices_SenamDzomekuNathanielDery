
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nixtla import NixtlaClient, TimeGPT
import plotly.graph_objects as go



# Initialize Nixtla client
nixtla_client = NixtlaClient(api_key = 'nixtla-tok-AGxYtllDVvOFafOqgELYnxfkJZDTUFgGgK0jEWZWDCfTt2G2b4U7YM1Yd2Svik9PKr914Ef5Ye9Kp7AQ')


# Function to load and preprocess data from GitHub
def load_data():
    url = 'https://github.com/senam-art/senam-art-PredictingMarketPrices_SenamDzomekuNathanielDery/raw/main/deployment_data.csv'
    df = pd.read_csv(url)

    # Convert 'ds' to datetime format (assuming 'MM-YYYY' format)
    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')
    # Convert to 'YYYY-MM-DD' format, start of the month
    df['ds'] = df['ds'].dt.to_period('M').dt.to_timestamp()
    df = df.rename(columns={'riceusdprice': 'y'})
    return df


def convert_to_start_of_month(df, date_column):
    '''This function is mainly to convert the date column to datetime format'''
    # Convert the column to datetime format
    df[date_column] = pd.to_datetime(df[date_column], format='%Y-%m-%d')

    # Convert to 'YYYY-MM-DD' format, set to the start of the month
    df[date_column] = df[date_column].dt.to_period('M').dt.to_timestamp()

    return df


# Function to forecast with TimeGPT
def forecast_with_timegpt(forecast_period):
    historical_df = load_data()

    # Define the forecast horizon
    forecast_horizon = forecast_period

    df_maize = historical_df[['ds', 'maizeusdprice']]
    df_inf = historical_df[['ds', 'Inflation']]
    df_exchangerate = historical_df[['ds', 'exchangerate']]
    df_rainfall = historical_df[['ds', 'Rainfall']]

    #Forecast for extrogeneous features
    maize_forecast = nixtla_client.forecast(df=df_maize, h=forecast_horizon, target_col='maizeusdprice')

    inflation_forecast = nixtla_client.forecast(df=df_inf, h=forecast_horizon, target_col='Inflation')

    exchangerate_forecast = nixtla_client.forecast(df=df_exchangerate, h=forecast_horizon, target_col='exchangerate')

    rainfall_forecast = nixtla_client.forecast(df=df_rainfall, h=forecast_horizon, target_col='Rainfall')

    # Rename columns for clarity
    maize_forecast = maize_forecast.rename(columns={'TimeGPT': 'maizeusdprice_forecast'})
    inflation_forecast = inflation_forecast.rename(columns={'TimeGPT': 'Inflation_forecast'})
    exchangerate_forecast = exchangerate_forecast.rename(columns={'TimeGPT': 'exchangerate_forecast'})
    rainfall_forecast = rainfall_forecast.rename(columns={'TimeGPT': 'Rainfall_forecast'})


    #convert 'ds' column to date format and 1st of month
    convert_to_start_of_month(maize_forecast, 'ds')
    convert_to_start_of_month(inflation_forecast, 'ds')
    convert_to_start_of_month(exchangerate_forecast,'ds')
    convert_to_start_of_month(rainfall_forecast,'ds')




    # Merge forecasts
    X_df = maize_forecast.merge(inflation_forecast, on='ds').merge(exchangerate_forecast, on='ds').merge(rainfall_forecast, on='ds')
    # Convert 'ds' to datetime format (assuming 'MM-YYYY' format)
    X_df['ds'] = pd.to_datetime(X_df['ds'], format='%Y-%m-%d')

    # Extract forecasted values from X_df
    maize_forecastv = X_df['maizeusdprice_forecast'].values
    exchangerate_forecastv = X_df['exchangerate_forecast'].values
    inflation_forecastv = X_df['Inflation_forecast'].values
    rainfall_forecastv = X_df['Rainfall_forecast'].values


    # Create future dates for extrogenous df
    last_date = historical_df['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_horizon, freq='M')


    # Create DataFrame with future dates and forecasted values
    forecast_exogenous_df = pd.DataFrame({
        'ds': future_dates,
        'maizeusdprice': maize_forecastv,
        'exchangerate': exchangerate_forecastv,
        'Inflation': inflation_forecastv,
        'Rainfall': rainfall_forecastv
    })

    convert_to_start_of_month(forecast_exogenous_df,'ds')

    # Forecasting(Training/Prediction)
    # Perform forecasting
    timegpt_forecast = nixtla_client.forecast(
        df=historical_df,
        h=forecast_horizon,
        finetune_steps=19,
        model='timegpt-1-long-horizon',
        time_col='ds',
        target_col='y'
    )

    #DataFrame with future dates and forecasted values
    graph_forecast_df = pd.DataFrame({
        'ds': future_dates,
        'y': timegpt_forecast['TimeGPT']
    })

    # Combine historical and forecasted data
    combined_df = pd.concat([historical_df, graph_forecast_df])

    return timegpt_forecast, combined_df


# Main function for the Streamlit app
# Main function for the Streamlit app
def main():

    # Add explanatory text to the sidebar
    st.sidebar.markdown("""
        ### Forecasting Information
        In this application, you can forecast Prices of rice measured as 50kg for a specified number of months into the future. 

        - **Forecast Period**: Use the slider to select the number of months you want to forecast. The range is from 1 to 36 months.
        - **Limit**: The forecasting period is capped at 36 months to ensure the accuracy and reliability of predictions. Longer forecasts may become less accurate due to uncertainties and changes in market conditions.
        - **Get Started** : Click 'Generate Forecast' to get started 
        - ***Scroll for forecasts***
        """)


    st.title('Rice Price Forecaster')
    st.write(
        'Welcome to the Rice Price Forecasting App! You can view existing rice price data in Ghana and generate forecasts. These predictions are for 50kg of Rice in Ghana.')

    st.sidebar.header('Forecasting Parameters')
    forecast_months = st.sidebar.slider('Forecast Period (months)', 1, 36, 12)

    # Load data and show existing prices
    df_main = load_data()
    display_rice_prices = df_main[['ds', 'y']]
    rice_prices= display_rice_prices.copy()
    rice_prices.rename(columns={'ds': 'DateStamp', 'y': 'Price in USD'}, inplace=True)
    rice_prices.set_index('DateStamp', inplace = True )

    # Plot existing rice price data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=display_rice_prices['ds'], y=display_rice_prices['y'],
                             mode='lines+markers',
                             name='Rice Prices',
                             hoverinfo='x+y',
                             marker=dict(size=6)))

    fig.update_layout(title='Rice Prices Over Time',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis=dict(tickangle=-45),
                      hovermode='x unified')

    st.plotly_chart(fig, use_container_width=True)

    # Display DataFrame
    st.write('### Existing Rice Price Data')
    st.dataframe(rice_prices,use_container_width =True)




    if st.sidebar.button('Generate Forecast'):
        with st.spinner('Generating forecast...'):
            # Forecasting
            st.write('### Forecasting')
            forecast_data,forecast_data_concat = forecast_with_timegpt(forecast_months)
            st.write(f'### Forecast for the next {forecast_months} months')

            #change column name
            # change column name
            forecast_data.rename(columns={'ds': 'DateStamp', 'TimeGPT': 'Price in USD'}, inplace=True)

            forecast_data_concat.rename(columns={'ds': 'DateStamp', 'TimeGPT': 'Price in USD'}, inplace=True)


            # Display forecast DataFrame
            st.dataframe(forecast_data,use_container_width =True)



            # Create a figure
            fig_forecast = go.Figure()

            # Plot historical data
            fig_forecast.add_trace(go.Scatter(x=df_main['ds'], y=df_main['y'],
                                              mode='lines+markers',
                                              name='Historical Prices',  # Updated name for clarity
                                              hoverinfo='x+y',
                                              marker=dict(size=6, color='blue'),
                                              line = dict(color='blue')
                                              ))

            # Plot forecasted data
            fig_forecast.add_trace(go.Scatter(x=forecast_data['DateStamp'], y=forecast_data['Price in USD'],
                                              mode='lines+markers',
                                              name='Forecasted Prices',
                                              hoverinfo='x+y',
                                              marker=dict(size=6, color='red'),
                                              line = dict(color='red')
                                                ))


            # Update layout
            fig_forecast.update_layout(title='Rice Prices and Forecast',
                                       xaxis_title='Date',
                                       yaxis_title='Price',
                                       xaxis=dict(tickangle=-45),
                                       hovermode='x unified')

            # Display the plot in Streamlit
            st.plotly_chart(fig_forecast, use_container_width=True)

            #Plot detailed graph
            # Create a figure
            fig_detailed_forecast = go.Figure()
            fig_detailed_forecast.add_trace(go.Scatter(x=forecast_data['DateStamp'], y=forecast_data['Price in USD'],
                                                           mode='lines+markers',
                                                           name='Detailed Forecasted Prices',
                                                           hoverinfo='x+y',
                                                           marker=dict(size=6, color='green'),
                                                           line=dict(color='green')))

            fig_detailed_forecast.update_layout(title='Detailed Forecasted Rice Prices',
                                                    xaxis_title='Date',
                                                    yaxis_title='Price',
                                                    xaxis=dict(tickangle=-45),
                                                    hovermode='x unified')

            # Display the plots in Streamlit
            st.plotly_chart(fig_detailed_forecast, use_container_width=True)


if __name__ == '__main__':
    main()