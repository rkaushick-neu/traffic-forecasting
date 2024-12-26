# Traffic Forecasting

## Datasets
1. [Traffic_data](data/TrafficDataTimeSeriesAnalysis.csv)
2. [Traffic Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset)

## Jupyter Notebooks
1. [traffic_forecast_1.ipynb](./traffic_forecast_nb1.ipynb): Built for the Dataset #1
2. traffic_forecast_2.ipynb: Built for the Dataset #2

## Metrics to Evaluate Models
1. **Mean Absolute Error (MAE)**
   MAE measures the average magnitude of the errors in a set of predictions, without considering their direction. It’s the average over the test sample of the absolute differences between prediction and actual observation.

    ```math
     \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
     ```

2. **Root Mean Squared Error (RMSE)**
   RMSE is the square root of the mean squared error. It provides a measure of how spread out these residuals are, and it is in the same units as the target variable.

   ```math
   \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} 
   ```
3. **Mean Absolute Percentage Error (MAPE)**
   MAPE measures the size of the error in terms of percentage. It is calculated as the average of the absolute percentage errors.

   ```math
   \text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| 
   ```

## Machine Learning Models

### Prophet

### Linear Regression

### Decision Trees

### Random Forest

## Deep Learning Models

### Recurrent Neural Networks (RNN)

### Gated Recurrent Units (GRU)

## Statistical Models

### Seasonal Auto-Regressive Integrated Moving Average (SARIMA) 

### Exponential Smoothing (Holt-Winters)

### Fourier Transform

