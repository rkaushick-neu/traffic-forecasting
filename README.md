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

# Project 1: Beginner Level on Dataset #1

## Machine Learning Model Results
### Prophet from Facebook (Meta)
Comparing Prophet Models:
<table>
    <tr>
        <th>Model</th>
        <th>MAE</th>
        <th>RMSE</th>
        <th>MAPE</th>
        <th>R-squared</th>
    </tr>
    <tr>
        <td>Prophet Model 1 (Baseline)</td>
        <td >1745.596</td>
        <td style="color: red;">2349.592</td>
        <td style="color: green;">16.84 %</td>
        <td style="color: red;">0.295</td>
    </tr>
    <tr>
        <td style="color: green;">Prophet Hyperparameter Tuned (GridSearchCV)</td>
        <td style="color: green;">1718.466</td>
        <td style="color: green;">2284.065</td>
        <td>17.03 %</td>
        <td style="color: green;">0.334</td>
    </tr>
    <tr>
        <td >Prophet Hyperparameter Tuned (Bayesian Optimization)</td>
        <td style="color: red;">1768.046</td>
        <td>2334.266</td>
        <td style="color: red;">17.35 %</td>
        <td>0.304</td>
    </tr>
</table>

<small>Note: Hyper-parameters were optimized for RMSE</small>

It was surprising to see that the Bayesian Optimization led to results which were just barely better than the baseline model in terms of the RMSE score & R-squared, and worse in terms of MAE & MAPE. 

We can see that the second model (in this case with GridSearchCV) was the overall best performing model because of the following reasons:
1. It has the best performance on 3 out of 4 metrics:
   - lowest MAE, 
   - lowest RMSE  
   - only slightly worse MAPE compared to Model 1.
   - highest R-squared
2. The higher R-squared indicates it captures more of the underlying patterns in the data.

### Random Forest
WIP

## Deep Learning Models
WIP

### Recurrent Neural Networks (RNN)
WIP

## Statistical Models
### Seasonal Auto-Regressive Integrated Moving Average (SARIMA) 
WIP

# Project 2: Intermediate Level on Dataset #2
## Machine Learning Models
WIP

## Deep Learning Models
WIP

## Statistical Models
WIP
