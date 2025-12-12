import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from datetime import datetime, timedelta

# Mock data for prediction
def generate_mock_data():
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    volumes = np.random.uniform(100000, 200000, 30) + np.sin(np.arange(30) * 0.2) * 10000
    return pd.DataFrame({'date': dates, 'volume': volumes})

def predict_water_volume(days_ahead=30):
    df = generate_mock_data()
    df['days'] = (df['date'] - df['date'].min()).dt.days

    # Linear Regression
    lr = LinearRegression()
    lr.fit(df[['days']], df['volume'])
    future_days = np.arange(df['days'].max() + 1, df['days'].max() + days_ahead + 1)
    lr_predictions = lr.predict(future_days.reshape(-1, 1))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(df[['days']], df['volume'])
    rf_predictions = rf.predict(future_days.reshape(-1, 1))

    future_dates = [df['date'].max() + timedelta(days=i) for i in range(1, days_ahead + 1)]

    return {
        'dates': future_dates,
        'linear_regression': lr_predictions.tolist(),
        'random_forest': rf_predictions.tolist()
    }