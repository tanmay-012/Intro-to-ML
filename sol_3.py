import numpy as np
import pickle as pkl
import time as tm
import os
import pandas as pd


# Define your prediction method here
# df is a dataframe containing timestamps, weather data and potentials
def my_predict( df_test ):
	# Return both sets of predictions
    with open('best_knn_model.pkl', 'rb') as f:
        knn = pkl.load(f)

    df_test['hour'] = pd.to_datetime(df_test['Time']).dt.hour
    X_test = df_test[['no2op1', 'no2op2', 'o3op1', 'o3op2', 'temp', 'humidity', 'hour']]

    y_pred = knn.predict(X_test)
    pred_o3=y_pred[:,0]
    pred_no2=y_pred[:,1]
    return (pred_o3, pred_no2)