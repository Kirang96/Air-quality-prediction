# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 17:03:03 2021

@author: user
"""
#%% Initialization
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
#%%
# importing prediction input and sequencing it

all_data = pd.read_csv(r'models/input_cluster.csv')
all_data = all_data.iloc[:,1:]

#%%
app = Flask(__name__)

lstm = tf.keras.models.load_model("models/lstm_60_90_38rmse.h5")
kmeans = pickle.load(open('models/kmeans.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    date = request.form.get("date")
    loc = request.form.get("loc")
    lat_long = []
    for items in loc.split(','):
        lat_long.append(float(items))
    
    location = kmeans.predict([lat_long])
    location = int(location)
    
    
    last_day = pd.to_datetime('23-03-2020')
    selected_day = pd.to_datetime(date)
    days_until_today = (selected_day-last_day).days

     #Getting location specific data
     # all_data_specific = all_data
    all_data_specific = all_data.loc[all_data.loc[:,'cluster'] == location]
     #scaling the selected data
    training_set = all_data_specific.values
    scaler = StandardScaler()
    training_set_scaled = scaler.fit_transform(training_set)
    scaler_predict = StandardScaler()
    scaler_predict.fit_transform(training_set[:, 0:1])
    train = training_set_scaled
     #splitting the data to sequences
    x_train = []
    y_train = []

    n_future = 500   # Number of days we want top predict into the future
    n_past = 500   # Number of past days we want to use to predict the future

    for i in range(n_past, len(train) - n_future +1):
        x_train.append(train[i - n_past:i, 0:train.shape[1]])
        y_train.append(train[i + n_future - 1:i + n_future, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

     # making prediction
    preds = lstm.predict(x_train[-50:])
    future_predictions = scaler_predict.inverse_transform(preds)[0,:days_until_today]
    today = future_predictions[-1]
    today = round(today,2)
    color="white"
    text="default text"
    if today <= 50:
        result = "Good"
        color = "rgb(0,117,0)"
        text = "Minimal impact"
        
        
    elif 51 < today < 100:
        result = "Satisfactory"
        color = "rgb(126,189,1)"
        text = "May cause minor breathing discomfort to sensitive people"
        
        
        
    elif 101< today <200:
        result = "Moderate"
        color = "rgb(242,215,2)"
        text = "May cause minor breathing discomfort to the people with lung diseases such as asthma and discomfort to people with heart disease, children and other adults."
        
        
    elif 201 < today <300:
        result = "Poor"
        color = "rgb(244,119,1)"
        text = "May cause minor breathing discomfort to the people on prolonged exposure and discomfort to people with heart diseases with short exposure."
       
        
        
    elif 301< today <400:
        color = "rgb(218,33,51)"
        result = "Very poor"
        text = "May cause respiratory illness to the people on prolonged exposure. Effect may be more pronounced in people with lung and hear diseases."
        
        
        
    else:
        result = "Severe"
        color = "rgb(158,25,20)"
        text = "May cause respiratory effects even on healthy people and serious health impacts on people with lung or heart diseases. The health impacts may be experienced even during light physical activity."
        
    return render_template("predict.html", result=result, prediction=today, text=text, color=color)



if __name__=='__main__':
    app.run(debug=True)
