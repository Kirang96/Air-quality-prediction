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
#%%
# importing prediction input and sequencing it

all_data = pd.read_csv('data/input_for_app.csv')
all_data = all_data.iloc[:,1:]

#%%
app = Flask(__name__)

lstm = tf.keras.models.load_model("models/conv_lstm_hyb.h5")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    date = request.form.get("date")
    loc = request.form.get("loc")
    
    last_day = pd.to_datetime('23-03-2020')
    selected_day = pd.to_datetime(date)
    days_until_today = (selected_day-last_day).days

    #Getting location specific data
    # all_data_specific = all_data
    all_data_specific = all_data.loc[(all_data.iloc[:,14]==float(loc))|(all_data.iloc[:,14]==12.20607688061421)]
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
    n_past = 50   # Number of past days we want to use to predict the future

    for i in range(n_past, len(train) - n_future +1):
        x_train.append(train[i - n_past:i, 0:train.shape[1]])
        y_train.append(train[i + n_future - 1:i + n_future, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # making prediction
    preds = lstm.predict(x_train[-50:])
    future_predictions = scaler_predict.inverse_transform(preds)[0,:days_until_today]
    today = future_predictions[-1]
    color="white"
    text="default text"
    if today <= 50:
        result = "Good"
        color = "green"
        text = "good People with lung diseases, such as asthma, chronic bronchitis, and emphysema. <br>Children, including teenagers.<br>Active people of all ages who exercise or work extensively outdoors<br>Some healthy people are more sensitive to ozone"
        
        
    elif 50 < today < 100:
        result = "Moderate"
        color = "rgb(226, 226, 6)"
        text = "moderate People with lung diseases, such as asthma, chronic bronchitis, and emphysema. <br>Children, including teenagers.<br>Active people of all ages who exercise or work extensively outdoors<br>Some healthy people are more sensitive to ozone"
        
        
        
    elif 100< today <150:
        result = "Unhealthy for sensitive groups"
        color = "orange"
        text = "unhealthy1 People with lung diseases, such as asthma, chronic bronchitis, and emphysema. <br>Children, including teenagers.<br>Active people of all ages who exercise or work extensively outdoors<br>Some healthy people are more sensitive to ozone"
        
        
    elif 150 < today <200:
        result = "Unhealthy"
        color = "rgb(161, 1, 1)"
        text = "unhealthy2 People with lung diseases, such as asthma, chronic bronchitis, and emphysema. <br>Children, including teenagers.<br>Active people of all ages who exercise or work extensively outdoors<br>Some healthy people are more sensitive to ozone"
        
        
        
    elif 200< today <250:
        color = "purple"
        result = "Very unhealthy"
        text = "purple People with lung diseases, such as asthma, chronic bronchitis, and emphysema. <br>Children, including teenagers.<br>Active people of all ages who exercise or work extensively outdoors<br>Some healthy people are more sensitive to ozone"
        
        
        
    else:
        result = "hazardous!"
        color = "maroon"
        text = "hazardous People with lung diseases, such as asthma, chronic bronchitis, and emphysema. <br>Children, including teenagers.<br>Active people of all ages who exercise or work extensively outdoors<br>Some healthy people are more sensitive to ozone"
        
    return render_template("index.html", prediction=today, result=result, color=color, text=text)




if __name__=='__main__':
    app.run(debug=True)
