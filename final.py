# -*- coding: utf-8 -*-
"""
Created on Wed May 19 18:41:55 2021

@author: Kiran
"""

#%% Importing all the packages

import time
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import haversine
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import dataframe_image as dfi
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
import glob

#%% getting the dates

# # Uncomment this column if we're using haversine distance function


# # assign dataset names
# list_of_names = ['eloor_1','eloor_2','irumpanam','kalamassery','mg','south','vytilla']
  
# # create empty list
# dataframes_list = []
  
# # append datasets into teh list
# for i in range(len(list_of_names)):
#     temp_df = pd.read_csv("./data/processed/"+list_of_names[i]+".csv", index_col='Date')
#     dataframes_list.append(temp_df)
    
# eloor_1,eloor_2,irumpanam,kalamassery,mg,south,vytilla = dataframes_list[:]
 
# all_data = pd.concat([eloor_1,eloor_2,irumpanam,kalamassery,mg,south,vytilla])

#%%

# use this if we're using the kmeans for locations

all_data = pd.read_csv(r"data/final_data_preprocessed/final_data.csv", parse_dates=(['Date']), index_col='Date')

#%% Taking a look at the data

# Shape of the full data
print(all_data.shape)

# Describing the data
data_descr = all_data.describe()
dfi.export(data_descr, 'data_description.png')
# information about the data
all_data.info()

#%% Visualising the data
# Finding correlations between the data

# correlation of AQI with Min T, Max T, min windspeed,  max wind speed, specific humidity, relative humidity, precipitation, surface pressure
df = all_data.loc[:, ['AQI','Temp Min','Wind Speed Max','Specific Humidity','Relative Humidity','Precipitation','Surface Pressure']]
corrmat = df.corr()
fig, ax = plt.subplots(figsize=(11,11))
sns.heatmap(corrmat)



#%% View all columns
print(all_data.columns)

#%% Sorting all the columns
all_data = all_data.reindex(columns=['AQI', 'SO2', 'NO2', 'PM2.5', 'PM10', 'Precipitation','Specific Humidity', 'Relative Humidity', 'Surface Pressure', 'Temp Max', 'Temp Min', 'Wind Speed Max', 'Wind Speed Min', 'Population', 'No.I', 'MIM', 'MIS', 'SM', 'SS', 'MEM', 'MES', 'cluster'])
all_data_copy = all_data.copy()


#%% Using RFE to find the best features for the model

estimator = RandomForestRegressor()

rfe = RFE(estimator=estimator, step=1, n_features_to_select=15)

xvals = all_data.iloc[:,0:-1]
yvals = all_data.iloc[:,0]

rfe= rfe.fit(xvals,yvals)

selected_features = pd.DataFrame({'Features':list(xvals.columns),
                                  'Ranking':rfe.ranking_ })
selected_features.sort_values(by='Ranking')

all_data = rfe.transform(xvals)

all_data.shape
all_data = pd.DataFrame(all_data)

#%% add the cluster column
clust = all_data_copy['cluster']
all_data['cluster'] = clust.values
#%% dataframe ranks plotting

selected_features = selected_features.sort_values('Ranking')
selected_features_styled = selected_features.style.background_gradient(low=1)
dfi.export(selected_features_styled, 'df_styled.png')
selected_features.dfi.export('df.png')

#%% Finding the trend and seasonality

all_data[0].plot(figsize=(20,6))
aqi = pd.DataFrame(all_data[0].values)
aqi.index = pd.to_datetime(aqi.index)
results = seasonal_decompose(aqi[0], period=1000)
results.plot();


#%% Normalising the dataset

all_data = all_data.astype('float')
#%% saving dataframe for app

target_input = pd.DataFrame(all_data)
target_input.to_csv('models/input_cluster.csv')

#%% Using multiple features (predictors)
training_set = all_data.values

scaler = StandardScaler()
training_set_scaled = scaler.fit_transform(training_set)

scaler_predict = StandardScaler()
scaler_predict.fit_transform(training_set[:, 0:1])

# pickle.dump(scaler_predict, open('models/transform.pkl', "wb"))
#%% Splitting the data to train, test and valid

# split_length = int(0.8*len(training_set_scaled))
# train = [:split_length, :]
# test = training_set_scaled[split_length:, :]

train = training_set_scaled

train2 = pd.DataFrame(train)
train_x = train2
train_y = train2.iloc[:,0]
#%% creating a sequence of datapoints 

x_train = []
y_train = []

n_future = 60   # Number of days we want top predict into the future
n_past = 90   # Number of past days we want to use to predict the future

for i in range(n_past, len(train) - n_future +1):
    x_train.append(train[i - n_past:i, 0:all_data.shape[1]])
    y_train.append(train[i + n_future - 1:i + n_future, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

print('x_train shape == {}.'.format(x_train.shape))
print('y_train shape == {}.'.format(y_train.shape))


#%% creating an feed forward model

ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=[None,all_data.shape[1]]), 
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(n_future, activation='linear')
    ])

ann.compile(loss=tf.keras.losses.Huber(), optimizer='Adam', metrics =['mae','mse'])

#%% fitting the ann

ann_history = ann.fit(x_train, y_train,validation_split=0.2,epochs=100)
ann.summary()

#%% Rnn

rnn = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(units=64, return_sequences=True,input_shape=[None, all_data.shape[1]]), 
    tf.keras.layers.SimpleRNN(units=64, return_sequences=True), 
    tf.keras.layers.SimpleRNN(units=32, return_sequences=False), 
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(n_future, activation='linear'),
    tf.keras.layers.Lambda(lambda x: x*100.0)
    ])

rnn.summary()

rnn.compile(loss=tf.keras.losses.Huber(), optimizer='Adam', metrics =['mae','mse'])
rnn_history = rnn.fit(x_train, y_train,validation_split=0.2,epochs=100)


#%% lstm


lstm = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=32, return_sequences=True,input_shape=[None, all_data.shape[1]]), 
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.LSTM(units=32, return_sequences=False), 
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(n_future, activation='linear'),
    tf.keras.layers.Lambda(lambda x: x*100.0)
    ])

lstm.summary()
lstm.compile(loss=tf.keras.losses.Huber(), optimizer='Adam', metrics =['mae','mse'])
lstm_history = lstm.fit(x_train, y_train,validation_split=0.2,epochs=100)

#%% Creating an conv_LSTM model


batch_size = 32

cnn_lstm = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, all_data.shape[1]]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32, return_sequences=False, dropout=0.25)), 
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(n_future, activation='linear'),
    tf.keras.layers.Lambda(lambda x: x*100.0)
    ])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.Adam(lr=1e-8)
cnn_lstm.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics =['mae','mse'])

cnn_lstm_history = cnn_lstm.fit(x_train, y_train,
                        validation_split=0.2,
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[lr_schedule]
                        )

#%% Finding the correct learning rate 

plt.semilogx(cnn_lstm_history.history["lr"], lstm_history.history["loss"])
plt.title('learning rate v loss')
plt.xlabel('learning rate')
plt.ylabel('loss')
#plt.axis([1e-7, 1e-3, 0, 10])

#%% The correct learning rate is found to be 3*1e-4. So training the same model with that learning rate for 500 epochs.
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1)
log_dir = "logs/"
# !tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=15, verbose=1)

batch_size = 64

cnn_lstm = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, all_data.shape[1]]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=16,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=16, return_sequences=False, dropout=0.25)), 
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(n_future, activation='linear'),
    tf.keras.layers.Lambda(lambda x: x*100.0)
    ])
cnn_lstm.summary()
optimizer = tf.keras.optimizers.Adam(lr=1e-5)
cnn_lstm.compile(loss=tf.keras.losses.Huber(), optimizer='Adam', metrics =['mae','mse'])



#%% fitting the model
cnn_lstm_history = cnn_lstm.fit(x_train, y_train,
                        validation_split=0.2,
                        epochs=100,
                        batch_size=batch_size
                        )

 #%% saving model to disk
# Uncomment when necessary
# lstm.summary()
cnn_lstm.save('models/lstm_60_90_42rmse.h5')
lstm.save('models/lstm_60_90_38rmse.h5')

#%% Loading the weights to the model
# cnn_lstm = tf.keras.models.load_model("models/conv_lstm_hyb-40rmse.h5")


#%% Visualising the model
import pydot
from keras.utils.vis_utils import plot_model
plot_model(cnn_lstm, to_file='lstm_plot.png', show_shapes=True, show_layer_names=True)
#%% plotting the loss of lstm
train_loss = lstm_history.history['loss']
val_loss = lstm_history.history['val_loss']

plt.plot(train_loss, 'r', label='training loss')
plt.plot(val_loss, 'g', label='validation loss')
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc=0)
plt.figure()
plt.show()




#%% Perform predictions on train data
n_future = 60

predictions_train = lstm.predict(x_train[-300:])   #7784
training_predictions = scaler_predict.inverse_transform(predictions_train)

#%% real values 

real_values = train[-300:,0]
real_values = scaler_predict.inverse_transform(real_values)


#%% Plotting everything
plt.plot(training_predictions[:2000,0], 'r', label='forcasted')
plt.plot(real_values[:2000], 'b', label='real')
#plt.axis([0,400,0,200])
plt.title('real and predicted')
plt.legend(loc=0)
plt.figure(figsize=(50,50))

plt.show()

#%%mape
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#%% find the rmse value and mape

mse = mean_squared_error(real_values, training_predictions[:,0])
rmse = np.sqrt(mse)
print(rmse)

mape = mean_absolute_percentage_error(real_values, training_predictions[:,0])
print(mape)

#%% Creating predictions

preds = cnn_lstm.predict(x_train[-500:])


#%% Inverse transforming the values and plotting the values

last_day = pd.to_datetime('23-03-2020')
selected_day = pd.to_datetime('20-06-2021')
days_until_today = (selected_day-last_day).days


future_predictions = scaler_predict.inverse_transform(preds)[:days_until_today,0]
plt.plot(future_predictions)   

#%% creating dates
def datetime_to_timestamp(x):
    '''
        x : a given datetime value (datetime.date)
    '''
    return time.strptime(x.strftime('%Y%m%d'), '%Y%m%d')

datelist_future = pd.date_range('04-05-2020', periods=days_until_today, freq='1d').tolist()
datelist_future = pd.to_datetime(datelist_future, dayfirst=True)

#%% Creating a dataframe with date of the predicted values

final = pd.DataFrame(data = future_predictions, columns=['AQI'], index=datelist_future)

today = future_predictions[-1]
print('AQI for the given day is:', today)


last_values = x_train[-50]










