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
#%% Getting the data

real_data = pd.read_excel(r'.\data\NEWTEMP_last.xlsx')
all_data = real_data.copy()

# #%% getting rid of 2020
all_data = all_data.set_index(all_data['date'])
all_data = all_data.drop(['date'], axis=1)
# all_data = all_data.loc['2017-01-03':'2019-12-30']

# all_data = all_data.drop(['AQI'], axis=1)
#%% Data cleaning

# Dropping the last 5 rows of NaN values
# all_data = all_data.drop(all_data.index[-5:])

# Looking for null values

print(all_data.isnull().sum())

# Dropping rows with NaN values

all_data = all_data.dropna(axis=0)

# Null values are now removed. Print the null sum again to check.

# There are values with '*' and '#'. To remove that, I'm replacing them with NaN value and then filling those NaN values with mean of the column.

all_data = all_data.replace(dict.fromkeys(['*','**','***','#','##','###','####','#####'], 'NaN'))
print(all_data.isin(['*','**','***','#','##','###','####','#####']).sum())

# There are also values with '.', '/', '*' at the end. So stripping the data off of special charaters from the end
strip_cols = all_data.select_dtypes(object).columns
all_data[strip_cols] = all_data[strip_cols].apply(lambda x: x.astype(str).str.rstrip('/.+'))

# There are no nans now, but there are some nan values as strings. So converting all nan strings to np.nan
all_data = all_data.replace('NaN', np.nan)

# check the number of nans again
print(all_data.isnull().sum())

# Converting all the object type columns to float
all_data.iloc[:,[4,5,6,7,8,14,15]] = all_data.iloc[:,[4,5,6,7,8,14,15]].astype(float)

# Imputing all the nan values with mean of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
all_data.iloc[:,4:] = imputer.fit_transform(all_data.iloc[:,4:])

# check the number of nans and dtypes again

print(all_data.isnull().sum())
print(all_data.info())

#%% Handling location data
# I'm using Haversine distance approximation for changing the lat long to kms. 
# I'm fixing one of the lat long to a point and calculating all the station locations as a distance to that point.

fixed_lat = np.array(9.96521)
fixed_long = np.array(76.292)

all_data["distance"] = all_data.apply(lambda row : haversine.haversine((fixed_lat,fixed_long),(row["lat"], row["long"])), axis=1)
all_data = all_data.drop(['lat','long'], axis=1)

#%% Changing the date and cols

# all_data['date'] = pd.to_datetime(all_data.DATE) + all_data.HOUR.astype('timedelta64[h]')
# all_data['date'] = pd.to_datetime(all_data['date'], dayfirst=True)
# all_data = all_data.drop(['HOUR'], axis=1)

#%%
# datelist_train = all_data['DATE']
# cols = list(all_data)[1:22]
# all_data = all_data.drop(['DATE'], axis=1)
# # Last date is 2019/12/31
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
df = all_data.iloc[:, [7,8,10,11,12,13,14]]
corrmat = df.corr()
fig, ax = plt.subplots(figsize=(11,11))
sns.heatmap(corrmat)

#%% Finding any ouliers in the dataset

# Using boxplots to find outliers

figure, axes = plt.subplots(2,2)
axes[0,0].boxplot(all_data['RSPM'])
axes[0,1].boxplot(all_data['AQI'])
axes[1,0].boxplot(all_data['SO2'])
axes[1,1].boxplot(all_data['NO2'])
figure.tight_layout()

# Removing outliers

all_data['z-score'] = (all_data.AQI - all_data.AQI.mean())/all_data.AQI.std()
outliers = all_data[(all_data['z-score']<-3) | (all_data['z-score']>3)]
all_data = all_data[(all_data['z-score']>-3)& (all_data['z-score']<3)]
all_data = all_data.drop('z-score', axis=1)
#%% View all columns
print(all_data.columns)

#%% Sorting all the columns
all_data = all_data.reindex(columns=['AQI','Year', 'Month', 'Day', 'SO2', 'NO2', 'RSPM', 'SPM', 'Tmin', 'Tmax', 'P', 'SH', 'RH', 'SP', 'WSmax', 'Wsmin', 'POP', 'No I', 'MIM', 'MIS', 'SM', 'SS', 'MEM', 'MES', 'distance'])
all_data_copy = all_data.copy()

#%% Using RFE to find the best features for the model


estimator = RandomForestRegressor()

rfe = RFE(estimator=estimator, step=1, n_features_to_select=15)

xvals = all_data.iloc[:,0:]
yvals = all_data.iloc[:,0]

rfe= rfe.fit(xvals,yvals)

selected_features = pd.DataFrame({'Features':list(xvals.columns),
                                  'Ranking':rfe.ranking_ })
selected_features.sort_values(by='Ranking')

all_data = rfe.transform(xvals)

all_data.shape
all_data = pd.DataFrame(all_data)

#%% dataframe ranks plotting

selected_features = selected_features.sort_values('Ranking')
selected_features_styled = selected_features.style.background_gradient(low=1)
dfi.export(selected_features_styled, 'df_styled.png')
selected_features.dfi.export('df.png')

#%% Finding the trend and seasonality

all_data[0].plot(figsize=(20,6))
aqi = pd.DataFrame(all_data[0].values)
aqi.index = pd.to_datetime(aqi.index)
results = seasonal_decompose(aqi[0], period=150,model='multiplicative')
results.plot();


#%% Normalising the dataset

all_data = all_data.astype('float')
#%% saving dataframe for app

target_input = pd.DataFrame(all_data)
target_input.to_csv('models/input.csv')

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

#%% creating a sequence of datapoints 

x_train = []
y_train = []

n_future = 500   # Number of days we want top predict into the future
n_past = 50   # Number of past days we want to use to predict the future

for i in range(n_past, len(train) - n_future +1):
    x_train.append(train[i - n_past:i, 0:all_data.shape[1]])
    y_train.append(train[i + n_future - 1:i + n_future, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

print('x_train shape == {}.'.format(x_train.shape))
print('y_train shape == {}.'.format(y_train.shape))


#%% Creating an conv_LSTM model


batch_size = 32

lstm = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, all_data.shape[1]]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32,return_sequences=True)),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32, return_sequences=False, dropout=0.25)), 
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(500, activation='linear'),
    tf.keras.layers.Lambda(lambda x: x*100.0)
    ])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.Adam(lr=1e-8)
lstm.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics =['mae'])

lstm_history = lstm.fit(x_train, y_train,
                        validation_split=0.2,
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[lr_schedule]
                        )

#%% Finding the correct learning rate 

plt.semilogx(lstm_history.history["lr"], lstm_history.history["loss"])
plt.title('learning rate v loss')
plt.xlabel('learning rate')
plt.ylabel('loss')
#plt.axis([1e-8, 1e-4, 0, 10])

#%% The correct learning rate is found to be 3*1e-4. So training the same model with that learning rate for 500 epochs.
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
log_dir = "logs/"
# !tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=25, verbose=1)

def multiplier(x):
    return x*100.0

batch_size = 64

lstm = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(units=20,return_sequences=True),
    tf.keras.layers.SimpleRNN(units=20,return_sequences=True),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.SimpleRNN(units=32),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(500, activation='linear'),
    tf.keras.layers.Lambda(multiplier)
    ])

optimizer = tf.keras.optimizers.Adam(lr=5*1e-4)
lstm.compile(loss=tf.keras.losses.Huber(), optimizer='Adam', metrics =['mae'])


#%% fitting the model
lstm_history = lstm.fit(x_train, y_train,
                        validation_split=0.2,
                        epochs=50,
                        batch_size=batch_size,
                        callbacks=[rlr,es]
                        )

#%% saving model to disk
# Uncomment when necessary
# lstm.summary()
# lstm.save('models/conv_lstm_hyb-40rmse.h5')

#%% Loading the weights to the model
lstm = tf.keras.models.load_model("models/conv_lstm_hyb-40rmse.h5")


#%% Visualising the model

# from keras.utils.vis_utils import plot_model
# plot_model(lstm, to_file='lstm_plot.png', show_shapes=True, show_layer_names=True)
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

predictions_train = lstm.predict(x_train[-n_past:])   #7784
training_predictions = scaler_predict.inverse_transform(predictions_train)

#%% real values 

real_values = train[-n_past:,0]
real_values = scaler_predict.inverse_transform(real_values)


#%% Plotting everything
plt.plot(training_predictions[:,0], 'r', label='forcasted')
plt.plot(real_values, 'b', label='real')
#plt.axis([0,400,0,200])
plt.title('real and predicted')
plt.legend(loc=0)
plt.figure(figsize=(50,50))

plt.show()

#%% find the rmse value

mse = mean_squared_error(real_values, training_predictions[:,0])
rmse = np.sqrt(mse)
print(rmse)



#%% Creating predictions

preds = lstm.predict(x_train[-50:])


#%% Inverse transforming the values and plotting the values

last_day = pd.to_datetime('23-03-2020')
selected_day = pd.to_datetime('15-06-2021')
days_until_today = (selected_day-last_day).days


future_predictions = scaler_predict.inverse_transform(preds)[0,:days_until_today]
plt.plot(future_predictions[-100:])   

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
