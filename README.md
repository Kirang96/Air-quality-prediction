# End-to-end Air-quality-prediction project
Air quality prediction using deep learning: A case study of Kochi

### TABLE OF CONTENTS:

- [Demo](https://github.com/Kirang96/Air-quality-prediction#demo)
- [What is the project about?](https://github.com/Kirang96/Air-quality-prediction#what-is-the-project-about)
- [Why Kochi and how is air quality relevant?](https://github.com/Kirang96/Air-quality-prediction#why-kochi-and-how-is-air-quality-relevant)
- [Data collection](https://github.com/Kirang96/Air-quality-prediction#data-collection)
- [Project structure](https://github.com/Kirang96/Air-quality-prediction#project-structure)
- [Hybrid 1D CNN LSTM network](https://github.com/Kirang96/Air-quality-prediction#hybrid-1d-cnn-lstm-network)
- [Deployment using Heroku](https://github.com/Kirang96/Air-quality-prediction#deployment-using-heroku)
- [Tools and technologies used](https://github.com/Kirang96/Air-quality-prediction#tools-and-technologies-used)
- [License](https://github.com/Kirang96/Air-quality-prediction#license)

## DEMO

This demonstrates the [hosted AQI website](http://aqi-kochi.herokuapp.com) making predictions with the given inputs. 

![AQI website making predictions](https://user-images.githubusercontent.com/29313141/123675222-1a061b80-d860-11eb-8031-1b477584e68e.png)


## WHAT IS THE PROJECT ABOUT?

- The quality of air we breathe has a huge impacts on our physical and mental health. By taking small measures such as closing windows, avoiding exercises in high pollution hours and using face masks, we will be able to protect ourselves and our families from dangers of air pollution.
- This project allows it's users to forcast air quality index (AQI) in specified regions around Kochi.
- It also explains the adverse effects the poor quality air might cause to a person so that he could protect himself accordingly.
- This website is hosted on Heroku platform and is running using a model trained on top of Keras API.
- The model used is a hybrid CNN LSTM network trained with a dataset of 11046 rows and 16 columns and trained for 100 epochs.

## WHY KOCHI AND HOW IS AIR QUALITY RELEVANT?

- The air we breathe can have a huge impact on our lives.
- Air pollution can cause both short term and long term effects to our body
- When we breathe polluted air, pollutants can enter the bloodstream and be carried to our internal organs such as the brain. This can    cause severe health problems such as asthma, cardiovascular diseases and even cancer and reduces the quality and number of years of life. 
- Air pollution not only harms human beings, but the planet Earth as a whole.

![kochi_map](https://user-images.githubusercontent.com/29313141/123814245-2c8d5d00-d913-11eb-9161-16bc361adf6d.png)

- Kochi is a major port city on the Malabar Coast of India bordering the Laccadive Sea, which is a part of the Arabian Sea.
- It is the most densely populated city in Kerala. It has very densely populated urban agglomeration.
- It also has a geographical area of 285 km2 with population density of 4876 persons/km2 which is 13 times average population of India.
- Kochi has also become a rapidly growing industrial, economic, transportation and cultural center in India.
- Hence such a densely populated area should be carefully studied so that the people in Kochi could take better care of themselves.

## DATA COLLECTION

- Under National Air Monitoring Programme (NAMP) by Central Pollution Control Board (CPCB), there are 7 monitoring stations in the area by Kerala State Pollution Control Board (KSPCB).
- 2 Commercial, 2 Residential and 3 Industrial.
- Area boundary was fixed based on the locations of the seven monitoring stations.
- The study area witnessed massive construction of Kochi Metro and other several bridges, roads and shopping malls.

| NAME OF THE STATION  | LAT | LONG | TYPE |
| -------------------- | --- | -----|------|
| SOUTH | 9.96519|76.29201|COMMERCIAL|
| M.G ROAD  | 9.96291  | 76.28591 | COMMERCIAL |
| VYTILLA  | 9.95823  | 76.32523 | RESIDENTIAL |
| IRUMPANAM  | 9.98793  | 76.35069 | INDUSTRIAL |
| KALAMASSERY  | 10.0502  | 76.31237 | INDUSTRIAL |
| ELOOR 1  | 10.07969  | 76.29677 | RESIDENTIAL |
| ELOOR 2  | 10.07435  | 76.30394 | INDUSTRIAL |

![KOCHI - MAP](https://user-images.githubusercontent.com/29313141/123958249-7127ff80-d9ca-11eb-9db4-667ca2cfd7a0.png)

The data ranges from 2017 to 2020 and are of 4 different types which are related to the air quality.
- **Air quality** data includes the amount of 'SO2', 'NO2', 'PM2.5' and 'PM10' in the atmosphere.
- **Meteorological** data includes Temperature, wind speed, surface pressure, precipitation and humidity.
- **Industry** data includes number of industries and number of manufacturing and service units.
- And the **population** data of each area is also included.

The data was collected from different sources:

| TYPE OF DATA  | SOURCE |
| ------------- | ------ |
| AIR QUALITY | KERALA POLLUTION CONTROL BOARD|
| METEOROLOGICAL | NASA POWER GRID|
| INDUSTRY | KERALA POLLUTION CONTROL BOARD|
| POPULATION | CENSUS 2011 |

## PROJECT STRUCTURE

- Data cleaning and engineering
   - Handling null values: 
       - by removal of rows
       - by imputation method
       - resampling the data and interpolating the values
   - Handling outliers:
     - outlier detection by plotting box plots
     - outlier removal using z-score elimination method
   - Handling the geographical coordinate data:
      - LAT and LONG columns are used to create different clusters by using KMeans clustering
- Visualising the data
   - Plotting histograms of different columns
   - Creating a correlation heat map matrix
   - Finding the trend and seasonality
 - Feature selection
    - RFE is used with Random Forest regressor as the estimator to select the important features
 - Data standardisation using StandardScaler
 - Data splitting
    - Data is split into train, test and split initially. But the whole data was used in the final training since it is a time series problem and the recent data are important for new predictions
 - Data sequencing
     - Data is converted into a supervised problem by sequencing the data into chunks of x and y values
     - Look back will have 90 values which will be used to predict 120 values.
 - Fixing the model parameters and metrices
    - Loss function used is Huberloss.
    - Optimization used is Adam whose learning rate is fixed by using LearningRateScheduler for the final model.
    - activation functions used for hidden layers are relu for FFN and default tanh for all RNN networks. All output layers has linear activation functions.
    - epochs are set to 100 with early stopping.
    - MAE and MSE are used for epoch wise training.
    - RMSE and MAPE are used as model metrics.
 - Creating and training models
     - Feed forward network (Baseline model)
     - Simple RNN network
     - LSTM network
     - Network with 1D CNN as input layer with LSTM layers
 - Testing the model on test data and plotting the results
 - Saving the model to disk.

## Hybrid 1D CNN LSTM network

- Finalised model is the hybrid 1D CNN LSTM network.
- This model works well with the time series data
- CNN layer is able to detect the features in the data with the multiple number of filters used.
- These features are then input to the hidden LSTM layers which uses a cell state. LSTM layers will be able to detect the inter-dependencies between multiple columns in the dataset and save the relevant information in the cell state.

## Deployment using Heroku

Deployment was done using Heroku free package which offer upto 500MB storage space.
The required files for heroku deployment are:
 - Python file containing the source code of the program
 - requirements file with all the dependencies to be installed on the heroku server.
 - Procfile which shows heroku which file to run first
 - models required during the execution

Steps followed during deployment are according to the Heroku user guide.

## Tools and technologies used
![python-programming-language](https://user-images.githubusercontent.com/29313141/124320295-3aecaa80-db99-11eb-8f13-fa8d20385277.png)
![bootstrap](https://user-images.githubusercontent.com/29313141/124320192-0ed12980-db99-11eb-8db2-c95b18e8ab2e.png)
![flask](https://user-images.githubusercontent.com/29313141/124320195-10025680-db99-11eb-935b-a9492d21a240.png)
![gunicorn](https://user-images.githubusercontent.com/29313141/124320196-109aed00-db99-11eb-9043-e000eefc4a26.png)
![spyder](https://user-images.githubusercontent.com/29313141/124320203-11cc1a00-db99-11eb-8f69-49f569723d03.png)
![heroku](https://user-images.githubusercontent.com/29313141/124322169-9b311b80-db9c-11eb-9105-5348c5e106a8.png)
![html_css](https://user-images.githubusercontent.com/29313141/124320199-11338380-db99-11eb-8805-448038a27048.jpg)
![Tensorflow and keras](https://user-images.githubusercontent.com/29313141/124320205-1264b080-db99-11eb-93f8-8ffcd5ea37bb.jpg)

## License

### Mozilla Public License 2.0

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
Permissions of this weak copyleft license are conditioned on making available source code of licensed files and modifications of those files under the same license (or in certain cases, one of the GNU licenses). Copyright and license notices must be preserved. Contributors provide an express grant of patent rights. However, a larger work using the licensed work may be distributed under different terms and without source code for files added in the larger work.
