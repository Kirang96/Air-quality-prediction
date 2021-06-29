# End-to-end Air-quality-prediction project
Air quality prediction using deep learning: A case study of Kochi

Table of contents:

- Demo
- What is the project about?
- Why Kochi and how is air quality relevant?
- Data collection
- Project structure
- Hybrid 1D CNN LSTM network
- Deployment using Heroku
- Directory tree
- Tools and technologies used
- Credits
- License

## Demo

This demonstrates the [hosted AQI website](http://aqi-kochi.herokuapp.com) making predictions with the given inputs. 

![AQI website making predictions](https://user-images.githubusercontent.com/29313141/123675222-1a061b80-d860-11eb-8031-1b477584e68e.png)


## What is the project about?

- The quality of air we breathe has a huge impacts on our physical and mental health. By taking small measures such as closing windows, avoiding exercises in high pollution hours and using face masks, we will be able to protect ourselves and our families from dangers of air pollution.
- This project allows it's users to forcast air quality index (AQI) in specified regions around Kochi.
- It also explains the adverse effects the poor quality air might cause to a person so that he could protect himself accordingly.
- This website is hosted on Heroku platform and is running using a model trained on top of Keras API.
- The model used is a hybrid CNN LSTM network trained with a dataset of 11046 rows and 16 columns and trained for 100 epochs.

## Why Kochi and how is air quality relevant?

- The air we breathe can 
- Air pollution can cause both short term and long term effects to our body
- When we breathe polluted air, pollutants can enter the bloodstream and be carried to our internal organs such as the brain. This can    cause severe health problems such as asthma, cardiovascular diseases and even cancer and reduces the quality and number of years of life. 
- Air pollution not only harms human beings, but the planet Earth as a whole.

![kochi_map](https://user-images.githubusercontent.com/29313141/123814245-2c8d5d00-d913-11eb-9161-16bc361adf6d.png)

- Kochi is a major port city on the Malabar Coast of India bordering the Laccadive Sea, which is a part of the Arabian Sea.
- It is the most densely populated city in Kerala. It has very densely populated urban agglomeration.
- It also has a geographical area of 285 km2 with population density of 4876 persons/km2 which is 13 times average population of India.
- Kochi has also become a rapidly growing industrial, economic, transportation and cultural center in India.
- Hence such a densely populated area should be carefully studied so that the people in Kochi could take better care of themselves.

## Data collection



