# Keras-library
Regression Model using Keras library
Regression Models with Keras
Introduction
As we discussed in the videos, despite the popularity of more powerful libraries such as PyToch and TensorFlow, they are not easy to use and have a steep learning curve. So, for people who are just starting to learn deep learning, there is no better library to use other than the Keras library.

Keras is a high-level API for building deep learning models. It has gained favor for its ease of use and syntactic simplicity facilitating fast development. As you will see in this lab and the other labs in this course, building a very complex deep learning network can be achieved with Keras with only few lines of code. You will appreciate Keras even more, once you learn how to build deep models using PyTorch and TensorFlow in the other courses.

So, in this lab, you will learn how to use the Keras library to build a regression model.

Table of Contents
Download and Clean Dataset
Import Keras
Build a Neural Network
Train and Test the Network

Download and Clean Dataset
Let's start by importing the pandas and the Numpy libraries.

import pandas as pd
import numpy as np
We will be playing around with the same dataset that we used in the videos.

The dataset is about the compressive strength of different samples of concrete based on the volumes of the different ingredients that were used to make them. Ingredients include:

1. Cement

2. Blast Furnace Slag

3. Fly Ash

4. Water

5. Superplasticizer

6. Coarse Aggregate

7. Fine Aggregate

Let's download the data and read it into a pandas dataframe.

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()
So the first concrete sample has 540 cubic meter of cement, 0 cubic meter of blast furnace slag, 0 cubic meter of fly ash, 162 cubic meter of water, 2.5 cubic meter of superplaticizer, 1040 cubic meter of coarse aggregate, 676 cubic meter of fine aggregate. Such a concrete mix which is 28 days old, has a compressive strength of 79.99 MPa.

Let's check how many data points we have.
concrete_data.shape
So, there are approximately 1000 samples to train our model on. Because of the few samples, we have to be careful not to overfit the training data.

Let's check the dataset for any missing values.

concrete_data.describe()
concrete_data.isnull().sum()
The data looks very clean and is ready to be used to build our model.

Split data into predictors and target
The target variable in this problem is the concrete sample strength. Therefore, our predictors will be all the other columns.

concrete_data_columns = concrete_data.columns
​
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

Let's do a quick sanity check of the predictors and the target dataframes.

predictors.head()
target.head()
Finally, the last step is to normalize the data by substracting the mean and dividing by the standard deviation.

predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()
Let's save the number of predictors to n_cols since we will need this number when building our network.

n_cols = predictors_norm.shape[1] # number of predictors


Import Keras
Recall from the videos that Keras normally runs on top of a low-level library such as TensorFlow. This means that to be able to use the Keras library, you will have to install TensorFlow first and when you import the Keras library, it will be explicitly displayed what backend was used to install the Keras library. In CC Labs, we used TensorFlow as the backend to install Keras, so it should clearly print that when we import Keras.

Let's go ahead and import the Keras library
import keras
As you can see, the TensorFlow backend was used to install the Keras library.

Let's import the rest of the packages from the Keras library that we will need to build our regressoin model.

from keras.models import Sequential
from keras.layers import Dense

Build a Neural Network
Let's define a function that defines our regression model for us so that we can conveniently call it to create our model.

# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
The above function create a model that has two hidden layers, each of 50 hidden units.



Train and Test the Network
Let's call the function now to create our model.

# build the model
model = regression_model()
Next, we will train and test the model at the same time using the fit method. We will leave out 30% of the data for validation and we will train the model for 100 epochs.

# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)
You can refer to this link to learn about other functions that you can use for prediction or evaluation.

Feel free to vary the following and note what impact each change has on the model's performance:

Increase or decreate number of neurons in hidden layers
Add more hidden layers
Increase number of epochs
