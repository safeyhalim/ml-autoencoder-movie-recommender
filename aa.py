# AutoEncoders Movie Recommender

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn # PyTorch neural network module
import torch.nn.parallel # PyTorch neural network parallel computing module
import torch.optim as optim # PyTroch optimizer
import torch.utils.data # PyTorch utilities
from torch.autograd import Variable # PyTorch Stochastic Gradient Descent module

# How the AutoEncoder Movie Recommender works?
# The AutoEncoder will be trained so that it will identify some features from the ratings and identify similar users 
# based on their ratings (according to the features it detects). Then, we feed the AutoEncoder with a user and her ratings and based 
# on that and on the movie features it learned, it will predict the ratings (between 1 and 5) of this user for movies that she hasn't rated

# Importing the dataset
# Note: the separator of the dataset is double colon '::'
# The dataset file has no header row (that's why the header is set to None)
# engine is set to python (the default is 'c'), because the python engine is more complete
# (has all the features while the c engine doesn't). This ensures that the dataset is correctly imported
# encoding: latin-1 because some of the movies have some special characters (the default is utf-8)
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the testing set
# The 100k dataset is used for the training/testing sets. 
# The MovieLens 100k dataset is already split into 5 training/testing set pairs u1 to u5
# (each is a diffrent splits of the 100k ratings) to allow for a 5 fold cross-validation analysis
# For this implementation, we are not interested in the cross validation (we want just to focus on building the AutoEncoders), that's why we will just 
# use the training and the testing sets of u1 (The data is split into 80% training set and 20% test set)
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int') # converting the training_set from a dataframe into array, because we are going to use PyTorch tensors (which expect arrays)
# Same for the test set
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')
