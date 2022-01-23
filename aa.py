# Stacked AutoEncoders Movie Recommender

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn # PyTorch neural network module
import torch.nn.parallel # PyTorch neural network parallel computing module
import torch.optim as optim # PyTroch optimizer
import torch.utils.data # PyTorch utilities
from torch.autograd import Variable # PyTorch Stochastic Gradient Descent module

# How the Stacked AutoEncoder Movie Recommender works?
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

# Getting the total number of users and movies in the dataset
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0]))) # the total number of users is the highest user ID (column 0) either in the training set or the testing set (because both the training and the testing sets have all the number of users: IDs from 1 to max)
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1]))) # Same logic applies for the total number of movies

# Like RBM, Autoencoder expects a matrix of observations as an input. Therefore, we are transforming the training set and the testing set into two matrices, each contains all the users and all the movies
# The rows of each matrix are the users, and the columns are the movies. Each cell of the matrix corresponds to a rating of 
# this user to this movie. If the user doesn't have a rating for a particular movie, a 0 is inserted in this cell.
# Note: because we are going to use PyTorch later, we won't create a 2D numpy array, instead we will create a list of lists (because this is what PyTorch expects)
def convert(data):
    new_data = []
    for id_user in range(1, nb_users + 1): # user Ids start at 1 (not 0)
        id_movies = data[:, 1][data[:, 0] == id_user] # take all the movie ids (column 1), such that the corresponding user Id (column 0) is equal to this user Id
        user_ratings = data[:, 2][data[:, 0] == id_user] # take the ratings all the ratings of this user
        ratings = np.zeros(nb_movies) # To make sure to have zeros for the movies that the user hasn't rated, we initialize an array of zeros 
        ratings[id_movies - 1] = user_ratings # Then replace the zeros with the user's ratings in the right indices (note the -1 because the arrays are indexed starting with 0, but the movies indices start with 1)
        new_data.append(list(ratings)) # Adding this user's movie ratings to the matrix (Since we want to create a list of list and NOT a 2D Numpy array, we need at the end to convert the np array of ratings into a list )
    return new_data

# Applying to the convert method to the training set and the testing set
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into PyTorch Tensors
# Tensors are arrays that contain elements of a single data type. It's a multidimensional array (matrix). 
# We can still use Numpy arrays for the same purpose, but PyTorch Tensors are more efficient (Also more efficient than TensorFlow Tensors)
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network (the Autoencoder)
# To create an Autoencoder, we need to define: 
# the layers (how many layers), how many nodes in the layers, an activation function, a criterion, and an optimizer function
# Creating a class for the Stacked Autoencoder
class SAE(nn.Module): # The class inherits from the class Module of PyTorch's nn module
    def __init__(self, ):
        super(SAE, self).__init__()
        # Starting with the connection between the input vector (the observations) and the first hidden layer (the first encoded vector) by applying PyTorch.nn Linear Transformation
        # The first full connection in the Autoencoder
        # nb_movies: the input vector
        # 20: (Chosen experimentally) the number of nodes in the first hidden layer (encoded vector) of the Autoencoder -> Features that will be detected by the Autoencoder
        self.fc1 = nn.Linear(nb_movies, 20)
        # The second full connection in the Autoencoder
        # Connecting the first hidden layer to the second hidden layer
        # 20: the number of nodes in the first hidden layer
        # 10: (Chosen experimentally) the number of nodes in the second hidden layer
        self.fc2 = nn.Linear(20, 10) 
        # The third full connection in the Autoencoder (the first part of the decoding)
        # The input vector is the output of the previous hidden layer.
        # Here we are starting to decode, so we make it symmetric. Therefore, we chose the output number of nodes to be 
        # the same as the output of the first hidden layer
        self.fc3 = nn.Linear(10, 20)
        # The fourth (and last) full connection: the output layer (last part of the decoding), we are reconstructing the input vector
        # So the output vector number of nodes must be the same as the input vector (the number of movies)
        self.fc4 = nn.Linear(20, nb_movies)
        # Defining the activation function
        # Experimentally, we are chosing the Sigmoid activation function between the all four full connections of the Autoencoder
        self.activation = nn.Sigmoid() # This varaible is a reference to a method

    # The method does the forward propagation (where the encoding and the decoding take place)
    # x: Input vector which represents observations from one user (movie ratings of one user)
    def forward(self, x):
        x = self.activation(self.fc1(x)) # Results in the first encoded vector after running the input through the first full connection by applying the activation function
        x = self.activation(self.fc2(x)) # Second encoded vector
        x = self.activation(self.fc3(x)) # First decoded vector
        x = self.fc4(x) # Second decoded vector (the output vector). Note: In the final decoding step in which we construct the output vector (attempt to reconstruct the input vector), we don't apply the activation function: the vector goes directly into the last full connection
        return x # The output (the vector of predicted ratings)


# Creating the Stacked Autoencoder model
sae = SAE()
criterion = nn.MSELoss() # Defining a criterion of the loss function (which is in this case the Mean Square Error)
# The optimizer will apply Stochastic Gradient Descent after each Epoch to reduce the error (the loss)
# The RMSprop optimizer was experimentally chosen (Another option was the Adam optimizer, but the RMSprop led to better results)
# The first argument is all the parameters from the sae object
# lr: The learning rate (value was chosen experimentally)
# weight_decay: Value is chosen experimentally. The decay is used to reduce the learning rate after each Epoch in order to regulate the convergence
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the Stacked Autoencoder (SAE)
nb_epoch = 200 # Number of Epochs (Experimentally chosen)
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # A counter of the users who rated at least one movie
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) # Because PyTorch expects a batch and not a vector (tensor) as input
        # We transform the vector into a batch, by adding a fake dimension using the Variable class and the unsqueeze method
        # Cloning the input data
        target = input.clone()
        if torch.sum(target.data > 0) > 0: # (the .data property returns the tensor inside the Variable object). Returns the number of ratings > 0. In other words, if the user rated at least one movie, proceed, otherwise do skip this user
            output = sae(input) # The forward method can be called like this because it's callable in the parent class of the SAE class (nn.Module)
            target.require_grad = False # We won't need to use the target vector when we compute the gradient (while doing backpropagation --> see loss function and loss.backward() calls below)
            output[target == 0] = 0 # Setting the predicted ratings whose corresponding target values are zeros to zeros, 
            # because those output values (even if they are not zeros) should not participate in the error calculation 
            # or the weights adjustment by the optimizer. Therefore, we set them to zero to save some memory
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10) # We divide the total number of movies by the number of movies that have non-zero ratings (remember .data property returns the tensor from the Variable object)
            # we add the 1e-10 (a very small number that won't cause a difference in the computation by protects against the possiblity of dividing by zero)
            # The mean corrector represents the average of the error, but by only considering the movies that were rated. We do this
            # because the in the target vector, we considered only the movies that have non-zero rating
            loss.backward() # This call will tell in which direction the weights should be updated (whether the weights need to be increased or decreased)
            # loss.data contains the MSE between the output and the target, we adjust that with the mean corrector we computed above 
            # to take account only for the rated movies, then we take the square root because we are interested in calculating the RMSE (the State of the Art Error)
            train_loss += np.sqrt(loss.data * mean_corrector)
            s += 1. # Since this is a user who rated at least one movie, increment the relevant counter
            optimizer.step() # Updates the weights. The difference between the optimizer.step() method and the previous call
            # of loss.backward() is that loss.backward() decides whether to increase or decrease the weights, while optimizer.step() decides
            # the amount by which the weights will be increased or decreased
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s)) # The average train loss for this epoch (average: that's why we are dividing by the number of observations for which training took place, therefore dividing by s)
    # With more epochs, we can end up with better results, but on the other hand this may lead to difference in loss between the training and test sets

# Testing the Stacked Autoencoder (SAE):
# Same as the code of the training's inner loop
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0) # The input is still from the training set and not from the test set, because the input is the movie ratings of this user. We want to predict the ratings of the movies this user hasn't watched yet (in the input vector, the ratings with 0 values)
    # Then we will compare those predicted ratings to the corresponding ratings in the test set
    target = Variable(test_set[id_user]) # The target we will compare to is the user's ratings in the test set
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data * mean_corrector)
        s += 1.
print('Test loss: ' + str(test_loss / s)) # Our goal is to get a test loss that is below 1 star (means that a predicted movie rating is on average different from the actual rating by less than one star)
# E.g. a test loss of 0.95 is less than 1, which means less than 1 star, so it's a good movie recommender

