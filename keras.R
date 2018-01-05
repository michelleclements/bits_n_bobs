#' # Keras in R
#' Notes taken from Deep Learning with R by Francois Chollet & J.J. Allaire

library(tidyverse)
library(keras)

#' ## MNIST data prep ---------------------------------------------------------

#' get test and train from keras package
mnist <- dataset_mnist()

train_images <- mnist$train$x
train_labels <- mnist$train$y

test_images <- mnist$test$x
test_labels <- mnist$test$y

# can also use c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist  

#' ### look at the structure of the data  

#' #### train_images  
#' data is an array
class(train_images) 
#' 3D array
length(dim(train_images)) 
#' 3D array - 60k samples, each 28*28
dim(train_images) 
#' contains integers
typeof(train_images) 
#' integers range from 0 to 255
summary(train_images) 

#' we can view one digit
digit <- train_images[23, , ] 
plot(as.raster(digit, max = 255))


#'#### train labels
#' data is an array
class(train_labels) 
#' 1D array - 60k samples
dim(train_labels) 
#' contains integers
typeof(train_labels) 
#' integers range from 0 to 9
summary(train_labels) 

#'#### test dataset
#' same structure as training but 10k labels
str(test_images) 
str(test_labels)

#'### tranformations to the data
#' Data has to be in tensor form.  

#' reshape the array to preserve the structure of the 28*28 image
train_images <- array_reshape(train_images, c(60000, 28 * 28))

#' divide by 255 (max value in dataset) so all values are between zero and 1. 
#' small input values help the model behave better
train_images <- train_images / 255

#' data is an matrix
class(train_images) 
#' 60k * 784 matrix
dim(train_images) 
#' contains doubles
str(train_images) 

#' do the same to the test images
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255

#' make the labels categorical
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)


#' ## Basic network on MNIST dataset -----------------------------------------------------------
#' ### Network architecture

network <- keras_model_sequential() %>% # going to be a sequential model
  layer_dense(units = 512, # dense layer with 512 nodes in first layer
              activation = "relu", # transformation function in first layer is relu 
              input_shape = c(28 * 28) # expect each data sample to be 28*28 
              ) %>%
  layer_dense(units = 10, # second layer has 10 units (one for each digit)
              activation = "softmax" # transformation function is softmax  - create P(being in class) for 10 classes
              )


#' ### Network compilation

network %>% compile(
  optimizer = "rmsprop", # use rmsprop optimiser in gradient descent (pretty standard)
  loss = "categorical_crossentropy", # minimise categorical cross entropy with gradient descent 
  metrics = c("accuracy") # also measure the accuracy at each step 
)


#' ### Network fit

network %>% fit(train_images, # use train_images as predictor variables
                train_labels, # use train_labels as response variable
                epochs = 5, # run optimiser for 5 passes through the network
                batch_size = 128 # run separate batches each with 128 samples
                )


#' ### Network Output

#' Getting metrics

network %>% evaluate(test_images, test_labels)

#' Getting predictions (for first 10 only)
network %>% predict_classes(test_images[1:10, ])


#' ## Understanding network parameters -----------------------------------------------------------
#' ### Data prep  
#'  * Data must be passed to keras as tensors  
#'  * Numeric variables should be either scaled between 0-1 or normalised to have a mean = 0 and sd = 1.    
#'  * Training, validation and test data should be scaled using mean and sd from training data.  
#'  * Categorical variables should be one-hot encoded (use to_categorical() function).  
#'  * Missing values: input as 0 - as long as 0 isn't already meaningful. Network will learn that missing isn't informative.  
#'  * You might have to simulate missing values if there aren't any in the training dataset, but they will be expected in test or live data.  
#'  * Deep neural nets don't really require feature engineering - they should find the features themselves.  
#'  
#' Example of scaling  
# mean <- apply(train_data, 2, mean)  
# std <- apply(train_data, 2, sd)  
# train_data <- scale(train_data, center = mean, scale = std)  
# test_data <- scale(test_data, center = mean, scale = std)  
#' 
#' 
#' ### Network architecture
# network <- keras_model_sequential() %>% 
#   layer_dense(units = 512, 
#               activation = "relu", 
#               input_shape = c(28 * 28) , 
#               kernel_regularizer = regularizer_l2(0.001)
#   ) %>%
#   layer_dropout(rate = 0.3) %>% 
#   layer_dense(units = 10,
#               activation = "softmax" 
#   )

#' #### Types of layers
#' * Dense layers (layer_dense): all possible connections between nodes. Used for standard sample and features data including sentiment analysis. 
#' * Recurrent layers (e.g. layer_lstm): Used for 3D tensors e.g. timeseries data
#' * Convolution layers (e.g. layer_cov_2): Used for 4D tensors  
#' * Drop out layers (layer_dropout): used to prevent overfitting (see below)
#' 
#' #### Activation functions
#' * Are functions applied to weights x param values + bias (intercept) on each node.  
#' * Activation functions are non-linear and are required to create interactions in the model.  
#' * Generally can use relu for intermediate layers.  
#' * "relu": rectified linear unit - makes negative values zero and leaves postives values as is.  
#' * The final layer activation function must always output the data in the correct format.  
#' * "sigmoid": squashes all values to be between zero and one, which can be interpreted as a probability. Negative values are < 0.5 and positive values > 0.5 
#' * "softmax": outputs probabilities of belonging to number of categories in the layer.  
#' * no activation function: outputs linear response (regression). Use layer_dense(units = 1)
#' 
#' #### Number of layers and units in a layer  
#' * Can be thought of as how much freedom you're giving the network.
#' * More units leads to a more complicated network but potentially computationally expensive and risks overfitting.
#' * Too few units in a layer creates an information bottleneck.
#' * Models with less samples should have a smaller network to prevent overfitting.  
#' * Sometimes helpful to overfit and then pare back.  
#' * See chapter 4 for more details.
#' 
#' #### Input shape
#' * Only needs to be stated for the first layer. 
#' * For vectors, use e.g. c(1000) if the training set has 1000 vars.  
#' 
#' #### Kernel regularizer
#' * Reduces size of weights to prevent overfitting (large weights are probably random variation).  
#' * Makes the weight values more regular.  
#' * Add a cost to the loss function associated with the size of the weights
#' * L1 regularization—The cost added is proportional to the absolute value of the weight coefficients (the L1 norm of the weights).   
#' * L2 regularization—The cost added is proportional to the square of the value of the weight coefficients (the L2 norm of the weights).  
#' * L2 regularization is also called weight decay - weight decay is mathematically the same as L2 regularization.  
#' * Regularizer_l2(0.001): every coefficient in the weight matrix of the layer will add 0.001 x weight_coefficient_value to the total loss of the network. 
#' * The loss will be much higher for training than test as penalty is only added during training.  
#' * Add both with regularizer_l1_l2(l1 = 0.001, l2 = 0.001).  
#' 
#' 
#' #### Layer dropout
#' * Layer drop out randomly sets a certain proportion of the output features of the layer to zero. 
#' * Stops particular nodes collaborating together to fit spurious interactions and features.      
#'    
#' ### Network compilation  
# network %>% compile(
#   optimizer = "rmsprop", 
#   loss = "categorical_crossentropy", 
#   metrics = c("accuracy") 
# )
#'   
#' #### Optimizer 
#' Optimization is by stochastic gradient descent. Params control type of SGD.  rmsprop is normally fine.   
#'   
#' (Mini-batch) SGD overview    
#' Aim: to find the weights that best fit the data:  
#' * Draw a batch of training samples and response variable at random (controlled by batch_size in fit()).   
#' * Run the network with random weights (parameters) to predict response var.  
#' * Compute the loss on the network - how far predicted is away from response - using loss function from compile().  
#' * Compute the gradient of the loss with respect to the network weights using differentiation (a backward pass).     
#' * Move the parameters down the slope a bit (update the weights) to reduce the loss.   
#' * Run the network with these updated weights/parameters. An epoch is complete when the weights are updated and the loss calculated.    
#' * Keep repeating computing the loss, moving weights down the slope, until reached the required number of epochs (controlled by epochs in fit()).   
#'     
#' Notes    
#' * Big batches are more accurate but also more expensive - need to find a balance.   
#' * The requirement to find the slope means the loss functions have to be differentiable with respect to the weights.    
#' * Change the learning rate (how big a jump down the slope) using e.g. optimizer = optimizer_rmsprop(lr = 0.0001).   
#'       
#'     
#' #### Loss function     
#' * How success will be measured. The value to be minimised.    
#' * It's normally fine just to use standard loss functions.    
#' * "binary_crossentropy": two-class classification.  
#' * "categorical_crossentropy": multi-class classification with one-hot encoded response variable (measures the distance between two probability distributions).  
#' * "sparse_categorical_crossentropy": multi-class classification with integer response variable.   
#' * "mse": mean squared error for regression.   
#' * CTC: sequence learning.     
#'   
#'     
#' #### Metrics
#' * Other values - not optimised - to measure in the model.  
#' * "accuracy": for categorical variables.  
#' * "mae": mean absolute error for regression.   
#'   
#'     
#'   
#' ### Network fit
# network %>% fit(train_images, 
#                 train_labels,
#                 epochs = 5, 
#                 batch_size = 512, 
#                 validation_data = list(x_val, y_val)
# )
#'
#' #### Epochs
#' * How many backwards passes through the data to do.  
#' * Too small and the model will miss important features.  
#' * Too big and the model will overfit. 
#' * Use validation data to see when the model started overfitting.   
#' 
#' 
#' #### Validation data  
#' * Measure loss and metrics on validation data also.  
#' * Make sure validation data really is independent!
#' * Need to have created x_val and y_val already.  
#' * Watch about overfitting to the validation data.   
#' * See p87 & 99 for k-fold validation  (mainly used when you don't have enough data to create a reliable validation dataset). 
#' 
#' 
#' ### Network output
#' * Can plot training and validation metrics using plot(history). str(history) lets you see params over epochs. Can call as.data.frame(history).
#' * model %>% predict(x_data) predicts y values.
#' 
#' 
#' ### Summary of last activation layer and loss function
#' * Binary classification: activation = sigmoid, loss = binary_crossentropy.  
#' * Multiclass, single-label classification: activation =  softmax, loss = categorical_crossentropy.  
#' * Multiclass, multilabel classification: activation =  sigmoid, loss = binary_crossentropy.  
#' * Regression to arbitrary values: activation =  None, loss = mse.  
#' * Regression to values between 0 and 1: activation =  sigmoid, loss = mse or binary_crossentrop.  
#' 
#' 
