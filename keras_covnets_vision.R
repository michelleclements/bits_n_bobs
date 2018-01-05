#' Covnets with keras
#' taken from deep learning with R
#' 
#' ## 2D covnet with MNIST
#' 
#' data prep
library(keras)
mnist <- dataset_mnist()
c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist
train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images / 255
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)


#' model architecture
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(3, 3), 
                activation = "relu",
                input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, 
                kernel_size = c(3, 3), 
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, 
                kernel_size = c(3, 3), 
                activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(units = 64, 
              activation = "relu") %>%
  layer_dense(units = 10, 
              activation = "softmax")


#' model compilation
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

#' model fit
model %>% fit(
  train_images, train_labels,
  epochs = 5, batch_size=64
)

#' model evaluation 
results <- model %>% evaluate(test_images, test_labels)
results


#' ## Covnet summary:  
#' * Difference is in the model architecture.  
#' * Convoluational layers learn local patters in the feature input space (data), rather than the global patterns learnt in dense layers.  
#' * The kernel_size tells the layer what size to look at. above is 3x3 squares.  
#' * Patterns can then be recognised anywhere in the image and not just e.g. in the bottom corner. They are said to be translation invariant.  
#' * Covnets can learn spatial hierarchies - each successive layer zooms in on the image a bit more.  
#' 
#' 
#' ### Covnet details: 
#' * Convoulations operate over feature maps - 3D tensors.  
#' * Feature maps have two spatial axes - height and width - and a depth/channels axis.  
#' * Black and white images have a single channel.  
#' * RGB impages have three channels and the dimension of the depth axis is 3.  
#' * The conv layer produces an output feature map.  
#' * The output feature map is a 3D tensor with width, height and filter.  
#' * Filters encode certain features within the data.  
#' * The first layer above takes in feature map of size 28x28x1 tensor and outputs feature map of size 26x26x32.  
#' * 26 is the number of 3 pixel groups along a single length 28 row (from kernel_size).  
#' * 32 is the number of filters specified.  
#' * Each filter is a 26x26 grid of values. This is the response map of the filter over the input.  
#' * Feature map: the colleciton of 2D spatial maps produced by each filter over the input.  
#' 
#' 
#' The main parameters varied in covnets are:
#' * The size of the patches looked at (kernel_size). Typically 3x3 or 5x5.   
#' * The number of filters. Also known as the depth of the output feature map. 
#' 
#' 

