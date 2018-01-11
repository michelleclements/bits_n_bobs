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
#' * Covnets require less image pre-processing than other options.  
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
#' * Feature map: the collection of 2D spatial maps produced by each filter over the input.  
#' * The covnet slides over all squares on the input map.  
#' * Within each square, the window height, window depth and input depth are extracted.  
#' * The square is then transformed into a 1D vector of length output_depth.  
#' * The transformation is by the convolutional kernel: a tensor product with teh same learned weight matrix.  
#' * The output vectors from all of the squares are then assembled into a 3D output feature map.  
#' * Note that information from one pixel is now in multiple adjacent points of the feature map.  
#' * You can use padding to get the output feature map the same height and width as the input. Use padding = "same".  
#' * Multiple covnets keep bringing in local information from a little bit further out? 
#'  
#' 
#' 
#' The main parameters varied in covnets are:
#' * The size of the patches looked at (kernel_size). Typically 3x3 or 5x5.   
#' * The number of filters. Also known as the depth of the output feature map. Tends to be more in deeper layers.   
#' * Also, padding = same to get output height and width the same as the input.  
#' * Note that as the network gets deeper, the height and width gets less but the number of filters increases.  
#' 
#' 
#' ### Max pooling layer  
#' * Max pooling reduces the height and width.  
#' * It takes its input feature map and creates windows across the feature space.  
#' * e.g. a 2x2 pool size above will lay 2x2 grids across the whole 26x26 space.  
#' * It will then take the maximum value of all 4 blocks that are in each grid.  
#' * e.g. above will output a 13x13 grid. 
#' * cov layers use stride = 1, so that there's overlap. Max pooling uses 2x2 and stride = 2 so there's no overlap.  
#' * Average pooling can also be used, but max tends to be more effective.  
#' 
#' Max pooling has two effects: 
#' 1. It reduces the number of co-effecients to process and helps to reduce overfitting.   
#' 2. It makes each node look at a relatively bigger space which induces spatial hierarchies. 
#' 
#' ### End the cov net with a classifier   
#' * use layer_flatten to bring dimensions down.  
#' * then optionally add a layer_dropout
#' * then layer_dese with activation relu and at least as many units as filters in the cov2d layer  
#' * then layer_dense with the required output activation function.  
#' 
#' ### Improving the power of covnets
#' * covnets can be trained with relatively small amounts of data if the problem is simple.  
#' 
#' Data augmentation can help mitigate overfitting:
#' * Augment the data by adding in samples that are randomly changed from original samples (e.g. squishing, rotating).  
#' * Samples are then not independent, but additional info might help the model. 
#' * Might help to add a layer_dropout after layer_flatten to avoid overfitting.  
#'  
#' Using pre-trained networks (e.g. ImageNet) can also help to improve models.  
#' * feature extraction with a pre-trained network - add a new classifier onto an existing trained convolutional base.  
#' * fine-tuning a pre-trained network - fine tune the final covnet after doing feature extraction above.       


#' end of chapter 5 has some stuff on visualising covnets.  
