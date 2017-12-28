#' # Decision Trees
#' Michelle Clements
#' 
#' `r Sys.Date()`
#' 
#' 
#' ## Overview & resources
#' This is my notes on decision trees
#' 
#' I have borrowed heavily from 
#' * Hands-On machine learning with scikit-learn and tensorflow by Aurelien Geron
#' * https://rstudio-pubs-static.s3.amazonaws.com/195428_16074a4e980747c4bc05af6c0bb305a9.html
#' This one could be useful: https://rpubs.com/saqib/rpart
#' 
#' 
#' ## Building a single decision tree using Caret
#' First load the required libraries
library(tidyverse) # for general data manipulation
library(caret) # for machine learning
library(rpart.plot) # for plotting decision trees
theme_set(theme_classic(base_size = 12)) # setting options for graphs


#' We're going to use the mushroom dataset to look at regression trees
#' We're just going to read in the training dataset created previously
#' To keep things simple, we're just going to use petal length and petal width as predictor vars
mush <- read_csv("data/training_raw.csv")


#' run a very simple tree
tree_simple <- train(class ~., # class is the response variable
                     method = "rpart", 
                     data = mush
                     )


#' visualised using prp()
prp(tree_simple$finalModel, 
    box.palette = "Blues", # colour bluein
    digits = 3 # display 3 digits
    )

#' look at output using name$finalModel
#' the * shows the terminal (or leaf) nodes which are the final bucket  
#' You can follow down the numbered nodes to the very bottom  
#' the first part shows the decision rule (or 'root' for the very bottom)  
#' The next number shows how many sampes fall into this bucket (e.g. leaf 6 has 54 samples)  
#' The following number shows how many samples are misclassified (e.g. leaf 6 has 5 of 54 samples misclassified)  
#' The text shows the classification assigned to the leaf (e.g. node 6 is assigned 'versicolor')  
#' The numbers in brackets show what proportion of samples in the bucket are of each class - this can be seen as the probability that any given sample in this bucket is each class.   
#' 

tree_simple$finalModel

print(tree_simple)

iris_results <- iris_petal %>% 
  mutate(predict_simple = predict(tree_simple), # use prediction function to get predictions
         correct_simple = predict_simple == Species # create a true/false for whether prediction was correct
         )


#'
#'graph predicitons

ggplot(iris_results, 
       aes(x = Petal.Length, 
           y = Petal.Width, 
           colour = correct_simple, 
           shape = Species)) +
  geom_point()


#'## Notes on basic decision trees
#'* decision trees don't require feature scaling or centering  
#' The tree above uses the CART algorithm - Classification And Regression Tree
#'*  CART  only produces yes/no options - a node can only have two children  
#' CART searches for the variable and threshold that produces the purest two subsets
#' It stops when it reaches the maximum depth parameter or when it cannot find a split that will reduce impurity
#' 
#' 
#' 
#' # Ensemble models
#' Ensemble models are when different models are combined. 
#' It's a popular addition to regression trees. There are two main ways to do ensemble models:  
#' 1. Train different types of models on the same data set - models have to be as different (independent) as possible
#' #' Combining the results can be done in a two main ways:  
#' a. Hard voting: each model gets a 'vote' and the class with the most vote wins
#' b. Soft voting: average probability of each class across all models and pick the class with the highest average (needs probs to be outputted)
#' 
#' 
#' 2. Train the same type of model on different subsets of the dataset
#' 
#' Sampling can be:
#' a. with replacement: bagging (bootstrap aggregating)
#' b. without replacement: pasting
#' 
#' Prediction can then be done onece the models are aggregated:
#' 1. Statistical mode (same as hard voting) for classification
#' 2. Average for regression
#' 
#' 
tree_bagged <- train(Species ~., # species is the response variable
                     method = "treebag", 
                     data = iris_petal
)

tree_bagged$finalModel

iris_results <- iris_results %>% 
  mutate(predict_bagged = predict(tree_bagged), # use prediction function to get predictions
         correct_bagged = predict_bagged == Species # create a true/false for whether prediction was correct
  )

#' note that you can't print out one tree as there's not a single one
#' 
#' 

#' ## Random forests
#' Random Forests use bagging/pasting. 
#' However, random forests don't search for the single best split in the data.
#' Instead, they search for the best split among a random subset of features 
#' 
#' 
tree_randomForest <- train(Species~.,
                           data = iris_petal,
                           method = "rf", # random forest
                           trControl = trainControl(method = "cv", # resampling method is cross validation 
                                                    number = 5), # 5 splits in the corss validation
                           prox = TRUE,
                           allowParallel = TRUE)
print(tree_randomForest)

print(tree_randomForest$finalModel)

iris_results <- iris_results %>% 
  mutate(predict_randomForest = predict(tree_randomForest), # use prediction function to get predictions
         correct_randomForest = predict_randomForest == Species # create a true/false for whether prediction was correct
  )



#' Get the importance of each variable with
importance <- varImp(tree_randomForest, scale=FALSE)
# 
# This tells us how much each variable decreases the average Gini index, a measure of how important the variable is to the model. Essentially, it estimates the impact a variable has on the model by comparing prediction accuracy rates for models with and without the variable. Larger values indicate higher importance of the variable. Here we see that the gender variable Sexmale is most important.
# http://cfss.uchicago.edu/stat004_decision_trees.html



# below does out of bag
age_sex_rf <- train(Survived ~ Age + Sex, data = titanic_rf_data,
                    method = "rf",
                    ntree = 200,
                    trControl = trainControl(method = "oob"))

#'## Boosting
#' Boosted models are ensemble models that are trained sequentially - each one tries to fix the errors of the one before it.
#' The two main boosting methods are AdaBoost (adaptive boosting) and gradient boosting
#' 
#' ## AdaBoost
#' 1. Create a model
#' 2. Predict on the training dataset
#' 3. Increase the weight on misclassified observations
#' 4. Train again on updated weights
#' 5. repeat steps 2-4
#' 
#' Note that AdaBoosting doesn't parallelise so doesn't scale well
#' https://qizeresearch.wordpress.com/2013/12/05/short-example-for-adaboost/
#' 
#' 
#' ## Gradient Boost
#' 1. Create a model
#' 2. Predict on the training dataset
#' 3. Calculate the residual errors from the first model
#' 4. Train another model on residual errors
#' 5. repeat steps 2-4
#' https://www.r-bloggers.com/gradient-boosting-in-r/
#' There's a learning rate hyper param - a low value (e.g. 0.1) will need more trees to fit to the training set but the results will usually generalise better. This is a type of shrinkage
#' Too many trees can result in overfitting though
#' There's ways you can optimise the number of trees by finding the lowest error in the validation dataset
#' 
#' 
#' #' Stacking
#' short for Stacked Generalisation
#' How to aggregate lots of different models?
#' Create a model!
#' 1. Split the dataset in two
#' 2. Use first subset to train predictors in first layer - create multiple models
#' 3. Feed the predictors from each of the models from 2 into a model
#' 
#' You can extend this by having multiple layers, so you can blend together outputs from different types of models and blenders trained using different types of models
#' 
#' 
#' XGBoost
#' https://cran.r-project.org/web/packages/xgboost/vignettes/xgboostPresentation.html
#' https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
#' http://xgboost.readthedocs.io/en/latest/model.html
#' 
#' Implements gradient boosting, using regularised model formulation to control for overfitting
#' Also, is super fast as it tries to optimise your computer's resources
#' 
#' 
#' 
#' 