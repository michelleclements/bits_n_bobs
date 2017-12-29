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
library(xgboost) # for connecting to XGBoost API
library(Matrix) # for creating matrices for XGBoost
library(ROCR) # for ROC curves
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

mush_results <- mush %>% 
  mutate(predict_simple = predict(tree_simple, newdata = mush), # use prediction function to get predictions
         correct_simple = predict_simple == class # create a true/false for whether prediction was correct
         )


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
tree_bagged <- train(class ~., # class is the response variable
                     method = "treebag", 
                     data = mush
)

tree_bagged$finalModel

mush_results <- mush_results %>% 
  mutate(predict_bagged = predict(tree_bagged, newdata = mush), # use prediction function to get predictions
         correct_bagged = predict_bagged == class # create a true/false for whether prediction was correct
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
tree_randomForest <- train(class~.,
                           data = mush,
                           method = "rf", # random forest
                           trControl = trainControl(method = "cv", # resampling method is cross validation 
                                                    number = 5), # 5 splits in the corss validation
                           prox = TRUE,
                           allowParallel = TRUE)
print(tree_randomForest)

print(tree_randomForest$finalModel)

mush_results <- mush_results %>% 
  mutate(predict_randomForest = predict(tree_randomForest, newdata = mush), # use prediction function to get predictions
         correct_randomForest = predict_randomForest == class # create a true/false for whether prediction was correct
  )

predict.train(tree_randomForest, newdata = mush, type = "prob")

#' Get the importance of each variable with
importance <- varImp(tree_randomForest, scale=FALSE)
# 
# This tells us how much each variable decreases the average Gini index, a measure of how important the variable is to the model. Essentially, it estimates the impact a variable has on the model by comparing prediction accuracy rates for models with and without the variable. Larger values indicate higher importance of the variable. Here we see that the gender variable Sexmale is most important.
# http://cfss.uchicago.edu/stat004_decision_trees.html

tree_randomForest$finalModel


#'### Random forest with out of bag sampling - boostrapped samples with validaton on approx 1/3 of samples not included in bootstrap

# below does out of bag
tree_randomForestOob <- train(class ~ ., data = mush,
                    method = "rf",
                    ntree = 200,
                    trControl = trainControl(method = "oob"))


print(tree_randomForestOob)

print(tree_randomForestOob$finalModel)

mush_results <- mush_results %>% 
  mutate(predict_randomForestOob = predict(tree_randomForestOob, newdata = mush), # use prediction function to get predictions
         correct_randomForestOob = predict_randomForestOob == class # create a true/false for whether prediction was correct
  )

#' Get the importance of each variable with
importanceOOb <- varImp(tree_randomForestOob, scale=FALSE)

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
#' ## XGBoost
#' https://cran.r-project.org/web/packages/xgboost/vignettes/xgboostPresentation.html
#' https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
#' http://xgboost.readthedocs.io/en/latest/model.html
#' 
#' Implements gradient boosting, using regularised model formulation to control for overfitting
#' Very good explanation is available here http://xgboost.readthedocs.io/en/latest/model.html
#' Also, is super fast as it tries to optimise your computer's resources
#' ### Preparing data for XGBoost
#' XGBoost needs the   
#' * response variable as a vector
#' * predictor variables at a matrix  
#' 
#' XGBoost also likes the data in data.frame so check that first
mush <- data.frame(mush)
#' 
#' Create design matrix:  
#' * 0/1 to indicate presence of each level of response variable
#' * class ~ indicates that class is response var and shouldn't be included
#' * -1 to not put in an intercept
mush_dsgn_mtx <- sparse.model.matrix(class ~ .-1, data = mush)

#' create 0/1 output_vector which is 1 when class = e and 0 otherwise
mush_response <- as.numeric(mush$class == "e")
#'


tree_xgb <- xgboost(data = mush_dsgn_mtx, 
               label = mush_response, 
               eta = 0.1, # learning rate - low takes longer but more robust to overfitting
               max_depth = 2, # max depth of tree
               nround = 5, # max number of interations
               subsample = 0.5, # randomly collect half data for fitting
               colsample_bytree = 0.5,
               seed = 1,
               objective = "binary:logistic",
               nthread = 3, 
               verbose = 2 # prints extra info about the trees
)

importance_xgb <- xgb.importance(feature_names = colnames(mush_dsgn_mtx), model = tree_xgb)

xgb.plot.importance(importance_matrix = importance_xgb)

#' * Gain is the improvement in accuracy brought by a feature to the branches it is on. 
#' The idea is that before adding a new split on a feature X to the branch there was some wrongly classified elements, 
#' after adding the split on this feature, there are two new branches, and each of these branch is more accurate 
#' (one branch saying if your observation is on this branch then it should be classified as 1, and the other branch saying the exact opposite).  
#' * Cover measures the relative quantity of observations concerned by a feature.  
#' * Frequency is a simpler way to measure the Gain. It just counts the number of times a feature is used in all generated trees. You should not use it (unless you know why you want to use it).

head(importance_xgb)


#' ## Getting predictions from XGBoost
mush_results <- mush_results %>% 
  mutate(predict_xgb = predict(tree_xgb, newdata = mush_dsgn_mtx) # use prediction function to get predictions
  )


hist(mush_results$predict_xgb)


#' ## Random forests with XGBoost
#' Random forests compute independent trees, wheras boosted models try to improve the tree before.  
#' Boosted models can give more consistent results when predictors are correlated
#' as the model will stick with the first variable it picks up.  
#' In contrast, different random forests may pick different things.  
#' Which you prefer may depend on the data
tree_xgb_rfor <- xgboost(data = mush_dsgn_mtx, 
                         label = mush_response, 
                         max_depth = 4, 
                         num_parallel_tree = 1000, 
                         subsample = 0.5, 
                         colsample_bytree =0.5, 
                         nrounds = 1, 
                         objective = "binary:logistic")

#' #### Params in XGBoost
#' Taken from https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/
#' Parameters used in Xgboost
#'  I understand, by now, you would be highly curious to know about various parameters used in xgboost model. So, there are three types of parameters: General Parameters, Booster Parameters and Task Parameters.
#'  
#'  ###### General parameters 
#'  refers to which booster we are using to do boosting. The commonly used are tree or linear model
#'  * silent : The default value is 0. You need to specify 0 for printing running messages, 1 for silent mode.
#'  * booster : The default value is gbtree. You need to specify the booster to use: gbtree (tree based) or gblinear (linear function).
#'  * num_pbuffer : This is set automatically by xgboost, no need to be set by user. Read documentation of xgboost for more details.
#'  * num_feature : This is set automatically by xgboost, no need to be set by user.
#'  
#'  ##### Booster parameters 
#'  depends on which booster you have chosen  
#'  The tree specific parameters:  
#'  
#'  * eta : The default value is set to 0.3. You need to specify step size shrinkage used in update to prevents overfitting. 
#'  After each boosting step, we can directly get the weights of new features. and eta actually shrinks the feature weights to make the boosting process more conservative. 
#'  The range is 0 to 1. Low eta value means model is more robust to overfitting.
#'  * gamma : The default value is set to 0. You need to specify minimum loss reduction required to make a further partition on a leaf node of the tree. 
#'  The larger, the more conservative the algorithm will be. The range is 0 to ∞. Larger the gamma more conservative the algorithm is.  
#'  * max_depth : The default value is set to 6. You need to specify the maximum depth of a tree. The range is 1 to ∞.  
#'  * min_child_weight : The default value is set to 1. You need to specify the minimum sum of instance weight(hessian) needed in a child. 
#'  If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. 
#'  In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be. The range is 0 to ∞.
#'  * max_delta_step : The default value is set to 0. Maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint.
#'   If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. 
#'   Set it to value of 1-10 might help control the update.The range is 0 to ∞.  
#'  * subsample : The default value is set to 1. You need to specify the subsample ratio of the training instance. 
#'  Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting. The range is 0 to 1.  
#'  * colsample_bytree : The default value is set to 1. You need to specify the subsample ratio of columns when constructing each tree. The range is 0 to 1.  
#'  
#'  
#'  * Linear Booster Specific Parameters:  
#'  * lambda and alpha : These are regularization term on weights. Lambda default value assumed is 1 and alpha is 0.  
#'  * lambda_bias : L2 regularization term on bias and has a default value of 0. 
#'  
#'    
#'  ##### Learning Task parameters 
#'  Decides on the learning scenario, for example, regression tasks may use different parameters with ranking tasks.   
#'  * base_score : The default value is set to 0.5 . You need to specify the initial prediction score of all instances, global bias.  
#'  * objective : The default value is set to reg:linear . You need to specify the type of learner you want which includes linear regression, logistic regression, poisson regression etc.  
#'  * eval_metric : You need to specify the evaluation metrics for validation data, a default metric will be assigned according to objective
#'  (rmse for regression, and error for classification, mean average precision for ranking)  
#'  * seed : As always here you specify the seed to reproduce the same set of outputs.

#'   
#'                                                                                                                                          
#'   ### Testing validity of ouputs
#'   Test with chi-squared test for categorical vars and correlation for continuous
chisq.test(mush$odor == "n", mush$class)                                                                                                                                                                                                                                                                                                                                                                                                                  

#' ## Impact of changing category boundaries
#' 
#' Create a dataset with some metrics for a large number of potential cutoffs.  
#' 
#' use package ROCR to get this data
#' 
#
rocr_xgb <- ROCR::prediction(predictions = mush_results$predict_xgb, 
                             labels = mush_response)

performance(pred,"tpr","fpr")

performance_xgb <- data_frame(cut = rocr_xgb@cutoffs[[1]], 
                              fpr = performance(rocr_xgb, "tpr")@y.values[[1]], 
                              tpr = performance(rocr_xgb, "fpr")@y.values[[1]], 
                              prec = performance(rocr_xgb, "prec")@y.values[[1]])


ggplot(performance_xgb, aes(x = cut, y = fpr)) + geom_point()


ggplot(performance_xgb, aes(x = cut)) + geom_line(aes(y = prec)) + geom_line(aes(y = tpr))



performance(rocr_xgb, "tpr")

plot(ROCR::performance(prediction.obj	 = performance_xgb,
                 measure = 'tpr',
                 x.measure = 'fpr'))

ROCRpred <- prediction(pred,obs)




library(precrec)


# Calculate ROC and Precision-Recall curves
a <- precrec::evalmod(scores = mush_results$predict_xgb, labels = mush_results$response)


summary(mush_results$predict_xgb[mush_results$predict_xgb < 0.5])
summary(mush_results$predict_xgb[mush_results$predict_xgb > 0.5])
