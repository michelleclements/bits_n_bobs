# learning-machine-learning

This repo contains some of my notes from machine learning. The examples are often lifted from books and blogs. I've tried to credit main sources but sometimes I don't know where I have got things from. 

This file contains a summary of the main highlights of each file. 

## Decision_trees

* Single CART decision tree finds the best feature and split to make the buckets as pure as possible
* Ensemble models are when multiple models are combined. There's lots of different ways to do this. 

### Ensemble models
There are two main ways to do ensemble models:  
1. Train different types of models on the same data set - models have to be as different (independent) as possible  
Combining the results can be done in a two main ways:  
a. Hard voting: each model gets a 'vote' and the class with the most vote wins  
b. Soft voting: average probability of each class across all models and pick the class with the highest average (needs probs to be outputted)  

2. Train the same type of model on different subsets of the dataset  

Sampling can be:  
a. with replacement: bagging (bootstrap aggregating)  
b. without replacement: pasting  

Prediction can then be done once the models are aggregated:  
1. Statistical mode (same as hard voting) for classification  
2. Average for regression    
3. Build a model to combine different predictions (see stacking)  

### Random forests  
Random Forests use bagging/pasting.  
However, random forests don't search for the single best split in the data.  
Instead, they search for the best split among a random subset of features  

### Boosting
Boosted models are ensemble models that are trained sequentially - each one tries to fix the errors of the one before it.
The two main boosting methods are AdaBoost (adaptive boosting) and gradient boosting

* AdaBoost: increases weights on misclassified observations and then retrains and repeats  
* GradientBoost: creates sequential models on the residuals from the previous models  

watch for overfitting!

### Stacked Generalisation / Stacking

Use a model to aggregate outputs from other models. 
1. Split the dataset in two (important so independent data going into second model)
2. Use first subset to train predictors in first layer - create multiple models
3. Feed the predictors from each of the models from 2 into a model

You can extend this by having multiple layers, so you can blend together outputs from different types of models and blenders trained using different types of models