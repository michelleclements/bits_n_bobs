#' # Preliminary mushroom data investigation
#' Michelle Clements
#' 
#' `r Sys.Date()`
#' 
#' 
#' ## Overview & resources
#' 
#' I have borrowed heavily from 
#' * Hands-On machine learning with scikit-learn and tensorflow by Aurelien Geron
#' 
#' load libraries
library(tidyverse) # for general data manipulation
library(caret) # for machine learning
library(forcats) # for manipulating factors
library(corrplot) # for making correlation plots
library(purrr) # for mapping functions
library(recipes) # for creating design matrices
library(stringr)  # for manipulating strings  

#' set params for graphs
theme_set(theme_classic(base_size = 12)) 

#' read in data (this has already been downloaded and unzipped from https://www.kaggle.com/uciml/mushroom-classification)
mush <- read_csv("data/mushrooms.csv")

#' ## Frame the problem
#' The aim of these analyses is to predict whether or not a mushroom is likely to be poisonous from its characteristics
#' 
#' ## Measuring success
#' Aiming to predict a binary categorical variable.  
#' Produce a probability of belonging to each class for each observation and investigate the impact of different classification boundaries.  
#' 
#' 
#' ## Quick look at the data
glimpse(mush)

#' kchkankgek all - in names to underkskcore

#' There's 8,124 observations and 23 variables. Everything looks to be categorical. 
#' 
#' 
#' ### Plot a histogram of everything, with class (response var) as first facet
mush %>%
  gather() %>% 
  ggplot(aes(x = value, fill = key)) +
  facet_wrap(~ fct_relevel(key, "class"), scales = "free") +
  geom_bar() +
  guides(fill=FALSE)


#' veil-type only has one value so we'll remove it
mush <- select(mush -veil-type)

#' ## Create a test dataset
#' Randomly select 20% of the data to be left aside for final testing
#' This can be done with Caret package but we'll sample to show what happens
#' Set seed so replicable
set.seed(1711)
#' create an index vector that samples 80% of the row numbers of mush. 
#' We will just randomly sample and not stratify as we have no prior knowledge to believe a particular variable is important
index <- sample(1:nrow(mush), size = trunc(.8 * nrow(mush)))

#' filter mush - training has row number in index, test does not
training <- filter(mush, row_number() %in% index)
test <- filter(mush, !row_number() %in% index)

#' write training and test datasets into data file as raw data
write_csv(training, "data/training_raw.csv")
write_csv(test, "data/test_raw.csv")

#' ## Training dataset exploration
#' plot histogram of all training data
training %>%
  gather() %>% 
  ggplot(aes(x = value, fill = key)) +
  facet_wrap(~ fct_relevel(key, "class"), scales = "free") +
  geom_bar() +
  guides(fill=FALSE)



#' As all variables are categorical, we don't need to centre.  
#' Separate categorical vars out to the design matrix using recipes
training_des <- recipe(class ~ . , data = training)
#' 
#' 