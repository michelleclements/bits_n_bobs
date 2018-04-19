library(tidyverse)
library(stringr)
# set up parameters
n_zones <- 4
n_weeks <- 2
days_of_week <- c("M", "Tu", "W", "Th", "F", "Sa", "Su")

# set means and sd's
mean_by_day <- c(30, 40, 50, 70, 100, 110, 70)
within_day_sd <- 10
zone_to_zone_sd <- 20
within_zone_sd <- 5
treatment_difference <- 10
within_treatment_sd <- 5
error_sd <- 5

# set seed so results are replicable
set.seed(103)

# Setting up basic structure ----------------------------------------------

# base dataset with each zone, week and day combo
df <- crossing(zone = paste0("z", 1:n_zones), 
               week = c(1:n_weeks), 
               day = factor(days_of_week, levels = days_of_week))

# Setting up effects -------------------------------------------------------

# create day effect manually as it's definitely not 
day_effect <- data_frame(
  day = factor(days_of_week, levels = days_of_week), 
  day_mean = mean_by_day, 
  day_sd = rep(within_day_sd, 7)
)

# create a zone effect - just adjusts day effect up or down so has mean of 0
# assume the zone means follow a normal distribution
zone_effect <- data_frame(
  zone = paste0("z", 1:n_zones), 
  zone_mean = rnorm(n = n_zones, 
                    mean = 0, 
                    sd = zone_to_zone_sd), 
  zone_sd = within_zone_sd
  
)

treatment_effect <- crossing(
  week = c(1:n_weeks), 
  day = factor(days_of_week, levels = days_of_week)
  )


# going to create two different patterns 
# one T then C, and other C then T and then sort out later
treatment_effect <- treatment_effect %>% 
  mutate(pattern1 = rep(c("C", "T"), n_weeks * length(days_of_week) / 2),
         pattern2 = rep(c("T", "C"), n_weeks * length(days_of_week) / 2))

# Combining data and sampling  --------------------------------------------
# join on effects
df <- df %>% 
  inner_join(day_effect, by = "day") %>% 
  inner_join(zone_effect, by = "zone") %>% 
  inner_join(treatment_effect, by = c("week", "day"))

# assign treatment pattern 1 to first half of zones, and treatment pattern2 to second half
# add in treatment effects and sd
df <- df %>% 
  mutate(treatment = ifelse(as.numeric(str_remove(zone, "z")) <= n_zones / 2, 
                            pattern1, 
                            pattern2), 
         treatment_mean = ifelse(treatment == "C", 0, treatment_difference), 
         treatment_sd = within_treatment_sd) %>% 
  select(-pattern1, -pattern2)

# sample effect for each part and then sum and sample sales in each zone
# sometimes goes below zero, but principle still valid
df <- df %>% 
  mutate(
    day_effect = rnorm(n = nrow(df), mean = day_mean, sd = day_sd), 
         zone_effect = rnorm(n = nrow(df), mean = zone_mean, sd = zone_sd), 
         treatment_effect = rnorm(n = nrow(df), mean = treatment_mean, sd = treatment_sd), 
         effect_sum = day_effect + zone_effect + treatment_effect, 
         sales = rnorm(n = nrow(df), mean = effect_sum, sd = error_sd)
         )


# Make some plots ---------------------------------------------------------

# average sales per day grouped by zone
df %>% 
  group_by(zone, day) %>% 
  summarise(mean = mean(sales)) %>% 
  ggplot() + 
  geom_line(aes(x = day, y = mean, group = zone, colour = zone))

# average sales per day by treatment
df %>% 
  group_by(treatment, day) %>% 
  summarise(mean = mean(sales)) %>% 
  ggplot() + 
  geom_line(aes(x = day, y = mean, group = treatment, colour = treatment))


# Analysis ----------------------------------------------------------------

# number of observations in experiment
nrow(df)

# raw effect estimate
mean(df$sales[df$treatment == "T"]) - mean(df$sales[df$treatment == "C"])

# do a t-test on treatment
t.test(df$sales ~ df$treatment)

# do an anova
summary(aov(sales ~ treatment + day, data = df))




