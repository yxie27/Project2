Project 2
================
Yilin Xie
July 3, 2020

  - [Introduction](#introduction)
      - [Data preprocessing](#data-preprocessing)
      - [Data split](#data-split)
      - [Summarizations](#summarizations)
  - [Ensemble model fit](#ensemble-model-fit)
      - [On train set](#on-train-set)
      - [On test set](#on-test-set)
  - [Linear regression fit](#linear-regression-fit)
      - [On train set](#on-train-set-1)
      - [On test set](#on-test-set-1)
  - [Conclusions](#conclusions)

## Introduction

The perpose of this project is going to analyze an online news
popularity data set
[here](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)
and predict **shares** by backward linear regression and random forest.
Firstly I read it into R session and determine which variables that I
would deal with.

``` r
#Load the data
news <- read.csv("/Users/yilinxie/Desktop/ST558/Project/Project2/OnlineNewsPopularity.csv")
head(news)
```

    ##                                                              url timedelta
    ## 1   http://mashable.com/2013/01/07/amazon-instant-video-browser/       731
    ## 2    http://mashable.com/2013/01/07/ap-samsung-sponsored-tweets/       731
    ## 3 http://mashable.com/2013/01/07/apple-40-billion-app-downloads/       731
    ## 4       http://mashable.com/2013/01/07/astronaut-notre-dame-bcs/       731
    ## 5               http://mashable.com/2013/01/07/att-u-verse-apps/       731
    ## 6               http://mashable.com/2013/01/07/beewi-smart-toys/       731
    ##   n_tokens_title n_tokens_content n_unique_tokens n_non_stop_words
    ## 1             12              219       0.6635945                1
    ## 2              9              255       0.6047431                1
    ## 3              9              211       0.5751295                1
    ## 4              9              531       0.5037879                1
    ## 5             13             1072       0.4156456                1
    ## 6             10              370       0.5598886                1
    ##   n_non_stop_unique_tokens num_hrefs num_self_hrefs num_imgs num_videos
    ## 1                0.8153846         4              2        1          0
    ## 2                0.7919463         3              1        1          0
    ## 3                0.6638655         3              1        1          0
    ## 4                0.6656347         9              0        1          0
    ## 5                0.5408895        19             19       20          0
    ## 6                0.6981982         2              2        0          0
    ##   average_token_length num_keywords data_channel_is_lifestyle
    ## 1             4.680365            5                         0
    ## 2             4.913725            4                         0
    ## 3             4.393365            6                         0
    ## 4             4.404896            7                         0
    ## 5             4.682836            7                         0
    ## 6             4.359459            9                         0
    ##   data_channel_is_entertainment data_channel_is_bus data_channel_is_socmed
    ## 1                             1                   0                      0
    ## 2                             0                   1                      0
    ## 3                             0                   1                      0
    ## 4                             1                   0                      0
    ## 5                             0                   0                      0
    ## 6                             0                   0                      0
    ##   data_channel_is_tech data_channel_is_world kw_min_min kw_max_min kw_avg_min
    ## 1                    0                     0          0          0          0
    ## 2                    0                     0          0          0          0
    ## 3                    0                     0          0          0          0
    ## 4                    0                     0          0          0          0
    ## 5                    1                     0          0          0          0
    ## 6                    1                     0          0          0          0
    ##   kw_min_max kw_max_max kw_avg_max kw_min_avg kw_max_avg kw_avg_avg
    ## 1          0          0          0          0          0          0
    ## 2          0          0          0          0          0          0
    ## 3          0          0          0          0          0          0
    ## 4          0          0          0          0          0          0
    ## 5          0          0          0          0          0          0
    ## 6          0          0          0          0          0          0
    ##   self_reference_min_shares self_reference_max_shares
    ## 1                       496                       496
    ## 2                         0                         0
    ## 3                       918                       918
    ## 4                         0                         0
    ## 5                       545                     16000
    ## 6                      8500                      8500
    ##   self_reference_avg_sharess weekday_is_monday weekday_is_tuesday
    ## 1                    496.000                 1                  0
    ## 2                      0.000                 1                  0
    ## 3                    918.000                 1                  0
    ## 4                      0.000                 1                  0
    ## 5                   3151.158                 1                  0
    ## 6                   8500.000                 1                  0
    ##   weekday_is_wednesday weekday_is_thursday weekday_is_friday
    ## 1                    0                   0                 0
    ## 2                    0                   0                 0
    ## 3                    0                   0                 0
    ## 4                    0                   0                 0
    ## 5                    0                   0                 0
    ## 6                    0                   0                 0
    ##   weekday_is_saturday weekday_is_sunday is_weekend     LDA_00     LDA_01
    ## 1                   0                 0          0 0.50033120 0.37827893
    ## 2                   0                 0          0 0.79975569 0.05004668
    ## 3                   0                 0          0 0.21779229 0.03333446
    ## 4                   0                 0          0 0.02857322 0.41929964
    ## 5                   0                 0          0 0.02863281 0.02879355
    ## 6                   0                 0          0 0.02224528 0.30671758
    ##       LDA_02     LDA_03     LDA_04 global_subjectivity
    ## 1 0.04000468 0.04126265 0.04012254           0.5216171
    ## 2 0.05009625 0.05010067 0.05000071           0.3412458
    ## 3 0.03335142 0.03333354 0.68218829           0.7022222
    ## 4 0.49465083 0.02890472 0.02857160           0.4298497
    ## 5 0.02857518 0.02857168 0.88542678           0.5135021
    ## 6 0.02223128 0.02222429 0.62658158           0.4374086
    ##   global_sentiment_polarity global_rate_positive_words
    ## 1                0.09256198                 0.04566210
    ## 2                0.14894781                 0.04313725
    ## 3                0.32333333                 0.05687204
    ## 4                0.10070467                 0.04143126
    ## 5                0.28100348                 0.07462687
    ## 6                0.07118419                 0.02972973
    ##   global_rate_negative_words rate_positive_words rate_negative_words
    ## 1                0.013698630           0.7692308           0.2307692
    ## 2                0.015686275           0.7333333           0.2666667
    ## 3                0.009478673           0.8571429           0.1428571
    ## 4                0.020715631           0.6666667           0.3333333
    ## 5                0.012126866           0.8602151           0.1397849
    ## 6                0.027027027           0.5238095           0.4761905
    ##   avg_positive_polarity min_positive_polarity max_positive_polarity
    ## 1             0.3786364            0.10000000                   0.7
    ## 2             0.2869146            0.03333333                   0.7
    ## 3             0.4958333            0.10000000                   1.0
    ## 4             0.3859652            0.13636364                   0.8
    ## 5             0.4111274            0.03333333                   1.0
    ## 6             0.3506100            0.13636364                   0.6
    ##   avg_negative_polarity min_negative_polarity max_negative_polarity
    ## 1            -0.3500000                -0.600            -0.2000000
    ## 2            -0.1187500                -0.125            -0.1000000
    ## 3            -0.4666667                -0.800            -0.1333333
    ## 4            -0.3696970                -0.600            -0.1666667
    ## 5            -0.2201923                -0.500            -0.0500000
    ## 6            -0.1950000                -0.400            -0.1000000
    ##   title_subjectivity title_sentiment_polarity abs_title_subjectivity
    ## 1          0.5000000               -0.1875000             0.00000000
    ## 2          0.0000000                0.0000000             0.50000000
    ## 3          0.0000000                0.0000000             0.50000000
    ## 4          0.0000000                0.0000000             0.50000000
    ## 5          0.4545455                0.1363636             0.04545455
    ## 6          0.6428571                0.2142857             0.14285714
    ##   abs_title_sentiment_polarity shares
    ## 1                    0.1875000    593
    ## 2                    0.0000000    711
    ## 3                    0.0000000   1500
    ## 4                    0.0000000   1200
    ## 5                    0.1363636    505
    ## 6                    0.2142857    855

``` r
#detect NA value
sum(is.na(news$shares) )
```

    ## [1] 0

As shown above, it is a huge data set with 39644 rows and 61 colomns. It
contains factor variables, numeric variables and dummy variables. The
task is to predict the 61st variable **shares**. Then I’m going to
preprocess the data to get it in the form I need.

### Data preprocessing

``` r
library(tidyverse)
## Filters out the data for the specified weekday
news <- mutate(news, weekday = ifelse(weekday_is_monday == "1", "monday", ifelse(weekday_is_tuesday == "1", "tuesday", ifelse(weekday_is_wednesday == "1", "wednesday", ifelse(weekday_is_thursday == "1", "thursday", ifelse(weekday_is_friday, "friday", ifelse(weekday_is_saturday == "1", "saturday", "sunday")))))))
news1 <- filter(news, weekday == params$weekday)
## Remove the useless colomns
news1 <- select(news1, -url, -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, -is_weekend, -weekday)
r = nrow(news1)
c = ncol(news1)
r
```

    ## [1] 5701

``` r
c
```

    ## [1] 52

Since r\>\>c, this is not a high dimension data set. I would use and to
predict **shares** by the entire variables.

### Data split

``` r
#Set seed to make work reproducible
set.seed(123)
#randomly sample from the data
sub <- sample(1:r, 0.7*r)
#store training dataset (70% of the data) and test dataset (30% of the data)
train <- news1[sub,]
test <- news1[-sub,]
```

### Summarizations

We can look at the distribution of **shares** through the histogram and
see some statistics of the total variables in a summary table.

``` r
#number of rows of training dataset
nrow(train)
```

    ## [1] 3990

``` r
#draw histograms
hist(train$shares)
```

![](README_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
hist(log(train$shares))
```

![](README_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

``` r
#summary the training dataset
t(summary(train))
```

    ##                                                                    
    ##   timedelta                   Min.   : 13        1st Qu.:167       
    ## n_tokens_title                Min.   : 3.00      1st Qu.: 9.00     
    ## n_tokens_content              Min.   :   0.0     1st Qu.: 235.0    
    ## n_unique_tokens               Min.   :0.0000     1st Qu.:0.4741    
    ## n_non_stop_words              Min.   :0.0000     1st Qu.:1.0000    
    ## n_non_stop_unique_tokens      Min.   :0.0000     1st Qu.:0.6287    
    ##   num_hrefs                   Min.   :  0.00     1st Qu.:  4.00    
    ## num_self_hrefs                Min.   :  0.000    1st Qu.:  1.000   
    ##    num_imgs                   Min.   :  0.000    1st Qu.:  1.000   
    ##   num_videos                  Min.   : 0.000     1st Qu.: 0.000    
    ## average_token_length          Min.   :0.000      1st Qu.:4.472     
    ##  num_keywords                 Min.   : 1.000     1st Qu.: 6.000    
    ## data_channel_is_lifestyle     Min.   :0.00000    1st Qu.:0.00000   
    ## data_channel_is_entertainment Min.   :0.0000     1st Qu.:0.0000    
    ## data_channel_is_bus           Min.   :0.0000     1st Qu.:0.0000    
    ## data_channel_is_socmed        Min.   :0.00000    1st Qu.:0.00000   
    ## data_channel_is_tech          Min.   :0.0000     1st Qu.:0.0000    
    ## data_channel_is_world         Min.   :0.0000     1st Qu.:0.0000    
    ##   kw_min_min                  Min.   : -1.00     1st Qu.: -1.00    
    ##   kw_max_min                  Min.   :     0.0   1st Qu.:   444.0  
    ##   kw_avg_min                  Min.   :   -1.0    1st Qu.:  139.8   
    ##   kw_min_max                  Min.   :     0     1st Qu.:     0    
    ##   kw_max_max                  Min.   : 28000     1st Qu.:843300    
    ##   kw_avg_max                  Min.   :  5911     1st Qu.:172411    
    ##   kw_min_avg                  Min.   :   0       1st Qu.:   0      
    ##   kw_max_avg                  Min.   :  2195     1st Qu.:  3569    
    ##   kw_avg_avg                  Min.   :  776.1    1st Qu.: 2368.5   
    ## self_reference_min_shares     Min.   :     0.0   1st Qu.:   653.2  
    ## self_reference_max_shares     Min.   :     0     1st Qu.:  1000    
    ## self_reference_avg_sharess    Min.   :     0.0   1st Qu.:   956.2  
    ##     LDA_00                    Min.   :0.01818    1st Qu.:0.02506   
    ##     LDA_01                    Min.   :0.01818    1st Qu.:0.02502   
    ##     LDA_02                    Min.   :0.01818    1st Qu.:0.02857   
    ##     LDA_03                    Min.   :0.01818    1st Qu.:0.02857   
    ##     LDA_04                    Min.   :0.01819    1st Qu.:0.02857   
    ## global_subjectivity           Min.   :0.0000     1st Qu.:0.3990    
    ## global_sentiment_polarity     Min.   :-0.36425   1st Qu.: 0.05489  
    ## global_rate_positive_words    Min.   :0.00000    1st Qu.:0.02751   
    ## global_rate_negative_words    Min.   :0.000000   1st Qu.:0.009901  
    ## rate_positive_words           Min.   :0.0000     1st Qu.:0.5926    
    ## rate_negative_words           Min.   :0.0000     1st Qu.:0.2000    
    ## avg_positive_polarity         Min.   :0.0000     1st Qu.:0.3070    
    ## min_positive_polarity         Min.   :0.00000    1st Qu.:0.05000   
    ## max_positive_polarity         Min.   :0.0000     1st Qu.:0.6000    
    ## avg_negative_polarity         Min.   :-1.0000    1st Qu.:-0.3321   
    ## min_negative_polarity         Min.   :-1.000     1st Qu.:-0.700    
    ## max_negative_polarity         Min.   :-1.0000    1st Qu.:-0.1250   
    ## title_subjectivity            Min.   :0.0000     1st Qu.:0.0000    
    ## title_sentiment_polarity      Min.   :-1.00000   1st Qu.: 0.00000  
    ## abs_title_subjectivity        Min.   :0.0000     1st Qu.:0.1667    
    ## abs_title_sentiment_polarity  Min.   :0.0000     1st Qu.:0.0000    
    ##     shares                    Min.   :    35     1st Qu.:   981    
    ##                                                                    
    ##   timedelta                   Median :335        Mean   :355       
    ## n_tokens_title                Median :10.00      Mean   :10.37     
    ## n_tokens_content              Median : 404.0     Mean   : 532.5    
    ## n_unique_tokens               Median :0.5441     Mean   :0.5357    
    ## n_non_stop_words              Median :1.0000     Mean   :0.9707    
    ## n_non_stop_unique_tokens      Median :0.6948     Mean   :0.6782    
    ##   num_hrefs                   Median :  7.00     Mean   : 10.84    
    ## num_self_hrefs                Median :  2.000    Mean   :  3.062   
    ##    num_imgs                   Median :  1.000    Mean   :  4.421   
    ##   num_videos                  Median : 0.000     Mean   : 1.304    
    ## average_token_length          Median :4.661      Mean   :4.550     
    ##  num_keywords                 Median : 7.000     Mean   : 7.188    
    ## data_channel_is_lifestyle     Median :0.00000    Mean   :0.05439   
    ## data_channel_is_entertainment Median :0.0000     Mean   :0.1697    
    ## data_channel_is_bus           Median :0.0000     Mean   :0.1491    
    ## data_channel_is_socmed        Median :0.00000    Mean   :0.05865   
    ## data_channel_is_tech          Median :0.0000     Mean   :0.1719    
    ## data_channel_is_world         Median :0.0000     Mean   :0.2258    
    ##   kw_min_min                  Median : -1.00     Mean   : 27.27    
    ##   kw_max_min                  Median :   657.5   Mean   :  1134.6  
    ##   kw_avg_min                  Median :  232.6    Mean   :  316.4   
    ##   kw_min_max                  Median :  1400     Mean   : 14351    
    ##   kw_max_max                  Median :843300     Mean   :751527    
    ##   kw_avg_max                  Median :246277     Mean   :260923    
    ##   kw_min_avg                  Median :1020       Mean   :1106      
    ##   kw_max_avg                  Median :  4396     Mean   :  5663    
    ##   kw_avg_avg                  Median : 2863.5    Mean   : 3148.7   
    ## self_reference_min_shares     Median :  1200.0   Mean   :  4214.1  
    ## self_reference_max_shares     Median :  2800     Mean   : 11000    
    ## self_reference_avg_sharess    Median :  2200.0   Mean   :  6844.7  
    ##     LDA_00                    Median :0.03336    Mean   :0.17745   
    ##     LDA_01                    Median :0.03334    Mean   :0.13491   
    ##     LDA_02                    Median :0.04006    Mean   :0.22749   
    ##     LDA_03                    Median :0.04000    Mean   :0.23436   
    ##     LDA_04                    Median :0.04001    Mean   :0.22578   
    ## global_subjectivity           Median :0.4553     Mean   :0.4461    
    ## global_sentiment_polarity     Median : 0.11452   Mean   : 0.11613  
    ## global_rate_positive_words    Median :0.03846    Mean   :0.03907   
    ## global_rate_negative_words    Median :0.015361   Mean   :0.016896  
    ## rate_positive_words           Median :0.7021     Mean   :0.6756    
    ## rate_negative_words           Median :0.2857     Mean   :0.2948    
    ## avg_positive_polarity         Median :0.3585     Mean   :0.3539    
    ## min_positive_polarity         Median :0.10000    Mean   :0.09691   
    ## max_positive_polarity         Median :0.8000     Mean   :0.7492    
    ## avg_negative_polarity         Median :-0.2586    Mean   :-0.2640   
    ## min_negative_polarity         Median :-0.500     Mean   :-0.523    
    ## max_negative_polarity         Median :-0.1000    Mean   :-0.1118   
    ## title_subjectivity            Median :0.1000     Mean   :0.2799    
    ## title_sentiment_polarity      Median : 0.00000   Mean   : 0.06773  
    ## abs_title_subjectivity        Median :0.5000     Mean   :0.3467    
    ## abs_title_sentiment_polarity  Median :0.0000     Mean   :0.1531    
    ##     shares                    Median :  1500     Mean   :  3314    
    ##                                                                    
    ##   timedelta                   3rd Qu.:545        Max.   :727       
    ## n_tokens_title                3rd Qu.:12.00      Max.   :23.00     
    ## n_tokens_content              3rd Qu.: 699.8     Max.   :7413.0    
    ## n_unique_tokens               3rd Qu.:0.6164     Max.   :1.0000    
    ## n_non_stop_words              3rd Qu.:1.0000     Max.   :1.0000    
    ## n_non_stop_unique_tokens      3rd Qu.:0.7602     Max.   :1.0000    
    ##   num_hrefs                   3rd Qu.: 14.00     Max.   :171.00    
    ## num_self_hrefs                3rd Qu.:  4.000    Max.   :116.000   
    ##    num_imgs                   3rd Qu.:  3.000    Max.   :108.000   
    ##   num_videos                  3rd Qu.: 1.000     Max.   :91.000    
    ## average_token_length          3rd Qu.:4.855      Max.   :6.486     
    ##  num_keywords                 3rd Qu.: 9.000     Max.   :10.000    
    ## data_channel_is_lifestyle     3rd Qu.:0.00000    Max.   :1.00000   
    ## data_channel_is_entertainment 3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_bus           3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_socmed        3rd Qu.:0.00000    Max.   :1.00000   
    ## data_channel_is_tech          3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_world         3rd Qu.:0.0000     Max.   :1.0000    
    ##   kw_min_min                  3rd Qu.:  4.00     Max.   :217.00    
    ##   kw_max_min                  3rd Qu.:  1000.0   Max.   :158900.0  
    ##   kw_avg_min                  3rd Qu.:  355.7    Max.   :39979.0   
    ##   kw_min_max                  3rd Qu.:  7400     Max.   :843300    
    ##   kw_max_max                  3rd Qu.:843300     Max.   :843300    
    ##   kw_avg_max                  3rd Qu.:333801     Max.   :843300    
    ##   kw_min_avg                  3rd Qu.:2009       Max.   :3584      
    ##   kw_max_avg                  3rd Qu.:  6116     Max.   :171030    
    ##   kw_avg_avg                  3rd Qu.: 3612.1    Max.   :37607.5   
    ## self_reference_min_shares     3rd Qu.:  2700.0   Max.   :690400.0  
    ## self_reference_max_shares     3rd Qu.:  7800     Max.   :843300    
    ## self_reference_avg_sharess    3rd Qu.:  5178.7   Max.   :690400.0  
    ##     LDA_00                    3rd Qu.:0.23256    Max.   :0.92699   
    ##     LDA_01                    3rd Qu.:0.13955    Max.   :0.91998   
    ##     LDA_02                    3rd Qu.:0.36885    Max.   :0.92000   
    ##     LDA_03                    3rd Qu.:0.41335    Max.   :0.92554   
    ##     LDA_04                    3rd Qu.:0.37236    Max.   :0.92653   
    ## global_subjectivity           3rd Qu.:0.5115     Max.   :0.8987    
    ## global_sentiment_polarity     3rd Qu.: 0.17494   Max.   : 0.61389  
    ## global_rate_positive_words    3rd Qu.:0.05000    Max.   :0.13699   
    ## global_rate_negative_words    3rd Qu.:0.022013   Max.   :0.136929  
    ## rate_positive_words           3rd Qu.:0.7944     Max.   :1.0000    
    ## rate_negative_words           3rd Qu.:0.3902     Max.   :1.0000    
    ## avg_positive_polarity         3rd Qu.:0.4116     Max.   :1.0000    
    ## min_positive_polarity         3rd Qu.:0.10000    Max.   :1.00000   
    ## max_positive_polarity         3rd Qu.:1.0000     Max.   :1.0000    
    ## avg_negative_polarity         3rd Qu.:-0.1887    Max.   : 0.0000   
    ## min_negative_polarity         3rd Qu.:-0.300     Max.   : 0.000    
    ## max_negative_polarity         3rd Qu.:-0.0500    Max.   : 0.0000   
    ## title_subjectivity            3rd Qu.:0.5000     Max.   :1.0000    
    ## title_sentiment_polarity      3rd Qu.: 0.13636   Max.   : 1.00000  
    ## abs_title_subjectivity        3rd Qu.:0.5000     Max.   :0.5000    
    ## abs_title_sentiment_polarity  3rd Qu.:0.2500     Max.   :1.0000    
    ##     shares                    3rd Qu.:  2800     Max.   :233400

## Ensemble model fit

A random forest is a forest constructed in a random way, and the forest
is composed of many unrelated decision trees. Therefore, in theory, the
performance of random forest is generally better than that of a single
decision tree, because the results of random forest are determined by
voting on the results of multiple decision trees. But here, I’m using a
random forest for regression.

### On train set

``` r
#load package
library(randomForest)
#Get random forest model fit on training dataset
rf <- randomForest(shares ~ ., data = train, importance=TRUE)
rf
```

    ## 
    ## Call:
    ##  randomForest(formula = shares ~ ., data = train, importance = TRUE) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 17
    ## 
    ##           Mean of squared residuals: 72700437
    ##                     % Var explained: -4.59

``` r
#variable importance measures
importance(rf)
```

    ##                                   %IncMSE IncNodePurity
    ## timedelta                      2.63437346    4451570463
    ## n_tokens_title                 0.68438770    3391978788
    ## n_tokens_content               5.11153771    7658278805
    ## n_unique_tokens                2.36379554    5692217558
    ## n_non_stop_words               4.52523677    7315160323
    ## n_non_stop_unique_tokens       1.81270210    4703703651
    ## num_hrefs                      2.27705939    4143313671
    ## num_self_hrefs                 3.09962569    2222576913
    ## num_imgs                       1.37624703    3930394391
    ## num_videos                     0.88810626   10208167869
    ## average_token_length           1.85985244    7290357123
    ## num_keywords                   3.74204358    3432150983
    ## data_channel_is_lifestyle      0.11131093     234588773
    ## data_channel_is_entertainment  2.84955130     540810812
    ## data_channel_is_bus            4.17164316     575921412
    ## data_channel_is_socmed         2.88869229     386202067
    ## data_channel_is_tech           2.81359088     212497276
    ## data_channel_is_world          3.74672730     382658327
    ## kw_min_min                     2.46509886     517496132
    ## kw_max_min                     4.52220830    4978836505
    ## kw_avg_min                     3.87271473    4803627592
    ## kw_min_max                     3.23658477    2007881893
    ## kw_max_max                     2.59215677     348890797
    ## kw_avg_max                     4.01708329    8099802994
    ## kw_min_avg                     6.82789809    3414347896
    ## kw_max_avg                     5.80166332   11101864663
    ## kw_avg_avg                     5.66805423   11755027390
    ## self_reference_min_shares      5.91269276    6415988559
    ## self_reference_max_shares      7.11325017    3503641193
    ## self_reference_avg_sharess     7.49738964    5760781383
    ## LDA_00                         5.67825867    7387680544
    ## LDA_01                         3.58565207   20545534730
    ## LDA_02                         5.41272186    7548434039
    ## LDA_03                         6.22318549   16102920859
    ## LDA_04                         4.83091118   12166852430
    ## global_subjectivity            3.79139784    5066410914
    ## global_sentiment_polarity      1.44037316    7119468483
    ## global_rate_positive_words     0.90586279    3198337058
    ## global_rate_negative_words     2.18660180    3057366784
    ## rate_positive_words            3.25460532    2964223517
    ## rate_negative_words            3.68124110    2712962533
    ## avg_positive_polarity          1.95725899    4281310830
    ## min_positive_polarity          3.52652493    2312923910
    ## max_positive_polarity          2.40740077    1579745931
    ## avg_negative_polarity         -0.22153050    3960127103
    ## min_negative_polarity          4.22938826    2131186481
    ## max_negative_polarity          0.86821884    2233982894
    ## title_subjectivity             3.30360709    1849445020
    ## title_sentiment_polarity       0.05598322    2033513840
    ## abs_title_subjectivity         2.09919865    2037698898
    ## abs_title_sentiment_polarity   2.02636816    1466390361

``` r
#draw dotplot of variable importance as measured by Random Forest
varImpPlot(rf)
```

![](README_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Calculate the predicted mean square error on the train set:

``` r
train.pred <- predict(rf, train[,-52])
mean((train.pred - train$shares)^2)
```

    ## [1] 16508932

So, the predicted mean square error on the training dataset is 68724773.

### On test set

``` r
rf.test <- predict(rf, newdata = test[,-52])
mean((test$shares-rf.test)^2)
```

    ## [1] 57685972

So, the predicted mean square error on the testing dataset is 80333539.

## Linear regression fit

I choose stepwise regression to fit this model, more specificly, the
backward way. This data set contains too many variables. And since I’m
not an expert on journalism, I can’t tell which factors should have a
real effect on the predicted variables. If I manually removed the
variables by their significance in the model and compared the
differences between the models, this would be a lot of work. So I want
to use a backward regression model based on the AIC criteria to
automatically determine which variables should be included or removed
from the model.

### On train set

``` r
#fit model
lm.step <- step(lm(shares ~ .,data = train))
```

    ## Start:  AIC=72004.17
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     LDA_04 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + rate_negative_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ## 
    ## Step:  AIC=72004.17
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - self_reference_avg_sharess     1       5748 2.6743e+11 72002
    ## - average_token_length           1      36131 2.6743e+11 72002
    ## - global_sentiment_polarity      1      59429 2.6743e+11 72002
    ## - rate_negative_words            1      80141 2.6743e+11 72002
    ## - self_reference_max_shares      1     360917 2.6743e+11 72002
    ## - rate_positive_words            1    1267918 2.6744e+11 72002
    ## - max_positive_polarity          1    1650036 2.6744e+11 72002
    ## - kw_avg_min                     1    3963247 2.6744e+11 72002
    ## - global_rate_negative_words     1    4747330 2.6744e+11 72002
    ## - kw_min_avg                     1    5201151 2.6744e+11 72002
    ## - kw_min_min                     1    6048425 2.6744e+11 72002
    ## - LDA_02                         1   14910070 2.6745e+11 72002
    ## - avg_negative_polarity          1   15357113 2.6745e+11 72002
    ## - kw_avg_max                     1   16601059 2.6745e+11 72002
    ## - kw_max_min                     1   17135307 2.6745e+11 72002
    ## - LDA_03                         1   17883410 2.6745e+11 72002
    ## - n_non_stop_words               1   18543551 2.6745e+11 72002
    ## - self_reference_min_shares      1   19081932 2.6745e+11 72002
    ## - num_hrefs                      1   19378324 2.6745e+11 72002
    ## - LDA_00                         1   20195918 2.6745e+11 72002
    ## - num_imgs                       1   22117523 2.6746e+11 72003
    ## - kw_max_avg                     1   22749364 2.6746e+11 72003
    ## - title_subjectivity             1   23502696 2.6746e+11 72003
    ## - avg_positive_polarity          1   28102237 2.6746e+11 72003
    ## - num_videos                     1   29215814 2.6746e+11 72003
    ## - n_tokens_title                 1   34500041 2.6747e+11 72003
    ## - data_channel_is_socmed         1   38072174 2.6747e+11 72003
    ## - abs_title_sentiment_polarity   1   38527793 2.6747e+11 72003
    ## - kw_avg_avg                     1   40332807 2.6747e+11 72003
    ## - kw_min_max                     1   44777612 2.6748e+11 72003
    ## - min_negative_polarity          1   45264140 2.6748e+11 72003
    ## - global_rate_positive_words     1   54024148 2.6749e+11 72003
    ## - kw_max_max                     1   64536729 2.6750e+11 72003
    ## - abs_title_subjectivity         1   72528681 2.6751e+11 72003
    ## - num_keywords                   1   76010510 2.6751e+11 72003
    ## - title_sentiment_polarity       1  108287103 2.6754e+11 72004
    ## - max_negative_polarity          1  110324756 2.6754e+11 72004
    ## - min_positive_polarity          1  115696619 2.6755e+11 72004
    ## - num_self_hrefs                 1  115758767 2.6755e+11 72004
    ## - LDA_01                         1  118121698 2.6755e+11 72004
    ## <none>                                        2.6743e+11 72004
    ## - global_subjectivity            1  154890132 2.6759e+11 72004
    ## - n_non_stop_unique_tokens       1  170599321 2.6761e+11 72005
    ## - data_channel_is_tech           1  196318730 2.6763e+11 72005
    ## - data_channel_is_lifestyle      1  234266584 2.6767e+11 72006
    ## - timedelta                      1  250984267 2.6769e+11 72006
    ## - data_channel_is_world          1  275800626 2.6771e+11 72006
    ## - n_unique_tokens                1  342712818 2.6778e+11 72007
    ## - data_channel_is_bus            1  485899506 2.6792e+11 72009
    ## - n_tokens_content               1  492331849 2.6793e+11 72010
    ## - data_channel_is_entertainment  1 1104378643 2.6854e+11 72019
    ## 
    ## Step:  AIC=72002.17
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - average_token_length           1      35942 2.6743e+11 72000
    ## - global_sentiment_polarity      1      59664 2.6743e+11 72000
    ## - rate_negative_words            1      80304 2.6743e+11 72000
    ## - rate_positive_words            1    1267794 2.6744e+11 72000
    ## - max_positive_polarity          1    1650995 2.6744e+11 72000
    ## - self_reference_max_shares      1    1940969 2.6744e+11 72000
    ## - kw_avg_min                     1    3961143 2.6744e+11 72000
    ## - global_rate_negative_words     1    4742996 2.6744e+11 72000
    ## - kw_min_avg                     1    5195479 2.6744e+11 72000
    ## - kw_min_min                     1    6056788 2.6744e+11 72000
    ## - LDA_02                         1   14906615 2.6745e+11 72000
    ## - avg_negative_polarity          1   15355430 2.6745e+11 72000
    ## - kw_avg_max                     1   16643413 2.6745e+11 72000
    ## - kw_max_min                     1   17209275 2.6745e+11 72000
    ## - LDA_03                         1   17898571 2.6745e+11 72000
    ## - n_non_stop_words               1   18540385 2.6745e+11 72000
    ## - num_hrefs                      1   19526528 2.6745e+11 72000
    ## - LDA_00                         1   20193572 2.6745e+11 72000
    ## - num_imgs                       1   22129292 2.6746e+11 72001
    ## - kw_max_avg                     1   22745915 2.6746e+11 72001
    ## - title_subjectivity             1   23499277 2.6746e+11 72001
    ## - avg_positive_polarity          1   28104630 2.6746e+11 72001
    ## - num_videos                     1   30121478 2.6746e+11 72001
    ## - n_tokens_title                 1   34548319 2.6747e+11 72001
    ## - data_channel_is_socmed         1   38071120 2.6747e+11 72001
    ## - abs_title_sentiment_polarity   1   38524325 2.6747e+11 72001
    ## - kw_avg_avg                     1   40345398 2.6747e+11 72001
    ## - kw_min_max                     1   44813297 2.6748e+11 72001
    ## - min_negative_polarity          1   45274302 2.6748e+11 72001
    ## - global_rate_positive_words     1   54056250 2.6749e+11 72001
    ## - kw_max_max                     1   64590418 2.6750e+11 72001
    ## - abs_title_subjectivity         1   72535108 2.6751e+11 72001
    ## - num_keywords                   1   76095077 2.6751e+11 72001
    ## - title_sentiment_polarity       1  108281355 2.6754e+11 72002
    ## - self_reference_min_shares      1  109358593 2.6754e+11 72002
    ## - max_negative_polarity          1  110338607 2.6754e+11 72002
    ## - min_positive_polarity          1  115691981 2.6755e+11 72002
    ## - LDA_01                         1  118118635 2.6755e+11 72002
    ## - num_self_hrefs                 1  118902357 2.6755e+11 72002
    ## <none>                                        2.6743e+11 72002
    ## - global_subjectivity            1  154933205 2.6759e+11 72002
    ## - n_non_stop_unique_tokens       1  170697040 2.6761e+11 72003
    ## - data_channel_is_tech           1  196355402 2.6763e+11 72003
    ## - data_channel_is_lifestyle      1  234364514 2.6767e+11 72004
    ## - timedelta                      1  250997664 2.6769e+11 72004
    ## - data_channel_is_world          1  276158983 2.6771e+11 72004
    ## - n_unique_tokens                1  342758296 2.6778e+11 72005
    ## - data_channel_is_bus            1  486525735 2.6792e+11 72007
    ## - n_tokens_content               1  492506700 2.6793e+11 72008
    ## - data_channel_is_entertainment  1 1104546494 2.6854e+11 72017
    ## 
    ## Step:  AIC=72000.17
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - global_sentiment_polarity      1      62660 2.6743e+11 71998
    ## - rate_negative_words            1      85399 2.6743e+11 71998
    ## - rate_positive_words            1    1251734 2.6744e+11 71998
    ## - max_positive_polarity          1    1647835 2.6744e+11 71998
    ## - self_reference_max_shares      1    1937511 2.6744e+11 71998
    ## - kw_avg_min                     1    3948921 2.6744e+11 71998
    ## - global_rate_negative_words     1    4716752 2.6744e+11 71998
    ## - kw_min_avg                     1    5181646 2.6744e+11 71998
    ## - kw_min_min                     1    6074821 2.6744e+11 71998
    ## - LDA_02                         1   15113402 2.6745e+11 71998
    ## - avg_negative_polarity          1   15374217 2.6745e+11 71998
    ## - kw_avg_max                     1   16669158 2.6745e+11 71998
    ## - kw_max_min                     1   17271076 2.6745e+11 71998
    ## - LDA_03                         1   17912575 2.6745e+11 71998
    ## - LDA_00                         1   20159375 2.6745e+11 71998
    ## - n_non_stop_words               1   20402640 2.6745e+11 71998
    ## - num_hrefs                      1   20757964 2.6746e+11 71998
    ## - num_imgs                       1   22203331 2.6746e+11 71999
    ## - kw_max_avg                     1   22811569 2.6746e+11 71999
    ## - title_subjectivity             1   23488370 2.6746e+11 71999
    ## - avg_positive_polarity          1   28068733 2.6746e+11 71999
    ## - num_videos                     1   30228370 2.6746e+11 71999
    ## - n_tokens_title                 1   35299848 2.6747e+11 71999
    ## - data_channel_is_socmed         1   38035601 2.6747e+11 71999
    ## - abs_title_sentiment_polarity   1   38532220 2.6747e+11 71999
    ## - kw_avg_avg                     1   40318215 2.6747e+11 71999
    ## - kw_min_max                     1   44865749 2.6748e+11 71999
    ## - min_negative_polarity          1   45269220 2.6748e+11 71999
    ## - global_rate_positive_words     1   54026768 2.6749e+11 71999
    ## - kw_max_max                     1   64642522 2.6750e+11 71999
    ## - abs_title_subjectivity         1   72621240 2.6751e+11 71999
    ## - num_keywords                   1   76069439 2.6751e+11 71999
    ## - title_sentiment_polarity       1  108467986 2.6754e+11 72000
    ## - self_reference_min_shares      1  109335023 2.6754e+11 72000
    ## - max_negative_polarity          1  110320228 2.6754e+11 72000
    ## - min_positive_polarity          1  116194760 2.6755e+11 72000
    ## - LDA_01                         1  118105802 2.6755e+11 72000
    ## - num_self_hrefs                 1  119261371 2.6755e+11 72000
    ## <none>                                        2.6743e+11 72000
    ## - global_subjectivity            1  155698416 2.6759e+11 72000
    ## - n_non_stop_unique_tokens       1  180977486 2.6762e+11 72001
    ## - data_channel_is_tech           1  196510922 2.6763e+11 72001
    ## - data_channel_is_lifestyle      1  234329335 2.6767e+11 72002
    ## - timedelta                      1  251751202 2.6769e+11 72002
    ## - data_channel_is_world          1  276657496 2.6771e+11 72002
    ## - n_unique_tokens                1  362864779 2.6780e+11 72004
    ## - data_channel_is_bus            1  486698974 2.6792e+11 72005
    ## - n_tokens_content               1  492940918 2.6793e+11 72006
    ## - data_channel_is_entertainment  1 1105684924 2.6854e+11 72015
    ## 
    ## Step:  AIC=71998.17
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + rate_negative_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - rate_negative_words            1      97877 2.6743e+11 71996
    ## - rate_positive_words            1    1248355 2.6744e+11 71996
    ## - max_positive_polarity          1    1669161 2.6744e+11 71996
    ## - self_reference_max_shares      1    1936356 2.6744e+11 71996
    ## - kw_avg_min                     1    3920583 2.6744e+11 71996
    ## - kw_min_avg                     1    5156354 2.6744e+11 71996
    ## - global_rate_negative_words     1    5733288 2.6744e+11 71996
    ## - kw_min_min                     1    6067022 2.6744e+11 71996
    ## - LDA_02                         1   15129969 2.6745e+11 71996
    ## - avg_negative_polarity          1   16535564 2.6745e+11 71996
    ## - kw_avg_max                     1   16644097 2.6745e+11 71996
    ## - kw_max_min                     1   17342164 2.6745e+11 71996
    ## - LDA_03                         1   17998763 2.6745e+11 71996
    ## - LDA_00                         1   20174925 2.6745e+11 71996
    ## - n_non_stop_words               1   20390779 2.6746e+11 71996
    ## - num_hrefs                      1   20695490 2.6746e+11 71996
    ## - num_imgs                       1   22181992 2.6746e+11 71997
    ## - kw_max_avg                     1   22834208 2.6746e+11 71997
    ## - title_subjectivity             1   23569922 2.6746e+11 71997
    ## - num_videos                     1   30218575 2.6746e+11 71997
    ## - n_tokens_title                 1   35285596 2.6747e+11 71997
    ## - data_channel_is_socmed         1   38016074 2.6747e+11 71997
    ## - abs_title_sentiment_polarity   1   38535017 2.6747e+11 71997
    ## - kw_avg_avg                     1   40278514 2.6747e+11 71997
    ## - avg_positive_polarity          1   44219773 2.6748e+11 71997
    ## - kw_min_max                     1   45044687 2.6748e+11 71997
    ## - min_negative_polarity          1   45446052 2.6748e+11 71997
    ## - global_rate_positive_words     1   62528775 2.6750e+11 71997
    ## - kw_max_max                     1   64591307 2.6750e+11 71997
    ## - abs_title_subjectivity         1   72633745 2.6751e+11 71997
    ## - num_keywords                   1   76011026 2.6751e+11 71997
    ## - self_reference_min_shares      1  109361002 2.6754e+11 71998
    ## - title_sentiment_polarity       1  109718522 2.6754e+11 71998
    ## - max_negative_polarity          1  112787411 2.6755e+11 71998
    ## - LDA_01                         1  118322518 2.6755e+11 71998
    ## - min_positive_polarity          1  119034983 2.6755e+11 71998
    ## - num_self_hrefs                 1  119477262 2.6755e+11 71998
    ## <none>                                        2.6743e+11 71998
    ## - global_subjectivity            1  161206917 2.6760e+11 71999
    ## - n_non_stop_unique_tokens       1  181067257 2.6762e+11 71999
    ## - data_channel_is_tech           1  196459479 2.6763e+11 71999
    ## - data_channel_is_lifestyle      1  234273888 2.6767e+11 72000
    ## - timedelta                      1  251970523 2.6769e+11 72000
    ## - data_channel_is_world          1  276604708 2.6771e+11 72000
    ## - n_unique_tokens                1  362819351 2.6780e+11 72002
    ## - data_channel_is_bus            1  486644960 2.6792e+11 72003
    ## - n_tokens_content               1  493388919 2.6793e+11 72004
    ## - data_channel_is_entertainment  1 1105705863 2.6854e+11 72013
    ## 
    ## Step:  AIC=71996.17
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - max_positive_polarity          1    1678809 2.6744e+11 71994
    ## - self_reference_max_shares      1    1939227 2.6744e+11 71994
    ## - kw_avg_min                     1    3926331 2.6744e+11 71994
    ## - kw_min_avg                     1    5156348 2.6744e+11 71994
    ## - kw_min_min                     1    6062460 2.6744e+11 71994
    ## - global_rate_negative_words     1    6155632 2.6744e+11 71994
    ## - LDA_02                         1   15134797 2.6745e+11 71994
    ## - avg_negative_polarity          1   16515326 2.6745e+11 71994
    ## - kw_avg_max                     1   16612741 2.6745e+11 71994
    ## - kw_max_min                     1   17332741 2.6745e+11 71994
    ## - LDA_03                         1   18067411 2.6745e+11 71994
    ## - LDA_00                         1   20168978 2.6745e+11 71994
    ## - num_hrefs                      1   20716037 2.6746e+11 71994
    ## - num_imgs                       1   22172985 2.6746e+11 71995
    ## - kw_max_avg                     1   22819229 2.6746e+11 71995
    ## - title_subjectivity             1   23576650 2.6746e+11 71995
    ## - rate_positive_words            1   25998934 2.6746e+11 71995
    ## - num_videos                     1   30267631 2.6747e+11 71995
    ## - n_tokens_title                 1   35245011 2.6747e+11 71995
    ## - data_channel_is_socmed         1   37991760 2.6747e+11 71995
    ## - abs_title_sentiment_polarity   1   38489206 2.6747e+11 71995
    ## - kw_avg_avg                     1   40309516 2.6748e+11 71995
    ## - avg_positive_polarity          1   44164110 2.6748e+11 71995
    ## - kw_min_max                     1   45025877 2.6748e+11 71995
    ## - min_negative_polarity          1   45411181 2.6748e+11 71995
    ## - global_rate_positive_words     1   62817577 2.6750e+11 71995
    ## - kw_max_max                     1   64569274 2.6750e+11 71995
    ## - abs_title_subjectivity         1   72598808 2.6751e+11 71995
    ## - num_keywords                   1   75948119 2.6751e+11 71995
    ## - self_reference_min_shares      1  109381512 2.6754e+11 71996
    ## - title_sentiment_polarity       1  109673306 2.6754e+11 71996
    ## - max_negative_polarity          1  112847682 2.6755e+11 71996
    ## - LDA_01                         1  118447884 2.6755e+11 71996
    ## - min_positive_polarity          1  119510983 2.6755e+11 71996
    ## - num_self_hrefs                 1  119561338 2.6755e+11 71996
    ## <none>                                        2.6743e+11 71996
    ## - global_subjectivity            1  161113627 2.6760e+11 71997
    ## - n_non_stop_unique_tokens       1  181386136 2.6762e+11 71997
    ## - data_channel_is_tech           1  196362917 2.6763e+11 71997
    ## - data_channel_is_lifestyle      1  234176229 2.6767e+11 71998
    ## - timedelta                      1  251873745 2.6769e+11 71998
    ## - data_channel_is_world          1  276583036 2.6771e+11 71998
    ## - n_non_stop_words               1  294296730 2.6773e+11 71999
    ## - n_unique_tokens                1  364444646 2.6780e+11 72000
    ## - data_channel_is_bus            1  486659225 2.6792e+11 72001
    ## - n_tokens_content               1  494259398 2.6793e+11 72002
    ## - data_channel_is_entertainment  1 1105780930 2.6854e+11 72011
    ## 
    ## Step:  AIC=71994.2
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - self_reference_max_shares      1    1988052 2.6744e+11 71992
    ## - kw_avg_min                     1    3938064 2.6744e+11 71992
    ## - kw_min_avg                     1    5143439 2.6744e+11 71992
    ## - global_rate_negative_words     1    5826326 2.6744e+11 71992
    ## - kw_min_min                     1    6035092 2.6744e+11 71992
    ## - LDA_02                         1   14973265 2.6745e+11 71992
    ## - avg_negative_polarity          1   16599997 2.6745e+11 71992
    ## - kw_avg_max                     1   16828555 2.6745e+11 71992
    ## - kw_max_min                     1   17207954 2.6745e+11 71992
    ## - LDA_03                         1   18328425 2.6745e+11 71992
    ## - num_hrefs                      1   19932691 2.6746e+11 71992
    ## - LDA_00                         1   20266201 2.6746e+11 71993
    ## - num_imgs                       1   22059557 2.6746e+11 71993
    ## - kw_max_avg                     1   22985887 2.6746e+11 71993
    ## - title_subjectivity             1   23592132 2.6746e+11 71993
    ## - rate_positive_words            1   27230065 2.6746e+11 71993
    ## - num_videos                     1   30611144 2.6747e+11 71993
    ## - n_tokens_title                 1   36107174 2.6747e+11 71993
    ## - data_channel_is_socmed         1   38511596 2.6747e+11 71993
    ## - abs_title_sentiment_polarity   1   38511897 2.6747e+11 71993
    ## - kw_avg_avg                     1   39858774 2.6748e+11 71993
    ## - kw_min_max                     1   45247123 2.6748e+11 71993
    ## - min_negative_polarity          1   46656107 2.6748e+11 71993
    ## - global_rate_positive_words     1   61221130 2.6750e+11 71993
    ## - kw_max_max                     1   64825151 2.6750e+11 71993
    ## - abs_title_subjectivity         1   72538992 2.6751e+11 71993
    ## - num_keywords                   1   76326968 2.6751e+11 71993
    ## - avg_positive_polarity          1   93744712 2.6753e+11 71994
    ## - self_reference_min_shares      1  108992726 2.6755e+11 71994
    ## - title_sentiment_polarity       1  109088690 2.6755e+11 71994
    ## - max_negative_polarity          1  114205906 2.6755e+11 71994
    ## - LDA_01                         1  119107683 2.6756e+11 71994
    ## - num_self_hrefs                 1  120462481 2.6756e+11 71994
    ## <none>                                        2.6744e+11 71994
    ## - min_positive_polarity          1  137330528 2.6757e+11 71994
    ## - global_subjectivity            1  159485250 2.6760e+11 71995
    ## - n_non_stop_unique_tokens       1  179709548 2.6762e+11 71995
    ## - data_channel_is_tech           1  196912877 2.6763e+11 71995
    ## - data_channel_is_lifestyle      1  235116644 2.6767e+11 71996
    ## - timedelta                      1  253327015 2.6769e+11 71996
    ## - data_channel_is_world          1  279250244 2.6772e+11 71996
    ## - n_non_stop_words               1  292657707 2.6773e+11 71997
    ## - n_unique_tokens                1  365515697 2.6780e+11 71998
    ## - data_channel_is_bus            1  489381856 2.6793e+11 71999
    ## - n_tokens_content               1  499192311 2.6794e+11 72000
    ## - data_channel_is_entertainment  1 1106999807 2.6854e+11 72009
    ## 
    ## Step:  AIC=71992.23
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_avg_min                     1    3187717 2.6744e+11 71990
    ## - kw_min_avg                     1    4938337 2.6744e+11 71990
    ## - global_rate_negative_words     1    5698350 2.6744e+11 71990
    ## - kw_min_min                     1    6016723 2.6744e+11 71990
    ## - LDA_02                         1   15366678 2.6745e+11 71990
    ## - kw_avg_max                     1   16647674 2.6746e+11 71990
    ## - avg_negative_polarity          1   16710288 2.6746e+11 71990
    ## - LDA_03                         1   18762880 2.6746e+11 71991
    ## - kw_max_min                     1   18872956 2.6746e+11 71991
    ## - LDA_00                         1   20093101 2.6746e+11 71991
    ## - num_hrefs                      1   20325798 2.6746e+11 71991
    ## - kw_max_avg                     1   22168371 2.6746e+11 71991
    ## - num_imgs                       1   22268932 2.6746e+11 71991
    ## - title_subjectivity             1   23141068 2.6746e+11 71991
    ## - rate_positive_words            1   27547844 2.6747e+11 71991
    ## - num_videos                     1   29021228 2.6747e+11 71991
    ## - n_tokens_title                 1   35932758 2.6747e+11 71991
    ## - kw_avg_avg                     1   38864406 2.6748e+11 71991
    ## - abs_title_sentiment_polarity   1   38896374 2.6748e+11 71991
    ## - data_channel_is_socmed         1   39333647 2.6748e+11 71991
    ## - kw_min_max                     1   44962647 2.6748e+11 71991
    ## - min_negative_polarity          1   46336688 2.6748e+11 71991
    ## - global_rate_positive_words     1   61445669 2.6750e+11 71991
    ## - kw_max_max                     1   65447233 2.6750e+11 71991
    ## - abs_title_subjectivity         1   71582250 2.6751e+11 71991
    ## - num_keywords                   1   77133291 2.6752e+11 71991
    ## - avg_positive_polarity          1   94377793 2.6753e+11 71992
    ## - title_sentiment_polarity       1  109243004 2.6755e+11 71992
    ## - max_negative_polarity          1  114414598 2.6755e+11 71992
    ## - LDA_01                         1  118915757 2.6756e+11 71992
    ## - num_self_hrefs                 1  127625868 2.6757e+11 71992
    ## - self_reference_min_shares      1  129268059 2.6757e+11 71992
    ## <none>                                        2.6744e+11 71992
    ## - min_positive_polarity          1  137379749 2.6758e+11 71992
    ## - global_subjectivity            1  158315575 2.6760e+11 71993
    ## - n_non_stop_unique_tokens       1  179745669 2.6762e+11 71993
    ## - data_channel_is_tech           1  200000754 2.6764e+11 71993
    ## - data_channel_is_lifestyle      1  236922805 2.6768e+11 71994
    ## - timedelta                      1  252328845 2.6769e+11 71994
    ## - data_channel_is_world          1  281054986 2.6772e+11 71994
    ## - n_non_stop_words               1  292633594 2.6773e+11 71995
    ## - n_unique_tokens                1  365272518 2.6780e+11 71996
    ## - data_channel_is_bus            1  492315317 2.6793e+11 71998
    ## - n_tokens_content               1  500598887 2.6794e+11 71998
    ## - data_channel_is_entertainment  1 1112321757 2.6855e+11 72007
    ## 
    ## Step:  AIC=71990.28
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + LDA_00 + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_min_avg                     1    3353915 2.6744e+11 71988
    ## - kw_min_min                     1    5600707 2.6745e+11 71988
    ## - global_rate_negative_words     1    5778813 2.6745e+11 71988
    ## - LDA_02                         1   15888311 2.6746e+11 71989
    ## - avg_negative_polarity          1   16883464 2.6746e+11 71989
    ## - LDA_03                         1   19746073 2.6746e+11 71989
    ## - LDA_00                         1   19854139 2.6746e+11 71989
    ## - num_hrefs                      1   20208182 2.6746e+11 71989
    ## - kw_avg_max                     1   20237689 2.6746e+11 71989
    ## - num_imgs                       1   21970603 2.6746e+11 71989
    ## - title_subjectivity             1   23333140 2.6746e+11 71989
    ## - rate_positive_words            1   27368628 2.6747e+11 71989
    ## - num_videos                     1   29372222 2.6747e+11 71989
    ## - kw_max_avg                     1   31036141 2.6747e+11 71989
    ## - n_tokens_title                 1   35551659 2.6748e+11 71989
    ## - kw_avg_avg                     1   36105952 2.6748e+11 71989
    ## - abs_title_sentiment_polarity   1   38517518 2.6748e+11 71989
    ## - data_channel_is_socmed         1   40238171 2.6748e+11 71989
    ## - kw_min_max                     1   45585333 2.6749e+11 71989
    ## - min_negative_polarity          1   46275792 2.6749e+11 71989
    ## - global_rate_positive_words     1   61390393 2.6750e+11 71989
    ## - kw_max_max                     1   65771927 2.6751e+11 71989
    ## - abs_title_subjectivity         1   72179608 2.6751e+11 71989
    ## - num_keywords                   1   82445299 2.6752e+11 71990
    ## - avg_positive_polarity          1   94441190 2.6754e+11 71990
    ## - title_sentiment_polarity       1  108953963 2.6755e+11 71990
    ## - max_negative_polarity          1  114599290 2.6756e+11 71990
    ## - LDA_01                         1  120160334 2.6756e+11 71990
    ## - num_self_hrefs                 1  128010390 2.6757e+11 71990
    ## - self_reference_min_shares      1  129120626 2.6757e+11 71990
    ## <none>                                        2.6744e+11 71990
    ## - min_positive_polarity          1  137568018 2.6758e+11 71990
    ## - global_subjectivity            1  160629133 2.6760e+11 71991
    ## - n_non_stop_unique_tokens       1  180665374 2.6762e+11 71991
    ## - data_channel_is_tech           1  201555486 2.6764e+11 71991
    ## - data_channel_is_lifestyle      1  236434443 2.6768e+11 71992
    ## - timedelta                      1  249584729 2.6769e+11 71992
    ## - data_channel_is_world          1  282528651 2.6772e+11 71992
    ## - n_non_stop_words               1  293265418 2.6773e+11 71993
    ## - kw_max_min                     1  320361828 2.6776e+11 71993
    ## - n_unique_tokens                1  366527104 2.6781e+11 71994
    ## - data_channel_is_bus            1  495581286 2.6794e+11 71996
    ## - n_tokens_content               1  501195310 2.6794e+11 71996
    ## - data_channel_is_entertainment  1 1124986148 2.6857e+11 72005
    ## 
    ## Step:  AIC=71988.33
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_min_max + kw_max_max + kw_avg_max + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + LDA_00 + LDA_01 + 
    ##     LDA_02 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_min_min                     1    5355702 2.6745e+11 71986
    ## - global_rate_negative_words     1    5650623 2.6745e+11 71986
    ## - LDA_02                         1   15796592 2.6746e+11 71987
    ## - avg_negative_polarity          1   16878147 2.6746e+11 71987
    ## - num_hrefs                      1   19700809 2.6746e+11 71987
    ## - kw_avg_max                     1   20418360 2.6747e+11 71987
    ## - LDA_00                         1   20445779 2.6747e+11 71987
    ## - LDA_03                         1   20923540 2.6747e+11 71987
    ## - num_imgs                       1   21704947 2.6747e+11 71987
    ## - title_subjectivity             1   22947676 2.6747e+11 71987
    ## - rate_positive_words            1   27652789 2.6747e+11 71987
    ## - num_videos                     1   29386298 2.6747e+11 71987
    ## - n_tokens_title                 1   35602012 2.6748e+11 71987
    ## - kw_avg_avg                     1   37383504 2.6748e+11 71987
    ## - abs_title_sentiment_polarity   1   38608484 2.6748e+11 71987
    ## - data_channel_is_socmed         1   41409998 2.6749e+11 71987
    ## - min_negative_polarity          1   46050719 2.6749e+11 71987
    ## - kw_min_max                     1   51856737 2.6750e+11 71987
    ## - kw_max_avg                     1   58169372 2.6750e+11 71987
    ## - global_rate_positive_words     1   61556226 2.6751e+11 71987
    ## - kw_max_max                     1   65989105 2.6751e+11 71987
    ## - abs_title_subjectivity         1   71819511 2.6752e+11 71987
    ## - num_keywords                   1   89728932 2.6753e+11 71988
    ## - avg_positive_polarity          1   93605603 2.6754e+11 71988
    ## - title_sentiment_polarity       1  108202981 2.6755e+11 71988
    ## - max_negative_polarity          1  114278771 2.6756e+11 71988
    ## - LDA_01                         1  122463238 2.6757e+11 71988
    ## - self_reference_min_shares      1  128530350 2.6757e+11 71988
    ## - num_self_hrefs                 1  132053949 2.6758e+11 71988
    ## <none>                                        2.6744e+11 71988
    ## - min_positive_polarity          1  137743997 2.6758e+11 71988
    ## - global_subjectivity            1  162159065 2.6761e+11 71989
    ## - n_non_stop_unique_tokens       1  179799305 2.6762e+11 71989
    ## - data_channel_is_tech           1  205806976 2.6765e+11 71989
    ## - data_channel_is_lifestyle      1  237623604 2.6768e+11 71990
    ## - timedelta                      1  250048470 2.6769e+11 71990
    ## - data_channel_is_world          1  289819911 2.6773e+11 71991
    ## - n_non_stop_words               1  293958323 2.6774e+11 71991
    ## - kw_max_min                     1  318353383 2.6776e+11 71991
    ## - n_unique_tokens                1  365291151 2.6781e+11 71992
    ## - n_tokens_content               1  498930996 2.6794e+11 71994
    ## - data_channel_is_bus            1  502762817 2.6795e+11 71994
    ## - data_channel_is_entertainment  1 1152984578 2.6860e+11 72003
    ## 
    ## Step:  AIC=71986.41
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_max_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - global_rate_negative_words     1    5800671 2.6746e+11 71984
    ## - LDA_02                         1   16572664 2.6747e+11 71985
    ## - avg_negative_polarity          1   16806438 2.6747e+11 71985
    ## - num_hrefs                      1   19731839 2.6747e+11 71985
    ## - kw_avg_max                     1   19897594 2.6747e+11 71985
    ## - LDA_00                         1   20325191 2.6747e+11 71985
    ## - num_imgs                       1   21013693 2.6747e+11 71985
    ## - LDA_03                         1   21124765 2.6747e+11 71985
    ## - title_subjectivity             1   22954611 2.6747e+11 71985
    ## - rate_positive_words            1   27493418 2.6748e+11 71985
    ## - num_videos                     1   31237549 2.6748e+11 71985
    ## - n_tokens_title                 1   35521155 2.6749e+11 71985
    ## - kw_avg_avg                     1   37687144 2.6749e+11 71985
    ## - abs_title_sentiment_polarity   1   38071566 2.6749e+11 71985
    ## - data_channel_is_socmed         1   41240965 2.6749e+11 71985
    ## - min_negative_polarity          1   46469175 2.6750e+11 71985
    ## - kw_min_max                     1   51139608 2.6750e+11 71985
    ## - kw_max_avg                     1   57723282 2.6751e+11 71985
    ## - global_rate_positive_words     1   60888842 2.6751e+11 71985
    ## - abs_title_subjectivity         1   71349877 2.6752e+11 71985
    ## - num_keywords                   1   90668374 2.6754e+11 71986
    ## - kw_max_max                     1   91247148 2.6754e+11 71986
    ## - avg_positive_polarity          1   92238514 2.6754e+11 71986
    ## - title_sentiment_polarity       1  107167278 2.6756e+11 71986
    ## - max_negative_polarity          1  114220654 2.6756e+11 71986
    ## - LDA_01                         1  125088322 2.6758e+11 71986
    ## - self_reference_min_shares      1  128730928 2.6758e+11 71986
    ## - num_self_hrefs                 1  134023343 2.6758e+11 71986
    ## <none>                                        2.6745e+11 71986
    ## - min_positive_polarity          1  137457089 2.6759e+11 71986
    ## - global_subjectivity            1  161440328 2.6761e+11 71987
    ## - n_non_stop_unique_tokens       1  179892221 2.6763e+11 71987
    ## - data_channel_is_tech           1  205919626 2.6766e+11 71987
    ## - data_channel_is_lifestyle      1  237201672 2.6769e+11 71988
    ## - timedelta                      1  260373601 2.6771e+11 71988
    ## - data_channel_is_world          1  287434868 2.6774e+11 71989
    ## - n_non_stop_words               1  292986185 2.6774e+11 71989
    ## - kw_max_min                     1  317759135 2.6777e+11 71989
    ## - n_unique_tokens                1  365418250 2.6782e+11 71990
    ## - n_tokens_content               1  498369343 2.6795e+11 71992
    ## - data_channel_is_bus            1  500899852 2.6795e+11 71992
    ## - data_channel_is_entertainment  1 1163101913 2.6861e+11 72002
    ## 
    ## Step:  AIC=71984.49
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_max_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_rate_positive_words + rate_positive_words + 
    ##     avg_positive_polarity + min_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - LDA_02                         1   16658427 2.6747e+11 71983
    ## - avg_negative_polarity          1   17010790 2.6747e+11 71983
    ## - num_hrefs                      1   18076667 2.6747e+11 71983
    ## - kw_avg_max                     1   20027802 2.6748e+11 71983
    ## - LDA_00                         1   20264140 2.6748e+11 71983
    ## - num_imgs                       1   20989540 2.6748e+11 71983
    ## - LDA_03                         1   21044369 2.6748e+11 71983
    ## - title_subjectivity             1   22907451 2.6748e+11 71983
    ## - num_videos                     1   29210068 2.6749e+11 71983
    ## - n_tokens_title                 1   35368110 2.6749e+11 71983
    ## - kw_avg_avg                     1   37271996 2.6749e+11 71983
    ## - abs_title_sentiment_polarity   1   39465489 2.6750e+11 71983
    ## - data_channel_is_socmed         1   40210812 2.6750e+11 71983
    ## - min_negative_polarity          1   44773542 2.6750e+11 71983
    ## - kw_min_max                     1   52208699 2.6751e+11 71983
    ## - kw_max_avg                     1   58196959 2.6751e+11 71983
    ## - abs_title_subjectivity         1   69411210 2.6753e+11 71984
    ## - avg_positive_polarity          1   89582744 2.6755e+11 71984
    ## - num_keywords                   1   90519756 2.6755e+11 71984
    ## - kw_max_max                     1   90557485 2.6755e+11 71984
    ## - max_negative_polarity          1  111460577 2.6757e+11 71984
    ## - title_sentiment_polarity       1  113799953 2.6757e+11 71984
    ## - LDA_01                         1  123791467 2.6758e+11 71984
    ## - self_reference_min_shares      1  128768372 2.6758e+11 71984
    ## - min_positive_polarity          1  133853069 2.6759e+11 71984
    ## - num_self_hrefs                 1  133885700 2.6759e+11 71984
    ## <none>                                        2.6746e+11 71984
    ## - global_rate_positive_words     1  143862033 2.6760e+11 71985
    ## - global_subjectivity            1  156541416 2.6761e+11 71985
    ## - n_non_stop_unique_tokens       1  178248912 2.6763e+11 71985
    ## - rate_positive_words            1  184596202 2.6764e+11 71985
    ## - data_channel_is_tech           1  205320381 2.6766e+11 71986
    ## - data_channel_is_lifestyle      1  236543123 2.6769e+11 71986
    ## - timedelta                      1  257369030 2.6771e+11 71986
    ## - data_channel_is_world          1  284704348 2.6774e+11 71987
    ## - kw_max_min                     1  317767245 2.6777e+11 71987
    ## - n_unique_tokens                1  362449791 2.6782e+11 71988
    ## - n_non_stop_words               1  498421003 2.6795e+11 71990
    ## - n_tokens_content               1  498737218 2.6795e+11 71990
    ## - data_channel_is_bus            1  498825052 2.6795e+11 71990
    ## - data_channel_is_entertainment  1 1160320187 2.6862e+11 72000
    ## 
    ## Step:  AIC=71982.74
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_max_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + LDA_00 + LDA_01 + LDA_03 + global_subjectivity + 
    ##     global_rate_positive_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - avg_negative_polarity          1   16854229 2.6749e+11 71981
    ## - kw_avg_max                     1   17404726 2.6749e+11 71981
    ## - num_hrefs                      1   19514769 2.6749e+11 71981
    ## - num_imgs                       1   20363847 2.6749e+11 71981
    ## - title_subjectivity             1   24444744 2.6750e+11 71981
    ## - num_videos                     1   30515700 2.6750e+11 71981
    ## - n_tokens_title                 1   35736570 2.6751e+11 71981
    ## - abs_title_sentiment_polarity   1   37266224 2.6751e+11 71981
    ## - data_channel_is_socmed         1   41021690 2.6751e+11 71981
    ## - kw_avg_avg                     1   42729414 2.6752e+11 71981
    ## - min_negative_polarity          1   43913759 2.6752e+11 71981
    ## - LDA_00                         1   47315728 2.6752e+11 71981
    ## - kw_min_max                     1   50380349 2.6752e+11 71981
    ## - kw_max_avg                     1   53629070 2.6753e+11 71982
    ## - LDA_03                         1   54287584 2.6753e+11 71982
    ## - abs_title_subjectivity         1   70183632 2.6754e+11 71982
    ## - avg_positive_polarity          1   87489360 2.6756e+11 71982
    ## - kw_max_max                     1   90779049 2.6756e+11 71982
    ## - num_keywords                   1   93930899 2.6757e+11 71982
    ## - max_negative_polarity          1  111743624 2.6758e+11 71982
    ## - title_sentiment_polarity       1  112735066 2.6759e+11 71982
    ## - self_reference_min_shares      1  129497871 2.6760e+11 71983
    ## - num_self_hrefs                 1  130958075 2.6760e+11 71983
    ## - min_positive_polarity          1  133014185 2.6761e+11 71983
    ## <none>                                        2.6747e+11 71983
    ## - global_rate_positive_words     1  143011818 2.6762e+11 71983
    ## - global_subjectivity            1  163224771 2.6764e+11 71983
    ## - n_non_stop_unique_tokens       1  175870170 2.6765e+11 71983
    ## - rate_positive_words            1  187331136 2.6766e+11 71984
    ## - data_channel_is_tech           1  189540527 2.6766e+11 71984
    ## - LDA_01                         1  204142438 2.6768e+11 71984
    ## - data_channel_is_lifestyle      1  219885669 2.6769e+11 71984
    ## - timedelta                      1  253276062 2.6773e+11 71985
    ## - kw_max_min                     1  320883112 2.6779e+11 71986
    ## - n_unique_tokens                1  359128017 2.6783e+11 71986
    ## - data_channel_is_world          1  370984220 2.6784e+11 71986
    ## - data_channel_is_bus            1  485975885 2.6796e+11 71988
    ## - n_tokens_content               1  493979380 2.6797e+11 71988
    ## - n_non_stop_words               1  501644554 2.6797e+11 71988
    ## - data_channel_is_entertainment  1 1166933807 2.6864e+11 71998
    ## 
    ## Step:  AIC=71980.99
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_max_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + LDA_00 + LDA_01 + LDA_03 + global_subjectivity + 
    ##     global_rate_positive_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_avg_max                     1   18056409 2.6751e+11 71979
    ## - num_hrefs                      1   18543202 2.6751e+11 71979
    ## - num_imgs                       1   18969159 2.6751e+11 71979
    ## - title_subjectivity             1   25474756 2.6752e+11 71979
    ## - num_videos                     1   32628670 2.6752e+11 71979
    ## - abs_title_sentiment_polarity   1   34039163 2.6752e+11 71980
    ## - n_tokens_title                 1   34703727 2.6752e+11 71980
    ## - kw_avg_avg                     1   42073105 2.6753e+11 71980
    ## - data_channel_is_socmed         1   42530881 2.6753e+11 71980
    ## - LDA_00                         1   48115041 2.6754e+11 71980
    ## - kw_min_max                     1   51291628 2.6754e+11 71980
    ## - LDA_03                         1   54679669 2.6754e+11 71980
    ## - kw_max_avg                     1   55433091 2.6755e+11 71980
    ## - abs_title_subjectivity         1   70429201 2.6756e+11 71980
    ## - avg_positive_polarity          1   85725034 2.6758e+11 71980
    ## - kw_max_max                     1   88280509 2.6758e+11 71980
    ## - num_keywords                   1   91987823 2.6758e+11 71980
    ## - title_sentiment_polarity       1  105542835 2.6760e+11 71981
    ## - self_reference_min_shares      1  128738281 2.6762e+11 71981
    ## - num_self_hrefs                 1  130406673 2.6762e+11 71981
    ## - max_negative_polarity          1  133597110 2.6762e+11 71981
    ## <none>                                        2.6749e+11 71981
    ## - min_positive_polarity          1  136867357 2.6763e+11 71981
    ## - global_rate_positive_words     1  142352222 2.6763e+11 71981
    ## - n_non_stop_unique_tokens       1  174849715 2.6766e+11 71982
    ## - rate_positive_words            1  179033934 2.6767e+11 71982
    ## - global_subjectivity            1  189590314 2.6768e+11 71982
    ## - data_channel_is_tech           1  190599362 2.6768e+11 71982
    ## - LDA_01                         1  205793198 2.6770e+11 71982
    ## - data_channel_is_lifestyle      1  219451529 2.6771e+11 71982
    ## - min_negative_polarity          1  238852640 2.6773e+11 71983
    ## - timedelta                      1  252172304 2.6774e+11 71983
    ## - kw_max_min                     1  320808471 2.6781e+11 71984
    ## - n_unique_tokens                1  356439544 2.6785e+11 71984
    ## - data_channel_is_world          1  374062485 2.6786e+11 71985
    ## - n_tokens_content               1  477533699 2.6797e+11 71986
    ## - data_channel_is_bus            1  489267422 2.6798e+11 71986
    ## - n_non_stop_words               1  494136041 2.6798e+11 71986
    ## - data_channel_is_entertainment  1 1158412233 2.6865e+11 71996
    ## 
    ## Step:  AIC=71979.26
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_max_min + 
    ##     kw_min_max + kw_max_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_00 + LDA_01 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_imgs                       1   20703273 2.6753e+11 71978
    ## - num_hrefs                      1   20901987 2.6753e+11 71978
    ## - title_subjectivity             1   25485170 2.6753e+11 71978
    ## - abs_title_sentiment_polarity   1   33610316 2.6754e+11 71978
    ## - kw_min_max                     1   35817307 2.6754e+11 71978
    ## - n_tokens_title                 1   36849027 2.6754e+11 71978
    ## - num_videos                     1   38182338 2.6755e+11 71978
    ## - kw_max_avg                     1   45421925 2.6755e+11 71978
    ## - LDA_00                         1   49477916 2.6756e+11 71978
    ## - data_channel_is_socmed         1   55663393 2.6756e+11 71978
    ## - LDA_03                         1   61018676 2.6757e+11 71978
    ## - kw_avg_avg                     1   65778458 2.6757e+11 71978
    ## - abs_title_subjectivity         1   70209395 2.6758e+11 71978
    ## - num_keywords                   1   74471372 2.6758e+11 71978
    ## - avg_positive_polarity          1   84764762 2.6759e+11 71979
    ## - title_sentiment_polarity       1  106265968 2.6761e+11 71979
    ## - num_self_hrefs                 1  126332368 2.6763e+11 71979
    ## - self_reference_min_shares      1  131673893 2.6764e+11 71979
    ## - max_negative_polarity          1  132379045 2.6764e+11 71979
    ## <none>                                        2.6751e+11 71979
    ## - min_positive_polarity          1  138945812 2.6765e+11 71979
    ## - global_rate_positive_words     1  143515852 2.6765e+11 71979
    ## - kw_max_max                     1  165953044 2.6767e+11 71980
    ## - n_non_stop_unique_tokens       1  175386099 2.6768e+11 71980
    ## - rate_positive_words            1  180736023 2.6769e+11 71980
    ## - global_subjectivity            1  190151457 2.6770e+11 71980
    ## - LDA_01                         1  203093620 2.6771e+11 71980
    ## - data_channel_is_tech           1  204091017 2.6771e+11 71980
    ## - timedelta                      1  234208389 2.6774e+11 71981
    ## - min_negative_polarity          1  238757682 2.6775e+11 71981
    ## - data_channel_is_lifestyle      1  242919203 2.6775e+11 71981
    ## - kw_max_min                     1  336498960 2.6784e+11 71982
    ## - n_unique_tokens                1  364879878 2.6787e+11 71983
    ## - data_channel_is_world          1  400625897 2.6791e+11 71983
    ## - data_channel_is_bus            1  483832209 2.6799e+11 71984
    ## - n_tokens_content               1  490687206 2.6800e+11 71985
    ## - n_non_stop_words               1  514912011 2.6802e+11 71985
    ## - data_channel_is_entertainment  1 1283380169 2.6879e+11 71996
    ## 
    ## Step:  AIC=71977.57
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_max_min + 
    ##     kw_min_max + kw_max_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_00 + LDA_01 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_hrefs                      1   21252560 2.6755e+11 71976
    ## - title_subjectivity             1   25541555 2.6755e+11 71976
    ## - abs_title_sentiment_polarity   1   33256786 2.6756e+11 71976
    ## - kw_min_max                     1   35731794 2.6756e+11 71976
    ## - n_tokens_title                 1   37440698 2.6757e+11 71976
    ## - kw_max_avg                     1   47190667 2.6758e+11 71976
    ## - LDA_00                         1   49496543 2.6758e+11 71976
    ## - num_videos                     1   53752300 2.6758e+11 71976
    ## - data_channel_is_socmed         1   54042959 2.6758e+11 71976
    ## - LDA_03                         1   55125590 2.6758e+11 71976
    ## - kw_avg_avg                     1   61504889 2.6759e+11 71976
    ## - abs_title_subjectivity         1   68085372 2.6760e+11 71977
    ## - num_keywords                   1   73262028 2.6760e+11 71977
    ## - avg_positive_polarity          1   79871232 2.6761e+11 71977
    ## - title_sentiment_polarity       1  102868065 2.6763e+11 71977
    ## - max_negative_polarity          1  131015433 2.6766e+11 71978
    ## - self_reference_min_shares      1  131345266 2.6766e+11 71978
    ## - num_self_hrefs                 1  131359590 2.6766e+11 71978
    ## <none>                                        2.6753e+11 71978
    ## - min_positive_polarity          1  135037429 2.6766e+11 71978
    ## - global_rate_positive_words     1  135898394 2.6766e+11 71978
    ## - n_non_stop_unique_tokens       1  154697066 2.6768e+11 71978
    ## - kw_max_max                     1  163089859 2.6769e+11 71978
    ## - rate_positive_words            1  180362074 2.6771e+11 71978
    ## - global_subjectivity            1  190106228 2.6772e+11 71978
    ## - LDA_01                         1  195259012 2.6772e+11 71978
    ## - data_channel_is_tech           1  204077431 2.6773e+11 71979
    ## - timedelta                      1  229560863 2.6776e+11 71979
    ## - min_negative_polarity          1  238883367 2.6777e+11 71979
    ## - data_channel_is_lifestyle      1  240915587 2.6777e+11 71979
    ## - kw_max_min                     1  330449373 2.6786e+11 71980
    ## - n_unique_tokens                1  344182444 2.6787e+11 71981
    ## - data_channel_is_world          1  398643583 2.6793e+11 71982
    ## - n_tokens_content               1  472101310 2.6800e+11 71983
    ## - data_channel_is_bus            1  477758311 2.6801e+11 71983
    ## - n_non_stop_words               1  567771374 2.6810e+11 71984
    ## - data_channel_is_entertainment  1 1276691045 2.6881e+11 71995
    ## 
    ## Step:  AIC=71975.89
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_self_hrefs + 
    ##     num_videos + num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_min_max + kw_max_max + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + LDA_00 + 
    ##     LDA_01 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - title_subjectivity             1   25756438 2.6758e+11 71974
    ## - abs_title_sentiment_polarity   1   33049637 2.6758e+11 71974
    ## - kw_min_max                     1   34656659 2.6758e+11 71974
    ## - n_tokens_title                 1   42010235 2.6759e+11 71975
    ## - kw_max_avg                     1   49129890 2.6760e+11 71975
    ## - LDA_00                         1   49243541 2.6760e+11 71975
    ## - num_videos                     1   50023101 2.6760e+11 71975
    ## - data_channel_is_socmed         1   52218308 2.6760e+11 71975
    ## - LDA_03                         1   55196587 2.6760e+11 71975
    ## - kw_avg_avg                     1   57484568 2.6761e+11 71975
    ## - abs_title_subjectivity         1   67456760 2.6762e+11 71975
    ## - num_keywords                   1   69967178 2.6762e+11 71975
    ## - avg_positive_polarity          1   72705161 2.6762e+11 71975
    ## - title_sentiment_polarity       1  102806902 2.6765e+11 71975
    ## - global_rate_positive_words     1  123193031 2.6767e+11 71976
    ## - max_negative_polarity          1  123825967 2.6767e+11 71976
    ## - min_positive_polarity          1  125541345 2.6768e+11 71976
    ## - self_reference_min_shares      1  133359794 2.6768e+11 71976
    ## <none>                                        2.6755e+11 71976
    ## - n_non_stop_unique_tokens       1  135731890 2.6769e+11 71976
    ## - kw_max_max                     1  160585341 2.6771e+11 71976
    ## - rate_positive_words            1  171380505 2.6772e+11 71976
    ## - global_subjectivity            1  182389106 2.6773e+11 71977
    ## - num_self_hrefs                 1  183481550 2.6773e+11 71977
    ## - data_channel_is_tech           1  194743135 2.6774e+11 71977
    ## - LDA_01                         1  198595552 2.6775e+11 71977
    ## - timedelta                      1  225186477 2.6777e+11 71977
    ## - min_negative_polarity          1  230390932 2.6778e+11 71977
    ## - data_channel_is_lifestyle      1  242334935 2.6779e+11 71977
    ## - n_unique_tokens                1  324059066 2.6787e+11 71979
    ## - kw_max_min                     1  325515480 2.6788e+11 71979
    ## - data_channel_is_world          1  392794521 2.6794e+11 71980
    ## - n_tokens_content               1  450889393 2.6800e+11 71981
    ## - data_channel_is_bus            1  468484205 2.6802e+11 71981
    ## - n_non_stop_words               1  586531485 2.6814e+11 71983
    ## - data_channel_is_entertainment  1 1258668732 2.6881e+11 71993
    ## 
    ## Step:  AIC=71974.27
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_self_hrefs + 
    ##     num_videos + num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_min_max + kw_max_max + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + LDA_00 + 
    ##     LDA_01 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_min_max                     1   35137721 2.6761e+11 71973
    ## - n_tokens_title                 1   40649572 2.6762e+11 71973
    ## - abs_title_subjectivity         1   48286079 2.6762e+11 71973
    ## - kw_max_avg                     1   48798988 2.6762e+11 71973
    ## - LDA_00                         1   50602831 2.6763e+11 71973
    ## - num_videos                     1   50662170 2.6763e+11 71973
    ## - data_channel_is_socmed         1   52076533 2.6763e+11 71973
    ## - LDA_03                         1   52795953 2.6763e+11 71973
    ## - kw_avg_avg                     1   57360116 2.6763e+11 71973
    ## - num_keywords                   1   69919598 2.6765e+11 71973
    ## - avg_positive_polarity          1   76139765 2.6765e+11 71973
    ## - title_sentiment_polarity       1  117956291 2.6769e+11 71974
    ## - min_positive_polarity          1  123100049 2.6770e+11 71974
    ## - max_negative_polarity          1  124843696 2.6770e+11 71974
    ## - global_rate_positive_words     1  128252566 2.6770e+11 71974
    ## - self_reference_min_shares      1  133675072 2.6771e+11 71974
    ## <none>                                        2.6758e+11 71974
    ## - n_non_stop_unique_tokens       1  134578799 2.6771e+11 71974
    ## - abs_title_sentiment_polarity   1  142588999 2.6772e+11 71974
    ## - kw_max_max                     1  166874658 2.6774e+11 71975
    ## - global_subjectivity            1  169614803 2.6775e+11 71975
    ## - rate_positive_words            1  175325210 2.6775e+11 71975
    ## - num_self_hrefs                 1  186906031 2.6776e+11 71975
    ## - data_channel_is_tech           1  194987644 2.6777e+11 71975
    ## - LDA_01                         1  197457840 2.6777e+11 71975
    ## - timedelta                      1  230993593 2.6781e+11 71976
    ## - min_negative_polarity          1  235544368 2.6781e+11 71976
    ## - data_channel_is_lifestyle      1  241492330 2.6782e+11 71976
    ## - n_unique_tokens                1  321651841 2.6790e+11 71977
    ## - kw_max_min                     1  324762975 2.6790e+11 71977
    ## - data_channel_is_world          1  394768850 2.6797e+11 71978
    ## - n_tokens_content               1  445628075 2.6802e+11 71979
    ## - data_channel_is_bus            1  472117757 2.6805e+11 71979
    ## - n_non_stop_words               1  579051595 2.6815e+11 71981
    ## - data_channel_is_entertainment  1 1256004589 2.6883e+11 71991
    ## 
    ## Step:  AIC=71972.8
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_self_hrefs + 
    ##     num_videos + num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_max_max + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + LDA_00 + LDA_01 + 
    ##     LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - n_tokens_title                 1   39746462 2.6765e+11 71971
    ## - kw_avg_avg                     1   42207758 2.6765e+11 71971
    ## - num_videos                     1   47144893 2.6766e+11 71971
    ## - abs_title_subjectivity         1   49065727 2.6766e+11 71972
    ## - LDA_00                         1   51569234 2.6766e+11 71972
    ## - LDA_03                         1   55223417 2.6767e+11 71972
    ## - data_channel_is_socmed         1   56275394 2.6767e+11 71972
    ## - kw_max_avg                     1   57419417 2.6767e+11 71972
    ## - avg_positive_polarity          1   76189717 2.6769e+11 71972
    ## - num_keywords                   1  104795216 2.6772e+11 71972
    ## - title_sentiment_polarity       1  113579725 2.6772e+11 71972
    ## - max_negative_polarity          1  124075362 2.6773e+11 71973
    ## - min_positive_polarity          1  125943108 2.6774e+11 71973
    ## - global_rate_positive_words     1  130324857 2.6774e+11 71973
    ## - n_non_stop_unique_tokens       1  133190593 2.6774e+11 71973
    ## <none>                                        2.6761e+11 71973
    ## - self_reference_min_shares      1  134687567 2.6775e+11 71973
    ## - abs_title_sentiment_polarity   1  141316520 2.6775e+11 71973
    ## - kw_max_max                     1  169332280 2.6778e+11 71973
    ## - global_subjectivity            1  175569331 2.6779e+11 71973
    ## - rate_positive_words            1  176291679 2.6779e+11 71973
    ## - num_self_hrefs                 1  187477758 2.6780e+11 71974
    ## - LDA_01                         1  199452051 2.6781e+11 71974
    ## - data_channel_is_tech           1  200685995 2.6781e+11 71974
    ## - min_negative_polarity          1  232007910 2.6784e+11 71974
    ## - timedelta                      1  233361140 2.6784e+11 71974
    ## - data_channel_is_lifestyle      1  247064617 2.6786e+11 71974
    ## - kw_max_min                     1  305007978 2.6792e+11 71975
    ## - n_unique_tokens                1  318196897 2.6793e+11 71976
    ## - data_channel_is_world          1  403055154 2.6801e+11 71977
    ## - n_tokens_content               1  442600523 2.6805e+11 71977
    ## - data_channel_is_bus            1  482065290 2.6809e+11 71978
    ## - n_non_stop_words               1  575418365 2.6819e+11 71979
    ## - data_channel_is_entertainment  1 1271004156 2.6888e+11 71990
    ## 
    ## Step:  AIC=71971.39
    ## shares ~ timedelta + n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_self_hrefs + num_videos + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_max_max + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + LDA_00 + LDA_01 + 
    ##     LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_avg_avg                     1   40256221 2.6769e+11 71970
    ## - num_videos                     1   50227302 2.6770e+11 71970
    ## - LDA_00                         1   50322980 2.6770e+11 71970
    ## - data_channel_is_socmed         1   56919056 2.6771e+11 71970
    ## - LDA_03                         1   59068235 2.6771e+11 71970
    ## - kw_max_avg                     1   60245318 2.6771e+11 71970
    ## - abs_title_subjectivity         1   63640421 2.6771e+11 71970
    ## - avg_positive_polarity          1   78779087 2.6773e+11 71971
    ## - num_keywords                   1  102991675 2.6775e+11 71971
    ## - title_sentiment_polarity       1  112636155 2.6776e+11 71971
    ## - max_negative_polarity          1  119545100 2.6777e+11 71971
    ## - min_positive_polarity          1  127516162 2.6778e+11 71971
    ## - n_non_stop_unique_tokens       1  128548509 2.6778e+11 71971
    ## - global_rate_positive_words     1  130268053 2.6778e+11 71971
    ## <none>                                        2.6765e+11 71971
    ## - self_reference_min_shares      1  134969648 2.6779e+11 71971
    ## - abs_title_sentiment_polarity   1  140993856 2.6779e+11 71971
    ## - kw_max_max                     1  163079169 2.6781e+11 71972
    ## - global_subjectivity            1  171977885 2.6782e+11 71972
    ## - rate_positive_words            1  172813208 2.6782e+11 71972
    ## - num_self_hrefs                 1  188197330 2.6784e+11 71972
    ## - data_channel_is_tech           1  195947323 2.6785e+11 71972
    ## - LDA_01                         1  203609933 2.6785e+11 71972
    ## - timedelta                      1  207223363 2.6786e+11 71972
    ## - min_negative_polarity          1  228547743 2.6788e+11 71973
    ## - data_channel_is_lifestyle      1  247022634 2.6790e+11 71973
    ## - kw_max_min                     1  305331779 2.6796e+11 71974
    ## - n_unique_tokens                1  308733362 2.6796e+11 71974
    ## - data_channel_is_world          1  392633611 2.6804e+11 71975
    ## - n_tokens_content               1  437840540 2.6809e+11 71976
    ## - data_channel_is_bus            1  473214974 2.6812e+11 71976
    ## - n_non_stop_words               1  568134739 2.6822e+11 71978
    ## - data_channel_is_entertainment  1 1240555857 2.6889e+11 71988
    ## 
    ## Step:  AIC=71969.99
    ## shares ~ timedelta + n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_self_hrefs + num_videos + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_max_max + kw_max_avg + 
    ##     self_reference_min_shares + LDA_00 + LDA_01 + LDA_03 + global_subjectivity + 
    ##     global_rate_positive_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_videos                     1   49704593 2.6774e+11 71969
    ## - LDA_00                         1   54615276 2.6775e+11 71969
    ## - abs_title_subjectivity         1   64284932 2.6775e+11 71969
    ## - data_channel_is_socmed         1   67885581 2.6776e+11 71969
    ## - LDA_03                         1   79589348 2.6777e+11 71969
    ## - avg_positive_polarity          1   81389553 2.6777e+11 71969
    ## - num_keywords                   1   84275436 2.6777e+11 71969
    ## - title_sentiment_polarity       1  110860020 2.6780e+11 71970
    ## - max_negative_polarity          1  119450278 2.6781e+11 71970
    ## - global_rate_positive_words     1  123493689 2.6781e+11 71970
    ## - min_positive_polarity          1  128458922 2.6782e+11 71970
    ## <none>                                        2.6769e+11 71970
    ## - n_non_stop_unique_tokens       1  137990721 2.6783e+11 71970
    ## - self_reference_min_shares      1  139791214 2.6783e+11 71970
    ## - abs_title_sentiment_polarity   1  143709205 2.6783e+11 71970
    ## - rate_positive_words            1  167896929 2.6786e+11 71970
    ## - global_subjectivity            1  179148752 2.6787e+11 71971
    ## - num_self_hrefs                 1  180279607 2.6787e+11 71971
    ## - timedelta                      1  190877223 2.6788e+11 71971
    ## - kw_max_max                     1  205997690 2.6790e+11 71971
    ## - LDA_01                         1  213800545 2.6790e+11 71971
    ## - data_channel_is_tech           1  219571182 2.6791e+11 71971
    ## - min_negative_polarity          1  226748864 2.6792e+11 71971
    ## - data_channel_is_lifestyle      1  258392232 2.6795e+11 71972
    ## - kw_max_min                     1  303250665 2.6799e+11 71973
    ## - n_unique_tokens                1  324191357 2.6801e+11 71973
    ## - n_tokens_content               1  444211252 2.6813e+11 71975
    ## - data_channel_is_world          1  468748983 2.6816e+11 71975
    ## - data_channel_is_bus            1  524149290 2.6821e+11 71976
    ## - kw_max_avg                     1  526453489 2.6822e+11 71976
    ## - n_non_stop_words               1  587088035 2.6828e+11 71977
    ## - data_channel_is_entertainment  1 1449329942 2.6914e+11 71990
    ## 
    ## Step:  AIC=71968.73
    ## shares ~ timedelta + n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_self_hrefs + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_max_max + kw_max_avg + 
    ##     self_reference_min_shares + LDA_00 + LDA_01 + LDA_03 + global_subjectivity + 
    ##     global_rate_positive_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - LDA_00                         1   53986454 2.6779e+11 71968
    ## - abs_title_subjectivity         1   65806419 2.6781e+11 71968
    ## - data_channel_is_socmed         1   69164988 2.6781e+11 71968
    ## - num_keywords                   1   80207579 2.6782e+11 71968
    ## - avg_positive_polarity          1   90030417 2.6783e+11 71968
    ## - LDA_03                         1   94731017 2.6783e+11 71968
    ## - global_rate_positive_words     1  114295355 2.6785e+11 71968
    ## - title_sentiment_polarity       1  116566663 2.6786e+11 71968
    ## - max_negative_polarity          1  128381848 2.6787e+11 71969
    ## <none>                                        2.6774e+11 71969
    ## - min_positive_polarity          1  137556479 2.6788e+11 71969
    ## - n_non_stop_unique_tokens       1  139217652 2.6788e+11 71969
    ## - self_reference_min_shares      1  139542725 2.6788e+11 71969
    ## - abs_title_sentiment_polarity   1  146217486 2.6789e+11 71969
    ## - rate_positive_words            1  157487857 2.6790e+11 71969
    ## - num_self_hrefs                 1  162714540 2.6790e+11 71969
    ## - global_subjectivity            1  177568303 2.6792e+11 71969
    ## - timedelta                      1  194807261 2.6794e+11 71970
    ## - kw_max_max                     1  204970971 2.6795e+11 71970
    ## - LDA_01                         1  206516324 2.6795e+11 71970
    ## - data_channel_is_tech           1  223309011 2.6796e+11 71970
    ## - min_negative_polarity          1  239555324 2.6798e+11 71970
    ## - data_channel_is_lifestyle      1  263532467 2.6800e+11 71971
    ## - kw_max_min                     1  295882150 2.6804e+11 71971
    ## - n_unique_tokens                1  338706366 2.6808e+11 71972
    ## - n_tokens_content               1  468835749 2.6821e+11 71974
    ## - data_channel_is_world          1  470132562 2.6821e+11 71974
    ## - kw_max_avg                     1  519640607 2.6826e+11 71974
    ## - data_channel_is_bus            1  524892967 2.6827e+11 71975
    ## - n_non_stop_words               1  604392627 2.6834e+11 71976
    ## - data_channel_is_entertainment  1 1427948749 2.6917e+11 71988
    ## 
    ## Step:  AIC=71967.53
    ## shares ~ timedelta + n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_self_hrefs + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_max_max + kw_max_avg + 
    ##     self_reference_min_shares + LDA_01 + LDA_03 + global_subjectivity + 
    ##     global_rate_positive_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - data_channel_is_socmed         1   48999600 2.6784e+11 71966
    ## - LDA_03                         1   63332926 2.6786e+11 71966
    ## - abs_title_subjectivity         1   65919294 2.6786e+11 71967
    ## - num_keywords                   1   77743401 2.6787e+11 71967
    ## - avg_positive_polarity          1   96175567 2.6789e+11 71967
    ## - global_rate_positive_words     1  109871701 2.6790e+11 71967
    ## - title_sentiment_polarity       1  114717830 2.6791e+11 71967
    ## - max_negative_polarity          1  125736748 2.6792e+11 71967
    ## <none>                                        2.6779e+11 71968
    ## - self_reference_min_shares      1  139635029 2.6793e+11 71968
    ## - n_non_stop_unique_tokens       1  140757306 2.6793e+11 71968
    ## - abs_title_sentiment_polarity   1  144784757 2.6794e+11 71968
    ## - min_positive_polarity          1  150300153 2.6794e+11 71968
    ## - rate_positive_words            1  159793379 2.6795e+11 71968
    ## - LDA_01                         1  165552960 2.6796e+11 71968
    ## - num_self_hrefs                 1  166866290 2.6796e+11 71968
    ## - global_subjectivity            1  174941250 2.6797e+11 71968
    ## - timedelta                      1  196483859 2.6799e+11 71968
    ## - kw_max_max                     1  204693286 2.6800e+11 71969
    ## - min_negative_polarity          1  241018660 2.6804e+11 71969
    ## - data_channel_is_lifestyle      1  285191188 2.6808e+11 71970
    ## - data_channel_is_tech           1  286252620 2.6808e+11 71970
    ## - kw_max_min                     1  296577019 2.6809e+11 71970
    ## - n_unique_tokens                1  344922747 2.6814e+11 71971
    ## - n_tokens_content               1  472027989 2.6827e+11 71973
    ## - data_channel_is_bus            1  478606082 2.6827e+11 71973
    ## - kw_max_avg                     1  529811983 2.6832e+11 71973
    ## - data_channel_is_world          1  571440048 2.6837e+11 71974
    ## - n_non_stop_words               1  617075992 2.6841e+11 71975
    ## - data_channel_is_entertainment  1 1426910572 2.6922e+11 71987
    ## 
    ## Step:  AIC=71966.26
    ## shares ~ timedelta + n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_self_hrefs + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_tech + data_channel_is_world + 
    ##     kw_max_min + kw_max_max + kw_max_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - abs_title_subjectivity         1   70021099 2.6791e+11 71965
    ## - avg_positive_polarity          1   95775495 2.6794e+11 71966
    ## - num_keywords                   1   95836435 2.6794e+11 71966
    ## - title_sentiment_polarity       1  110010722 2.6795e+11 71966
    ## - global_rate_positive_words     1  111783005 2.6796e+11 71966
    ## - max_negative_polarity          1  127446615 2.6797e+11 71966
    ## <none>                                        2.6784e+11 71966
    ## - self_reference_min_shares      1  139455412 2.6798e+11 71966
    ## - min_positive_polarity          1  143649773 2.6799e+11 71966
    ## - abs_title_sentiment_polarity   1  144942171 2.6799e+11 71966
    ## - rate_positive_words            1  151082699 2.6799e+11 71967
    ## - n_non_stop_unique_tokens       1  153369954 2.6800e+11 71967
    ## - num_self_hrefs                 1  171099719 2.6801e+11 71967
    ## - global_subjectivity            1  184887387 2.6803e+11 71967
    ## - timedelta                      1  194225267 2.6804e+11 71967
    ## - LDA_03                         1  199696469 2.6804e+11 71967
    ## - kw_max_max                     1  203950666 2.6805e+11 71967
    ## - data_channel_is_lifestyle      1  242699974 2.6809e+11 71968
    ## - min_negative_polarity          1  252088384 2.6810e+11 71968
    ## - data_channel_is_tech           1  266604912 2.6811e+11 71968
    ## - LDA_01                         1  286783183 2.6813e+11 71969
    ## - kw_max_min                     1  306067193 2.6815e+11 71969
    ## - n_unique_tokens                1  367548177 2.6821e+11 71970
    ## - n_tokens_content               1  474517348 2.6832e+11 71971
    ## - data_channel_is_bus            1  523917452 2.6837e+11 71972
    ## - kw_max_avg                     1  537794354 2.6838e+11 71972
    ## - n_non_stop_words               1  631495701 2.6847e+11 71974
    ## - data_channel_is_world          1  652924816 2.6850e+11 71974
    ## - data_channel_is_entertainment  1 1439880458 2.6928e+11 71986
    ## 
    ## Step:  AIC=71965.31
    ## shares ~ timedelta + n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_self_hrefs + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_tech + data_channel_is_world + 
    ##     kw_max_min + kw_max_max + kw_max_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - avg_positive_polarity          1   85087672 2.6800e+11 71965
    ## - global_rate_positive_words     1   89630206 2.6800e+11 71965
    ## - abs_title_sentiment_polarity   1   97605742 2.6801e+11 71965
    ## - num_keywords                   1   97852207 2.6801e+11 71965
    ## - title_sentiment_polarity       1  121321738 2.6803e+11 71965
    ## - max_negative_polarity          1  128868290 2.6804e+11 71965
    ## <none>                                        2.6791e+11 71965
    ## - min_positive_polarity          1  136673676 2.6805e+11 71965
    ## - self_reference_min_shares      1  140673787 2.6805e+11 71965
    ## - rate_positive_words            1  141333633 2.6805e+11 71965
    ## - n_non_stop_unique_tokens       1  161254745 2.6807e+11 71966
    ## - global_subjectivity            1  175474308 2.6809e+11 71966
    ## - num_self_hrefs                 1  175576713 2.6809e+11 71966
    ## - timedelta                      1  186449902 2.6810e+11 71966
    ## - LDA_03                         1  194143385 2.6811e+11 71966
    ## - kw_max_max                     1  203792662 2.6812e+11 71966
    ## - min_negative_polarity          1  241704096 2.6815e+11 71967
    ## - data_channel_is_lifestyle      1  250552432 2.6816e+11 71967
    ## - data_channel_is_tech           1  271729689 2.6818e+11 71967
    ## - LDA_01                         1  285855852 2.6820e+11 71968
    ## - kw_max_min                     1  307298135 2.6822e+11 71968
    ## - n_unique_tokens                1  374962856 2.6829e+11 71969
    ## - n_tokens_content               1  472602349 2.6839e+11 71970
    ## - data_channel_is_bus            1  523700990 2.6844e+11 71971
    ## - kw_max_avg                     1  538079754 2.6845e+11 71971
    ## - n_non_stop_words               1  609053515 2.6852e+11 71972
    ## - data_channel_is_world          1  659226630 2.6857e+11 71973
    ## - data_channel_is_entertainment  1 1422408575 2.6934e+11 71984
    ## 
    ## Step:  AIC=71964.57
    ## shares ~ timedelta + n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_self_hrefs + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_tech + data_channel_is_world + 
    ##     kw_max_min + kw_max_max + kw_max_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     rate_positive_words + min_positive_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - min_positive_polarity          1   69393021 2.6807e+11 71964
    ## - global_rate_positive_words     1   75212883 2.6807e+11 71964
    ## - abs_title_sentiment_polarity   1   90830788 2.6809e+11 71964
    ## - num_keywords                   1   97586794 2.6810e+11 71964
    ## <none>                                        2.6800e+11 71965
    ## - title_sentiment_polarity       1  135520739 2.6813e+11 71965
    ## - max_negative_polarity          1  139502312 2.6814e+11 71965
    ## - self_reference_min_shares      1  148637907 2.6815e+11 71965
    ## - n_non_stop_unique_tokens       1  151184528 2.6815e+11 71965
    ## - rate_positive_words            1  164034773 2.6816e+11 71965
    ## - num_self_hrefs                 1  171946104 2.6817e+11 71965
    ## - timedelta                      1  197970388 2.6820e+11 71966
    ## - kw_max_max                     1  206471908 2.6820e+11 71966
    ## - LDA_03                         1  207259181 2.6821e+11 71966
    ## - data_channel_is_lifestyle      1  249306454 2.6825e+11 71966
    ## - global_subjectivity            1  258714759 2.6826e+11 71966
    ## - min_negative_polarity          1  259381737 2.6826e+11 71966
    ## - data_channel_is_tech           1  281092604 2.6828e+11 71967
    ## - LDA_01                         1  283094417 2.6828e+11 71967
    ## - kw_max_min                     1  314884211 2.6831e+11 71967
    ## - n_unique_tokens                1  352685920 2.6835e+11 71968
    ## - n_tokens_content               1  497889248 2.6850e+11 71970
    ## - data_channel_is_bus            1  531596848 2.6853e+11 71970
    ## - kw_max_avg                     1  552427211 2.6855e+11 71971
    ## - n_non_stop_words               1  555531689 2.6855e+11 71971
    ## - data_channel_is_world          1  679914381 2.6868e+11 71973
    ## - data_channel_is_entertainment  1 1414609972 2.6941e+11 71984
    ## 
    ## Step:  AIC=71963.61
    ## shares ~ timedelta + n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_self_hrefs + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_tech + data_channel_is_world + 
    ##     kw_max_min + kw_max_max + kw_max_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     rate_positive_words + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - global_rate_positive_words     1   50896745 2.6812e+11 71962
    ## - abs_title_sentiment_polarity   1   92197405 2.6816e+11 71963
    ## - num_keywords                   1   93975263 2.6816e+11 71963
    ## - n_non_stop_unique_tokens       1  131115216 2.6820e+11 71964
    ## - title_sentiment_polarity       1  132308169 2.6820e+11 71964
    ## <none>                                        2.6807e+11 71964
    ## - max_negative_polarity          1  145821057 2.6821e+11 71964
    ## - self_reference_min_shares      1  150995520 2.6822e+11 71964
    ## - num_self_hrefs                 1  170333688 2.6824e+11 71964
    ## - rate_positive_words            1  189627954 2.6826e+11 71964
    ## - LDA_03                         1  197783936 2.6827e+11 71965
    ## - timedelta                      1  198152723 2.6827e+11 71965
    ## - kw_max_max                     1  211402760 2.6828e+11 71965
    ## - global_subjectivity            1  218318303 2.6829e+11 71965
    ## - data_channel_is_lifestyle      1  263660645 2.6833e+11 71966
    ## - LDA_01                         1  280645366 2.6835e+11 71966
    ## - min_negative_polarity          1  288238523 2.6836e+11 71966
    ## - data_channel_is_tech           1  300251804 2.6837e+11 71966
    ## - n_unique_tokens                1  307651284 2.6838e+11 71966
    ## - kw_max_min                     1  318600868 2.6839e+11 71966
    ## - n_tokens_content               1  502565512 2.6857e+11 71969
    ## - data_channel_is_bus            1  538981834 2.6861e+11 71970
    ## - kw_max_avg                     1  551903738 2.6862e+11 71970
    ## - n_non_stop_words               1  581343631 2.6865e+11 71970
    ## - data_channel_is_world          1  679413655 2.6875e+11 71972
    ## - data_channel_is_entertainment  1 1410478831 2.6948e+11 71983
    ## 
    ## Step:  AIC=71962.36
    ## shares ~ timedelta + n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_self_hrefs + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_tech + data_channel_is_world + 
    ##     kw_max_min + kw_max_max + kw_max_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_03 + global_subjectivity + rate_positive_words + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_keywords                   1   85201122 2.6820e+11 71962
    ## - abs_title_sentiment_polarity   1  103336248 2.6822e+11 71962
    ## - max_negative_polarity          1  127200434 2.6825e+11 71962
    ## - title_sentiment_polarity       1  127549455 2.6825e+11 71962
    ## <none>                                        2.6812e+11 71962
    ## - rate_positive_words            1  138757736 2.6826e+11 71962
    ## - n_non_stop_unique_tokens       1  145963790 2.6826e+11 71963
    ## - self_reference_min_shares      1  155285427 2.6827e+11 71963
    ## - num_self_hrefs                 1  164354945 2.6828e+11 71963
    ## - global_subjectivity            1  186147980 2.6830e+11 71963
    ## - timedelta                      1  190535342 2.6831e+11 71963
    ## - LDA_03                         1  212172013 2.6833e+11 71964
    ## - kw_max_max                     1  217041370 2.6834e+11 71964
    ## - data_channel_is_lifestyle      1  249781208 2.6837e+11 71964
    ## - min_negative_polarity          1  253105762 2.6837e+11 71964
    ## - LDA_01                         1  278988907 2.6840e+11 71965
    ## - data_channel_is_tech           1  286513879 2.6841e+11 71965
    ## - n_unique_tokens                1  304043697 2.6842e+11 71965
    ## - kw_max_min                     1  323565093 2.6844e+11 71965
    ## - n_tokens_content               1  481617989 2.6860e+11 71968
    ## - data_channel_is_bus            1  518450198 2.6864e+11 71968
    ## - n_non_stop_words               1  532353478 2.6865e+11 71968
    ## - kw_max_avg                     1  554234231 2.6867e+11 71969
    ## - data_channel_is_world          1  638681932 2.6876e+11 71970
    ## - data_channel_is_entertainment  1 1391152202 2.6951e+11 71981
    ## 
    ## Step:  AIC=71961.63
    ## shares ~ timedelta + n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_self_hrefs + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_max_max + kw_max_avg + 
    ##     self_reference_min_shares + LDA_01 + LDA_03 + global_subjectivity + 
    ##     rate_positive_words + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - abs_title_sentiment_polarity   1   98795306 2.6830e+11 71961
    ## - title_sentiment_polarity       1  125823552 2.6833e+11 71962
    ## <none>                                        2.6820e+11 71962
    ## - max_negative_polarity          1  136914296 2.6834e+11 71962
    ## - self_reference_min_shares      1  154365626 2.6836e+11 71962
    ## - rate_positive_words            1  155665050 2.6836e+11 71962
    ## - num_self_hrefs                 1  155929709 2.6836e+11 71962
    ## - n_non_stop_unique_tokens       1  161168173 2.6836e+11 71962
    ## - global_subjectivity            1  198256811 2.6840e+11 71963
    ## - timedelta                      1  208882200 2.6841e+11 71963
    ## - data_channel_is_lifestyle      1  216038538 2.6842e+11 71963
    ## - LDA_03                         1  225240385 2.6843e+11 71963
    ## - kw_max_max                     1  228052207 2.6843e+11 71963
    ## - data_channel_is_tech           1  255898130 2.6846e+11 71963
    ## - min_negative_polarity          1  259998903 2.6846e+11 71963
    ## - LDA_01                         1  284912506 2.6849e+11 71964
    ## - n_unique_tokens                1  322489364 2.6853e+11 71964
    ## - kw_max_min                     1  327700357 2.6853e+11 71965
    ## - n_tokens_content               1  498214015 2.6870e+11 71967
    ## - data_channel_is_bus            1  528641965 2.6873e+11 71967
    ## - n_non_stop_words               1  556335499 2.6876e+11 71968
    ## - kw_max_avg                     1  590910978 2.6879e+11 71968
    ## - data_channel_is_world          1  598083480 2.6880e+11 71969
    ## - data_channel_is_entertainment  1 1407316673 2.6961e+11 71981
    ## 
    ## Step:  AIC=71961.1
    ## shares ~ timedelta + n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_self_hrefs + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_max_max + kw_max_avg + 
    ##     self_reference_min_shares + LDA_01 + LDA_03 + global_subjectivity + 
    ##     rate_positive_words + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - title_sentiment_polarity       1   60479757 2.6836e+11 71960
    ## <none>                                        2.6830e+11 71961
    ## - max_negative_polarity          1  141038944 2.6844e+11 71961
    ## - self_reference_min_shares      1  150567617 2.6845e+11 71961
    ## - n_non_stop_unique_tokens       1  150585720 2.6845e+11 71961
    ## - num_self_hrefs                 1  152931984 2.6846e+11 71961
    ## - rate_positive_words            1  164540405 2.6847e+11 71962
    ## - global_subjectivity            1  169763966 2.6847e+11 71962
    ## - timedelta                      1  207113586 2.6851e+11 71962
    ## - data_channel_is_lifestyle      1  211335217 2.6851e+11 71962
    ## - LDA_03                         1  215450898 2.6852e+11 71962
    ## - kw_max_max                     1  230079793 2.6853e+11 71963
    ## - min_negative_polarity          1  239608138 2.6854e+11 71963
    ## - data_channel_is_tech           1  251602264 2.6855e+11 71963
    ## - LDA_01                         1  268695465 2.6857e+11 71963
    ## - n_unique_tokens                1  310240345 2.6861e+11 71964
    ## - kw_max_min                     1  324040534 2.6863e+11 71964
    ## - n_tokens_content               1  503721958 2.6881e+11 71967
    ## - data_channel_is_bus            1  523468682 2.6883e+11 71967
    ## - n_non_stop_words               1  538887300 2.6884e+11 71967
    ## - kw_max_avg                     1  582904700 2.6889e+11 71968
    ## - data_channel_is_world          1  601454101 2.6890e+11 71968
    ## - data_channel_is_entertainment  1 1384461855 2.6969e+11 71980
    ## 
    ## Step:  AIC=71960
    ## shares ~ timedelta + n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_self_hrefs + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_max_max + kw_max_avg + 
    ##     self_reference_min_shares + LDA_01 + LDA_03 + global_subjectivity + 
    ##     rate_positive_words + min_negative_polarity + max_negative_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## <none>                                        2.6836e+11 71960
    ## - max_negative_polarity          1  141787044 2.6850e+11 71960
    ## - num_self_hrefs                 1  151906487 2.6851e+11 71960
    ## - n_non_stop_unique_tokens       1  154088302 2.6852e+11 71960
    ## - self_reference_min_shares      1  154839418 2.6852e+11 71960
    ## - global_subjectivity            1  166690675 2.6853e+11 71960
    ## - rate_positive_words            1  203737976 2.6857e+11 71961
    ## - timedelta                      1  203790059 2.6857e+11 71961
    ## - data_channel_is_lifestyle      1  211486216 2.6857e+11 71961
    ## - kw_max_max                     1  220077586 2.6858e+11 71961
    ## - LDA_03                         1  221706820 2.6858e+11 71961
    ## - min_negative_polarity          1  233715610 2.6860e+11 71961
    ## - data_channel_is_tech           1  257844400 2.6862e+11 71962
    ## - LDA_01                         1  270649277 2.6863e+11 71962
    ## - n_unique_tokens                1  309862662 2.6867e+11 71963
    ## - kw_max_min                     1  328316445 2.6869e+11 71963
    ## - n_tokens_content               1  508161468 2.6887e+11 71966
    ## - data_channel_is_bus            1  525811263 2.6889e+11 71966
    ## - n_non_stop_words               1  548377573 2.6891e+11 71966
    ## - kw_max_avg                     1  589648512 2.6895e+11 71967
    ## - data_channel_is_world          1  609144442 2.6897e+11 71967
    ## - data_channel_is_entertainment  1 1401358576 2.6976e+11 71979

``` r
#summary the model
summary(lm.step)
```

    ## 
    ## Call:
    ## lm(formula = shares ~ timedelta + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_self_hrefs + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_tech + data_channel_is_world + 
    ##     kw_max_min + kw_max_max + kw_max_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_03 + global_subjectivity + rate_positive_words + 
    ##     min_negative_polarity + max_negative_polarity, data = train)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -14850  -2234  -1186    -96 227051 
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                    2.630e+03  1.211e+03   2.171  0.02998 *  
    ## timedelta                      1.444e+00  8.316e-01   1.736  0.08267 .  
    ## n_tokens_content               1.262e+00  4.604e-01   2.741  0.00615 ** 
    ## n_unique_tokens                8.275e+03  3.866e+03   2.140  0.03238 *  
    ## n_non_stop_words              -5.194e+03  1.824e+03  -2.848  0.00443 ** 
    ## n_non_stop_unique_tokens      -4.828e+03  3.199e+03  -1.509  0.13127    
    ## num_self_hrefs                -5.872e+01  3.918e+01  -1.499  0.13403    
    ## data_channel_is_lifestyle     -1.272e+03  7.192e+02  -1.768  0.07708 .  
    ## data_channel_is_entertainment -2.268e+03  4.983e+02  -4.552 5.47e-06 ***
    ## data_channel_is_bus           -1.598e+03  5.732e+02  -2.788  0.00532 ** 
    ## data_channel_is_tech          -1.102e+03  5.646e+02  -1.953  0.05094 .  
    ## data_channel_is_world         -1.658e+03  5.524e+02  -3.001  0.00271 ** 
    ## kw_max_min                    -8.918e-02  4.048e-02  -2.203  0.02763 *  
    ## kw_max_max                     1.454e-03  8.062e-04   1.804  0.07132 .  
    ## kw_max_avg                     8.826e-02  2.989e-02   2.953  0.00317 ** 
    ## self_reference_min_shares      8.562e-03  5.658e-03   1.513  0.13034    
    ## LDA_01                         1.858e+03  9.288e+02   2.000  0.04552 *  
    ## LDA_03                         1.346e+03  7.436e+02   1.811  0.07028 .  
    ## global_subjectivity            2.594e+03  1.652e+03   1.570  0.11651    
    ## rate_positive_words            1.753e+03  1.010e+03   1.736  0.08270 .  
    ## min_negative_polarity         -1.173e+03  6.312e+02  -1.859  0.06311 .  
    ## max_negative_polarity          2.059e+03  1.422e+03   1.448  0.14772    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 8224 on 3968 degrees of freedom
    ## Multiple R-squared:  0.03237,    Adjusted R-squared:  0.02725 
    ## F-statistic: 6.321 on 21 and 3968 DF,  p-value: < 2.2e-16

Calculate the predicted mean square error on the train set:

``` r
train.pred <- predict(lm.step, train[,-52])
mean((train.pred - train$shares)^2)
```

    ## [1] 67258920

So, the predicted mean square error on the training dataset is
264157333.

### On test set

Calculate the predicted mean square error on the test set:

``` r
test.pred <- predict(lm.step, test[,-52])
mean((test.pred - test$shares)^2)
```

    ## [1] 56821777

So, the predicted mean square error on the testing dataset is 85833271.

## Conclusions

In this project, random forest and backward stepwise regression were
used to predict the number of shares in social networks (popularity)
about articles published by Mashable in a period of two years. Random
forest is a kind of ensemble learning, which uses decision tree as weak
classifier. Backward stepwise regression gradually removes the less
significant variables on the basis that all variables are included in
the regression model. As for data collected about Monday, shares of
referenced articles in Mashable, average keyword and average shares of
worst keyword are variables considered important to the results by both
models. The random forest showed slight overfitting and backward
stepwise regression showed severe underfitting. However, the two
eventually performed similarly on the test set with random forest
slightly superior to backward stepwise regression. But this is not the
strictest way to compare the performance of the two models. Multiple
data sets can be divided and predicted to calculate the mean MSE.
