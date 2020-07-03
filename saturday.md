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

    ## [1] 2453

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

    ## [1] 1717

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
    ##   timedelta                   Min.   : 12.0      1st Qu.:166.0     
    ## n_tokens_title                Min.   : 5.00      1st Qu.: 9.00     
    ## n_tokens_content              Min.   :   0.0     1st Qu.: 275.0    
    ## n_unique_tokens               Min.   :0.0000     1st Qu.:0.4572    
    ## n_non_stop_words              Min.   :0.0000     1st Qu.:1.0000    
    ## n_non_stop_unique_tokens      Min.   :0.0000     1st Qu.:0.6099    
    ##   num_hrefs                   Min.   :  0.0      1st Qu.:  5.0     
    ## num_self_hrefs                Min.   : 0.000     1st Qu.: 1.000    
    ##    num_imgs                   Min.   :  0.000    1st Qu.:  1.000   
    ##   num_videos                  Min.   : 0.000     1st Qu.: 0.000    
    ## average_token_length          Min.   :0.000      1st Qu.:4.485     
    ##  num_keywords                 Min.   : 1.000     1st Qu.: 6.000    
    ## data_channel_is_lifestyle     Min.   :0.00000    1st Qu.:0.00000   
    ## data_channel_is_entertainment Min.   :0.0000     1st Qu.:0.0000    
    ## data_channel_is_bus           Min.   :0.00000    1st Qu.:0.00000   
    ## data_channel_is_socmed        Min.   :0.00000    1st Qu.:0.00000   
    ## data_channel_is_tech          Min.   :0.0000     1st Qu.:0.0000    
    ## data_channel_is_world         Min.   :0.0000     1st Qu.:0.0000    
    ##   kw_min_min                  Min.   : -1.00     1st Qu.: -1.00    
    ##   kw_max_min                  Min.   :    0      1st Qu.:  461     
    ##   kw_avg_min                  Min.   :  -1.0     1st Qu.: 143.2    
    ##   kw_min_max                  Min.   :     0     1st Qu.:     0    
    ##   kw_max_max                  Min.   : 37400     1st Qu.:843300    
    ##   kw_avg_max                  Min.   :  7178     1st Qu.:171138    
    ##   kw_min_avg                  Min.   :   0       1st Qu.:   0      
    ##   kw_max_avg                  Min.   :  2414     1st Qu.:  3582    
    ##   kw_avg_avg                  Min.   : 1115      1st Qu.: 2549     
    ## self_reference_min_shares     Min.   :     0     1st Qu.:   678    
    ## self_reference_max_shares     Min.   :     0     1st Qu.:  1100    
    ## self_reference_avg_sharess    Min.   :     0     1st Qu.:  1000    
    ##     LDA_00                    Min.   :0.01843    1st Qu.:0.02500   
    ##     LDA_01                    Min.   :0.01819    1st Qu.:0.02275   
    ##     LDA_02                    Min.   :0.01822    1st Qu.:0.02500   
    ##     LDA_03                    Min.   :0.01820    1st Qu.:0.02502   
    ##     LDA_04                    Min.   :0.01820    1st Qu.:0.02857   
    ## global_subjectivity           Min.   :0.0000     1st Qu.:0.4064    
    ## global_sentiment_polarity     Min.   :-0.39375   1st Qu.: 0.06188  
    ## global_rate_positive_words    Min.   :0.00000    1st Qu.:0.02899   
    ## global_rate_negative_words    Min.   :0.00000    1st Qu.:0.01033   
    ## rate_positive_words           Min.   :0.0000     1st Qu.:0.6000    
    ## rate_negative_words           Min.   :0.0000     1st Qu.:0.1860    
    ## avg_positive_polarity         Min.   :0.0000     1st Qu.:0.3145    
    ## min_positive_polarity         Min.   :0.00000    1st Qu.:0.05000   
    ## max_positive_polarity         Min.   :0.0000     1st Qu.:0.6000    
    ## avg_negative_polarity         Min.   :-1.0000    1st Qu.:-0.3380   
    ## min_negative_polarity         Min.   :-1.0000    1st Qu.:-0.8000   
    ## max_negative_polarity         Min.   :-1.0000    1st Qu.:-0.1250   
    ## title_subjectivity            Min.   :0.0000     1st Qu.:0.0000    
    ## title_sentiment_polarity      Min.   :-1.00000   1st Qu.: 0.00000  
    ## abs_title_subjectivity        Min.   :0.0000     1st Qu.:0.1250    
    ## abs_title_sentiment_polarity  Min.   :0.00000    1st Qu.:0.00000   
    ##     shares                    Min.   :    43     1st Qu.:  1400    
    ##                                                                    
    ##   timedelta                   Median :341.0      Mean   :353.7     
    ## n_tokens_title                Median :10.00      Mean   :10.28     
    ## n_tokens_content              Median : 500.0     Mean   : 604.8    
    ## n_unique_tokens               Median :0.5188     Mean   :0.5135    
    ## n_non_stop_words              Median :1.0000     Mean   :0.9633    
    ## n_non_stop_unique_tokens      Median :0.6742     Mean   :0.6552    
    ##   num_hrefs                   Median : 10.0      Mean   : 13.1     
    ## num_self_hrefs                Median : 3.000     Mean   : 3.936    
    ##    num_imgs                   Median :  1.000    Mean   :  5.579   
    ##   num_videos                  Median : 0.000     Mean   : 1.086    
    ## average_token_length          Median :4.665      Mean   :4.517     
    ##  num_keywords                 Median : 8.000     Mean   : 7.581    
    ## data_channel_is_lifestyle     Median :0.00000    Mean   :0.07513   
    ## data_channel_is_entertainment Median :0.0000     Mean   :0.1602    
    ## data_channel_is_bus           Median :0.00000    Mean   :0.09493   
    ## data_channel_is_socmed        Median :0.00000    Mean   :0.07455   
    ## data_channel_is_tech          Median :0.0000     Mean   :0.2149    
    ## data_channel_is_world         Median :0.0000     Mean   :0.2009    
    ##   kw_min_min                  Median : -1.00     Mean   : 23.84    
    ##   kw_max_min                  Median :  686      Mean   : 1047     
    ##   kw_avg_min                  Median : 245.5     Mean   : 295.9    
    ##   kw_min_max                  Median :  2000     Mean   : 14500    
    ##   kw_max_max                  Median :843300     Mean   :765046    
    ##   kw_avg_max                  Median :241071     Mean   :253288    
    ##   kw_min_avg                  Median :1300       Mean   :1278      
    ##   kw_max_avg                  Median :  4737     Mean   :  6096    
    ##   kw_avg_avg                  Median : 3057      Mean   : 3324     
    ## self_reference_min_shares     Median :  1300     Mean   :  3492    
    ## self_reference_max_shares     Median :  3000     Mean   : 10585    
    ## self_reference_avg_sharess    Median :  2350     Mean   :  5894    
    ##     LDA_00                    Median :0.03333    Mean   :0.16361   
    ##     LDA_01                    Median :0.03334    Mean   :0.14040   
    ##     LDA_02                    Median :0.04000    Mean   :0.21234   
    ##     LDA_03                    Median :0.04000    Mean   :0.23097   
    ##     LDA_04                    Median :0.05000    Mean   :0.25267   
    ## global_subjectivity           Median :0.4614     Mean   :0.4509    
    ## global_sentiment_polarity     Median : 0.12457   Mean   : 0.12502  
    ## global_rate_positive_words    Median :0.04110    Mean   :0.04118   
    ## global_rate_negative_words    Median :0.01584    Mean   :0.01685   
    ## rate_positive_words           Median :0.7143     Mean   :0.6790    
    ## rate_negative_words           Median :0.2735     Mean   :0.2843    
    ## avg_positive_polarity         Median :0.3667     Mean   :0.3591    
    ## min_positive_polarity         Median :0.10000    Mean   :0.09035   
    ## max_positive_polarity         Median :0.8000     Mean   :0.7815    
    ## avg_negative_polarity         Median :-0.2615    Mean   :-0.2681   
    ## min_negative_polarity         Median :-0.5000    Mean   :-0.5515   
    ## max_negative_polarity         Median :-0.1000    Mean   :-0.1059   
    ## title_subjectivity            Median :0.2333     Mean   :0.2986    
    ## title_sentiment_polarity      Median : 0.00000   Mean   : 0.09905  
    ## abs_title_subjectivity        Median :0.4333     Mean   :0.3252    
    ## abs_title_sentiment_polarity  Median :0.05714    Mean   :0.17648   
    ##     shares                    Median :  2000     Mean   :  4266    
    ##                                                                    
    ##   timedelta                   3rd Qu.:537.0      Max.   :726.0     
    ## n_tokens_title                3rd Qu.:12.00      Max.   :18.00     
    ## n_tokens_content              3rd Qu.: 781.0     Max.   :7034.0    
    ## n_unique_tokens               3rd Qu.:0.5939     Max.   :0.9574    
    ## n_non_stop_words              3rd Qu.:1.0000     Max.   :1.0000    
    ## n_non_stop_unique_tokens      3rd Qu.:0.7378     Max.   :1.0000    
    ##   num_hrefs                   3rd Qu.: 17.0      Max.   :104.0     
    ## num_self_hrefs                3rd Qu.: 4.000     Max.   :74.000    
    ##    num_imgs                   3rd Qu.:  9.000    Max.   :101.000   
    ##   num_videos                  3rd Qu.: 1.000     Max.   :50.000    
    ## average_token_length          3rd Qu.:4.858      Max.   :5.957     
    ##  num_keywords                 3rd Qu.: 9.000     Max.   :10.000    
    ## data_channel_is_lifestyle     3rd Qu.:0.00000    Max.   :1.00000   
    ## data_channel_is_entertainment 3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_bus           3rd Qu.:0.00000    Max.   :1.00000   
    ## data_channel_is_socmed        3rd Qu.:0.00000    Max.   :1.00000   
    ## data_channel_is_tech          3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_world         3rd Qu.:0.0000     Max.   :1.0000    
    ##   kw_min_min                  3rd Qu.:  4.00     Max.   :217.00    
    ##   kw_max_min                  3rd Qu.: 1100      Max.   :50000     
    ##   kw_avg_min                  3rd Qu.: 363.0     Max.   :8540.8    
    ##   kw_min_max                  3rd Qu.: 10800     Max.   :843300    
    ##   kw_max_max                  3rd Qu.:843300     Max.   :843300    
    ##   kw_avg_max                  3rd Qu.:319533     Max.   :843300    
    ##   kw_min_avg                  3rd Qu.:2207       Max.   :3594      
    ##   kw_max_avg                  3rd Qu.:  6830     Max.   :237967    
    ##   kw_avg_avg                  3rd Qu.: 3877      Max.   :36717     
    ## self_reference_min_shares     3rd Qu.:  2700     Max.   :158900    
    ## self_reference_max_shares     3rd Qu.:  8900     Max.   :837700    
    ## self_reference_avg_sharess    3rd Qu.:  5500     Max.   :309412    
    ##     LDA_00                    3rd Qu.:0.18109    Max.   :0.91998   
    ##     LDA_01                    3rd Qu.:0.15788    Max.   :0.91986   
    ##     LDA_02                    3rd Qu.:0.32749    Max.   :0.92000   
    ##     LDA_03                    3rd Qu.:0.39976    Max.   :0.91997   
    ##     LDA_04                    3rd Qu.:0.45281    Max.   :0.91999   
    ## global_subjectivity           3rd Qu.:0.5188     Max.   :0.8179    
    ## global_sentiment_polarity     3rd Qu.: 0.19080   Max.   : 0.60000  
    ## global_rate_positive_words    3rd Qu.:0.05289    Max.   :0.13065   
    ## global_rate_negative_words    3rd Qu.:0.02190    Max.   :0.13983   
    ## rate_positive_words           3rd Qu.:0.8000     Max.   :1.0000    
    ## rate_negative_words           3rd Qu.:0.3750     Max.   :1.0000    
    ## avg_positive_polarity         3rd Qu.:0.4203     Max.   :0.8333    
    ## min_positive_polarity         3rd Qu.:0.10000    Max.   :0.80000   
    ## max_positive_polarity         3rd Qu.:1.0000     Max.   :1.0000    
    ## avg_negative_polarity         3rd Qu.:-0.2000    Max.   : 0.0000   
    ## min_negative_polarity         3rd Qu.:-0.3889    Max.   : 0.0000   
    ## max_negative_polarity         3rd Qu.:-0.0500    Max.   : 0.0000   
    ## title_subjectivity            3rd Qu.:0.5000     Max.   :1.0000    
    ## title_sentiment_polarity      3rd Qu.: 0.25000   Max.   : 1.00000  
    ## abs_title_subjectivity        3rd Qu.:0.5000     Max.   :0.5000    
    ## abs_title_sentiment_polarity  3rd Qu.:0.30000    Max.   :1.00000   
    ##     shares                    3rd Qu.:  3600     Max.   :617900

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
    ##           Mean of squared residuals: 278902686
    ##                     % Var explained: -2.75

``` r
#variable importance measures
importance(rf)
```

    ##                                   %IncMSE IncNodePurity
    ## timedelta                      6.83419801   18990801897
    ## n_tokens_title                 1.04182285    2379624638
    ## n_tokens_content               2.38984298    2428830284
    ## n_unique_tokens                2.89559665    6259634741
    ## n_non_stop_words               2.23062935    4099320338
    ## n_non_stop_unique_tokens       2.39740683    4228182289
    ## num_hrefs                      1.66018814    4588066894
    ## num_self_hrefs                 2.47730700    1237260103
    ## num_imgs                       1.93961428    8214535688
    ## num_videos                    -0.07600338    2699229530
    ## average_token_length          -0.34549808    4594671503
    ## num_keywords                   0.63195297    2835428149
    ## data_channel_is_lifestyle      0.12753454      64630640
    ## data_channel_is_entertainment  1.74031340     394788815
    ## data_channel_is_bus            0.01540499     206183159
    ## data_channel_is_socmed         1.02867094      38161456
    ## data_channel_is_tech           1.44736933      50224337
    ## data_channel_is_world          1.51700630      68324778
    ## kw_min_min                     2.50857688    1437815598
    ## kw_max_min                     2.70620321   15419050577
    ## kw_avg_min                     2.38320737    7649893540
    ## kw_min_max                     2.83125681   42860100303
    ## kw_max_max                     5.48373351    9188193716
    ## kw_avg_max                     8.32445518  103072923579
    ## kw_min_avg                     1.36955769    4472860463
    ## kw_max_avg                     2.85411956   11996294569
    ## kw_avg_avg                     4.25964233    7348318955
    ## self_reference_min_shares      0.65345472    7422505796
    ## self_reference_max_shares      2.00324903   10518077483
    ## self_reference_avg_sharess     2.20712592   17033815299
    ## LDA_00                         0.92232168    5682920413
    ## LDA_01                         0.10288954    9096019427
    ## LDA_02                         2.67020318   14731317763
    ## LDA_03                         2.67573484   17473595303
    ## LDA_04                         0.82939870    8125397633
    ## global_subjectivity           -0.38553309   16636158752
    ## global_sentiment_polarity      1.82577310    3662755445
    ## global_rate_positive_words     1.19001470    3811187250
    ## global_rate_negative_words    -0.68196006    7826674422
    ## rate_positive_words           -1.43760151    8530848787
    ## rate_negative_words            0.05391239    4908153329
    ## avg_positive_polarity         -1.15406417    2480178636
    ## min_positive_polarity          1.17129716    2267890307
    ## max_positive_polarity          1.81716804    1195569370
    ## avg_negative_polarity          0.82734751    1346934598
    ## min_negative_polarity          3.30610744    3274668385
    ## max_negative_polarity          2.82651340    9502216180
    ## title_subjectivity             0.88499450    1321359391
    ## title_sentiment_polarity       1.95999574    2997113216
    ## abs_title_subjectivity         2.99131163     535914181
    ## abs_title_sentiment_polarity   2.35462950    1782642558

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

    ## [1] 53469834

So, the predicted mean square error on the training dataset is 68724773.

### On test set

``` r
rf.test <- predict(rf, newdata = test[,-52])
mean((test$shares-rf.test)^2)
```

    ## [1] 51390173

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

    ## Start:  AIC=33379.67
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
    ## Step:  AIC=33379.67
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
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ## 
    ## Step:  AIC=33379.67
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
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - self_reference_max_shares      1      96856 4.4922e+11 33378
    ## - avg_positive_polarity          1     722799 4.4922e+11 33378
    ## - n_non_stop_words               1    1440845 4.4922e+11 33378
    ## - title_sentiment_polarity       1    1858683 4.4922e+11 33378
    ## - LDA_00                         1    2406026 4.4922e+11 33378
    ## - kw_max_max                     1    3309719 4.4923e+11 33378
    ## - avg_negative_polarity          1    3807360 4.4923e+11 33378
    ## - kw_max_avg                     1    3812850 4.4923e+11 33378
    ## - kw_avg_avg                     1    5708707 4.4923e+11 33378
    ## - n_non_stop_unique_tokens       1    9342022 4.4923e+11 33378
    ## - abs_title_sentiment_polarity   1   10506819 4.4923e+11 33378
    ## - self_reference_avg_sharess     1   13395600 4.4924e+11 33378
    ## - min_negative_polarity          1   18845031 4.4924e+11 33378
    ## - global_sentiment_polarity      1   35844725 4.4926e+11 33378
    ## - kw_max_min                     1   40975666 4.4926e+11 33378
    ## - n_tokens_content               1   47625001 4.4927e+11 33378
    ## - n_unique_tokens                1   54026371 4.4928e+11 33378
    ## - kw_avg_min                     1   57511513 4.4928e+11 33378
    ## - num_imgs                       1   59260423 4.4928e+11 33378
    ## - num_hrefs                      1   81270217 4.4930e+11 33378
    ## - timedelta                      1   82017696 4.4930e+11 33378
    ## - LDA_02                         1   83104131 4.4931e+11 33378
    ## - num_keywords                   1   85183959 4.4931e+11 33378
    ## - LDA_01                         1   86010827 4.4931e+11 33378
    ## - LDA_03                         1   90259487 4.4931e+11 33378
    ## - title_subjectivity             1   92070261 4.4931e+11 33378
    ## - max_negative_polarity          1  101758257 4.4932e+11 33378
    ## - num_videos                     1  107139314 4.4933e+11 33378
    ## - average_token_length           1  113487800 4.4934e+11 33378
    ## - max_positive_polarity          1  128759188 4.4935e+11 33378
    ## - n_tokens_title                 1  174375371 4.4940e+11 33378
    ## - global_rate_positive_words     1  183390480 4.4941e+11 33378
    ## - min_positive_polarity          1  186383592 4.4941e+11 33378
    ## - kw_min_max                     1  205861942 4.4943e+11 33378
    ## - rate_positive_words            1  252282510 4.4947e+11 33379
    ## - self_reference_min_shares      1  296979017 4.4952e+11 33379
    ## - data_channel_is_bus            1  351634833 4.4957e+11 33379
    ## - global_rate_negative_words     1  364857372 4.4959e+11 33379
    ## - kw_avg_max                     1  370415759 4.4959e+11 33379
    ## - num_self_hrefs                 1  407222182 4.4963e+11 33379
    ## - kw_min_avg                     1  418909674 4.4964e+11 33379
    ## - global_subjectivity            1  496210490 4.4972e+11 33380
    ## <none>                                        4.4922e+11 33380
    ## - abs_title_subjectivity         1  836734958 4.5006e+11 33381
    ## - data_channel_is_tech           1  980932631 4.5020e+11 33381
    ## - data_channel_is_lifestyle      1 1041477923 4.5026e+11 33382
    ## - data_channel_is_entertainment  1 1112609703 4.5033e+11 33382
    ## - kw_min_min                     1 1142054835 4.5036e+11 33382
    ## - data_channel_is_socmed         1 1442790328 4.5067e+11 33383
    ## - data_channel_is_world          1 1650799884 4.5087e+11 33384
    ## 
    ## Step:  AIC=33377.67
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - avg_positive_polarity          1     724229 4.4922e+11 33376
    ## - n_non_stop_words               1    1458718 4.4922e+11 33376
    ## - title_sentiment_polarity       1    1841799 4.4922e+11 33376
    ## - LDA_00                         1    2427133 4.4922e+11 33376
    ## - kw_max_max                     1    3294228 4.4923e+11 33376
    ## - avg_negative_polarity          1    3746487 4.4923e+11 33376
    ## - kw_max_avg                     1    4155516 4.4923e+11 33376
    ## - kw_avg_avg                     1    5639300 4.4923e+11 33376
    ## - n_non_stop_unique_tokens       1    9322615 4.4923e+11 33376
    ## - abs_title_sentiment_polarity   1   10489530 4.4923e+11 33376
    ## - min_negative_polarity          1   18780016 4.4924e+11 33376
    ## - global_sentiment_polarity      1   35803872 4.4926e+11 33376
    ## - kw_max_min                     1   41003271 4.4926e+11 33376
    ## - n_tokens_content               1   47614277 4.4927e+11 33376
    ## - n_unique_tokens                1   54021102 4.4928e+11 33376
    ## - kw_avg_min                     1   57615594 4.4928e+11 33376
    ## - num_imgs                       1   59278841 4.4928e+11 33376
    ## - num_hrefs                      1   81200986 4.4930e+11 33376
    ## - timedelta                      1   82028174 4.4930e+11 33376
    ## - LDA_02                         1   83218752 4.4931e+11 33376
    ## - num_keywords                   1   85105805 4.4931e+11 33376
    ## - LDA_01                         1   85948096 4.4931e+11 33376
    ## - self_reference_avg_sharess     1   86423735 4.4931e+11 33376
    ## - LDA_03                         1   90554144 4.4931e+11 33376
    ## - title_subjectivity             1   92085556 4.4931e+11 33376
    ## - max_negative_polarity          1  101725841 4.4932e+11 33376
    ## - num_videos                     1  107907056 4.4933e+11 33376
    ## - average_token_length           1  113570159 4.4934e+11 33376
    ## - max_positive_polarity          1  128668777 4.4935e+11 33376
    ## - n_tokens_title                 1  174293766 4.4940e+11 33376
    ## - global_rate_positive_words     1  183336823 4.4941e+11 33376
    ## - min_positive_polarity          1  186348437 4.4941e+11 33376
    ## - kw_min_max                     1  205944008 4.4943e+11 33376
    ## - rate_positive_words            1  252287736 4.4947e+11 33377
    ## - data_channel_is_bus            1  351568481 4.4957e+11 33377
    ## - global_rate_negative_words     1  364795651 4.4959e+11 33377
    ## - kw_avg_max                     1  370534011 4.4959e+11 33377
    ## - kw_min_avg                     1  419164006 4.4964e+11 33377
    ## - num_self_hrefs                 1  426952430 4.4965e+11 33377
    ## - self_reference_min_shares      1  448243055 4.4967e+11 33377
    ## - global_subjectivity            1  497716260 4.4972e+11 33378
    ## <none>                                        4.4922e+11 33378
    ## - abs_title_subjectivity         1  836640139 4.5006e+11 33379
    ## - data_channel_is_tech           1  981133038 4.5020e+11 33379
    ## - data_channel_is_lifestyle      1 1041398865 4.5026e+11 33380
    ## - data_channel_is_entertainment  1 1112515078 4.5033e+11 33380
    ## - kw_min_min                     1 1142189553 4.5036e+11 33380
    ## - data_channel_is_socmed         1 1442734595 4.5067e+11 33381
    ## - data_channel_is_world          1 1650906272 4.5087e+11 33382
    ## 
    ## Step:  AIC=33375.67
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - n_non_stop_words               1    1652797 4.4922e+11 33374
    ## - title_sentiment_polarity       1    1702748 4.4922e+11 33374
    ## - LDA_00                         1    2416944 4.4923e+11 33374
    ## - kw_max_max                     1    3376767 4.4923e+11 33374
    ## - kw_max_avg                     1    4034728 4.4923e+11 33374
    ## - avg_negative_polarity          1    4526622 4.4923e+11 33374
    ## - kw_avg_avg                     1    5763925 4.4923e+11 33374
    ## - n_non_stop_unique_tokens       1    9892588 4.4923e+11 33374
    ## - abs_title_sentiment_polarity   1   10001451 4.4923e+11 33374
    ## - min_negative_polarity          1   19187811 4.4924e+11 33374
    ## - kw_max_min                     1   41537339 4.4926e+11 33374
    ## - global_sentiment_polarity      1   46921411 4.4927e+11 33374
    ## - n_tokens_content               1   48097883 4.4927e+11 33374
    ## - n_unique_tokens                1   53562210 4.4928e+11 33374
    ## - kw_avg_min                     1   57942372 4.4928e+11 33374
    ## - num_imgs                       1   59094946 4.4928e+11 33374
    ## - num_hrefs                      1   81440320 4.4930e+11 33374
    ## - timedelta                      1   81862222 4.4930e+11 33374
    ## - LDA_02                         1   82814403 4.4931e+11 33374
    ## - num_keywords                   1   85201224 4.4931e+11 33374
    ## - LDA_01                         1   85729509 4.4931e+11 33374
    ## - self_reference_avg_sharess     1   86799272 4.4931e+11 33374
    ## - LDA_03                         1   90145492 4.4931e+11 33374
    ## - title_subjectivity             1   93031771 4.4932e+11 33374
    ## - max_negative_polarity          1  103618709 4.4933e+11 33374
    ## - num_videos                     1  107546398 4.4933e+11 33374
    ## - average_token_length           1  113242895 4.4934e+11 33374
    ## - max_positive_polarity          1  174829845 4.4940e+11 33374
    ## - n_tokens_title                 1  175512991 4.4940e+11 33374
    ## - global_rate_positive_words     1  192004284 4.4942e+11 33374
    ## - kw_min_max                     1  207036977 4.4943e+11 33374
    ## - min_positive_polarity          1  250669857 4.4947e+11 33375
    ## - rate_positive_words            1  262362348 4.4949e+11 33375
    ## - data_channel_is_bus            1  351466498 4.4957e+11 33375
    ## - kw_avg_max                     1  372026539 4.4960e+11 33375
    ## - global_rate_negative_words     1  373228957 4.4960e+11 33375
    ## - kw_min_avg                     1  419325330 4.4964e+11 33375
    ## - num_self_hrefs                 1  427360029 4.4965e+11 33375
    ## - self_reference_min_shares      1  447676595 4.4967e+11 33375
    ## - global_subjectivity            1  513367322 4.4974e+11 33376
    ## <none>                                        4.4922e+11 33376
    ## - abs_title_subjectivity         1  836591645 4.5006e+11 33377
    ## - data_channel_is_tech           1  980411892 4.5020e+11 33377
    ## - data_channel_is_lifestyle      1 1041634598 4.5026e+11 33378
    ## - data_channel_is_entertainment  1 1111951560 4.5034e+11 33378
    ## - kw_min_min                     1 1141478855 4.5036e+11 33378
    ## - data_channel_is_socmed         1 1442038634 4.5067e+11 33379
    ## - data_channel_is_world          1 1650476404 4.5087e+11 33380
    ## 
    ## Step:  AIC=33373.68
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - title_sentiment_polarity       1    1615457 4.4923e+11 33372
    ## - LDA_00                         1    2728722 4.4923e+11 33372
    ## - kw_max_max                     1    3341651 4.4923e+11 33372
    ## - kw_max_avg                     1    3921891 4.4923e+11 33372
    ## - avg_negative_polarity          1    4532231 4.4923e+11 33372
    ## - kw_avg_avg                     1    5876818 4.4923e+11 33372
    ## - abs_title_sentiment_polarity   1    9555033 4.4923e+11 33372
    ## - n_non_stop_unique_tokens       1   13905891 4.4924e+11 33372
    ## - min_negative_polarity          1   18776508 4.4924e+11 33372
    ## - kw_max_min                     1   42247469 4.4927e+11 33372
    ## - global_sentiment_polarity      1   49159612 4.4927e+11 33372
    ## - n_tokens_content               1   51797815 4.4928e+11 33372
    ## - n_unique_tokens                1   51909416 4.4928e+11 33372
    ## - kw_avg_min                     1   58785344 4.4928e+11 33372
    ## - num_imgs                       1   60104018 4.4928e+11 33372
    ## - timedelta                      1   82482218 4.4931e+11 33372
    ## - num_hrefs                      1   83454719 4.4931e+11 33372
    ## - LDA_02                         1   84232811 4.4931e+11 33372
    ## - LDA_01                         1   84540426 4.4931e+11 33372
    ## - num_keywords                   1   85523841 4.4931e+11 33372
    ## - self_reference_avg_sharess     1   87381212 4.4931e+11 33372
    ## - LDA_03                         1   91614510 4.4932e+11 33372
    ## - title_subjectivity             1   94544128 4.4932e+11 33372
    ## - max_negative_polarity          1  103819449 4.4933e+11 33372
    ## - num_videos                     1  106482250 4.4933e+11 33372
    ## - n_tokens_title                 1  173865488 4.4940e+11 33372
    ## - max_positive_polarity          1  181587601 4.4941e+11 33372
    ## - global_rate_positive_words     1  193938336 4.4942e+11 33372
    ## - kw_min_max                     1  206822583 4.4943e+11 33372
    ## - average_token_length           1  221301381 4.4945e+11 33373
    ## - min_positive_polarity          1  263133835 4.4949e+11 33373
    ## - rate_positive_words            1  307390752 4.4953e+11 33373
    ## - data_channel_is_bus            1  351656004 4.4958e+11 33373
    ## - kw_avg_max                     1  372936797 4.4960e+11 33373
    ## - global_rate_negative_words     1  415620412 4.4964e+11 33373
    ## - kw_min_avg                     1  417882438 4.4964e+11 33373
    ## - num_self_hrefs                 1  427064011 4.4965e+11 33373
    ## - self_reference_min_shares      1  446736444 4.4967e+11 33373
    ## <none>                                        4.4922e+11 33374
    ## - global_subjectivity            1  535737318 4.4976e+11 33374
    ## - abs_title_subjectivity         1  836191733 4.5006e+11 33375
    ## - data_channel_is_tech           1  980862267 4.5021e+11 33375
    ## - data_channel_is_lifestyle      1 1042840892 4.5027e+11 33376
    ## - data_channel_is_entertainment  1 1117249123 4.5034e+11 33376
    ## - kw_min_min                     1 1140870951 4.5037e+11 33376
    ## - data_channel_is_socmed         1 1442305224 4.5067e+11 33377
    ## - data_channel_is_world          1 1648936924 4.5087e+11 33378
    ## 
    ## Step:  AIC=33371.68
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - LDA_00                         1    2765428 4.4923e+11 33370
    ## - kw_max_max                     1    3417582 4.4923e+11 33370
    ## - kw_max_avg                     1    4069828 4.4923e+11 33370
    ## - avg_negative_polarity          1    4164755 4.4923e+11 33370
    ## - kw_avg_avg                     1    5708199 4.4923e+11 33370
    ## - abs_title_sentiment_polarity   1    7946135 4.4923e+11 33370
    ## - n_non_stop_unique_tokens       1   14098066 4.4924e+11 33370
    ## - min_negative_polarity          1   18035883 4.4924e+11 33370
    ## - kw_max_min                     1   42410394 4.4927e+11 33370
    ## - global_sentiment_polarity      1   47699029 4.4927e+11 33370
    ## - n_unique_tokens                1   51429309 4.4928e+11 33370
    ## - n_tokens_content               1   51477332 4.4928e+11 33370
    ## - kw_avg_min                     1   58985145 4.4929e+11 33370
    ## - num_imgs                       1   61337440 4.4929e+11 33370
    ## - timedelta                      1   82219388 4.4931e+11 33370
    ## - LDA_02                         1   83995499 4.4931e+11 33370
    ## - num_hrefs                      1   84277285 4.4931e+11 33370
    ## - LDA_01                         1   84754261 4.4931e+11 33370
    ## - num_keywords                   1   85606437 4.4931e+11 33370
    ## - self_reference_avg_sharess     1   86795535 4.4931e+11 33370
    ## - LDA_03                         1   92010380 4.4932e+11 33370
    ## - title_subjectivity             1   97177542 4.4932e+11 33370
    ## - max_negative_polarity          1  102839420 4.4933e+11 33370
    ## - num_videos                     1  107230856 4.4933e+11 33370
    ## - n_tokens_title                 1  174966247 4.4940e+11 33370
    ## - max_positive_polarity          1  180581178 4.4941e+11 33370
    ## - global_rate_positive_words     1  193883904 4.4942e+11 33370
    ## - kw_min_max                     1  206039544 4.4943e+11 33370
    ## - average_token_length           1  221143207 4.4945e+11 33371
    ## - min_positive_polarity          1  263561458 4.4949e+11 33371
    ## - rate_positive_words            1  306775218 4.4953e+11 33371
    ## - data_channel_is_bus            1  350887334 4.4958e+11 33371
    ## - kw_avg_max                     1  373927913 4.4960e+11 33371
    ## - global_rate_negative_words     1  416120732 4.4964e+11 33371
    ## - kw_min_avg                     1  417257309 4.4964e+11 33371
    ## - num_self_hrefs                 1  427163775 4.4965e+11 33371
    ## - self_reference_min_shares      1  449832907 4.4968e+11 33371
    ## <none>                                        4.4923e+11 33372
    ## - global_subjectivity            1  534197332 4.4976e+11 33372
    ## - abs_title_subjectivity         1  852629989 4.5008e+11 33373
    ## - data_channel_is_tech           1  981650505 4.5021e+11 33373
    ## - data_channel_is_lifestyle      1 1045063632 4.5027e+11 33374
    ## - data_channel_is_entertainment  1 1115677341 4.5034e+11 33374
    ## - kw_min_min                     1 1139258274 4.5037e+11 33374
    ## - data_channel_is_socmed         1 1442815422 4.5067e+11 33375
    ## - data_channel_is_world          1 1647752071 4.5087e+11 33376
    ## 
    ## Step:  AIC=33369.69
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_avg_sharess + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_max_max                     1    3582881 4.4923e+11 33368
    ## - kw_max_avg                     1    4014291 4.4923e+11 33368
    ## - avg_negative_polarity          1    4114677 4.4923e+11 33368
    ## - kw_avg_avg                     1    5796113 4.4923e+11 33368
    ## - abs_title_sentiment_polarity   1    7667446 4.4924e+11 33368
    ## - n_non_stop_unique_tokens       1   13834058 4.4924e+11 33368
    ## - min_negative_polarity          1   18128151 4.4925e+11 33368
    ## - kw_max_min                     1   41853951 4.4927e+11 33368
    ## - global_sentiment_polarity      1   48143931 4.4928e+11 33368
    ## - n_tokens_content               1   51604716 4.4928e+11 33368
    ## - n_unique_tokens                1   51706405 4.4928e+11 33368
    ## - kw_avg_min                     1   58152128 4.4929e+11 33368
    ## - num_imgs                       1   62894525 4.4929e+11 33368
    ## - timedelta                      1   80907389 4.4931e+11 33368
    ## - num_keywords                   1   85180710 4.4931e+11 33368
    ## - num_hrefs                      1   86783978 4.4932e+11 33368
    ## - LDA_02                         1   87073649 4.4932e+11 33368
    ## - self_reference_avg_sharess     1   87893919 4.4932e+11 33368
    ## - LDA_03                         1   96052254 4.4933e+11 33368
    ## - title_subjectivity             1   97833621 4.4933e+11 33368
    ## - max_negative_polarity          1  103072905 4.4933e+11 33368
    ## - num_videos                     1  107751659 4.4934e+11 33368
    ## - LDA_01                         1  115949660 4.4935e+11 33368
    ## - n_tokens_title                 1  175287403 4.4940e+11 33368
    ## - max_positive_polarity          1  179733387 4.4941e+11 33368
    ## - global_rate_positive_words     1  192789942 4.4942e+11 33368
    ## - kw_min_max                     1  204669852 4.4943e+11 33368
    ## - average_token_length           1  223051105 4.4945e+11 33369
    ## - min_positive_polarity          1  264347576 4.4949e+11 33369
    ## - rate_positive_words            1  305172305 4.4953e+11 33369
    ## - kw_avg_max                     1  372046420 4.4960e+11 33369
    ## - data_channel_is_bus            1  393386024 4.4962e+11 33369
    ## - global_rate_negative_words     1  415146120 4.4964e+11 33369
    ## - kw_min_avg                     1  415433413 4.4964e+11 33369
    ## - num_self_hrefs                 1  428539235 4.4966e+11 33369
    ## - self_reference_min_shares      1  449672321 4.4968e+11 33369
    ## <none>                                        4.4923e+11 33370
    ## - global_subjectivity            1  542431618 4.4977e+11 33370
    ## - abs_title_subjectivity         1  856555606 4.5009e+11 33371
    ## - data_channel_is_lifestyle      1 1094617425 4.5032e+11 33372
    ## - data_channel_is_entertainment  1 1114031664 4.5034e+11 33372
    ## - data_channel_is_tech           1 1136279516 4.5037e+11 33372
    ## - kw_min_min                     1 1138879386 4.5037e+11 33372
    ## - data_channel_is_socmed         1 1468454939 4.5070e+11 33373
    ## - data_channel_is_world          1 1653795079 4.5088e+11 33374
    ## 
    ## Step:  AIC=33367.71
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - avg_negative_polarity          1    3713545 4.4924e+11 33366
    ## - kw_max_avg                     1    4188928 4.4924e+11 33366
    ## - kw_avg_avg                     1    5670461 4.4924e+11 33366
    ## - abs_title_sentiment_polarity   1    7602085 4.4924e+11 33366
    ## - n_non_stop_unique_tokens       1   13713402 4.4925e+11 33366
    ## - min_negative_polarity          1   17708137 4.4925e+11 33366
    ## - kw_max_min                     1   41392949 4.4927e+11 33366
    ## - global_sentiment_polarity      1   48716727 4.4928e+11 33366
    ## - n_tokens_content               1   50950583 4.4928e+11 33366
    ## - n_unique_tokens                1   51237492 4.4928e+11 33366
    ## - kw_avg_min                     1   57588759 4.4929e+11 33366
    ## - num_imgs                       1   62835838 4.4930e+11 33366
    ## - num_keywords                   1   82085023 4.4931e+11 33366
    ## - num_hrefs                      1   84703340 4.4932e+11 33366
    ## - LDA_02                         1   86944488 4.4932e+11 33366
    ## - self_reference_avg_sharess     1   87445341 4.4932e+11 33366
    ## - timedelta                      1   89165003 4.4932e+11 33366
    ## - LDA_03                         1   97166223 4.4933e+11 33366
    ## - title_subjectivity             1   99156183 4.4933e+11 33366
    ## - max_negative_polarity          1  102877723 4.4934e+11 33366
    ## - num_videos                     1  105857015 4.4934e+11 33366
    ## - LDA_01                         1  118660714 4.4935e+11 33366
    ## - n_tokens_title                 1  176831049 4.4941e+11 33366
    ## - max_positive_polarity          1  178486197 4.4941e+11 33366
    ## - global_rate_positive_words     1  189999368 4.4942e+11 33366
    ## - kw_min_max                     1  213839232 4.4945e+11 33367
    ## - average_token_length           1  223729404 4.4946e+11 33367
    ## - min_positive_polarity          1  265536091 4.4950e+11 33367
    ## - rate_positive_words            1  302419806 4.4954e+11 33367
    ## - data_channel_is_bus            1  392941859 4.4963e+11 33367
    ## - global_rate_negative_words     1  411831357 4.4964e+11 33367
    ## - kw_min_avg                     1  416368476 4.4965e+11 33367
    ## - kw_avg_max                     1  422205553 4.4965e+11 33367
    ## - num_self_hrefs                 1  425709253 4.4966e+11 33367
    ## - self_reference_min_shares      1  447928463 4.4968e+11 33367
    ## <none>                                        4.4923e+11 33368
    ## - global_subjectivity            1  551421826 4.4978e+11 33368
    ## - abs_title_subjectivity         1  863771851 4.5010e+11 33369
    ## - data_channel_is_lifestyle      1 1102746823 4.5034e+11 33370
    ## - data_channel_is_entertainment  1 1118663318 4.5035e+11 33370
    ## - data_channel_is_tech           1 1146026300 4.5038e+11 33370
    ## - data_channel_is_socmed         1 1484227265 4.5072e+11 33371
    ## - data_channel_is_world          1 1664475391 4.5090e+11 33372
    ## - kw_min_min                     1 2516086181 4.5175e+11 33375
    ## 
    ## Step:  AIC=33365.72
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + min_positive_polarity + max_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_max_avg                     1    4227695 4.4924e+11 33364
    ## - kw_avg_avg                     1    5645950 4.4924e+11 33364
    ## - abs_title_sentiment_polarity   1    6765643 4.4924e+11 33364
    ## - n_non_stop_unique_tokens       1   14303565 4.4925e+11 33364
    ## - min_negative_polarity          1   16226880 4.4925e+11 33364
    ## - kw_max_min                     1   41204866 4.4928e+11 33364
    ## - n_tokens_content               1   48093842 4.4928e+11 33364
    ## - n_unique_tokens                1   50729258 4.4929e+11 33364
    ## - kw_avg_min                     1   57403882 4.4929e+11 33364
    ## - global_sentiment_polarity      1   63204833 4.4930e+11 33364
    ## - num_imgs                       1   64305231 4.4930e+11 33364
    ## - num_keywords                   1   81652494 4.4932e+11 33364
    ## - num_hrefs                      1   83903381 4.4932e+11 33364
    ## - LDA_02                         1   86490006 4.4932e+11 33364
    ## - self_reference_avg_sharess     1   86585990 4.4932e+11 33364
    ## - timedelta                      1   87933853 4.4932e+11 33364
    ## - LDA_03                         1   95606424 4.4933e+11 33364
    ## - title_subjectivity             1  101138096 4.4934e+11 33364
    ## - num_videos                     1  106627462 4.4934e+11 33364
    ## - LDA_01                         1  120187900 4.4936e+11 33364
    ## - max_negative_polarity          1  171100281 4.4941e+11 33364
    ## - n_tokens_title                 1  175737811 4.4941e+11 33364
    ## - max_positive_polarity          1  190038407 4.4943e+11 33364
    ## - global_rate_positive_words     1  194206733 4.4943e+11 33364
    ## - kw_min_max                     1  214683427 4.4945e+11 33365
    ## - average_token_length           1  227548173 4.4946e+11 33365
    ## - min_positive_polarity          1  265803932 4.4950e+11 33365
    ## - rate_positive_words            1  301453070 4.4954e+11 33365
    ## - data_channel_is_bus            1  394201522 4.4963e+11 33365
    ## - kw_min_avg                     1  416658017 4.4965e+11 33365
    ## - kw_avg_max                     1  421638807 4.4966e+11 33365
    ## - global_rate_negative_words     1  421716099 4.4966e+11 33365
    ## - num_self_hrefs                 1  428681391 4.4967e+11 33365
    ## - self_reference_min_shares      1  448075666 4.4968e+11 33365
    ## <none>                                        4.4924e+11 33366
    ## - global_subjectivity            1  616701590 4.4985e+11 33366
    ## - abs_title_subjectivity         1  862530597 4.5010e+11 33367
    ## - data_channel_is_lifestyle      1 1108433856 4.5034e+11 33368
    ## - data_channel_is_entertainment  1 1127530589 4.5036e+11 33368
    ## - data_channel_is_tech           1 1148618003 4.5039e+11 33368
    ## - data_channel_is_socmed         1 1489325673 4.5073e+11 33369
    ## - data_channel_is_world          1 1664852406 4.5090e+11 33370
    ## - kw_min_min                     1 2521093217 4.5176e+11 33373
    ## 
    ## Step:  AIC=33363.74
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_avg_max + kw_min_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + min_positive_polarity + max_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - abs_title_sentiment_polarity   1    6726997 4.4925e+11 33362
    ## - n_non_stop_unique_tokens       1   14381225 4.4926e+11 33362
    ## - min_negative_polarity          1   15755245 4.4926e+11 33362
    ## - kw_max_min                     1   43077122 4.4928e+11 33362
    ## - n_tokens_content               1   47441286 4.4929e+11 33362
    ## - n_unique_tokens                1   50729386 4.4929e+11 33362
    ## - kw_avg_min                     1   58380015 4.4930e+11 33362
    ## - global_sentiment_polarity      1   62628823 4.4930e+11 33362
    ## - num_imgs                       1   65572489 4.4931e+11 33362
    ## - num_keywords                   1   79632393 4.4932e+11 33362
    ## - num_hrefs                      1   82294407 4.4932e+11 33362
    ## - LDA_02                         1   89376145 4.4933e+11 33362
    ## - timedelta                      1   90354829 4.4933e+11 33362
    ## - LDA_03                         1   91703461 4.4933e+11 33362
    ## - self_reference_avg_sharess     1   93622442 4.4933e+11 33362
    ## - title_subjectivity             1  100408606 4.4934e+11 33362
    ## - num_videos                     1  106865605 4.4935e+11 33362
    ## - kw_avg_avg                     1  123842550 4.4936e+11 33362
    ## - LDA_01                         1  124906402 4.4937e+11 33362
    ## - max_negative_polarity          1  170342008 4.4941e+11 33362
    ## - n_tokens_title                 1  178540865 4.4942e+11 33362
    ## - max_positive_polarity          1  189693211 4.4943e+11 33362
    ## - global_rate_positive_words     1  195151699 4.4944e+11 33362
    ## - kw_min_max                     1  217169830 4.4946e+11 33363
    ## - average_token_length           1  228678740 4.4947e+11 33363
    ## - min_positive_polarity          1  267316629 4.4951e+11 33363
    ## - rate_positive_words            1  303480880 4.4954e+11 33363
    ## - data_channel_is_bus            1  391529938 4.4963e+11 33363
    ## - global_rate_negative_words     1  421922852 4.4966e+11 33363
    ## - num_self_hrefs                 1  425397846 4.4967e+11 33363
    ## - self_reference_min_shares      1  443886045 4.4968e+11 33363
    ## - kw_avg_max                     1  449831877 4.4969e+11 33363
    ## - kw_min_avg                     1  505839064 4.4975e+11 33364
    ## <none>                                        4.4924e+11 33364
    ## - global_subjectivity            1  618089035 4.4986e+11 33364
    ## - abs_title_subjectivity         1  858303988 4.5010e+11 33365
    ## - data_channel_is_lifestyle      1 1106422950 4.5035e+11 33366
    ## - data_channel_is_entertainment  1 1138336159 4.5038e+11 33366
    ## - data_channel_is_tech           1 1144405823 4.5039e+11 33366
    ## - data_channel_is_socmed         1 1487630116 4.5073e+11 33367
    ## - data_channel_is_world          1 1664001287 4.5090e+11 33368
    ## - kw_min_min                     1 2528713812 4.5177e+11 33371
    ## 
    ## Step:  AIC=33361.76
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_avg_max + kw_min_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + min_positive_polarity + max_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - n_non_stop_unique_tokens       1   14393735 4.4926e+11 33360
    ## - min_negative_polarity          1   16556344 4.4926e+11 33360
    ## - kw_max_min                     1   45770709 4.4929e+11 33360
    ## - n_tokens_content               1   46686136 4.4929e+11 33360
    ## - n_unique_tokens                1   50646718 4.4930e+11 33360
    ## - kw_avg_min                     1   61091012 4.4931e+11 33360
    ## - num_imgs                       1   64748993 4.4931e+11 33360
    ## - global_sentiment_polarity      1   65754063 4.4931e+11 33360
    ## - num_keywords                   1   81556461 4.4933e+11 33360
    ## - num_hrefs                      1   83594444 4.4933e+11 33360
    ## - LDA_02                         1   88280047 4.4934e+11 33360
    ## - timedelta                      1   88497664 4.4934e+11 33360
    ## - LDA_03                         1   92599791 4.4934e+11 33360
    ## - self_reference_avg_sharess     1   94041095 4.4934e+11 33360
    ## - num_videos                     1  105259941 4.4935e+11 33360
    ## - kw_avg_avg                     1  123833394 4.4937e+11 33360
    ## - LDA_01                         1  126661820 4.4937e+11 33360
    ## - n_tokens_title                 1  173856476 4.4942e+11 33360
    ## - max_negative_polarity          1  178292287 4.4943e+11 33360
    ## - max_positive_polarity          1  187357702 4.4943e+11 33360
    ## - global_rate_positive_words     1  194975614 4.4944e+11 33361
    ## - title_subjectivity             1  214433410 4.4946e+11 33361
    ## - kw_min_max                     1  223016143 4.4947e+11 33361
    ## - average_token_length           1  227816589 4.4948e+11 33361
    ## - min_positive_polarity          1  265735357 4.4951e+11 33361
    ## - rate_positive_words            1  301178968 4.4955e+11 33361
    ## - data_channel_is_bus            1  390315374 4.4964e+11 33361
    ## - global_rate_negative_words     1  421197556 4.4967e+11 33361
    ## - num_self_hrefs                 1  431717099 4.4968e+11 33361
    ## - self_reference_min_shares      1  442198525 4.4969e+11 33361
    ## - kw_avg_max                     1  453111410 4.4970e+11 33361
    ## - kw_min_avg                     1  508106636 4.4976e+11 33362
    ## <none>                                        4.4925e+11 33362
    ## - global_subjectivity            1  621924711 4.4987e+11 33362
    ## - abs_title_subjectivity         1  857462423 4.5010e+11 33363
    ## - data_channel_is_lifestyle      1 1107309439 4.5035e+11 33364
    ## - data_channel_is_tech           1 1139166446 4.5039e+11 33364
    ## - data_channel_is_entertainment  1 1143783500 4.5039e+11 33364
    ## - data_channel_is_socmed         1 1485103342 4.5073e+11 33365
    ## - data_channel_is_world          1 1660940413 4.5091e+11 33366
    ## - kw_min_min                     1 2556322196 4.5180e+11 33370
    ## 
    ## Step:  AIC=33359.82
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_avg_max + kw_min_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_avg_sharess + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + min_positive_polarity + 
    ##     max_positive_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - min_negative_polarity          1   10810896 4.4927e+11 33358
    ## - kw_max_min                     1   47304477 4.4931e+11 33358
    ## - num_imgs                       1   52046569 4.4931e+11 33358
    ## - kw_avg_min                     1   62732609 4.4932e+11 33358
    ## - global_sentiment_polarity      1   68344892 4.4933e+11 33358
    ## - n_tokens_content               1   70010843 4.4933e+11 33358
    ## - num_keywords                   1   81882296 4.4934e+11 33358
    ## - LDA_02                         1   87301392 4.4935e+11 33358
    ## - timedelta                      1   87308193 4.4935e+11 33358
    ## - self_reference_avg_sharess     1   94621218 4.4936e+11 33358
    ## - num_videos                     1   98745995 4.4936e+11 33358
    ## - LDA_03                         1  100190981 4.4936e+11 33358
    ## - num_hrefs                      1  100826485 4.4936e+11 33358
    ## - kw_avg_avg                     1  122838483 4.4938e+11 33358
    ## - LDA_01                         1  123334103 4.4939e+11 33358
    ## - n_tokens_title                 1  174281460 4.4944e+11 33358
    ## - global_rate_positive_words     1  184849757 4.4945e+11 33359
    ## - max_negative_polarity          1  203242404 4.4947e+11 33359
    ## - max_positive_polarity          1  213065729 4.4947e+11 33359
    ## - title_subjectivity             1  220441340 4.4948e+11 33359
    ## - kw_min_max                     1  221127493 4.4948e+11 33359
    ## - average_token_length           1  241214717 4.4950e+11 33359
    ## - min_positive_polarity          1  252644805 4.4951e+11 33359
    ## - rate_positive_words            1  287595847 4.4955e+11 33359
    ## - data_channel_is_bus            1  407256581 4.4967e+11 33359
    ## - global_rate_negative_words     1  409285380 4.4967e+11 33359
    ## - num_self_hrefs                 1  433792091 4.4970e+11 33359
    ## - kw_avg_max                     1  448771296 4.4971e+11 33360
    ## - self_reference_min_shares      1  451449280 4.4971e+11 33360
    ## - kw_min_avg                     1  502053208 4.4976e+11 33360
    ## - n_unique_tokens                1  515029359 4.4978e+11 33360
    ## <none>                                        4.4926e+11 33360
    ## - global_subjectivity            1  628034671 4.4989e+11 33360
    ## - abs_title_subjectivity         1  848635715 4.5011e+11 33361
    ## - data_channel_is_lifestyle      1 1139289099 4.5040e+11 33362
    ## - data_channel_is_tech           1 1164837680 4.5043e+11 33362
    ## - data_channel_is_entertainment  1 1166452986 4.5043e+11 33362
    ## - data_channel_is_socmed         1 1493877059 4.5076e+11 33364
    ## - data_channel_is_world          1 1690946046 4.5095e+11 33364
    ## - kw_min_min                     1 2576741186 4.5184e+11 33368
    ## 
    ## Step:  AIC=33357.86
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_avg_max + kw_min_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_avg_sharess + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + min_positive_polarity + 
    ##     max_positive_polarity + max_negative_polarity + title_subjectivity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_max_min                     1   47216268 4.4932e+11 33356
    ## - num_imgs                       1   54065322 4.4933e+11 33356
    ## - global_sentiment_polarity      1   57541859 4.4933e+11 33356
    ## - n_tokens_content               1   61238378 4.4933e+11 33356
    ## - kw_avg_min                     1   62508079 4.4934e+11 33356
    ## - num_keywords                   1   81191387 4.4935e+11 33356
    ## - LDA_02                         1   88269128 4.4936e+11 33356
    ## - timedelta                      1   88815726 4.4936e+11 33356
    ## - self_reference_avg_sharess     1   96134751 4.4937e+11 33356
    ## - num_videos                     1   98419125 4.4937e+11 33356
    ## - LDA_03                         1  101465097 4.4937e+11 33356
    ## - num_hrefs                      1  105809558 4.4938e+11 33356
    ## - LDA_01                         1  123063644 4.4940e+11 33356
    ## - kw_avg_avg                     1  124423717 4.4940e+11 33356
    ## - n_tokens_title                 1  173130207 4.4945e+11 33357
    ## - global_rate_positive_words     1  175950793 4.4945e+11 33357
    ## - max_positive_polarity          1  202386499 4.4947e+11 33357
    ## - title_subjectivity             1  217479004 4.4949e+11 33357
    ## - kw_min_max                     1  218326113 4.4949e+11 33357
    ## - max_negative_polarity          1  219083386 4.4949e+11 33357
    ## - average_token_length           1  248091173 4.4952e+11 33357
    ## - min_positive_polarity          1  249610913 4.4952e+11 33357
    ## - rate_positive_words            1  282323188 4.4955e+11 33357
    ## - data_channel_is_bus            1  405303642 4.4968e+11 33357
    ## - global_rate_negative_words     1  409294545 4.4968e+11 33357
    ## - num_self_hrefs                 1  431820093 4.4970e+11 33358
    ## - kw_avg_max                     1  447470691 4.4972e+11 33358
    ## - self_reference_min_shares      1  450094921 4.4972e+11 33358
    ## - kw_min_avg                     1  502600090 4.4978e+11 33358
    ## <none>                                        4.4927e+11 33358
    ## - n_unique_tokens                1  534720974 4.4981e+11 33358
    ## - global_subjectivity            1  629230805 4.4990e+11 33358
    ## - abs_title_subjectivity         1  846089011 4.5012e+11 33359
    ## - data_channel_is_lifestyle      1 1134166026 4.5041e+11 33360
    ## - data_channel_is_entertainment  1 1158870140 4.5043e+11 33360
    ## - data_channel_is_tech           1 1169605109 4.5044e+11 33360
    ## - data_channel_is_socmed         1 1490700660 4.5076e+11 33362
    ## - data_channel_is_world          1 1683173560 4.5096e+11 33362
    ## - kw_min_min                     1 2578098891 4.5185e+11 33366
    ## 
    ## Step:  AIC=33356.04
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_avg_min + kw_min_max + 
    ##     kw_avg_max + kw_min_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_avg_sharess + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + min_positive_polarity + 
    ##     max_positive_polarity + max_negative_polarity + title_subjectivity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_avg_min                     1   24366068 4.4934e+11 33354
    ## - num_imgs                       1   55898791 4.4938e+11 33354
    ## - global_sentiment_polarity      1   57091379 4.4938e+11 33354
    ## - n_tokens_content               1   61561224 4.4938e+11 33354
    ## - timedelta                      1   67830377 4.4939e+11 33354
    ## - LDA_02                         1   85022366 4.4940e+11 33354
    ## - num_keywords                   1   95085234 4.4941e+11 33354
    ## - self_reference_avg_sharess     1   95302842 4.4942e+11 33354
    ## - num_videos                     1   96561409 4.4942e+11 33354
    ## - LDA_03                         1  100626526 4.4942e+11 33354
    ## - num_hrefs                      1  103355451 4.4942e+11 33354
    ## - kw_avg_avg                     1  119819731 4.4944e+11 33354
    ## - LDA_01                         1  120487866 4.4944e+11 33354
    ## - n_tokens_title                 1  164848385 4.4948e+11 33355
    ## - global_rate_positive_words     1  177702947 4.4950e+11 33355
    ## - max_positive_polarity          1  192999603 4.4951e+11 33355
    ## - max_negative_polarity          1  220805647 4.4954e+11 33355
    ## - kw_min_max                     1  221499246 4.4954e+11 33355
    ## - title_subjectivity             1  222359860 4.4954e+11 33355
    ## - min_positive_polarity          1  242680667 4.4956e+11 33355
    ## - average_token_length           1  247874694 4.4957e+11 33355
    ## - rate_positive_words            1  280728128 4.4960e+11 33355
    ## - kw_avg_max                     1  401387281 4.4972e+11 33356
    ## - global_rate_negative_words     1  409782112 4.4973e+11 33356
    ## - num_self_hrefs                 1  416167336 4.4974e+11 33356
    ## - data_channel_is_bus            1  430421388 4.4975e+11 33356
    ## - self_reference_min_shares      1  456639688 4.4978e+11 33356
    ## - kw_min_avg                     1  489303665 4.4981e+11 33356
    ## <none>                                        4.4932e+11 33356
    ## - n_unique_tokens                1  537520397 4.4986e+11 33356
    ## - global_subjectivity            1  647804058 4.4997e+11 33357
    ## - abs_title_subjectivity         1  853304687 4.5017e+11 33357
    ## - data_channel_is_lifestyle      1 1111920718 4.5043e+11 33358
    ## - data_channel_is_entertainment  1 1123199710 4.5044e+11 33358
    ## - data_channel_is_tech           1 1140319585 4.5046e+11 33358
    ## - data_channel_is_socmed         1 1496067020 4.5082e+11 33360
    ## - data_channel_is_world          1 1651130442 4.5097e+11 33360
    ## - kw_min_min                     1 2538804938 4.5186e+11 33364
    ## 
    ## Step:  AIC=33354.13
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_min_max + kw_avg_max + 
    ##     kw_min_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + min_positive_polarity + max_positive_polarity + 
    ##     max_negative_polarity + title_subjectivity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - global_sentiment_polarity      1   55769847 4.4940e+11 33352
    ## - num_imgs                       1   56308371 4.4940e+11 33352
    ## - n_tokens_content               1   61414371 4.4941e+11 33352
    ## - timedelta                      1   65704418 4.4941e+11 33352
    ## - LDA_02                         1   81151945 4.4943e+11 33352
    ## - num_keywords                   1   94740543 4.4944e+11 33352
    ## - self_reference_avg_sharess     1   95874927 4.4944e+11 33352
    ## - num_videos                     1   97051937 4.4944e+11 33353
    ## - kw_avg_avg                     1   99746663 4.4944e+11 33353
    ## - num_hrefs                      1  103217498 4.4945e+11 33353
    ## - LDA_03                         1  103354665 4.4945e+11 33353
    ## - LDA_01                         1  119396002 4.4946e+11 33353
    ## - n_tokens_title                 1  166523263 4.4951e+11 33353
    ## - global_rate_positive_words     1  181233558 4.4953e+11 33353
    ## - max_positive_polarity          1  191755736 4.4954e+11 33353
    ## - max_negative_polarity          1  219757161 4.4956e+11 33353
    ## - kw_min_max                     1  222593361 4.4957e+11 33353
    ## - title_subjectivity             1  223859373 4.4957e+11 33353
    ## - min_positive_polarity          1  241026311 4.4959e+11 33353
    ## - average_token_length           1  246488352 4.4959e+11 33353
    ## - rate_positive_words            1  287458814 4.4963e+11 33353
    ## - kw_avg_max                     1  378419370 4.4972e+11 33354
    ## - num_self_hrefs                 1  410792001 4.4975e+11 33354
    ## - global_rate_negative_words     1  414777839 4.4976e+11 33354
    ## - data_channel_is_bus            1  439474843 4.4978e+11 33354
    ## - self_reference_min_shares      1  454636280 4.4980e+11 33354
    ## - kw_min_avg                     1  507064390 4.4985e+11 33354
    ## <none>                                        4.4934e+11 33354
    ## - n_unique_tokens                1  537373612 4.4988e+11 33354
    ## - global_subjectivity            1  656644360 4.5000e+11 33355
    ## - abs_title_subjectivity         1  850487290 4.5019e+11 33355
    ## - data_channel_is_lifestyle      1 1103946501 4.5045e+11 33356
    ## - data_channel_is_entertainment  1 1115014003 4.5046e+11 33356
    ## - data_channel_is_tech           1 1137530305 4.5048e+11 33356
    ## - data_channel_is_socmed         1 1493156210 4.5084e+11 33358
    ## - data_channel_is_world          1 1634375590 4.5098e+11 33358
    ## - kw_min_min                     1 2523876632 4.5187e+11 33362
    ## 
    ## Step:  AIC=33352.35
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_min_max + kw_avg_max + 
    ##     kw_min_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + min_positive_polarity + 
    ##     max_positive_polarity + max_negative_polarity + title_subjectivity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_imgs                       1   49361342 4.4945e+11 33351
    ## - timedelta                      1   63308567 4.4946e+11 33351
    ## - n_tokens_content               1   75876327 4.4948e+11 33351
    ## - LDA_02                         1   83133448 4.4948e+11 33351
    ## - self_reference_avg_sharess     1   91060101 4.4949e+11 33351
    ## - num_videos                     1   91924920 4.4949e+11 33351
    ## - num_keywords                   1   93606513 4.4949e+11 33351
    ## - kw_avg_avg                     1  105970569 4.4951e+11 33351
    ## - LDA_01                         1  109072730 4.4951e+11 33351
    ## - LDA_03                         1  111466118 4.4951e+11 33351
    ## - num_hrefs                      1  113810796 4.4951e+11 33351
    ## - global_rate_positive_words     1  137662990 4.4954e+11 33351
    ## - max_positive_polarity          1  144032147 4.4954e+11 33351
    ## - n_tokens_title                 1  160778376 4.4956e+11 33351
    ## - max_negative_polarity          1  180191892 4.4958e+11 33351
    ## - min_positive_polarity          1  199945024 4.4960e+11 33351
    ## - average_token_length           1  206503601 4.4961e+11 33351
    ## - title_subjectivity             1  216927249 4.4962e+11 33351
    ## - kw_min_max                     1  224794570 4.4962e+11 33351
    ## - global_rate_negative_words     1  359952447 4.4976e+11 33352
    ## - rate_positive_words            1  372228199 4.4977e+11 33352
    ## - kw_avg_max                     1  380876286 4.4978e+11 33352
    ## - num_self_hrefs                 1  410692357 4.4981e+11 33352
    ## - data_channel_is_bus            1  425509982 4.4983e+11 33352
    ## - self_reference_min_shares      1  454305341 4.4985e+11 33352
    ## - kw_min_avg                     1  500267074 4.4990e+11 33352
    ## <none>                                        4.4940e+11 33352
    ## - n_unique_tokens                1  526200647 4.4993e+11 33352
    ## - global_subjectivity            1  623158159 4.5002e+11 33353
    ## - abs_title_subjectivity         1  833010339 4.5023e+11 33354
    ## - data_channel_is_lifestyle      1 1077393860 4.5048e+11 33354
    ## - data_channel_is_tech           1 1122292809 4.5052e+11 33355
    ## - data_channel_is_entertainment  1 1125264006 4.5053e+11 33355
    ## - data_channel_is_socmed         1 1482046506 4.5088e+11 33356
    ## - data_channel_is_world          1 1608489966 4.5101e+11 33356
    ## - kw_min_min                     1 2521929663 4.5192e+11 33360
    ## 
    ## Step:  AIC=33350.53
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_min_max + kw_avg_max + 
    ##     kw_min_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + min_positive_polarity + 
    ##     max_positive_polarity + max_negative_polarity + title_subjectivity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - timedelta                      1   59187036 4.4951e+11 33349
    ## - num_videos                     1   69979313 4.4952e+11 33349
    ## - LDA_02                         1   79489390 4.4953e+11 33349
    ## - LDA_03                         1   85323715 4.4953e+11 33349
    ## - num_keywords                   1   88718272 4.4954e+11 33349
    ## - self_reference_avg_sharess     1   89721164 4.4954e+11 33349
    ## - kw_avg_avg                     1   99982468 4.4955e+11 33349
    ## - num_hrefs                      1  104683464 4.4955e+11 33349
    ## - LDA_01                         1  123152190 4.4957e+11 33349
    ## - global_rate_positive_words     1  131602262 4.4958e+11 33349
    ## - max_positive_polarity          1  142459984 4.4959e+11 33349
    ## - n_tokens_content               1  146795593 4.4960e+11 33349
    ## - n_tokens_title                 1  159866105 4.4961e+11 33349
    ## - max_negative_polarity          1  174973264 4.4962e+11 33349
    ## - average_token_length           1  193959073 4.4964e+11 33349
    ## - min_positive_polarity          1  208955563 4.4966e+11 33349
    ## - title_subjectivity             1  213308185 4.4966e+11 33349
    ## - kw_min_max                     1  222606166 4.4967e+11 33349
    ## - kw_avg_max                     1  361228896 4.4981e+11 33350
    ## - global_rate_negative_words     1  377008207 4.4983e+11 33350
    ## - rate_positive_words            1  378935126 4.4983e+11 33350
    ## - num_self_hrefs                 1  390479953 4.4984e+11 33350
    ## - data_channel_is_bus            1  407710806 4.4986e+11 33350
    ## - self_reference_min_shares      1  453941271 4.4990e+11 33350
    ## - kw_min_avg                     1  487801877 4.4994e+11 33350
    ## - n_unique_tokens                1  501676990 4.4995e+11 33350
    ## <none>                                        4.4945e+11 33351
    ## - global_subjectivity            1  642454010 4.5009e+11 33351
    ## - abs_title_subjectivity         1  863765811 4.5031e+11 33352
    ## - data_channel_is_lifestyle      1 1088549372 4.5054e+11 33353
    ## - data_channel_is_entertainment  1 1105007177 4.5055e+11 33353
    ## - data_channel_is_tech           1 1131067171 4.5058e+11 33353
    ## - data_channel_is_socmed         1 1470913753 4.5092e+11 33354
    ## - data_channel_is_world          1 1583584364 4.5103e+11 33355
    ## - kw_min_min                     1 2560444021 4.5201e+11 33358
    ## 
    ## Step:  AIC=33348.76
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_min_max + kw_avg_max + 
    ##     kw_min_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + min_positive_polarity + 
    ##     max_positive_polarity + max_negative_polarity + title_subjectivity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_videos                     1   69377684 4.4958e+11 33347
    ## - num_keywords                   1   73379982 4.4958e+11 33347
    ## - LDA_02                         1   76079511 4.4958e+11 33347
    ## - LDA_03                         1   84625504 4.4959e+11 33347
    ## - kw_avg_avg                     1   97525122 4.4961e+11 33347
    ## - self_reference_avg_sharess     1   98105025 4.4961e+11 33347
    ## - num_hrefs                      1  101554651 4.4961e+11 33347
    ## - LDA_01                         1  123401410 4.4963e+11 33347
    ## - n_tokens_title                 1  123879367 4.4963e+11 33347
    ## - max_positive_polarity          1  134174918 4.4964e+11 33347
    ## - global_rate_positive_words     1  136638574 4.4965e+11 33347
    ## - n_tokens_content               1  145978939 4.4965e+11 33347
    ## - average_token_length           1  167777921 4.4968e+11 33347
    ## - max_negative_polarity          1  168157190 4.4968e+11 33347
    ## - min_positive_polarity          1  205371738 4.4971e+11 33348
    ## - title_subjectivity             1  206584538 4.4972e+11 33348
    ## - kw_min_max                     1  247009450 4.4976e+11 33348
    ## - num_self_hrefs                 1  371640061 4.4988e+11 33348
    ## - global_rate_negative_words     1  385894781 4.4989e+11 33348
    ## - rate_positive_words            1  396919824 4.4991e+11 33348
    ## - data_channel_is_bus            1  403554856 4.4991e+11 33348
    ## - self_reference_min_shares      1  441831596 4.4995e+11 33348
    ## - n_unique_tokens                1  461603405 4.4997e+11 33349
    ## - kw_min_avg                     1  480040276 4.4999e+11 33349
    ## - kw_avg_max                     1  514734621 4.5002e+11 33349
    ## <none>                                        4.4951e+11 33349
    ## - global_subjectivity            1  651369623 4.5016e+11 33349
    ## - abs_title_subjectivity         1  851357105 4.5036e+11 33350
    ## - data_channel_is_tech           1 1169446951 4.5068e+11 33351
    ## - data_channel_is_lifestyle      1 1171797566 4.5068e+11 33351
    ## - data_channel_is_entertainment  1 1184609933 4.5069e+11 33351
    ## - data_channel_is_socmed         1 1472971065 4.5098e+11 33352
    ## - data_channel_is_world          1 1693901568 4.5120e+11 33353
    ## - kw_min_min                     1 3245387127 4.5275e+11 33359
    ## 
    ## Step:  AIC=33347.02
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + average_token_length + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_min_max + kw_avg_max + 
    ##     kw_min_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + min_positive_polarity + 
    ##     max_positive_polarity + max_negative_polarity + title_subjectivity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_keywords                   1   66322228 4.4964e+11 33345
    ## - LDA_02                         1   68922698 4.4965e+11 33345
    ## - LDA_03                         1   80762570 4.4966e+11 33345
    ## - self_reference_avg_sharess     1   88334762 4.4967e+11 33345
    ## - num_hrefs                      1   91609429 4.4967e+11 33345
    ## - kw_avg_avg                     1  103345968 4.4968e+11 33345
    ## - LDA_01                         1  108404815 4.4969e+11 33345
    ## - n_tokens_title                 1  122118953 4.4970e+11 33345
    ## - global_rate_positive_words     1  127648301 4.4971e+11 33346
    ## - max_positive_polarity          1  141245992 4.4972e+11 33346
    ## - n_tokens_content               1  162026008 4.4974e+11 33346
    ## - max_negative_polarity          1  167108486 4.4974e+11 33346
    ## - average_token_length           1  201038058 4.4978e+11 33346
    ## - min_positive_polarity          1  206603791 4.4978e+11 33346
    ## - title_subjectivity             1  218381938 4.4980e+11 33346
    ## - kw_min_max                     1  253221991 4.4983e+11 33346
    ## - global_rate_negative_words     1  346067847 4.4992e+11 33346
    ## - num_self_hrefs                 1  366767243 4.4994e+11 33346
    ## - rate_positive_words            1  373995349 4.4995e+11 33346
    ## - data_channel_is_bus            1  384869233 4.4996e+11 33346
    ## - self_reference_min_shares      1  464585643 4.5004e+11 33347
    ## - kw_min_avg                     1  482553624 4.5006e+11 33347
    ## - n_unique_tokens                1  486842998 4.5006e+11 33347
    ## <none>                                        4.4958e+11 33347
    ## - kw_avg_max                     1  569519440 4.5015e+11 33347
    ## - global_subjectivity            1  644753056 4.5022e+11 33347
    ## - abs_title_subjectivity         1  867857364 4.5045e+11 33348
    ## - data_channel_is_tech           1 1137517818 4.5072e+11 33349
    ## - data_channel_is_lifestyle      1 1146463803 4.5072e+11 33349
    ## - data_channel_is_entertainment  1 1250969489 4.5083e+11 33350
    ## - data_channel_is_socmed         1 1440003953 4.5102e+11 33351
    ## - data_channel_is_world          1 1658329059 4.5124e+11 33351
    ## - kw_min_min                     1 3202975200 4.5278e+11 33357
    ## 
    ## Step:  AIC=33345.28
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_min_max + kw_avg_max + kw_min_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_avg_sharess + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + min_positive_polarity + max_positive_polarity + 
    ##     max_negative_polarity + title_subjectivity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - LDA_02                         1   51839875 4.4970e+11 33343
    ## - LDA_03                         1   83725570 4.4973e+11 33344
    ## - self_reference_avg_sharess     1   84339514 4.4973e+11 33344
    ## - num_hrefs                      1  102330196 4.4975e+11 33344
    ## - kw_avg_avg                     1  121008912 4.4977e+11 33344
    ## - LDA_01                         1  121807290 4.4977e+11 33344
    ## - global_rate_positive_words     1  127205988 4.4977e+11 33344
    ## - n_tokens_title                 1  141058175 4.4979e+11 33344
    ## - max_positive_polarity          1  142243327 4.4979e+11 33344
    ## - max_negative_polarity          1  164017695 4.4981e+11 33344
    ## - n_tokens_content               1  169632754 4.4981e+11 33344
    ## - average_token_length           1  190588008 4.4983e+11 33344
    ## - min_positive_polarity          1  204921681 4.4985e+11 33344
    ## - title_subjectivity             1  215864505 4.4986e+11 33344
    ## - kw_min_max                     1  234123710 4.4988e+11 33344
    ## - num_self_hrefs                 1  330375550 4.4997e+11 33345
    ## - global_rate_negative_words     1  347044548 4.4999e+11 33345
    ## - rate_positive_words            1  387658637 4.5003e+11 33345
    ## - data_channel_is_bus            1  411638819 4.5006e+11 33345
    ## - kw_min_avg                     1  426353692 4.5007e+11 33345
    ## - self_reference_min_shares      1  481509526 4.5013e+11 33345
    ## - n_unique_tokens                1  498095733 4.5014e+11 33345
    ## <none>                                        4.4964e+11 33345
    ## - global_subjectivity            1  634476235 4.5028e+11 33346
    ## - kw_avg_max                     1  792114084 4.5044e+11 33346
    ## - abs_title_subjectivity         1  857658084 4.5050e+11 33347
    ## - data_channel_is_lifestyle      1 1135729492 4.5078e+11 33348
    ## - data_channel_is_tech           1 1144423053 4.5079e+11 33348
    ## - data_channel_is_entertainment  1 1309578084 4.5095e+11 33348
    ## - data_channel_is_socmed         1 1506526646 4.5115e+11 33349
    ## - data_channel_is_world          1 1648989912 4.5129e+11 33350
    ## - kw_min_min                     1 3149392949 4.5279e+11 33355
    ## 
    ## Step:  AIC=33343.48
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_min_max + kw_avg_max + kw_min_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_avg_sharess + LDA_01 + LDA_03 + global_subjectivity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + min_positive_polarity + max_positive_polarity + 
    ##     max_negative_polarity + title_subjectivity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - LDA_03                         1   50113188 4.4975e+11 33342
    ## - self_reference_avg_sharess     1   85251984 4.4978e+11 33342
    ## - num_hrefs                      1  106677771 4.4980e+11 33342
    ## - kw_avg_avg                     1  114748766 4.4981e+11 33342
    ## - global_rate_positive_words     1  129501424 4.4983e+11 33342
    ## - max_positive_polarity          1  149848541 4.4985e+11 33342
    ## - n_tokens_title                 1  157753798 4.4985e+11 33342
    ## - max_negative_polarity          1  166069393 4.4986e+11 33342
    ## - n_tokens_content               1  166963966 4.4986e+11 33342
    ## - LDA_01                         1  191364302 4.4989e+11 33342
    ## - min_positive_polarity          1  200364779 4.4990e+11 33342
    ## - average_token_length           1  219899853 4.4992e+11 33342
    ## - title_subjectivity             1  221900808 4.4992e+11 33342
    ## - kw_min_max                     1  229424269 4.4993e+11 33342
    ## - global_rate_negative_words     1  340737097 4.5004e+11 33343
    ## - num_self_hrefs                 1  355398554 4.5005e+11 33343
    ## - rate_positive_words            1  371515975 4.5007e+11 33343
    ## - kw_min_avg                     1  409358368 4.5011e+11 33343
    ## - self_reference_min_shares      1  471434495 4.5017e+11 33343
    ## - n_unique_tokens                1  512022605 4.5021e+11 33343
    ## - data_channel_is_bus            1  523775056 4.5022e+11 33343
    ## <none>                                        4.4970e+11 33343
    ## - global_subjectivity            1  650324297 4.5035e+11 33344
    ## - kw_avg_max                     1  767572120 4.5046e+11 33344
    ## - abs_title_subjectivity         1  865870113 4.5056e+11 33345
    ## - data_channel_is_lifestyle      1 1263357362 4.5096e+11 33346
    ## - data_channel_is_entertainment  1 1286907274 4.5098e+11 33346
    ## - data_channel_is_tech           1 1298015112 4.5099e+11 33346
    ## - data_channel_is_socmed         1 1525978956 4.5122e+11 33347
    ## - data_channel_is_world          1 1717813551 4.5141e+11 33348
    ## - kw_min_min                     1 3145608371 4.5284e+11 33353
    ## 
    ## Step:  AIC=33341.67
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_min_max + kw_avg_max + kw_min_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_avg_sharess + LDA_01 + global_subjectivity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + min_positive_polarity + max_positive_polarity + 
    ##     max_negative_polarity + title_subjectivity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - self_reference_avg_sharess     1   83613324 4.4983e+11 33340
    ## - num_hrefs                      1  116287098 4.4986e+11 33340
    ## - global_rate_positive_words     1  134154599 4.4988e+11 33340
    ## - kw_avg_avg                     1  138685312 4.4988e+11 33340
    ## - max_positive_polarity          1  144081110 4.4989e+11 33340
    ## - n_tokens_title                 1  154938275 4.4990e+11 33340
    ## - n_tokens_content               1  167794013 4.4991e+11 33340
    ## - max_negative_polarity          1  170864756 4.4992e+11 33340
    ## - min_positive_polarity          1  193120819 4.4994e+11 33340
    ## - average_token_length           1  208050474 4.4995e+11 33340
    ## - kw_min_max                     1  222212869 4.4997e+11 33341
    ## - title_subjectivity             1  231127181 4.4998e+11 33341
    ## - global_rate_negative_words     1  335332795 4.5008e+11 33341
    ## - rate_positive_words            1  365621752 4.5011e+11 33341
    ## - num_self_hrefs                 1  374556999 4.5012e+11 33341
    ## - kw_min_avg                     1  397584138 4.5014e+11 33341
    ## - self_reference_min_shares      1  472884531 4.5022e+11 33341
    ## - n_unique_tokens                1  499661657 4.5025e+11 33342
    ## - LDA_01                         1  506816871 4.5025e+11 33342
    ## <none>                                        4.4975e+11 33342
    ## - global_subjectivity            1  617804961 4.5036e+11 33342
    ## - kw_avg_max                     1  723881313 4.5047e+11 33342
    ## - abs_title_subjectivity         1  876629688 4.5062e+11 33343
    ## - data_channel_is_bus            1 1366560924 4.5111e+11 33345
    ## - data_channel_is_entertainment  1 1370395963 4.5112e+11 33345
    ## - data_channel_is_lifestyle      1 2070071933 4.5182e+11 33348
    ## - data_channel_is_socmed         1 2442062832 4.5219e+11 33349
    ## - data_channel_is_tech           1 3182761316 4.5293e+11 33352
    ## - kw_min_min                     1 3286973474 4.5303e+11 33352
    ## - data_channel_is_world          1 3697651417 4.5344e+11 33354
    ## 
    ## Step:  AIC=33339.99
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_min_max + kw_avg_max + kw_min_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + global_subjectivity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + min_positive_polarity + 
    ##     max_positive_polarity + max_negative_polarity + title_subjectivity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_hrefs                      1  120976059 4.4995e+11 33338
    ## - global_rate_positive_words     1  129636581 4.4996e+11 33338
    ## - max_positive_polarity          1  142640483 4.4997e+11 33339
    ## - n_tokens_content               1  164598097 4.4999e+11 33339
    ## - n_tokens_title                 1  167027045 4.5000e+11 33339
    ## - max_negative_polarity          1  171886540 4.5000e+11 33339
    ## - kw_avg_avg                     1  181286581 4.5001e+11 33339
    ## - min_positive_polarity          1  200801627 4.5003e+11 33339
    ## - average_token_length           1  209627692 4.5004e+11 33339
    ## - kw_min_max                     1  215508260 4.5005e+11 33339
    ## - title_subjectivity             1  222751153 4.5005e+11 33339
    ## - global_rate_negative_words     1  332723172 4.5016e+11 33339
    ## - num_self_hrefs                 1  359799780 4.5019e+11 33339
    ## - rate_positive_words            1  361124844 4.5019e+11 33339
    ## - kw_min_avg                     1  378368756 4.5021e+11 33339
    ## - n_unique_tokens                1  495837608 4.5033e+11 33340
    ## - LDA_01                         1  514957473 4.5034e+11 33340
    ## <none>                                        4.4983e+11 33340
    ## - global_subjectivity            1  615586019 4.5045e+11 33340
    ## - kw_avg_max                     1  716350329 4.5055e+11 33341
    ## - abs_title_subjectivity         1  878408804 4.5071e+11 33341
    ## - self_reference_min_shares      1  980629760 4.5081e+11 33342
    ## - data_channel_is_entertainment  1 1341518288 4.5117e+11 33343
    ## - data_channel_is_bus            1 1377804752 4.5121e+11 33343
    ## - data_channel_is_lifestyle      1 2082596925 4.5191e+11 33346
    ## - data_channel_is_socmed         1 2413771549 4.5224e+11 33347
    ## - data_channel_is_tech           1 3137111963 4.5297e+11 33350
    ## - kw_min_min                     1 3272528929 4.5310e+11 33350
    ## - data_channel_is_world          1 3683235509 4.5351e+11 33352
    ## 
    ## Step:  AIC=33338.45
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_self_hrefs + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_min_max + kw_avg_max + kw_min_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + global_subjectivity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + min_positive_polarity + 
    ##     max_positive_polarity + max_negative_polarity + title_subjectivity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - max_positive_polarity          1  121083642 4.5007e+11 33337
    ## - global_rate_positive_words     1  131651844 4.5008e+11 33337
    ## - n_tokens_content               1  152147684 4.5010e+11 33337
    ## - n_tokens_title                 1  159823198 4.5011e+11 33337
    ## - max_negative_polarity          1  169673807 4.5012e+11 33337
    ## - kw_avg_avg                     1  198617059 4.5015e+11 33337
    ## - kw_min_max                     1  200301169 4.5015e+11 33337
    ## - min_positive_polarity          1  214506490 4.5017e+11 33337
    ## - title_subjectivity             1  240331393 4.5019e+11 33337
    ## - num_self_hrefs                 1  252387780 4.5020e+11 33337
    ## - average_token_length           1  303744834 4.5025e+11 33338
    ## - global_rate_negative_words     1  305781846 4.5026e+11 33338
    ## - rate_positive_words            1  332324978 4.5028e+11 33338
    ## - kw_min_avg                     1  383739254 4.5033e+11 33338
    ## <none>                                        4.4995e+11 33338
    ## - LDA_01                         1  556738822 4.5051e+11 33339
    ## - global_subjectivity            1  570613754 4.5052e+11 33339
    ## - n_unique_tokens                1  613053770 4.5056e+11 33339
    ## - kw_avg_max                     1  729901892 4.5068e+11 33339
    ## - abs_title_subjectivity         1  885087571 4.5084e+11 33340
    ## - self_reference_min_shares      1 1024848556 4.5098e+11 33340
    ## - data_channel_is_entertainment  1 1403257466 4.5135e+11 33342
    ## - data_channel_is_bus            1 1449188013 4.5140e+11 33342
    ## - data_channel_is_lifestyle      1 2100861286 4.5205e+11 33344
    ## - data_channel_is_socmed         1 2485591002 4.5244e+11 33346
    ## - kw_min_min                     1 3269200680 4.5322e+11 33349
    ## - data_channel_is_tech           1 3428539591 4.5338e+11 33349
    ## - data_channel_is_world          1 3880770680 4.5383e+11 33351
    ## 
    ## Step:  AIC=33336.91
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_self_hrefs + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_min_max + kw_avg_max + kw_min_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + global_subjectivity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + min_positive_polarity + 
    ##     max_negative_polarity + title_subjectivity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - global_rate_positive_words     1  134063048 4.5021e+11 33335
    ## - n_tokens_title                 1  160055720 4.5023e+11 33336
    ## - n_tokens_content               1  189942074 4.5026e+11 33336
    ## - kw_min_max                     1  200816636 4.5027e+11 33336
    ## - kw_avg_avg                     1  204637883 4.5028e+11 33336
    ## - max_negative_polarity          1  205796736 4.5028e+11 33336
    ## - min_positive_polarity          1  219522552 4.5029e+11 33336
    ## - global_rate_negative_words     1  239775104 4.5031e+11 33336
    ## - title_subjectivity             1  242208075 4.5031e+11 33336
    ## - rate_positive_words            1  255564872 4.5033e+11 33336
    ## - num_self_hrefs                 1  268270941 4.5034e+11 33336
    ## - average_token_length           1  277550785 4.5035e+11 33336
    ## - kw_min_avg                     1  368482644 4.5044e+11 33336
    ## <none>                                        4.5007e+11 33337
    ## - n_unique_tokens                1  531410713 4.5060e+11 33337
    ## - LDA_01                         1  534746583 4.5061e+11 33337
    ## - kw_avg_max                     1  729998884 4.5080e+11 33338
    ## - global_subjectivity            1  730295401 4.5080e+11 33338
    ## - abs_title_subjectivity         1  898894844 4.5097e+11 33338
    ## - self_reference_min_shares      1 1022793522 4.5109e+11 33339
    ## - data_channel_is_entertainment  1 1418114944 4.5149e+11 33340
    ## - data_channel_is_bus            1 1442088693 4.5151e+11 33340
    ## - data_channel_is_lifestyle      1 2108997539 4.5218e+11 33343
    ## - data_channel_is_socmed         1 2432548734 4.5250e+11 33344
    ## - kw_min_min                     1 3301363756 4.5337e+11 33347
    ## - data_channel_is_tech           1 3371545277 4.5344e+11 33348
    ## - data_channel_is_world          1 3810676204 4.5388e+11 33349
    ## 
    ## Step:  AIC=33335.42
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_self_hrefs + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_min_max + kw_avg_max + kw_min_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + global_subjectivity + global_rate_negative_words + 
    ##     rate_positive_words + min_positive_polarity + max_negative_polarity + 
    ##     title_subjectivity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - global_rate_negative_words     1  112207797 4.5032e+11 33334
    ## - rate_positive_words            1  121638697 4.5033e+11 33334
    ## - min_positive_polarity          1  171216227 4.5038e+11 33334
    ## - n_tokens_content               1  176678053 4.5038e+11 33334
    ## - n_tokens_title                 1  189733628 4.5040e+11 33334
    ## - kw_min_max                     1  208316607 4.5041e+11 33334
    ## - kw_avg_avg                     1  211376757 4.5042e+11 33334
    ## - title_subjectivity             1  230860344 4.5044e+11 33334
    ## - max_negative_polarity          1  243557263 4.5045e+11 33334
    ## - num_self_hrefs                 1  287911146 4.5049e+11 33335
    ## - kw_min_avg                     1  343340631 4.5055e+11 33335
    ## <none>                                        4.5021e+11 33335
    ## - n_unique_tokens                1  538836861 4.5074e+11 33335
    ## - LDA_01                         1  541935376 4.5075e+11 33335
    ## - average_token_length           1  603401928 4.5081e+11 33336
    ## - kw_avg_max                     1  707709793 4.5091e+11 33336
    ## - global_subjectivity            1  805111309 4.5101e+11 33336
    ## - abs_title_subjectivity         1  980303108 4.5119e+11 33337
    ## - self_reference_min_shares      1 1098101431 4.5130e+11 33338
    ## - data_channel_is_entertainment  1 1347231348 4.5155e+11 33339
    ## - data_channel_is_bus            1 1450324827 4.5166e+11 33339
    ## - data_channel_is_lifestyle      1 2056694987 4.5226e+11 33341
    ## - data_channel_is_socmed         1 2426623543 4.5263e+11 33343
    ## - kw_min_min                     1 3288062793 4.5349e+11 33346
    ## - data_channel_is_tech           1 3318062942 4.5352e+11 33346
    ## - data_channel_is_world          1 3742280043 4.5395e+11 33348
    ## 
    ## Step:  AIC=33333.85
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_self_hrefs + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_min_max + kw_avg_max + kw_min_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + global_subjectivity + rate_positive_words + min_positive_polarity + 
    ##     max_negative_polarity + title_subjectivity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - rate_positive_words            1   22810158 4.5034e+11 33332
    ## - n_tokens_content               1  128885712 4.5045e+11 33332
    ## - n_tokens_title                 1  184438112 4.5050e+11 33333
    ## - kw_min_max                     1  198221327 4.5052e+11 33333
    ## - max_negative_polarity          1  203990236 4.5052e+11 33333
    ## - kw_avg_avg                     1  206734787 4.5052e+11 33333
    ## - min_positive_polarity          1  223970558 4.5054e+11 33333
    ## - title_subjectivity             1  233257243 4.5055e+11 33333
    ## - num_self_hrefs                 1  291814261 4.5061e+11 33333
    ## - kw_min_avg                     1  361967454 4.5068e+11 33333
    ## - n_unique_tokens                1  450399159 4.5077e+11 33334
    ## <none>                                        4.5032e+11 33334
    ## - LDA_01                         1  546315375 4.5086e+11 33334
    ## - global_subjectivity            1  693387258 4.5101e+11 33334
    ## - kw_avg_max                     1  697308620 4.5102e+11 33335
    ## - average_token_length           1  836783942 4.5115e+11 33335
    ## - abs_title_subjectivity         1  928399190 4.5125e+11 33335
    ## - self_reference_min_shares      1 1111192544 4.5143e+11 33336
    ## - data_channel_is_entertainment  1 1280320390 4.5160e+11 33337
    ## - data_channel_is_bus            1 1397234710 4.5172e+11 33337
    ## - data_channel_is_lifestyle      1 2021449202 4.5234e+11 33340
    ## - data_channel_is_socmed         1 2351970617 4.5267e+11 33341
    ## - data_channel_is_tech           1 3241298400 4.5356e+11 33344
    ## - kw_min_min                     1 3375331383 4.5369e+11 33345
    ## - data_channel_is_world          1 3816304683 4.5413e+11 33346
    ## 
    ## Step:  AIC=33331.94
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_self_hrefs + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_min_max + kw_avg_max + kw_min_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + global_subjectivity + min_positive_polarity + max_negative_polarity + 
    ##     title_subjectivity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - n_tokens_content               1  128110750 4.5047e+11 33330
    ## - n_tokens_title                 1  187140904 4.5053e+11 33331
    ## - kw_avg_avg                     1  199820283 4.5054e+11 33331
    ## - kw_min_max                     1  201589082 4.5054e+11 33331
    ## - max_negative_polarity          1  217920103 4.5056e+11 33331
    ## - title_subjectivity             1  224468911 4.5057e+11 33331
    ## - min_positive_polarity          1  243947544 4.5058e+11 33331
    ## - num_self_hrefs                 1  282131100 4.5062e+11 33331
    ## - kw_min_avg                     1  363958323 4.5070e+11 33331
    ## - n_unique_tokens                1  441050187 4.5078e+11 33332
    ## <none>                                        4.5034e+11 33332
    ## - LDA_01                         1  544798820 4.5089e+11 33332
    ## - global_subjectivity            1  672003631 4.5101e+11 33332
    ## - kw_avg_max                     1  720108172 4.5106e+11 33333
    ## - abs_title_subjectivity         1  912993961 4.5125e+11 33333
    ## - average_token_length           1  992933666 4.5133e+11 33334
    ## - self_reference_min_shares      1 1108678399 4.5145e+11 33334
    ## - data_channel_is_entertainment  1 1273603316 4.5161e+11 33335
    ## - data_channel_is_bus            1 1374425259 4.5172e+11 33335
    ## - data_channel_is_lifestyle      1 1998743881 4.5234e+11 33338
    ## - data_channel_is_socmed         1 2330746352 4.5267e+11 33339
    ## - data_channel_is_tech           1 3240006609 4.5358e+11 33342
    ## - kw_min_min                     1 3358568056 4.5370e+11 33343
    ## - data_channel_is_world          1 3871608772 4.5421e+11 33345
    ## 
    ## Step:  AIC=33330.42
    ## shares ~ n_tokens_title + n_unique_tokens + num_self_hrefs + 
    ##     average_token_length + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_min_max + kw_avg_max + 
    ##     kw_min_avg + kw_avg_avg + self_reference_min_shares + LDA_01 + 
    ##     global_subjectivity + min_positive_polarity + max_negative_polarity + 
    ##     title_subjectivity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - n_tokens_title                 1  168357238 4.5064e+11 33329
    ## - kw_min_max                     1  200918964 4.5067e+11 33329
    ## - min_positive_polarity          1  212356677 4.5068e+11 33329
    ## - kw_avg_avg                     1  218418809 4.5069e+11 33329
    ## - title_subjectivity             1  226634319 4.5070e+11 33329
    ## - max_negative_polarity          1  251506453 4.5072e+11 33329
    ## - n_unique_tokens                1  316402751 4.5079e+11 33330
    ## - num_self_hrefs                 1  363479867 4.5083e+11 33330
    ## - kw_min_avg                     1  369979020 4.5084e+11 33330
    ## <none>                                        4.5047e+11 33330
    ## - LDA_01                         1  579928198 4.5105e+11 33331
    ## - kw_avg_max                     1  756352923 4.5123e+11 33331
    ## - global_subjectivity            1  798452873 4.5127e+11 33331
    ## - average_token_length           1  884913007 4.5135e+11 33332
    ## - abs_title_subjectivity         1  906709902 4.5138e+11 33332
    ## - self_reference_min_shares      1 1063254006 4.5153e+11 33332
    ## - data_channel_is_entertainment  1 1344418692 4.5181e+11 33334
    ## - data_channel_is_bus            1 1440057196 4.5191e+11 33334
    ## - data_channel_is_lifestyle      1 2098413241 4.5257e+11 33336
    ## - data_channel_is_socmed         1 2338106133 4.5281e+11 33337
    ## - data_channel_is_tech           1 3285548367 4.5375e+11 33341
    ## - kw_min_min                     1 3331522492 4.5380e+11 33341
    ## - data_channel_is_world          1 3869927568 4.5434e+11 33343
    ## 
    ## Step:  AIC=33329.07
    ## shares ~ n_unique_tokens + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_min_max + kw_avg_max + 
    ##     kw_min_avg + kw_avg_avg + self_reference_min_shares + LDA_01 + 
    ##     global_subjectivity + min_positive_polarity + max_negative_polarity + 
    ##     title_subjectivity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_min_max                     1  178319084 4.5082e+11 33328
    ## - min_positive_polarity          1  206698258 4.5084e+11 33328
    ## - kw_avg_avg                     1  210251237 4.5085e+11 33328
    ## - title_subjectivity             1  239345212 4.5088e+11 33328
    ## - max_negative_polarity          1  245135494 4.5088e+11 33328
    ## - n_unique_tokens                1  298689971 4.5094e+11 33328
    ## - kw_min_avg                     1  343885696 4.5098e+11 33328
    ## - num_self_hrefs                 1  370604343 4.5101e+11 33328
    ## <none>                                        4.5064e+11 33329
    ## - LDA_01                         1  615533631 4.5125e+11 33329
    ## - kw_avg_max                     1  664922801 4.5130e+11 33330
    ## - global_subjectivity            1  824474096 4.5146e+11 33330
    ## - abs_title_subjectivity         1  848770218 4.5149e+11 33330
    ## - average_token_length           1  864566883 4.5150e+11 33330
    ## - self_reference_min_shares      1 1032826355 4.5167e+11 33331
    ## - data_channel_is_entertainment  1 1224633093 4.5186e+11 33332
    ## - data_channel_is_bus            1 1400367596 4.5204e+11 33332
    ## - data_channel_is_lifestyle      1 2077817751 4.5272e+11 33335
    ## - data_channel_is_socmed         1 2280407184 4.5292e+11 33336
    ## - data_channel_is_tech           1 3180069413 4.5382e+11 33339
    ## - kw_min_min                     1 3393195312 4.5403e+11 33340
    ## - data_channel_is_world          1 3709251737 4.5435e+11 33341
    ## 
    ## Step:  AIC=33327.75
    ## shares ~ n_unique_tokens + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_avg_max + kw_min_avg + 
    ##     kw_avg_avg + self_reference_min_shares + LDA_01 + global_subjectivity + 
    ##     min_positive_polarity + max_negative_polarity + title_subjectivity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_avg_avg                     1  201929360 4.5102e+11 33327
    ## - min_positive_polarity          1  209734902 4.5103e+11 33327
    ## - max_negative_polarity          1  237569210 4.5105e+11 33327
    ## - title_subjectivity             1  264780675 4.5108e+11 33327
    ## - n_unique_tokens                1  309563433 4.5113e+11 33327
    ## - num_self_hrefs                 1  399832469 4.5122e+11 33327
    ## - kw_min_avg                     1  494998474 4.5131e+11 33328
    ## - kw_avg_max                     1  495639077 4.5131e+11 33328
    ## <none>                                        4.5082e+11 33328
    ## - LDA_01                         1  593220055 4.5141e+11 33328
    ## - global_subjectivity            1  855950372 4.5167e+11 33329
    ## - average_token_length           1  886453548 4.5170e+11 33329
    ## - abs_title_subjectivity         1  891474685 4.5171e+11 33329
    ## - self_reference_min_shares      1 1085458080 4.5190e+11 33330
    ## - data_channel_is_entertainment  1 1131112502 4.5195e+11 33330
    ## - data_channel_is_bus            1 1362746355 4.5218e+11 33331
    ## - data_channel_is_lifestyle      1 1983220697 4.5280e+11 33333
    ## - data_channel_is_socmed         1 2131431507 4.5295e+11 33334
    ## - data_channel_is_tech           1 3034701828 4.5385e+11 33337
    ## - data_channel_is_world          1 3574164451 4.5439e+11 33339
    ## - kw_min_min                     1 3842683287 4.5466e+11 33340
    ## 
    ## Step:  AIC=33326.51
    ## shares ~ n_unique_tokens + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_avg_max + kw_min_avg + 
    ##     self_reference_min_shares + LDA_01 + global_subjectivity + 
    ##     min_positive_polarity + max_negative_polarity + title_subjectivity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - min_positive_polarity          1  194009993 4.5121e+11 33325
    ## - max_negative_polarity          1  235359953 4.5125e+11 33325
    ## - title_subjectivity             1  263641718 4.5128e+11 33326
    ## - n_unique_tokens                1  308237756 4.5133e+11 33326
    ## - num_self_hrefs                 1  357390809 4.5138e+11 33326
    ## - kw_avg_max                     1  432495353 4.5145e+11 33326
    ## <none>                                        4.5102e+11 33327
    ## - LDA_01                         1  599759867 4.5162e+11 33327
    ## - kw_min_avg                     1  724602109 4.5174e+11 33327
    ## - global_subjectivity            1  831059375 4.5185e+11 33328
    ## - average_token_length           1  862317057 4.5188e+11 33328
    ## - abs_title_subjectivity         1  898999231 4.5192e+11 33328
    ## - self_reference_min_shares      1 1117768872 4.5214e+11 33329
    ## - data_channel_is_entertainment  1 1339867874 4.5236e+11 33330
    ## - data_channel_is_bus            1 1522583507 4.5254e+11 33330
    ## - data_channel_is_lifestyle      1 2028400354 4.5305e+11 33332
    ## - data_channel_is_socmed         1 2309258762 4.5333e+11 33333
    ## - data_channel_is_tech           1 3642465853 4.5466e+11 33338
    ## - kw_min_min                     1 3731647373 4.5475e+11 33339
    ## - data_channel_is_world          1 4315051457 4.5533e+11 33341
    ## 
    ## Step:  AIC=33325.25
    ## shares ~ n_unique_tokens + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_avg_max + kw_min_avg + 
    ##     self_reference_min_shares + LDA_01 + global_subjectivity + 
    ##     max_negative_polarity + title_subjectivity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - max_negative_polarity          1  215328514 4.5143e+11 33324
    ## - title_subjectivity             1  271917346 4.5148e+11 33324
    ## - num_self_hrefs                 1  326929364 4.5154e+11 33324
    ## - kw_avg_max                     1  422206341 4.5163e+11 33325
    ## - n_unique_tokens                1  463154771 4.5167e+11 33325
    ## <none>                                        4.5121e+11 33325
    ## - LDA_01                         1  585219384 4.5180e+11 33325
    ## - kw_min_avg                     1  703005782 4.5191e+11 33326
    ## - average_token_length           1  877561023 4.5209e+11 33327
    ## - global_subjectivity            1  903364458 4.5211e+11 33327
    ## - abs_title_subjectivity         1  907678988 4.5212e+11 33327
    ## - self_reference_min_shares      1 1086603494 4.5230e+11 33327
    ## - data_channel_is_entertainment  1 1243486361 4.5246e+11 33328
    ## - data_channel_is_bus            1 1386679451 4.5260e+11 33329
    ## - data_channel_is_lifestyle      1 1914483287 4.5313e+11 33331
    ## - data_channel_is_socmed         1 2156768748 4.5337e+11 33331
    ## - data_channel_is_tech           1 3503730421 4.5472e+11 33337
    ## - kw_min_min                     1 3786454353 4.5500e+11 33338
    ## - data_channel_is_world          1 4158037230 4.5537e+11 33339
    ## 
    ## Step:  AIC=33324.07
    ## shares ~ n_unique_tokens + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_avg_max + kw_min_avg + 
    ##     self_reference_min_shares + LDA_01 + global_subjectivity + 
    ##     title_subjectivity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - title_subjectivity             1  280725444 4.5171e+11 33323
    ## - num_self_hrefs                 1  334506718 4.5176e+11 33323
    ## - n_unique_tokens                1  345205666 4.5177e+11 33323
    ## - kw_avg_max                     1  409605253 4.5184e+11 33324
    ## <none>                                        4.5143e+11 33324
    ## - LDA_01                         1  593862344 4.5202e+11 33324
    ## - kw_min_avg                     1  679454022 4.5211e+11 33325
    ## - average_token_length           1  830309944 4.5226e+11 33325
    ## - global_subjectivity            1  835123054 4.5226e+11 33325
    ## - abs_title_subjectivity         1  915724958 4.5234e+11 33326
    ## - self_reference_min_shares      1 1059727992 4.5249e+11 33326
    ## - data_channel_is_entertainment  1 1303926586 4.5273e+11 33327
    ## - data_channel_is_bus            1 1483504489 4.5291e+11 33328
    ## - data_channel_is_lifestyle      1 1968281425 4.5340e+11 33330
    ## - data_channel_is_socmed         1 2176136743 4.5360e+11 33330
    ## - data_channel_is_tech           1 3648450413 4.5508e+11 33336
    ## - kw_min_min                     1 3791074380 4.5522e+11 33336
    ## - data_channel_is_world          1 4297312596 4.5572e+11 33338
    ## 
    ## Step:  AIC=33323.14
    ## shares ~ n_unique_tokens + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_avg_max + kw_min_avg + 
    ##     self_reference_min_shares + LDA_01 + global_subjectivity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - n_unique_tokens                1  341362110 4.5205e+11 33322
    ## - num_self_hrefs                 1  346298625 4.5205e+11 33322
    ## - kw_avg_max                     1  400655016 4.5211e+11 33323
    ## <none>                                        4.5171e+11 33323
    ## - LDA_01                         1  600290076 4.5231e+11 33323
    ## - abs_title_subjectivity         1  638207696 4.5235e+11 33324
    ## - kw_min_avg                     1  665862303 4.5237e+11 33324
    ## - global_subjectivity            1  683602461 4.5239e+11 33324
    ## - average_token_length           1  734888416 4.5244e+11 33324
    ## - self_reference_min_shares      1 1090696998 4.5280e+11 33325
    ## - data_channel_is_entertainment  1 1315902559 4.5302e+11 33326
    ## - data_channel_is_bus            1 1555780541 4.5326e+11 33327
    ## - data_channel_is_lifestyle      1 1944521608 4.5365e+11 33329
    ## - data_channel_is_socmed         1 2228740845 4.5394e+11 33330
    ## - data_channel_is_tech           1 3791038432 4.5550e+11 33335
    ## - kw_min_min                     1 3795066693 4.5550e+11 33336
    ## - data_channel_is_world          1 4274245843 4.5598e+11 33337
    ## 
    ## Step:  AIC=33322.44
    ## shares ~ num_self_hrefs + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_avg_max + kw_min_avg + self_reference_min_shares + LDA_01 + 
    ##     global_subjectivity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_self_hrefs                 1  237466457 4.5229e+11 33321
    ## - average_token_length           1  399744330 4.5245e+11 33322
    ## - kw_avg_max                     1  465632586 4.5251e+11 33322
    ## <none>                                        4.5205e+11 33322
    ## - abs_title_subjectivity         1  620897080 4.5267e+11 33323
    ## - LDA_01                         1  698317762 4.5275e+11 33323
    ## - kw_min_avg                     1  712983600 4.5276e+11 33323
    ## - global_subjectivity            1  802256589 4.5285e+11 33323
    ## - self_reference_min_shares      1 1048942570 4.5310e+11 33324
    ## - data_channel_is_entertainment  1 1146244923 4.5320e+11 33325
    ## - data_channel_is_bus            1 1306365950 4.5336e+11 33325
    ## - data_channel_is_lifestyle      1 1778536913 4.5383e+11 33327
    ## - data_channel_is_socmed         1 2048525189 4.5410e+11 33328
    ## - kw_min_min                     1 3531451244 4.5558e+11 33334
    ## - data_channel_is_tech           1 3557930334 4.5561e+11 33334
    ## - data_channel_is_world          1 4008160545 4.5606e+11 33336
    ## 
    ## Step:  AIC=33321.34
    ## shares ~ average_token_length + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_avg_max + kw_min_avg + 
    ##     self_reference_min_shares + LDA_01 + global_subjectivity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - average_token_length           1  355402015 4.5264e+11 33321
    ## - kw_avg_max                     1  427234352 4.5271e+11 33321
    ## <none>                                        4.5229e+11 33321
    ## - abs_title_subjectivity         1  628368506 4.5291e+11 33322
    ## - LDA_01                         1  659230905 4.5295e+11 33322
    ## - kw_min_avg                     1  676754472 4.5296e+11 33322
    ## - global_subjectivity            1  800118922 4.5309e+11 33322
    ## - self_reference_min_shares      1 1136446607 4.5342e+11 33324
    ## - data_channel_is_entertainment  1 1162734937 4.5345e+11 33324
    ## - data_channel_is_bus            1 1266911514 4.5355e+11 33324
    ## - data_channel_is_lifestyle      1 1719772575 4.5401e+11 33326
    ## - data_channel_is_socmed         1 2568653910 4.5486e+11 33329
    ## - kw_min_min                     1 3482161644 4.5577e+11 33333
    ## - data_channel_is_tech           1 3776520874 4.5606e+11 33334
    ## - data_channel_is_world          1 3914730314 4.5620e+11 33334
    ## 
    ## Step:  AIC=33320.69
    ## shares ~ data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_avg_max + kw_min_avg + 
    ##     self_reference_min_shares + LDA_01 + global_subjectivity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - global_subjectivity            1  445215425 4.5309e+11 33320
    ## - kw_avg_max                     1  445488055 4.5309e+11 33320
    ## <none>                                        4.5264e+11 33321
    ## - kw_min_avg                     1  639455030 4.5328e+11 33321
    ## - abs_title_subjectivity         1  746906228 4.5339e+11 33322
    ## - LDA_01                         1  808502459 4.5345e+11 33322
    ## - data_channel_is_entertainment  1  940191032 4.5358e+11 33322
    ## - data_channel_is_bus            1 1097968056 4.5374e+11 33323
    ## - self_reference_min_shares      1 1187221750 4.5383e+11 33323
    ## - data_channel_is_lifestyle      1 1650482501 4.5429e+11 33325
    ## - data_channel_is_socmed         1 2321434899 4.5496e+11 33327
    ## - kw_min_min                     1 3437350182 4.5608e+11 33332
    ## - data_channel_is_tech           1 3499830630 4.5614e+11 33332
    ## - data_channel_is_world          1 3559907008 4.5620e+11 33332
    ## 
    ## Step:  AIC=33320.38
    ## shares ~ data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_avg_max + kw_min_avg + 
    ##     self_reference_min_shares + LDA_01 + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_avg_max                     1  422795496 4.5351e+11 33320
    ## <none>                                        4.5309e+11 33320
    ## - kw_min_avg                     1  593064679 4.5368e+11 33321
    ## - abs_title_subjectivity         1  675788077 4.5376e+11 33321
    ## - LDA_01                         1  689706796 4.5378e+11 33321
    ## - data_channel_is_entertainment  1  947115881 4.5403e+11 33322
    ## - data_channel_is_bus            1 1054268169 4.5414e+11 33322
    ## - self_reference_min_shares      1 1074990194 4.5416e+11 33322
    ## - data_channel_is_lifestyle      1 1723456362 4.5481e+11 33325
    ## - data_channel_is_socmed         1 2249594249 4.5534e+11 33327
    ## - data_channel_is_world          1 3208472604 4.5630e+11 33330
    ## - kw_min_min                     1 3425776935 4.5651e+11 33331
    ## - data_channel_is_tech           1 3439452663 4.5653e+11 33331
    ## 
    ## Step:  AIC=33319.98
    ## shares ~ data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_min_avg + self_reference_min_shares + 
    ##     LDA_01 + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_min_avg                     1  364619828 4.5387e+11 33319
    ## <none>                                        4.5351e+11 33320
    ## - LDA_01                         1  574270791 4.5408e+11 33320
    ## - abs_title_subjectivity         1  697978228 4.5421e+11 33321
    ## - data_channel_is_entertainment  1  772160704 4.5428e+11 33321
    ## - data_channel_is_bus            1  936022825 4.5445e+11 33322
    ## - self_reference_min_shares      1 1080113354 4.5459e+11 33322
    ## - data_channel_is_lifestyle      1 1383576203 4.5489e+11 33323
    ## - data_channel_is_socmed         1 1894187268 4.5540e+11 33325
    ## - data_channel_is_world          1 2801090126 4.5631e+11 33329
    ## - data_channel_is_tech           1 3016853658 4.5653e+11 33329
    ## - kw_min_min                     1 6119737939 4.5963e+11 33341
    ## 
    ## Step:  AIC=33319.36
    ## shares ~ data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + self_reference_min_shares + 
    ##     LDA_01 + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## <none>                                        4.5387e+11 33319
    ## - LDA_01                         1  622718962 4.5450e+11 33320
    ## - abs_title_subjectivity         1  669676042 4.5454e+11 33320
    ## - data_channel_is_entertainment  1  815517697 4.5469e+11 33320
    ## - data_channel_is_bus            1  988121805 4.5486e+11 33321
    ## - self_reference_min_shares      1 1149031699 4.5502e+11 33322
    ## - data_channel_is_lifestyle      1 1450796307 4.5533e+11 33323
    ## - data_channel_is_socmed         1 1834705954 4.5571e+11 33324
    ## - data_channel_is_tech           1 3155218313 4.5703e+11 33329
    ## - data_channel_is_world          1 3173804336 4.5705e+11 33329
    ## - kw_min_min                     1 5843808651 4.5972e+11 33339

``` r
#summary the model
summary(lm.step)
```

    ## 
    ## Call:
    ## lm(formula = shares ~ data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + self_reference_min_shares + 
    ##     LDA_01 + abs_title_subjectivity, data = train)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -14356  -2730  -1215    272 604161 
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                    5.937e+03  1.253e+03   4.738 2.33e-06 ***
    ## data_channel_is_lifestyle     -4.060e+03  1.739e+03  -2.335 0.019648 *  
    ## data_channel_is_entertainment -2.542e+03  1.452e+03  -1.751 0.080159 .  
    ## data_channel_is_bus           -3.112e+03  1.615e+03  -1.927 0.054121 .  
    ## data_channel_is_socmed        -4.589e+03  1.747e+03  -2.626 0.008715 ** 
    ## data_channel_is_tech          -4.515e+03  1.311e+03  -3.444 0.000587 ***
    ## data_channel_is_world         -4.608e+03  1.334e+03  -3.454 0.000566 ***
    ## kw_min_min                     2.818e+01  6.013e+00   4.687 3.00e-06 ***
    ## self_reference_min_shares      9.649e-02  4.643e-02   2.078 0.037840 *  
    ## LDA_01                        -3.519e+03  2.300e+03  -1.530 0.126223    
    ## abs_title_subjectivity         3.258e+03  2.054e+03   1.587 0.112800    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 16310 on 1706 degrees of freedom
    ## Multiple R-squared:  0.02616,    Adjusted R-squared:  0.02046 
    ## F-statistic: 4.584 on 10 and 1706 DF,  p-value: 1.89e-06

Calculate the predicted mean square error on the train set:

``` r
train.pred <- predict(lm.step, train[,-52])
mean((train.pred - train$shares)^2)
```

    ## [1] 264341612

So, the predicted mean square error on the training dataset is
264157333.

### On test set

Calculate the predicted mean square error on the test set:

``` r
test.pred <- predict(lm.step, test[,-52])
mean((test.pred - test$shares)^2)
```

    ## [1] 48824152

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
