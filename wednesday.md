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

    ## [1] 7435

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

    ## [1] 5204

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
    ##   timedelta                   Min.   :  8        1st Qu.:169       
    ## n_tokens_title                Min.   : 4.00      1st Qu.: 9.00     
    ## n_tokens_content              Min.   :   0.0     1st Qu.: 241.0    
    ## n_unique_tokens               Min.   :0.0000     1st Qu.:0.4732    
    ## n_non_stop_words              Min.   :0.0000     1st Qu.:1.0000    
    ## n_non_stop_unique_tokens      Min.   :0.0000     1st Qu.:0.6290    
    ##   num_hrefs                   Min.   :  0.00     1st Qu.:  4.00    
    ## num_self_hrefs                Min.   : 0.000     1st Qu.: 1.000    
    ##    num_imgs                   Min.   : 0.000     1st Qu.: 1.000    
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
    ##   kw_max_min                  Min.   :     0     1st Qu.:   441    
    ##   kw_avg_min                  Min.   :   -1.0    1st Qu.:  138.8   
    ##   kw_min_max                  Min.   :     0     1st Qu.:     0    
    ##   kw_max_max                  Min.   : 17100     1st Qu.:690400    
    ##   kw_avg_max                  Min.   :  2240     1st Qu.:171094    
    ##   kw_min_avg                  Min.   :  -1       1st Qu.:   0      
    ##   kw_max_avg                  Min.   :  1953     1st Qu.:  3529    
    ##   kw_avg_avg                  Min.   :  424.3    1st Qu.: 2359.6   
    ## self_reference_min_shares     Min.   :     0.0   1st Qu.:   618.2  
    ## self_reference_max_shares     Min.   :     0     1st Qu.:  1000    
    ## self_reference_avg_sharess    Min.   :     0.0   1st Qu.:   938.5  
    ##     LDA_00                    Min.   :0.01818    1st Qu.:0.02520   
    ##     LDA_01                    Min.   :0.01819    1st Qu.:0.02502   
    ##     LDA_02                    Min.   :0.01819    1st Qu.:0.02857   
    ##     LDA_03                    Min.   :0.01820    1st Qu.:0.02857   
    ##     LDA_04                    Min.   :0.01818    1st Qu.:0.02858   
    ## global_subjectivity           Min.   :0.0000     1st Qu.:0.3944    
    ## global_sentiment_polarity     Min.   :-0.3750    1st Qu.: 0.0593   
    ## global_rate_positive_words    Min.   :0.00000    1st Qu.:0.02817   
    ## global_rate_negative_words    Min.   :0.000000   1st Qu.:0.009346  
    ## rate_positive_words           Min.   :0.0000     1st Qu.:0.6000    
    ## rate_negative_words           Min.   :0.0000     1st Qu.:0.1818    
    ## avg_positive_polarity         Min.   :0.0000     1st Qu.:0.3065    
    ## min_positive_polarity         Min.   :0.0000     1st Qu.:0.0500    
    ## max_positive_polarity         Min.   :0.0000     1st Qu.:0.6000    
    ## avg_negative_polarity         Min.   :-1.0000    1st Qu.:-0.3253   
    ## min_negative_polarity         Min.   :-1.0000    1st Qu.:-0.7000   
    ## max_negative_polarity         Min.   :-1.0000    1st Qu.:-0.1250   
    ## title_subjectivity            Min.   :0.0000     1st Qu.:0.0000    
    ## title_sentiment_polarity      Min.   :-1.00000   1st Qu.: 0.00000  
    ## abs_title_subjectivity        Min.   :0.0000     1st Qu.:0.1667    
    ## abs_title_sentiment_polarity  Min.   :0.0000     1st Qu.:0.0000    
    ##     shares                    Min.   :    23.0   1st Qu.:   880.8  
    ##                                                                    
    ##   timedelta                   Median :344        Mean   :360       
    ## n_tokens_title                Median :10.00      Mean   :10.43     
    ## n_tokens_content              Median : 396.0     Mean   : 528.3    
    ## n_unique_tokens               Median :0.5418     Mean   :0.5319    
    ## n_non_stop_words              Median :1.0000     Mean   :0.9669    
    ## n_non_stop_unique_tokens      Median :0.6939     Mean   :0.6744    
    ##   num_hrefs                   Median :  7.00     Mean   : 10.17    
    ## num_self_hrefs                Median : 2.000     Mean   : 3.121    
    ##    num_imgs                   Median : 1.000     Mean   : 4.063    
    ##   num_videos                  Median : 0.000     Mean   : 1.231    
    ## average_token_length          Median :4.660      Mean   :4.530     
    ##  num_keywords                 Median : 7.000     Mean   : 7.145    
    ## data_channel_is_lifestyle     Median :0.00000    Mean   :0.05342   
    ## data_channel_is_entertainment Median :0.0000     Mean   :0.1751    
    ## data_channel_is_bus           Median :0.0000     Mean   :0.1716    
    ## data_channel_is_socmed        Median :0.00000    Mean   :0.05515   
    ## data_channel_is_tech          Median :0.0000     Mean   :0.1897    
    ## data_channel_is_world         Median :0.0000     Mean   :0.2098    
    ##   kw_min_min                  Median : -1.00     Mean   : 27.28    
    ##   kw_max_min                  Median :   654     Mean   :  1162    
    ##   kw_avg_min                  Median :  236.9    Mean   :  313.9   
    ##   kw_min_max                  Median :  1300     Mean   : 14494    
    ##   kw_max_max                  Median :843300     Mean   :747108    
    ##   kw_avg_max                  Median :245305     Mean   :260170    
    ##   kw_min_avg                  Median :1008       Mean   :1093      
    ##   kw_max_avg                  Median :  4283     Mean   :  5581    
    ##   kw_avg_avg                  Median : 2831.6    Mean   : 3096.8   
    ## self_reference_min_shares     Median :  1200.0   Mean   :  3729.7  
    ## self_reference_max_shares     Median :  2800     Mean   : 10124    
    ## self_reference_avg_sharess    Median :  2200.0   Mean   :  6254.6  
    ##     LDA_00                    Median :0.03361    Mean   :0.19498   
    ##     LDA_01                    Median :0.03335    Mean   :0.13621   
    ##     LDA_02                    Median :0.04003    Mean   :0.21681   
    ##     LDA_03                    Median :0.04000    Mean   :0.21629   
    ##     LDA_04                    Median :0.05000    Mean   :0.23571   
    ## global_subjectivity           Median :0.4510     Mean   :0.4408    
    ## global_sentiment_polarity     Median : 0.1208    Mean   : 0.1194   
    ## global_rate_positive_words    Median :0.03904    Mean   :0.03949   
    ## global_rate_negative_words    Median :0.014925   Mean   :0.016294  
    ## rate_positive_words           Median :0.7143     Mean   :0.6832    
    ## rate_negative_words           Median :0.2759     Mean   :0.2837    
    ## avg_positive_polarity         Median :0.3587     Mean   :0.3522    
    ## min_positive_polarity         Median :0.1000     Mean   :0.0956    
    ## max_positive_polarity         Median :0.8000     Mean   :0.7478    
    ## avg_negative_polarity         Median :-0.2500    Mean   :-0.2557   
    ## min_negative_polarity         Median :-0.5000    Mean   :-0.5122   
    ## max_negative_polarity         Median :-0.1000    Mean   :-0.1072   
    ## title_subjectivity            Median :0.1000     Mean   :0.2764    
    ## title_sentiment_polarity      Median : 0.00000   Mean   : 0.06682  
    ## abs_title_subjectivity        Median :0.5000     Mean   :0.3469    
    ## abs_title_sentiment_polarity  Median :0.0000     Mean   :0.1532    
    ##     shares                    Median :  1300.0   Mean   :  3215.9  
    ##                                                                    
    ##   timedelta                   3rd Qu.:554        Max.   :729       
    ## n_tokens_title                3rd Qu.:12.00      Max.   :18.00     
    ## n_tokens_content              3rd Qu.: 697.0     Max.   :4747.0    
    ## n_unique_tokens               3rd Qu.:0.6128     Max.   :0.9714    
    ## n_non_stop_words              3rd Qu.:1.0000     Max.   :1.0000    
    ## n_non_stop_unique_tokens      3rd Qu.:0.7576     Max.   :1.0000    
    ##   num_hrefs                   3rd Qu.: 12.00     Max.   :150.00    
    ## num_self_hrefs                3rd Qu.: 4.000     Max.   :43.000    
    ##    num_imgs                   3rd Qu.: 3.000     Max.   :92.000    
    ##   num_videos                  3rd Qu.: 1.000     Max.   :73.000    
    ## average_token_length          3rd Qu.:4.852      Max.   :6.610     
    ##  num_keywords                 3rd Qu.: 9.000     Max.   :10.000    
    ## data_channel_is_lifestyle     3rd Qu.:0.00000    Max.   :1.00000   
    ## data_channel_is_entertainment 3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_bus           3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_socmed        3rd Qu.:0.00000    Max.   :1.00000   
    ## data_channel_is_tech          3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_world         3rd Qu.:0.0000     Max.   :1.0000    
    ##   kw_min_min                  3rd Qu.:  4.00     Max.   :294.00    
    ##   kw_max_min                  3rd Qu.:  1000     Max.   :102200    
    ##   kw_avg_min                  3rd Qu.:  357.5    Max.   :14716.9   
    ##   kw_min_max                  3rd Qu.:  7525     Max.   :843300    
    ##   kw_max_max                  3rd Qu.:843300     Max.   :843300    
    ##   kw_avg_max                  3rd Qu.:334325     Max.   :843300    
    ##   kw_min_avg                  3rd Qu.:1995       Max.   :3613      
    ##   kw_max_avg                  3rd Qu.:  5927     Max.   :135125    
    ##   kw_avg_avg                  3rd Qu.: 3537.5    Max.   :20377.6   
    ## self_reference_min_shares     3rd Qu.:  2700.0   Max.   :690400.0  
    ## self_reference_max_shares     3rd Qu.:  7900     Max.   :690400    
    ## self_reference_avg_sharess    3rd Qu.:  5100.0   Max.   :690400.0  
    ##     LDA_00                    3rd Qu.:0.27865    Max.   :0.92000   
    ##     LDA_01                    3rd Qu.:0.14416    Max.   :0.91998   
    ##     LDA_02                    3rd Qu.:0.32974    Max.   :0.92000   
    ##     LDA_03                    3rd Qu.:0.34821    Max.   :0.91998   
    ##     LDA_04                    3rd Qu.:0.40823    Max.   :0.92712   
    ## global_subjectivity           3rd Qu.:0.5057     Max.   :1.0000    
    ## global_sentiment_polarity     3rd Qu.: 0.1776    Max.   : 0.5667   
    ## global_rate_positive_words    3rd Qu.:0.05000    Max.   :0.15549   
    ## global_rate_negative_words    3rd Qu.:0.021572   Max.   :0.081395  
    ## rate_positive_words           3rd Qu.:0.8065     Max.   :1.0000    
    ## rate_negative_words           3rd Qu.:0.3831     Max.   :1.0000    
    ## avg_positive_polarity         3rd Qu.:0.4091     Max.   :1.0000    
    ## min_positive_polarity         3rd Qu.:0.1000     Max.   :1.0000    
    ## max_positive_polarity         3rd Qu.:1.0000     Max.   :1.0000    
    ## avg_negative_polarity         3rd Qu.:-0.1818    Max.   : 0.0000   
    ## min_negative_polarity         3rd Qu.:-0.3000    Max.   : 0.0000   
    ## max_negative_polarity         3rd Qu.:-0.0500    Max.   : 0.0000   
    ## title_subjectivity            3rd Qu.:0.5000     Max.   :1.0000    
    ## title_sentiment_polarity      3rd Qu.: 0.13636   Max.   : 1.00000  
    ## abs_title_subjectivity        3rd Qu.:0.5000     Max.   :0.5000    
    ## abs_title_sentiment_polarity  3rd Qu.:0.2500     Max.   :1.0000    
    ##     shares                    3rd Qu.:  2600.0   Max.   :663600.0

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
    ##           Mean of squared residuals: 154257853
    ##                     % Var explained: -8.61

``` r
#variable importance measures
importance(rf)
```

    ##                                   %IncMSE IncNodePurity
    ## timedelta                      2.13272156    4866592926
    ## n_tokens_title                -1.08826943    6927761573
    ## n_tokens_content               3.33846584   61127009974
    ## n_unique_tokens                3.00299226   30373389235
    ## n_non_stop_words               3.49461522   37282386348
    ## n_non_stop_unique_tokens       2.90886562   29412144282
    ## num_hrefs                      0.91888601    8694145597
    ## num_self_hrefs                 3.50658872    2755933614
    ## num_imgs                       1.59339834    9010875887
    ## num_videos                     1.86543449    3481775517
    ## average_token_length           4.33626470    7486245775
    ## num_keywords                   4.88781298    1524762046
    ## data_channel_is_lifestyle      3.30052779     161410523
    ## data_channel_is_entertainment  1.14538661     765383098
    ## data_channel_is_bus           -1.01315652    1208515596
    ## data_channel_is_socmed        -0.85678620     304696740
    ## data_channel_is_tech           2.55392790     237932084
    ## data_channel_is_world          2.93907637     208150117
    ## kw_min_min                     3.78284179    1151372622
    ## kw_max_min                     4.39295271    7024325179
    ## kw_avg_min                     2.72837017    7864547722
    ## kw_min_max                     7.89952266    3450850438
    ## kw_max_max                     2.16724718    1046804269
    ## kw_avg_max                     7.43217030    8998062095
    ## kw_min_avg                     4.85905340    9272820410
    ## kw_max_avg                     6.31946074    9372671056
    ## kw_avg_avg                     7.89517576   14455291181
    ## self_reference_min_shares      6.11913494   10263781424
    ## self_reference_max_shares      4.91640580   76156748053
    ## self_reference_avg_sharess     4.43297699   46931959531
    ## LDA_00                         1.78511926   11125921523
    ## LDA_01                         3.48267874   18348676756
    ## LDA_02                         1.34741017   29933355399
    ## LDA_03                         2.73456213   16490392638
    ## LDA_04                         8.34232490    8411775489
    ## global_subjectivity            2.73319619   10174825116
    ## global_sentiment_polarity      3.67010545    7472155341
    ## global_rate_positive_words     1.40734421    7374222903
    ## global_rate_negative_words     4.57148425    5779266155
    ## rate_positive_words            6.61903187    3686370212
    ## rate_negative_words            1.52173597    7543237026
    ## avg_positive_polarity          7.14990370    8881523174
    ## min_positive_polarity          2.51771297    1722787801
    ## max_positive_polarity          3.16937908   17819143973
    ## avg_negative_polarity          3.62056608    5497171100
    ## min_negative_polarity          1.82515282   18309821931
    ## max_negative_polarity          0.90626965    2014468701
    ## title_subjectivity            -0.52102671    7818201701
    ## title_sentiment_polarity      -0.01168499    5363772585
    ## abs_title_subjectivity        -1.94668118   17597074437
    ## abs_title_sentiment_polarity   2.14068602    3055503158

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

    ## [1] 36221984

So, the predicted mean square error on the training dataset is 68724773.

### On test set

``` r
rf.test <- predict(rf, newdata = test[,-52])
mean((test$shares-rf.test)^2)
```

    ## [1] 387039092

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

    ## Start:  AIC=97580.93
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
    ## Step:  AIC=97580.93
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
    ## Step:  AIC=97580.93
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
    ## - kw_min_max                     1     433164 7.1041e+11 97579
    ## - LDA_01                         1    1457241 7.1041e+11 97579
    ## - data_channel_is_tech           1    2309393 7.1041e+11 97579
    ## - kw_min_avg                     1    2513790 7.1041e+11 97579
    ## - global_subjectivity            1    5553979 7.1041e+11 97579
    ## - self_reference_max_shares      1   23975087 7.1043e+11 97579
    ## - abs_title_subjectivity         1   31909841 7.1044e+11 97579
    ## - kw_avg_max                     1   37238301 7.1045e+11 97579
    ## - avg_positive_polarity          1   37244533 7.1045e+11 97579
    ## - rate_positive_words            1   37871670 7.1045e+11 97579
    ## - LDA_02                         1   42142138 7.1045e+11 97579
    ## - kw_max_max                     1   42862003 7.1045e+11 97579
    ## - abs_title_sentiment_polarity   1   53314422 7.1046e+11 97579
    ## - self_reference_avg_sharess     1   60979439 7.1047e+11 97579
    ## - min_positive_polarity          1   63754189 7.1047e+11 97579
    ## - LDA_03                         1   74843264 7.1048e+11 97579
    ## - global_rate_negative_words     1   90418665 7.1050e+11 97580
    ## - n_tokens_title                 1  120948822 7.1053e+11 97580
    ## - n_non_stop_words               1  132591065 7.1054e+11 97580
    ## - data_channel_is_world          1  133182702 7.1054e+11 97580
    ## - data_channel_is_socmed         1  146799560 7.1056e+11 97580
    ## - self_reference_min_shares      1  166397590 7.1058e+11 97580
    ## - kw_min_min                     1  171993105 7.1058e+11 97580
    ## - timedelta                      1  191362472 7.1060e+11 97580
    ## - title_sentiment_polarity       1  230173769 7.1064e+11 97581
    ## - n_non_stop_unique_tokens       1  241802227 7.1065e+11 97581
    ## - num_keywords                   1  264228260 7.1067e+11 97581
    ## <none>                                        7.1041e+11 97581
    ## - kw_max_min                     1  276542702 7.1069e+11 97581
    ## - num_videos                     1  285412077 7.1069e+11 97581
    ## - kw_avg_min                     1  296853989 7.1071e+11 97581
    ## - global_rate_positive_words     1  301034080 7.1071e+11 97581
    ## - title_subjectivity             1  332835182 7.1074e+11 97581
    ## - average_token_length           1  360569152 7.1077e+11 97582
    ## - max_positive_polarity          1  390829244 7.1080e+11 97582
    ## - data_channel_is_lifestyle      1  426268809 7.1084e+11 97582
    ## - global_sentiment_polarity      1  436311937 7.1085e+11 97582
    ## - max_negative_polarity          1  529306598 7.1094e+11 97583
    ## - num_imgs                       1  574181260 7.1098e+11 97583
    ## - LDA_00                         1  583812523 7.1099e+11 97583
    ## - num_hrefs                      1  588972586 7.1100e+11 97583
    ## - data_channel_is_entertainment  1  591540348 7.1100e+11 97583
    ## - kw_max_avg                     1  617913665 7.1103e+11 97583
    ## - n_unique_tokens                1  667781276 7.1108e+11 97584
    ## - data_channel_is_bus            1  700007945 7.1111e+11 97584
    ## - num_self_hrefs                 1 1156677633 7.1157e+11 97587
    ## - kw_avg_avg                     1 1304200352 7.1171e+11 97588
    ## - avg_negative_polarity          1 1896991178 7.1231e+11 97593
    ## - min_negative_polarity          1 2287837720 7.1270e+11 97596
    ## - n_tokens_content               1 6923439328 7.1733e+11 97629
    ## 
    ## Step:  AIC=97578.93
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - LDA_01                         1    1487085 7.1041e+11 97577
    ## - kw_min_avg                     1    2230907 7.1041e+11 97577
    ## - data_channel_is_tech           1    2413560 7.1041e+11 97577
    ## - global_subjectivity            1    5613016 7.1041e+11 97577
    ## - self_reference_max_shares      1   23924975 7.1043e+11 97577
    ## - abs_title_subjectivity         1   31964066 7.1044e+11 97577
    ## - avg_positive_polarity          1   37294478 7.1045e+11 97577
    ## - rate_positive_words            1   37906780 7.1045e+11 97577
    ## - LDA_02                         1   41967092 7.1045e+11 97577
    ## - kw_max_max                     1   44410985 7.1045e+11 97577
    ## - kw_avg_max                     1   48337301 7.1046e+11 97577
    ## - abs_title_sentiment_polarity   1   53403556 7.1046e+11 97577
    ## - self_reference_avg_sharess     1   61036775 7.1047e+11 97577
    ## - min_positive_polarity          1   63858691 7.1047e+11 97577
    ## - LDA_03                         1   75032214 7.1048e+11 97577
    ## - global_rate_negative_words     1   90504278 7.1050e+11 97578
    ## - n_tokens_title                 1  121110823 7.1053e+11 97578
    ## - n_non_stop_words               1  132770197 7.1054e+11 97578
    ## - data_channel_is_world          1  134668540 7.1054e+11 97578
    ## - data_channel_is_socmed         1  150703983 7.1056e+11 97578
    ## - self_reference_min_shares      1  166253799 7.1058e+11 97578
    ## - kw_min_min                     1  171803291 7.1058e+11 97578
    ## - timedelta                      1  191620555 7.1060e+11 97578
    ## - title_sentiment_polarity       1  229954860 7.1064e+11 97579
    ## - n_non_stop_unique_tokens       1  241926354 7.1065e+11 97579
    ## - num_keywords                   1  265213938 7.1067e+11 97579
    ## <none>                                        7.1041e+11 97579
    ## - kw_max_min                     1  276452819 7.1069e+11 97579
    ## - num_videos                     1  285007897 7.1069e+11 97579
    ## - kw_avg_min                     1  296735647 7.1071e+11 97579
    ## - global_rate_positive_words     1  301342261 7.1071e+11 97579
    ## - title_subjectivity             1  332914792 7.1074e+11 97579
    ## - average_token_length           1  361130945 7.1077e+11 97580
    ## - max_positive_polarity          1  390955231 7.1080e+11 97580
    ## - data_channel_is_lifestyle      1  430333959 7.1084e+11 97580
    ## - global_sentiment_polarity      1  436697650 7.1085e+11 97580
    ## - max_negative_polarity          1  529906580 7.1094e+11 97581
    ## - num_imgs                       1  574755792 7.1098e+11 97581
    ## - LDA_00                         1  585084077 7.1099e+11 97581
    ## - num_hrefs                      1  588561393 7.1100e+11 97581
    ## - data_channel_is_entertainment  1  597872241 7.1101e+11 97581
    ## - kw_max_avg                     1  622668655 7.1103e+11 97581
    ## - n_unique_tokens                1  668332054 7.1108e+11 97582
    ## - data_channel_is_bus            1  700395123 7.1111e+11 97582
    ## - num_self_hrefs                 1 1156917779 7.1157e+11 97585
    ## - kw_avg_avg                     1 1316085786 7.1173e+11 97587
    ## - avg_negative_polarity          1 1900458699 7.1231e+11 97591
    ## - min_negative_polarity          1 2291530664 7.1270e+11 97594
    ## - n_tokens_content               1 6923345195 7.1733e+11 97627
    ## 
    ## Step:  AIC=97576.95
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - data_channel_is_tech           1    1059495 7.1041e+11 97575
    ## - kw_min_avg                     1    2453646 7.1041e+11 97575
    ## - global_subjectivity            1    5687461 7.1042e+11 97575
    ## - self_reference_max_shares      1   23918927 7.1043e+11 97575
    ## - abs_title_subjectivity         1   32253303 7.1044e+11 97575
    ## - avg_positive_polarity          1   37163082 7.1045e+11 97575
    ## - rate_positive_words            1   37742409 7.1045e+11 97575
    ## - LDA_02                         1   43325104 7.1045e+11 97575
    ## - kw_max_max                     1   44422644 7.1046e+11 97575
    ## - kw_avg_max                     1   47926748 7.1046e+11 97575
    ## - abs_title_sentiment_polarity   1   53486683 7.1046e+11 97575
    ## - self_reference_avg_sharess     1   61019528 7.1047e+11 97575
    ## - min_positive_polarity          1   64013572 7.1047e+11 97575
    ## - global_rate_negative_words     1   90119811 7.1050e+11 97576
    ## - n_tokens_title                 1  121329426 7.1053e+11 97576
    ## - n_non_stop_words               1  134015936 7.1054e+11 97576
    ## - data_channel_is_world          1  141000538 7.1055e+11 97576
    ## - data_channel_is_socmed         1  155729071 7.1057e+11 97576
    ## - self_reference_min_shares      1  166678810 7.1058e+11 97576
    ## - kw_min_min                     1  171414971 7.1058e+11 97576
    ## - LDA_03                         1  191508379 7.1060e+11 97576
    ## - timedelta                      1  192084534 7.1060e+11 97576
    ## - title_sentiment_polarity       1  229923974 7.1064e+11 97577
    ## - n_non_stop_unique_tokens       1  241843074 7.1065e+11 97577
    ## - num_keywords                   1  269291225 7.1068e+11 97577
    ## <none>                                        7.1041e+11 97577
    ## - kw_max_min                     1  276026960 7.1069e+11 97577
    ## - num_videos                     1  284509083 7.1070e+11 97577
    ## - kw_avg_min                     1  296307357 7.1071e+11 97577
    ## - global_rate_positive_words     1  300731158 7.1071e+11 97577
    ## - title_subjectivity             1  332575066 7.1074e+11 97577
    ## - average_token_length           1  362173425 7.1077e+11 97578
    ## - max_positive_polarity          1  390859845 7.1080e+11 97578
    ## - global_sentiment_polarity      1  435604483 7.1085e+11 97578
    ## - max_negative_polarity          1  529164941 7.1094e+11 97579
    ## - data_channel_is_lifestyle      1  531236322 7.1094e+11 97579
    ## - num_imgs                       1  574546549 7.1099e+11 97579
    ## - num_hrefs                      1  587975505 7.1100e+11 97579
    ## - kw_max_avg                     1  621215175 7.1103e+11 97579
    ## - data_channel_is_entertainment  1  632152585 7.1104e+11 97580
    ## - n_unique_tokens                1  667128898 7.1108e+11 97580
    ## - LDA_00                         1  725724897 7.1114e+11 97580
    ## - data_channel_is_bus            1  758095812 7.1117e+11 97580
    ## - num_self_hrefs                 1 1155439376 7.1157e+11 97583
    ## - kw_avg_avg                     1 1315675214 7.1173e+11 97585
    ## - avg_negative_polarity          1 1899155798 7.1231e+11 97589
    ## - min_negative_polarity          1 2290511124 7.1270e+11 97592
    ## - n_tokens_content               1 6921861760 7.1733e+11 97625
    ## 
    ## Step:  AIC=97574.95
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_min_avg                     1    2035192 7.1041e+11 97573
    ## - global_subjectivity            1    5701341 7.1042e+11 97573
    ## - self_reference_max_shares      1   24127843 7.1044e+11 97573
    ## - abs_title_subjectivity         1   32234937 7.1044e+11 97573
    ## - avg_positive_polarity          1   37214243 7.1045e+11 97573
    ## - rate_positive_words            1   37942992 7.1045e+11 97573
    ## - LDA_02                         1   42317535 7.1045e+11 97573
    ## - kw_max_max                     1   43994884 7.1046e+11 97573
    ## - kw_avg_max                     1   47082938 7.1046e+11 97573
    ## - abs_title_sentiment_polarity   1   53785476 7.1047e+11 97573
    ## - self_reference_avg_sharess     1   61135584 7.1047e+11 97573
    ## - min_positive_polarity          1   63556379 7.1048e+11 97573
    ## - global_rate_negative_words     1   90624501 7.1050e+11 97574
    ## - n_tokens_title                 1  121701935 7.1053e+11 97574
    ## - n_non_stop_words               1  132989813 7.1054e+11 97574
    ## - self_reference_min_shares      1  166731969 7.1058e+11 97574
    ## - kw_min_min                     1  173271799 7.1059e+11 97574
    ## - timedelta                      1  193593971 7.1061e+11 97574
    ## - data_channel_is_socmed         1  200549389 7.1061e+11 97574
    ## - data_channel_is_world          1  202189219 7.1061e+11 97574
    ## - title_sentiment_polarity       1  230166030 7.1064e+11 97575
    ## - n_non_stop_unique_tokens       1  245180181 7.1066e+11 97575
    ## - LDA_03                         1  269685422 7.1068e+11 97575
    ## - num_keywords                   1  269767857 7.1068e+11 97575
    ## <none>                                        7.1041e+11 97575
    ## - kw_max_min                     1  276243960 7.1069e+11 97575
    ## - num_videos                     1  286314214 7.1070e+11 97575
    ## - kw_avg_min                     1  297367647 7.1071e+11 97575
    ## - global_rate_positive_words     1  300768540 7.1071e+11 97575
    ## - title_subjectivity             1  334562195 7.1075e+11 97575
    ## - average_token_length           1  361311668 7.1077e+11 97576
    ## - max_positive_polarity          1  389838859 7.1080e+11 97576
    ## - global_sentiment_polarity      1  436916685 7.1085e+11 97576
    ## - max_negative_polarity          1  531252776 7.1094e+11 97577
    ## - num_imgs                       1  573495669 7.1099e+11 97577
    ## - num_hrefs                      1  592763974 7.1100e+11 97577
    ## - kw_max_avg                     1  651076132 7.1106e+11 97578
    ## - n_unique_tokens                1  679903991 7.1109e+11 97578
    ## - LDA_00                         1  739180381 7.1115e+11 97578
    ## - data_channel_is_lifestyle      1  784600330 7.1120e+11 97579
    ## - data_channel_is_bus            1 1102182413 7.1151e+11 97581
    ## - num_self_hrefs                 1 1166348663 7.1158e+11 97581
    ## - data_channel_is_entertainment  1 1175566366 7.1159e+11 97582
    ## - kw_avg_avg                     1 1420983667 7.1183e+11 97583
    ## - avg_negative_polarity          1 1908058713 7.1232e+11 97587
    ## - min_negative_polarity          1 2289558316 7.1270e+11 97590
    ## - n_tokens_content               1 6921524971 7.1733e+11 97623
    ## 
    ## Step:  AIC=97572.97
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - global_subjectivity            1    5981431 7.1042e+11 97571
    ## - self_reference_max_shares      1   24599861 7.1044e+11 97571
    ## - abs_title_subjectivity         1   32092646 7.1045e+11 97571
    ## - avg_positive_polarity          1   37289701 7.1045e+11 97571
    ## - rate_positive_words            1   38097202 7.1045e+11 97571
    ## - LDA_02                         1   41577592 7.1046e+11 97571
    ## - kw_max_max                     1   43435195 7.1046e+11 97571
    ## - kw_avg_max                     1   46823040 7.1046e+11 97571
    ## - abs_title_sentiment_polarity   1   53529992 7.1047e+11 97571
    ## - self_reference_avg_sharess     1   61964393 7.1048e+11 97571
    ## - min_positive_polarity          1   64200416 7.1048e+11 97571
    ## - global_rate_negative_words     1   90453512 7.1050e+11 97572
    ## - n_tokens_title                 1  121455910 7.1054e+11 97572
    ## - n_non_stop_words               1  134060619 7.1055e+11 97572
    ## - self_reference_min_shares      1  165087410 7.1058e+11 97572
    ## - kw_min_min                     1  175125206 7.1059e+11 97572
    ## - timedelta                      1  194005651 7.1061e+11 97572
    ## - data_channel_is_socmed         1  199708771 7.1061e+11 97572
    ## - data_channel_is_world          1  200686150 7.1061e+11 97572
    ## - title_sentiment_polarity       1  230183496 7.1064e+11 97573
    ## - n_non_stop_unique_tokens       1  245153199 7.1066e+11 97573
    ## - LDA_03                         1  269891649 7.1068e+11 97573
    ## <none>                                        7.1041e+11 97573
    ## - num_keywords                   1  273535307 7.1069e+11 97573
    ## - kw_max_min                     1  283017727 7.1070e+11 97573
    ## - num_videos                     1  285552493 7.1070e+11 97573
    ## - global_rate_positive_words     1  300114635 7.1071e+11 97573
    ## - kw_avg_min                     1  306013691 7.1072e+11 97573
    ## - title_subjectivity             1  334306109 7.1075e+11 97573
    ## - average_token_length           1  361725995 7.1078e+11 97574
    ## - max_positive_polarity          1  390542933 7.1080e+11 97574
    ## - global_sentiment_polarity      1  438440275 7.1085e+11 97574
    ## - max_negative_polarity          1  530404004 7.1094e+11 97575
    ## - num_imgs                       1  572605454 7.1099e+11 97575
    ## - num_hrefs                      1  590794376 7.1100e+11 97575
    ## - n_unique_tokens                1  679652350 7.1109e+11 97576
    ## - LDA_00                         1  737226840 7.1115e+11 97576
    ## - data_channel_is_lifestyle      1  784326709 7.1120e+11 97577
    ## - kw_max_avg                     1  974143350 7.1139e+11 97578
    ## - data_channel_is_bus            1 1100212456 7.1151e+11 97579
    ## - num_self_hrefs                 1 1166063525 7.1158e+11 97580
    ## - data_channel_is_entertainment  1 1174336715 7.1159e+11 97580
    ## - avg_negative_polarity          1 1907199428 7.1232e+11 97585
    ## - min_negative_polarity          1 2290423788 7.1270e+11 97588
    ## - kw_avg_avg                     1 2453712594 7.1287e+11 97589
    ## - n_tokens_content               1 6935213468 7.1735e+11 97622
    ## 
    ## Step:  AIC=97571.01
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_02 + LDA_03 + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - self_reference_max_shares      1   24384830 7.1044e+11 97569
    ## - abs_title_subjectivity         1   30459817 7.1045e+11 97569
    ## - avg_positive_polarity          1   39880344 7.1046e+11 97569
    ## - LDA_02                         1   40482523 7.1046e+11 97569
    ## - rate_positive_words            1   40959328 7.1046e+11 97569
    ## - kw_max_max                     1   42881474 7.1046e+11 97569
    ## - kw_avg_max                     1   46366572 7.1047e+11 97569
    ## - abs_title_sentiment_polarity   1   52450695 7.1047e+11 97569
    ## - self_reference_avg_sharess     1   61420199 7.1048e+11 97569
    ## - min_positive_polarity          1   65633329 7.1049e+11 97569
    ## - global_rate_negative_words     1   85419039 7.1051e+11 97570
    ## - n_tokens_title                 1  121871988 7.1054e+11 97570
    ## - n_non_stop_words               1  128759136 7.1055e+11 97570
    ## - self_reference_min_shares      1  164799407 7.1058e+11 97570
    ## - kw_min_min                     1  173795011 7.1059e+11 97570
    ## - timedelta                      1  194449415 7.1061e+11 97570
    ## - data_channel_is_world          1  198424296 7.1062e+11 97570
    ## - data_channel_is_socmed         1  198711523 7.1062e+11 97570
    ## - title_sentiment_polarity       1  228960357 7.1065e+11 97571
    ## - n_non_stop_unique_tokens       1  242674545 7.1066e+11 97571
    ## - LDA_03                         1  266998792 7.1069e+11 97571
    ## - num_keywords                   1  272373874 7.1069e+11 97571
    ## <none>                                        7.1042e+11 97571
    ## - num_videos                     1  281886740 7.1070e+11 97571
    ## - kw_max_min                     1  282850853 7.1070e+11 97571
    ## - global_rate_positive_words     1  297461379 7.1072e+11 97571
    ## - kw_avg_min                     1  305896006 7.1073e+11 97571
    ## - title_subjectivity             1  328876614 7.1075e+11 97571
    ## - average_token_length           1  358017573 7.1078e+11 97572
    ## - max_positive_polarity          1  386436699 7.1081e+11 97572
    ## - global_sentiment_polarity      1  434844970 7.1085e+11 97572
    ## - max_negative_polarity          1  524448930 7.1094e+11 97573
    ## - num_imgs                       1  575844252 7.1100e+11 97573
    ## - num_hrefs                      1  586667343 7.1101e+11 97573
    ## - n_unique_tokens                1  675480668 7.1110e+11 97574
    ## - LDA_00                         1  745609189 7.1117e+11 97574
    ## - data_channel_is_lifestyle      1  782390093 7.1120e+11 97575
    ## - kw_max_avg                     1  969397070 7.1139e+11 97576
    ## - data_channel_is_bus            1 1097695976 7.1152e+11 97577
    ## - num_self_hrefs                 1 1162466366 7.1158e+11 97578
    ## - data_channel_is_entertainment  1 1168355568 7.1159e+11 97578
    ## - avg_negative_polarity          1 1943687554 7.1236e+11 97583
    ## - min_negative_polarity          1 2290231883 7.1271e+11 97586
    ## - kw_avg_avg                     1 2448591337 7.1287e+11 97587
    ## - n_tokens_content               1 6938775087 7.1736e+11 97620
    ## 
    ## Step:  AIC=97569.19
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_02 + LDA_03 + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - abs_title_subjectivity         1   30695132 7.1048e+11 97567
    ## - LDA_02                         1   40037143 7.1048e+11 97567
    ## - rate_positive_words            1   40883831 7.1049e+11 97567
    ## - avg_positive_polarity          1   41021214 7.1049e+11 97567
    ## - kw_max_max                     1   43571323 7.1049e+11 97568
    ## - kw_avg_max                     1   47376356 7.1049e+11 97568
    ## - abs_title_sentiment_polarity   1   51981408 7.1050e+11 97568
    ## - min_positive_polarity          1   65307454 7.1051e+11 97568
    ## - self_reference_avg_sharess     1   66749620 7.1051e+11 97568
    ## - global_rate_negative_words     1   86159344 7.1053e+11 97568
    ## - n_tokens_title                 1  123562541 7.1057e+11 97568
    ## - n_non_stop_words               1  130287241 7.1057e+11 97568
    ## - kw_min_min                     1  174199814 7.1062e+11 97568
    ## - timedelta                      1  197361780 7.1064e+11 97569
    ## - data_channel_is_world          1  198205871 7.1064e+11 97569
    ## - data_channel_is_socmed         1  199763570 7.1064e+11 97569
    ## - title_sentiment_polarity       1  232809603 7.1068e+11 97569
    ## - n_non_stop_unique_tokens       1  243512104 7.1069e+11 97569
    ## - LDA_03                         1  268788942 7.1071e+11 97569
    ## - num_keywords                   1  270447410 7.1071e+11 97569
    ## <none>                                        7.1044e+11 97569
    ## - kw_max_min                     1  283948953 7.1073e+11 97569
    ## - num_videos                     1  285041449 7.1073e+11 97569
    ## - global_rate_positive_words     1  299903036 7.1074e+11 97569
    ## - kw_avg_min                     1  307874197 7.1075e+11 97569
    ## - title_subjectivity             1  325145759 7.1077e+11 97570
    ## - average_token_length           1  360351331 7.1080e+11 97570
    ## - max_positive_polarity          1  383209576 7.1083e+11 97570
    ## - self_reference_min_shares      1  427691211 7.1087e+11 97570
    ## - global_sentiment_polarity      1  435962217 7.1088e+11 97570
    ## - max_negative_polarity          1  526311202 7.1097e+11 97571
    ## - num_imgs                       1  572250699 7.1102e+11 97571
    ## - num_hrefs                      1  585423927 7.1103e+11 97571
    ## - n_unique_tokens                1  677013393 7.1112e+11 97572
    ## - LDA_00                         1  741138566 7.1119e+11 97573
    ## - data_channel_is_lifestyle      1  780723133 7.1123e+11 97573
    ## - kw_max_avg                     1  966579806 7.1141e+11 97574
    ## - data_channel_is_bus            1 1092607644 7.1154e+11 97575
    ## - data_channel_is_entertainment  1 1166054758 7.1161e+11 97576
    ## - num_self_hrefs                 1 1237930041 7.1168e+11 97576
    ## - avg_negative_polarity          1 1957468776 7.1240e+11 97582
    ## - min_negative_polarity          1 2318239224 7.1276e+11 97584
    ## - kw_avg_avg                     1 2444293782 7.1289e+11 97585
    ## - n_tokens_content               1 6953004280 7.1740e+11 97618
    ## 
    ## Step:  AIC=97567.42
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_02 + LDA_03 + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - LDA_02                         1   37692515 7.1051e+11 97566
    ## - avg_positive_polarity          1   38472378 7.1051e+11 97566
    ## - kw_max_max                     1   42189952 7.1052e+11 97566
    ## - rate_positive_words            1   42258888 7.1052e+11 97566
    ## - kw_avg_max                     1   46700378 7.1052e+11 97566
    ## - abs_title_sentiment_polarity   1   54319167 7.1053e+11 97566
    ## - self_reference_avg_sharess     1   66120036 7.1054e+11 97566
    ## - min_positive_polarity          1   67158651 7.1054e+11 97566
    ## - global_rate_negative_words     1   82537505 7.1056e+11 97566
    ## - n_tokens_title                 1  109666223 7.1058e+11 97566
    ## - n_non_stop_words               1  132210576 7.1061e+11 97566
    ## - kw_min_min                     1  171144860 7.1065e+11 97567
    ## - timedelta                      1  197360778 7.1067e+11 97567
    ## - data_channel_is_world          1  200584295 7.1068e+11 97567
    ## - data_channel_is_socmed         1  200612883 7.1068e+11 97567
    ## - title_sentiment_polarity       1  212904583 7.1069e+11 97567
    ## - n_non_stop_unique_tokens       1  240736889 7.1072e+11 97567
    ## <none>                                        7.1048e+11 97567
    ## - num_keywords                   1  276841227 7.1075e+11 97567
    ## - num_videos                     1  278083083 7.1075e+11 97567
    ## - LDA_03                         1  279894604 7.1075e+11 97567
    ## - kw_max_min                     1  283153830 7.1076e+11 97567
    ## - title_subjectivity             1  294698023 7.1077e+11 97568
    ## - kw_avg_min                     1  306364508 7.1078e+11 97568
    ## - global_rate_positive_words     1  308653300 7.1078e+11 97568
    ## - average_token_length           1  361495228 7.1084e+11 97568
    ## - max_positive_polarity          1  387538331 7.1086e+11 97568
    ## - self_reference_min_shares      1  430561167 7.1091e+11 97569
    ## - global_sentiment_polarity      1  444024703 7.1092e+11 97569
    ## - max_negative_polarity          1  537192504 7.1101e+11 97569
    ## - num_imgs                       1  570151829 7.1105e+11 97570
    ## - num_hrefs                      1  579707476 7.1105e+11 97570
    ## - n_unique_tokens                1  671654807 7.1115e+11 97570
    ## - LDA_00                         1  745176084 7.1122e+11 97571
    ## - data_channel_is_lifestyle      1  782828448 7.1126e+11 97571
    ## - kw_max_avg                     1  972207647 7.1145e+11 97573
    ## - data_channel_is_bus            1 1097064392 7.1157e+11 97573
    ## - data_channel_is_entertainment  1 1169900173 7.1164e+11 97574
    ## - num_self_hrefs                 1 1238273819 7.1171e+11 97574
    ## - avg_negative_polarity          1 1978997176 7.1245e+11 97580
    ## - min_negative_polarity          1 2316431038 7.1279e+11 97582
    ## - kw_avg_avg                     1 2449920802 7.1292e+11 97583
    ## - n_tokens_content               1 6944694364 7.1742e+11 97616
    ## 
    ## Step:  AIC=97565.69
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - rate_positive_words            1   40242419 7.1055e+11 97564
    ## - avg_positive_polarity          1   40794691 7.1055e+11 97564
    ## - kw_max_max                     1   43371023 7.1056e+11 97564
    ## - kw_avg_max                     1   52476155 7.1057e+11 97564
    ## - abs_title_sentiment_polarity   1   54680264 7.1057e+11 97564
    ## - self_reference_avg_sharess     1   63373704 7.1058e+11 97564
    ## - min_positive_polarity          1   63508873 7.1058e+11 97564
    ## - global_rate_negative_words     1   85908271 7.1060e+11 97564
    ## - n_tokens_title                 1  110481794 7.1062e+11 97565
    ## - n_non_stop_words               1  139600158 7.1065e+11 97565
    ## - kw_min_min                     1  176001481 7.1069e+11 97565
    ## - timedelta                      1  194623500 7.1071e+11 97565
    ## - title_sentiment_polarity       1  209194800 7.1072e+11 97565
    ## - n_non_stop_unique_tokens       1  239579705 7.1075e+11 97565
    ## - data_channel_is_socmed         1  247661986 7.1076e+11 97566
    ## <none>                                        7.1051e+11 97566
    ## - num_videos                     1  281065366 7.1079e+11 97566
    ## - num_keywords                   1  283031172 7.1080e+11 97566
    ## - kw_max_min                     1  284293105 7.1080e+11 97566
    ## - title_subjectivity             1  294251898 7.1081e+11 97566
    ## - kw_avg_min                     1  310134345 7.1082e+11 97566
    ## - global_rate_positive_words     1  313873073 7.1083e+11 97566
    ## - LDA_03                         1  333792712 7.1085e+11 97566
    ## - max_positive_polarity          1  384943159 7.1090e+11 97567
    ## - average_token_length           1  385747917 7.1090e+11 97567
    ## - self_reference_min_shares      1  437412591 7.1095e+11 97567
    ## - global_sentiment_polarity      1  454597014 7.1097e+11 97567
    ## - max_negative_polarity          1  540878230 7.1105e+11 97568
    ## - num_imgs                       1  568439013 7.1108e+11 97568
    ## - num_hrefs                      1  571766733 7.1108e+11 97568
    ## - n_unique_tokens                1  669474931 7.1118e+11 97569
    ## - data_channel_is_world          1  746699491 7.1126e+11 97569
    ## - data_channel_is_lifestyle      1  782312906 7.1130e+11 97569
    ## - LDA_00                         1  900215743 7.1141e+11 97570
    ## - kw_max_avg                     1 1001589388 7.1151e+11 97571
    ## - data_channel_is_bus            1 1158804297 7.1167e+11 97572
    ## - data_channel_is_entertainment  1 1169911587 7.1168e+11 97572
    ## - num_self_hrefs                 1 1216139196 7.1173e+11 97573
    ## - avg_negative_polarity          1 1993065036 7.1251e+11 97578
    ## - min_negative_polarity          1 2321637274 7.1283e+11 97581
    ## - kw_avg_avg                     1 2535286684 7.1305e+11 97582
    ## - n_tokens_content               1 6917298477 7.1743e+11 97614
    ## 
    ## Step:  AIC=97563.99
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - avg_positive_polarity          1   30707298 7.1058e+11 97562
    ## - kw_max_max                     1   42756840 7.1060e+11 97562
    ## - kw_avg_max                     1   52422925 7.1061e+11 97562
    ## - abs_title_sentiment_polarity   1   54329182 7.1061e+11 97562
    ## - min_positive_polarity          1   56815887 7.1061e+11 97562
    ## - self_reference_avg_sharess     1   63963442 7.1062e+11 97562
    ## - n_non_stop_words               1   99383019 7.1065e+11 97563
    ## - n_tokens_title                 1  109123448 7.1066e+11 97563
    ## - kw_min_min                     1  175802073 7.1073e+11 97563
    ## - timedelta                      1  186927982 7.1074e+11 97563
    ## - title_sentiment_polarity       1  211332515 7.1076e+11 97564
    ## - n_non_stop_unique_tokens       1  237324137 7.1079e+11 97564
    ## - data_channel_is_socmed         1  247847652 7.1080e+11 97564
    ## <none>                                        7.1055e+11 97564
    ## - num_keywords                   1  279988343 7.1083e+11 97564
    ## - kw_max_min                     1  283289788 7.1084e+11 97564
    ## - title_subjectivity             1  297163329 7.1085e+11 97564
    ## - kw_avg_min                     1  309575622 7.1086e+11 97564
    ## - num_videos                     1  312210873 7.1087e+11 97564
    ## - LDA_03                         1  341787790 7.1089e+11 97564
    ## - average_token_length           1  380391787 7.1093e+11 97565
    ## - max_positive_polarity          1  400211273 7.1095e+11 97565
    ## - global_sentiment_polarity      1  414405482 7.1097e+11 97565
    ## - global_rate_negative_words     1  419064994 7.1097e+11 97565
    ## - self_reference_min_shares      1  432051533 7.1099e+11 97565
    ## - max_negative_polarity          1  541097364 7.1109e+11 97566
    ## - num_imgs                       1  567653892 7.1112e+11 97566
    ## - num_hrefs                      1  581001195 7.1113e+11 97566
    ## - global_rate_positive_words     1  653337074 7.1121e+11 97567
    ## - n_unique_tokens                1  665614001 7.1122e+11 97567
    ## - data_channel_is_world          1  724897974 7.1128e+11 97567
    ## - data_channel_is_lifestyle      1  780953575 7.1133e+11 97568
    ## - LDA_00                         1  899254393 7.1145e+11 97569
    ## - kw_max_avg                     1  999002185 7.1155e+11 97569
    ## - data_channel_is_bus            1 1154569288 7.1171e+11 97570
    ## - data_channel_is_entertainment  1 1184088826 7.1174e+11 97571
    ## - num_self_hrefs                 1 1212068715 7.1177e+11 97571
    ## - avg_negative_polarity          1 1969060276 7.1252e+11 97576
    ## - min_negative_polarity          1 2287060346 7.1284e+11 97579
    ## - kw_avg_avg                     1 2534925256 7.1309e+11 97581
    ## - n_tokens_content               1 6912038294 7.1746e+11 97612
    ## 
    ## Step:  AIC=97562.21
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_max_max                     1   42792319 7.1063e+11 97561
    ## - kw_avg_max                     1   51743169 7.1064e+11 97561
    ## - abs_title_sentiment_polarity   1   57856164 7.1064e+11 97561
    ## - self_reference_avg_sharess     1   63533216 7.1065e+11 97561
    ## - n_non_stop_words               1   93187309 7.1068e+11 97561
    ## - n_tokens_title                 1  110859983 7.1069e+11 97561
    ## - min_positive_polarity          1  134108017 7.1072e+11 97561
    ## - kw_min_min                     1  174334578 7.1076e+11 97561
    ## - timedelta                      1  189199106 7.1077e+11 97562
    ## - title_sentiment_polarity       1  213376596 7.1080e+11 97562
    ## - n_non_stop_unique_tokens       1  239652118 7.1082e+11 97562
    ## - data_channel_is_socmed         1  252030964 7.1084e+11 97562
    ## <none>                                        7.1058e+11 97562
    ## - num_keywords                   1  277854604 7.1086e+11 97562
    ## - kw_max_min                     1  283777224 7.1087e+11 97562
    ## - title_subjectivity             1  298722989 7.1088e+11 97562
    ## - kw_avg_min                     1  310520582 7.1089e+11 97562
    ## - num_videos                     1  313128197 7.1090e+11 97563
    ## - LDA_03                         1  336756902 7.1092e+11 97563
    ## - average_token_length           1  383610121 7.1097e+11 97563
    ## - global_rate_negative_words     1  406980860 7.1099e+11 97563
    ## - self_reference_min_shares      1  437473103 7.1102e+11 97563
    ## - global_sentiment_polarity      1  451940282 7.1104e+11 97564
    ## - max_negative_polarity          1  521760758 7.1111e+11 97564
    ## - num_imgs                       1  571957045 7.1116e+11 97564
    ## - num_hrefs                      1  586760866 7.1117e+11 97565
    ## - n_unique_tokens                1  657872603 7.1124e+11 97565
    ## - global_rate_positive_words     1  663046756 7.1125e+11 97565
    ## - max_positive_polarity          1  683440169 7.1127e+11 97565
    ## - data_channel_is_world          1  724907062 7.1131e+11 97566
    ## - data_channel_is_lifestyle      1  790596200 7.1137e+11 97566
    ## - LDA_00                         1  896311541 7.1148e+11 97567
    ## - kw_max_avg                     1  996559723 7.1158e+11 97568
    ## - data_channel_is_bus            1 1165480662 7.1175e+11 97569
    ## - data_channel_is_entertainment  1 1192431299 7.1178e+11 97569
    ## - num_self_hrefs                 1 1212115861 7.1180e+11 97569
    ## - avg_negative_polarity          1 1951835652 7.1254e+11 97574
    ## - min_negative_polarity          1 2274263304 7.1286e+11 97577
    ## - kw_avg_avg                     1 2527542468 7.1311e+11 97579
    ## - n_tokens_content               1 6914684306 7.1750e+11 97611
    ## 
    ## Step:  AIC=97560.52
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_avg_max + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_avg_max                     1   33387676 7.1066e+11 97559
    ## - abs_title_sentiment_polarity   1   59188343 7.1069e+11 97559
    ## - self_reference_avg_sharess     1   61878114 7.1069e+11 97559
    ## - n_non_stop_words               1   93790127 7.1072e+11 97559
    ## - n_tokens_title                 1  111767134 7.1074e+11 97559
    ## - min_positive_polarity          1  134752737 7.1076e+11 97560
    ## - kw_min_min                     1  156240724 7.1078e+11 97560
    ## - timedelta                      1  160904814 7.1079e+11 97560
    ## - title_sentiment_polarity       1  217460375 7.1084e+11 97560
    ## - n_non_stop_unique_tokens       1  238744943 7.1087e+11 97560
    ## - data_channel_is_socmed         1  245854299 7.1087e+11 97560
    ## <none>                                        7.1063e+11 97561
    ## - kw_max_min                     1  288385362 7.1091e+11 97561
    ## - title_subjectivity             1  300558114 7.1093e+11 97561
    ## - LDA_03                         1  314957967 7.1094e+11 97561
    ## - kw_avg_min                     1  315092036 7.1094e+11 97561
    ## - num_videos                     1  320992701 7.1095e+11 97561
    ## - num_keywords                   1  324728897 7.1095e+11 97561
    ## - average_token_length           1  381500034 7.1101e+11 97561
    ## - global_rate_negative_words     1  411468434 7.1104e+11 97562
    ## - self_reference_min_shares      1  442347938 7.1107e+11 97562
    ## - global_sentiment_polarity      1  451669960 7.1108e+11 97562
    ## - max_negative_polarity          1  515290482 7.1114e+11 97562
    ## - num_imgs                       1  564186285 7.1119e+11 97563
    ## - num_hrefs                      1  592712558 7.1122e+11 97563
    ## - n_unique_tokens                1  656747803 7.1128e+11 97563
    ## - global_rate_positive_words     1  657095087 7.1128e+11 97563
    ## - max_positive_polarity          1  683009541 7.1131e+11 97564
    ## - data_channel_is_world          1  712452808 7.1134e+11 97564
    ## - data_channel_is_lifestyle      1  797719067 7.1142e+11 97564
    ## - LDA_00                         1  892540544 7.1152e+11 97565
    ## - kw_max_avg                     1 1005953166 7.1163e+11 97566
    ## - data_channel_is_entertainment  1 1164762917 7.1179e+11 97567
    ## - data_channel_is_bus            1 1171660815 7.1180e+11 97567
    ## - num_self_hrefs                 1 1212909781 7.1184e+11 97567
    ## - avg_negative_polarity          1 1942138911 7.1257e+11 97573
    ## - min_negative_polarity          1 2269131512 7.1290e+11 97575
    ## - kw_avg_avg                     1 2556710914 7.1318e+11 97577
    ## - n_tokens_content               1 6906480580 7.1753e+11 97609
    ## 
    ## Step:  AIC=97558.77
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - self_reference_avg_sharess     1   58174803 7.1072e+11 97557
    ## - abs_title_sentiment_polarity   1   58664111 7.1072e+11 97557
    ## - n_non_stop_words               1   97217216 7.1076e+11 97557
    ## - n_tokens_title                 1  108634524 7.1077e+11 97558
    ## - min_positive_polarity          1  131902143 7.1079e+11 97558
    ## - timedelta                      1  210038623 7.1087e+11 97558
    ## - title_sentiment_polarity       1  217708738 7.1088e+11 97558
    ## - kw_min_min                     1  226931717 7.1089e+11 97558
    ## - n_non_stop_unique_tokens       1  232154263 7.1089e+11 97558
    ## - data_channel_is_socmed         1  233878475 7.1089e+11 97558
    ## - kw_max_min                     1  259438933 7.1092e+11 97559
    ## <none>                                        7.1066e+11 97559
    ## - kw_avg_min                     1  283282919 7.1094e+11 97559
    ## - LDA_03                         1  290579160 7.1095e+11 97559
    ## - title_subjectivity             1  301522884 7.1096e+11 97559
    ## - num_videos                     1  348256558 7.1101e+11 97559
    ## - average_token_length           1  382542098 7.1104e+11 97560
    ## - global_rate_negative_words     1  410046182 7.1107e+11 97560
    ## - self_reference_min_shares      1  452194067 7.1111e+11 97560
    ## - global_sentiment_polarity      1  453271655 7.1111e+11 97560
    ## - num_keywords                   1  491548154 7.1115e+11 97560
    ## - max_negative_polarity          1  521525093 7.1118e+11 97561
    ## - num_imgs                       1  551316121 7.1121e+11 97561
    ## - num_hrefs                      1  619097418 7.1128e+11 97561
    ## - n_unique_tokens                1  642514884 7.1130e+11 97561
    ## - global_rate_positive_words     1  652681996 7.1131e+11 97562
    ## - max_positive_polarity          1  688191021 7.1135e+11 97562
    ## - data_channel_is_world          1  699047580 7.1136e+11 97562
    ## - data_channel_is_lifestyle      1  771812896 7.1143e+11 97562
    ## - LDA_00                         1  881844010 7.1154e+11 97563
    ## - kw_max_avg                     1  980651812 7.1164e+11 97564
    ## - data_channel_is_entertainment  1 1132768348 7.1179e+11 97565
    ## - num_self_hrefs                 1 1222075190 7.1188e+11 97566
    ## - data_channel_is_bus            1 1252608361 7.1191e+11 97566
    ## - avg_negative_polarity          1 1941165290 7.1260e+11 97571
    ## - min_negative_polarity          1 2264510272 7.1292e+11 97573
    ## - kw_avg_avg                     1 2795457443 7.1346e+11 97577
    ## - n_tokens_content               1 6876032669 7.1754e+11 97607
    ## 
    ## Step:  AIC=97557.19
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + LDA_00 + LDA_03 + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - abs_title_sentiment_polarity   1   59997672 7.1078e+11 97556
    ## - n_non_stop_words               1   98201965 7.1082e+11 97556
    ## - n_tokens_title                 1  110826574 7.1083e+11 97556
    ## - min_positive_polarity          1  136234435 7.1085e+11 97556
    ## - timedelta                      1  211392933 7.1093e+11 97557
    ## - title_sentiment_polarity       1  216549376 7.1093e+11 97557
    ## - kw_min_min                     1  224805699 7.1094e+11 97557
    ## - n_non_stop_unique_tokens       1  227350262 7.1095e+11 97557
    ## - data_channel_is_socmed         1  232330887 7.1095e+11 97557
    ## - kw_max_min                     1  258677049 7.1098e+11 97557
    ## <none>                                        7.1072e+11 97557
    ## - LDA_03                         1  284657795 7.1100e+11 97557
    ## - kw_avg_min                     1  284982831 7.1100e+11 97557
    ## - title_subjectivity             1  305574426 7.1102e+11 97557
    ## - num_videos                     1  340981840 7.1106e+11 97558
    ## - average_token_length           1  383724369 7.1110e+11 97558
    ## - global_rate_negative_words     1  407637064 7.1113e+11 97558
    ## - global_sentiment_polarity      1  453297394 7.1117e+11 97559
    ## - num_keywords                   1  500358047 7.1122e+11 97559
    ## - max_negative_polarity          1  515164792 7.1123e+11 97559
    ## - num_imgs                       1  546542024 7.1126e+11 97559
    ## - num_hrefs                      1  614381681 7.1133e+11 97560
    ## - n_unique_tokens                1  637058513 7.1136e+11 97560
    ## - global_rate_positive_words     1  658888259 7.1138e+11 97560
    ## - max_positive_polarity          1  683667806 7.1140e+11 97560
    ## - data_channel_is_world          1  701946177 7.1142e+11 97560
    ## - data_channel_is_lifestyle      1  778741177 7.1150e+11 97561
    ## - LDA_00                         1  881022330 7.1160e+11 97562
    ## - kw_max_avg                     1  982951072 7.1170e+11 97562
    ## - data_channel_is_entertainment  1 1136059105 7.1185e+11 97564
    ## - num_self_hrefs                 1 1188376120 7.1191e+11 97564
    ## - self_reference_min_shares      1 1212844466 7.1193e+11 97564
    ## - data_channel_is_bus            1 1243474724 7.1196e+11 97564
    ## - avg_negative_polarity          1 1931331045 7.1265e+11 97569
    ## - min_negative_polarity          1 2244426630 7.1296e+11 97572
    ## - kw_avg_avg                     1 2850494721 7.1357e+11 97576
    ## - n_tokens_content               1 6843929384 7.1756e+11 97605
    ## 
    ## Step:  AIC=97555.63
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + LDA_00 + LDA_03 + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - n_non_stop_words               1   98228679 7.1088e+11 97554
    ## - n_tokens_title                 1  117998265 7.1090e+11 97554
    ## - min_positive_polarity          1  133535129 7.1091e+11 97555
    ## - title_sentiment_polarity       1  165559281 7.1094e+11 97555
    ## - timedelta                      1  216559393 7.1099e+11 97555
    ## - n_non_stop_unique_tokens       1  218264522 7.1100e+11 97555
    ## - kw_min_min                     1  224293358 7.1100e+11 97555
    ## - data_channel_is_socmed         1  235921258 7.1101e+11 97555
    ## - kw_max_min                     1  260798688 7.1104e+11 97556
    ## <none>                                        7.1078e+11 97556
    ## - title_subjectivity             1  281029124 7.1106e+11 97556
    ## - LDA_03                         1  282026078 7.1106e+11 97556
    ## - kw_avg_min                     1  286612523 7.1106e+11 97556
    ## - num_videos                     1  339616506 7.1112e+11 97556
    ## - average_token_length           1  376646425 7.1115e+11 97556
    ## - global_rate_negative_words     1  400076200 7.1118e+11 97557
    ## - global_sentiment_polarity      1  440313009 7.1122e+11 97557
    ## - num_keywords                   1  499849601 7.1128e+11 97557
    ## - max_negative_polarity          1  500258412 7.1128e+11 97557
    ## - num_imgs                       1  536781085 7.1131e+11 97558
    ## - num_hrefs                      1  610763237 7.1139e+11 97558
    ## - n_unique_tokens                1  620762031 7.1140e+11 97558
    ## - global_rate_positive_words     1  646457360 7.1142e+11 97558
    ## - max_positive_polarity          1  694691740 7.1147e+11 97559
    ## - data_channel_is_world          1  709203844 7.1149e+11 97559
    ## - data_channel_is_lifestyle      1  786564088 7.1156e+11 97559
    ## - LDA_00                         1  888716862 7.1167e+11 97560
    ## - kw_max_avg                     1  996477107 7.1177e+11 97561
    ## - data_channel_is_entertainment  1 1140837985 7.1192e+11 97562
    ## - num_self_hrefs                 1 1187524452 7.1197e+11 97562
    ## - self_reference_min_shares      1 1223737759 7.1200e+11 97563
    ## - data_channel_is_bus            1 1256817745 7.1203e+11 97563
    ## - avg_negative_polarity          1 1898694884 7.1268e+11 97568
    ## - min_negative_polarity          1 2240996254 7.1302e+11 97570
    ## - kw_avg_avg                     1 2867533826 7.1365e+11 97575
    ## - n_tokens_content               1 6823223355 7.1760e+11 97603
    ## 
    ## Step:  AIC=97554.35
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + LDA_00 + 
    ##     LDA_03 + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - min_positive_polarity          1  114644865 7.1099e+11 97553
    ## - n_tokens_title                 1  138848668 7.1102e+11 97553
    ## - n_non_stop_unique_tokens       1  147572534 7.1102e+11 97553
    ## - title_sentiment_polarity       1  163211701 7.1104e+11 97554
    ## - timedelta                      1  219971479 7.1110e+11 97554
    ## - kw_min_min                     1  220481675 7.1110e+11 97554
    ## - data_channel_is_socmed         1  239784689 7.1112e+11 97554
    ## - kw_max_min                     1  262070900 7.1114e+11 97554
    ## - LDA_03                         1  270084641 7.1115e+11 97554
    ## <none>                                        7.1088e+11 97554
    ## - title_subjectivity             1  276171202 7.1115e+11 97554
    ## - kw_avg_min                     1  287796743 7.1116e+11 97554
    ## - num_videos                     1  344110763 7.1122e+11 97555
    ## - average_token_length           1  399198499 7.1128e+11 97555
    ## - global_rate_negative_words     1  434237906 7.1131e+11 97556
    ## - global_sentiment_polarity      1  457395074 7.1133e+11 97556
    ## - num_keywords                   1  471032296 7.1135e+11 97556
    ## - max_negative_polarity          1  497457516 7.1137e+11 97556
    ## - num_imgs                       1  504128871 7.1138e+11 97556
    ## - num_hrefs                      1  559831356 7.1144e+11 97556
    ## - n_unique_tokens                1  577993057 7.1145e+11 97557
    ## - global_rate_positive_words     1  620637688 7.1150e+11 97557
    ## - max_positive_polarity          1  641444263 7.1152e+11 97557
    ## - data_channel_is_lifestyle      1  799118864 7.1168e+11 97558
    ## - data_channel_is_world          1  835869370 7.1171e+11 97558
    ## - LDA_00                         1  883918023 7.1176e+11 97559
    ## - kw_max_avg                     1  966941089 7.1184e+11 97559
    ## - num_self_hrefs                 1 1132461770 7.1201e+11 97561
    ## - data_channel_is_entertainment  1 1170904922 7.1205e+11 97561
    ## - self_reference_min_shares      1 1248755099 7.1212e+11 97561
    ## - data_channel_is_bus            1 1296036165 7.1217e+11 97562
    ## - avg_negative_polarity          1 1946993084 7.1282e+11 97567
    ## - min_negative_polarity          1 2258255687 7.1313e+11 97569
    ## - kw_avg_avg                     1 2806931975 7.1368e+11 97573
    ## - n_tokens_content               1 7586732836 7.1846e+11 97608
    ## 
    ## Step:  AIC=97553.19
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + LDA_00 + 
    ##     LDA_03 + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - n_non_stop_unique_tokens       1  123387464 7.1111e+11 97552
    ## - n_tokens_title                 1  139187927 7.1113e+11 97552
    ## - title_sentiment_polarity       1  158078101 7.1115e+11 97552
    ## - kw_min_min                     1  221027170 7.1121e+11 97553
    ## - timedelta                      1  224037117 7.1121e+11 97553
    ## - data_channel_is_socmed         1  225835404 7.1122e+11 97553
    ## - title_subjectivity             1  261935413 7.1125e+11 97553
    ## - kw_max_min                     1  269931649 7.1126e+11 97553
    ## - LDA_03                         1  271336610 7.1126e+11 97553
    ## <none>                                        7.1099e+11 97553
    ## - kw_avg_min                     1  297235112 7.1129e+11 97553
    ## - num_videos                     1  342464649 7.1133e+11 97554
    ## - global_rate_negative_words     1  374675688 7.1137e+11 97554
    ## - global_sentiment_polarity      1  377059173 7.1137e+11 97554
    ## - average_token_length           1  415335471 7.1141e+11 97554
    ## - num_keywords                   1  475802793 7.1147e+11 97555
    ## - max_negative_polarity          1  498177476 7.1149e+11 97555
    ## - n_unique_tokens                1  501286909 7.1149e+11 97555
    ## - global_rate_positive_words     1  510336600 7.1150e+11 97555
    ## - num_imgs                       1  511953941 7.1150e+11 97555
    ## - num_hrefs                      1  608968559 7.1160e+11 97556
    ## - max_positive_polarity          1  622226300 7.1161e+11 97556
    ## - data_channel_is_lifestyle      1  803908436 7.1179e+11 97557
    ## - data_channel_is_world          1  805789468 7.1180e+11 97557
    ## - LDA_00                         1  911559792 7.1190e+11 97558
    ## - kw_max_avg                     1  953580157 7.1194e+11 97558
    ## - num_self_hrefs                 1 1124007640 7.1211e+11 97559
    ## - data_channel_is_entertainment  1 1130139599 7.1212e+11 97559
    ## - self_reference_min_shares      1 1252526080 7.1224e+11 97560
    ## - data_channel_is_bus            1 1290951672 7.1228e+11 97561
    ## - avg_negative_polarity          1 1912839345 7.1290e+11 97565
    ## - min_negative_polarity          1 2245413763 7.1324e+11 97568
    ## - kw_avg_avg                     1 2789468302 7.1378e+11 97572
    ## - n_tokens_content               1 7553756938 7.1854e+11 97606
    ## 
    ## Step:  AIC=97552.1
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + LDA_00 + LDA_03 + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - n_tokens_title                 1  131646582 7.1125e+11 97551
    ## - title_sentiment_polarity       1  160754489 7.1128e+11 97551
    ## - kw_min_min                     1  222621895 7.1134e+11 97552
    ## - data_channel_is_socmed         1  226667047 7.1134e+11 97552
    ## - timedelta                      1  230111965 7.1134e+11 97552
    ## - kw_max_min                     1  271513387 7.1139e+11 97552
    ## <none>                                        7.1111e+11 97552
    ## - title_subjectivity             1  280229591 7.1139e+11 97552
    ## - kw_avg_min                     1  300481574 7.1141e+11 97552
    ## - num_videos                     1  314508002 7.1143e+11 97552
    ## - LDA_03                         1  320028315 7.1143e+11 97552
    ## - global_sentiment_polarity      1  368577818 7.1148e+11 97553
    ## - global_rate_negative_words     1  374013732 7.1149e+11 97553
    ## - num_imgs                       1  409817471 7.1152e+11 97553
    ## - average_token_length           1  422501957 7.1154e+11 97553
    ## - max_negative_polarity          1  455426863 7.1157e+11 97553
    ## - num_keywords                   1  479071606 7.1159e+11 97554
    ## - global_rate_positive_words     1  519581010 7.1163e+11 97554
    ## - n_unique_tokens                1  703401128 7.1182e+11 97555
    ## - max_positive_polarity          1  720811474 7.1184e+11 97555
    ## - num_hrefs                      1  738603303 7.1185e+11 97555
    ## - data_channel_is_world          1  827342506 7.1194e+11 97556
    ## - data_channel_is_lifestyle      1  861786427 7.1198e+11 97556
    ## - LDA_00                         1  912137520 7.1203e+11 97557
    ## - kw_max_avg                     1  989407152 7.1210e+11 97557
    ## - num_self_hrefs                 1 1130446383 7.1224e+11 97558
    ## - data_channel_is_entertainment  1 1134867321 7.1225e+11 97558
    ## - self_reference_min_shares      1 1259375229 7.1237e+11 97559
    ## - data_channel_is_bus            1 1316591869 7.1243e+11 97560
    ## - avg_negative_polarity          1 1880373814 7.1299e+11 97564
    ## - min_negative_polarity          1 2351592194 7.1347e+11 97567
    ## - kw_avg_avg                     1 2887666137 7.1400e+11 97571
    ## - n_tokens_content               1 7793769757 7.1891e+11 97607
    ## 
    ## Step:  AIC=97551.06
    ## shares ~ timedelta + n_tokens_content + n_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + LDA_00 + LDA_03 + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - title_sentiment_polarity       1  166628435 7.1141e+11 97550
    ## - timedelta                      1  176405161 7.1142e+11 97550
    ## - kw_min_min                     1  236164058 7.1148e+11 97551
    ## - data_channel_is_socmed         1  241210813 7.1149e+11 97551
    ## - kw_max_min                     1  268000580 7.1151e+11 97551
    ## <none>                                        7.1125e+11 97551
    ## - kw_avg_min                     1  297654463 7.1154e+11 97551
    ## - num_videos                     1  302565179 7.1155e+11 97551
    ## - title_subjectivity             1  310191237 7.1156e+11 97551
    ## - LDA_03                         1  327620408 7.1157e+11 97551
    ## - global_sentiment_polarity      1  368064879 7.1161e+11 97552
    ## - global_rate_negative_words     1  374072269 7.1162e+11 97552
    ## - num_imgs                       1  407362084 7.1165e+11 97552
    ## - average_token_length           1  440342565 7.1169e+11 97552
    ## - max_negative_polarity          1  465078252 7.1171e+11 97552
    ## - num_keywords                   1  479205378 7.1173e+11 97553
    ## - global_rate_positive_words     1  530027977 7.1178e+11 97553
    ## - max_positive_polarity          1  707163829 7.1195e+11 97554
    ## - n_unique_tokens                1  714992236 7.1196e+11 97554
    ## - num_hrefs                      1  721337032 7.1197e+11 97554
    ## - data_channel_is_world          1  809302048 7.1206e+11 97555
    ## - data_channel_is_lifestyle      1  873983705 7.1212e+11 97555
    ## - LDA_00                         1  910143078 7.1216e+11 97556
    ## - kw_max_avg                     1  975090568 7.1222e+11 97556
    ## - data_channel_is_entertainment  1 1070970387 7.1232e+11 97557
    ## - num_self_hrefs                 1 1105754031 7.1235e+11 97557
    ## - self_reference_min_shares      1 1257574105 7.1250e+11 97558
    ## - data_channel_is_bus            1 1304401770 7.1255e+11 97559
    ## - avg_negative_polarity          1 1897021887 7.1314e+11 97563
    ## - min_negative_polarity          1 2362390426 7.1361e+11 97566
    ## - kw_avg_avg                     1 2849963842 7.1410e+11 97570
    ## - n_tokens_content               1 7788925615 7.1903e+11 97606
    ## 
    ## Step:  AIC=97550.28
    ## shares ~ timedelta + n_tokens_content + n_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + LDA_00 + LDA_03 + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - timedelta                      1  181212441 7.1159e+11 97550
    ## - kw_min_min                     1  235843664 7.1165e+11 97550
    ## - data_channel_is_socmed         1  240250061 7.1165e+11 97550
    ## <none>                                        7.1141e+11 97550
    ## - kw_max_min                     1  273638100 7.1169e+11 97550
    ## - num_videos                     1  279994376 7.1169e+11 97550
    ## - kw_avg_min                     1  301766706 7.1171e+11 97550
    ## - LDA_03                         1  318548522 7.1173e+11 97551
    ## - global_rate_negative_words     1  371659078 7.1178e+11 97551
    ## - num_imgs                       1  397718005 7.1181e+11 97551
    ## - global_sentiment_polarity      1  415130855 7.1183e+11 97551
    ## - average_token_length           1  424047358 7.1184e+11 97551
    ## - title_subjectivity             1  433796558 7.1185e+11 97551
    ## - max_negative_polarity          1  433849455 7.1185e+11 97551
    ## - num_keywords                   1  486280047 7.1190e+11 97552
    ## - global_rate_positive_words     1  516244472 7.1193e+11 97552
    ## - n_unique_tokens                1  686031861 7.1210e+11 97553
    ## - num_hrefs                      1  726073341 7.1214e+11 97554
    ## - max_positive_polarity          1  727281557 7.1214e+11 97554
    ## - data_channel_is_world          1  831182898 7.1224e+11 97554
    ## - data_channel_is_lifestyle      1  874909737 7.1229e+11 97555
    ## - LDA_00                         1  896417672 7.1231e+11 97555
    ## - kw_max_avg                     1  972662501 7.1239e+11 97555
    ## - data_channel_is_entertainment  1 1077248237 7.1249e+11 97556
    ## - num_self_hrefs                 1 1122192002 7.1253e+11 97556
    ## - self_reference_min_shares      1 1250082716 7.1266e+11 97557
    ## - data_channel_is_bus            1 1292951949 7.1271e+11 97558
    ## - avg_negative_polarity          1 1826465789 7.1324e+11 97562
    ## - min_negative_polarity          1 2310432496 7.1372e+11 97565
    ## - kw_avg_avg                     1 2869194678 7.1428e+11 97569
    ## - n_tokens_content               1 7775706484 7.1919e+11 97605
    ## 
    ## Step:  AIC=97549.6
    ## shares ~ n_tokens_content + n_unique_tokens + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + LDA_00 + LDA_03 + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_max_min                     1  186345190 7.1178e+11 97549
    ## - kw_avg_min                     1  202276284 7.1180e+11 97549
    ## - data_channel_is_socmed         1  243324580 7.1184e+11 97549
    ## <none>                                        7.1159e+11 97550
    ## - num_videos                     1  281204898 7.1187e+11 97550
    ## - LDA_03                         1  336341856 7.1193e+11 97550
    ## - global_rate_negative_words     1  380623720 7.1197e+11 97550
    ## - num_imgs                       1  383035480 7.1198e+11 97550
    ## - global_sentiment_polarity      1  423442636 7.1202e+11 97551
    ## - title_subjectivity             1  425081053 7.1202e+11 97551
    ## - average_token_length           1  431470026 7.1203e+11 97551
    ## - max_negative_polarity          1  439781165 7.1203e+11 97551
    ## - global_rate_positive_words     1  481130675 7.1207e+11 97551
    ## - num_keywords                   1  497252359 7.1209e+11 97551
    ## - kw_min_min                     1  549428524 7.1214e+11 97552
    ## - max_positive_polarity          1  723379685 7.1232e+11 97553
    ## - n_unique_tokens                1  743982839 7.1234e+11 97553
    ## - num_hrefs                      1  751555778 7.1235e+11 97553
    ## - data_channel_is_lifestyle      1  873881781 7.1247e+11 97554
    ## - kw_max_avg                     1  906968156 7.1250e+11 97554
    ## - LDA_00                         1  909535426 7.1250e+11 97554
    ## - data_channel_is_world          1  946350505 7.1254e+11 97555
    ## - num_self_hrefs                 1 1100751388 7.1269e+11 97556
    ## - data_channel_is_entertainment  1 1156014912 7.1275e+11 97556
    ## - self_reference_min_shares      1 1247474602 7.1284e+11 97557
    ## - data_channel_is_bus            1 1317912884 7.1291e+11 97557
    ## - avg_negative_polarity          1 1835452609 7.1343e+11 97561
    ## - min_negative_polarity          1 2354611327 7.1395e+11 97565
    ## - kw_avg_avg                     1 2717036437 7.1431e+11 97567
    ## - n_tokens_content               1 7766551346 7.1936e+11 97604
    ## 
    ## Step:  AIC=97548.97
    ## shares ~ n_tokens_content + n_unique_tokens + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_avg_min + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_00 + LDA_03 + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_avg_min                     1   16101985 7.1180e+11 97547
    ## - data_channel_is_socmed         1  266725562 7.1205e+11 97549
    ## - num_videos                     1  273553895 7.1205e+11 97549
    ## <none>                                        7.1178e+11 97549
    ## - LDA_03                         1  361904697 7.1214e+11 97550
    ## - global_rate_negative_words     1  380190045 7.1216e+11 97550
    ## - kw_min_min                     1  386637774 7.1217e+11 97550
    ## - num_imgs                       1  394921942 7.1218e+11 97550
    ## - global_sentiment_polarity      1  426712015 7.1221e+11 97550
    ## - title_subjectivity             1  432461899 7.1221e+11 97550
    ## - average_token_length           1  441698322 7.1222e+11 97550
    ## - max_negative_polarity          1  447647612 7.1223e+11 97550
    ## - num_keywords                   1  486595559 7.1227e+11 97551
    ## - global_rate_positive_words     1  503272528 7.1228e+11 97551
    ## - max_positive_polarity          1  720932215 7.1250e+11 97552
    ## - n_unique_tokens                1  752569788 7.1253e+11 97552
    ## - num_hrefs                      1  762560269 7.1254e+11 97553
    ## - kw_max_avg                     1  824026851 7.1260e+11 97553
    ## - data_channel_is_lifestyle      1  893459092 7.1267e+11 97553
    ## - LDA_00                         1  913944547 7.1269e+11 97554
    ## - data_channel_is_world          1  931574161 7.1271e+11 97554
    ## - num_self_hrefs                 1 1063958836 7.1284e+11 97555
    ## - self_reference_min_shares      1 1172974686 7.1295e+11 97556
    ## - data_channel_is_entertainment  1 1178266221 7.1296e+11 97556
    ## - data_channel_is_bus            1 1365597045 7.1315e+11 97557
    ## - avg_negative_polarity          1 1846862199 7.1363e+11 97560
    ## - min_negative_polarity          1 2362805161 7.1414e+11 97564
    ## - kw_avg_avg                     1 2648432272 7.1443e+11 97566
    ## - n_tokens_content               1 7788969443 7.1957e+11 97604
    ## 
    ## Step:  AIC=97547.08
    ## shares ~ n_tokens_content + n_unique_tokens + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_00 + LDA_03 + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - data_channel_is_socmed         1  269063089 7.1207e+11 97547
    ## - num_videos                     1  271473304 7.1207e+11 97547
    ## <none>                                        7.1180e+11 97547
    ## - kw_min_min                     1  370611716 7.1217e+11 97548
    ## - LDA_03                         1  379105457 7.1218e+11 97548
    ## - global_rate_negative_words     1  381051236 7.1218e+11 97548
    ## - num_imgs                       1  391359490 7.1219e+11 97548
    ## - global_sentiment_polarity      1  430531520 7.1223e+11 97548
    ## - title_subjectivity             1  430643257 7.1223e+11 97548
    ## - average_token_length           1  444407055 7.1224e+11 97548
    ## - max_negative_polarity          1  445769197 7.1224e+11 97548
    ## - num_keywords                   1  477811402 7.1227e+11 97549
    ## - global_rate_positive_words     1  505051807 7.1230e+11 97549
    ## - max_positive_polarity          1  722461514 7.1252e+11 97550
    ## - n_unique_tokens                1  749578627 7.1255e+11 97551
    ## - num_hrefs                      1  771277670 7.1257e+11 97551
    ## - data_channel_is_lifestyle      1  889050291 7.1269e+11 97552
    ## - LDA_00                         1  918087635 7.1271e+11 97552
    ## - kw_max_avg                     1  929411943 7.1273e+11 97552
    ## - data_channel_is_world          1  930594116 7.1273e+11 97552
    ## - num_self_hrefs                 1 1069461916 7.1287e+11 97553
    ## - self_reference_min_shares      1 1161040607 7.1296e+11 97554
    ## - data_channel_is_entertainment  1 1180467152 7.1298e+11 97554
    ## - data_channel_is_bus            1 1372511784 7.1317e+11 97555
    ## - avg_negative_polarity          1 1842699892 7.1364e+11 97559
    ## - min_negative_polarity          1 2354671372 7.1415e+11 97562
    ## - kw_avg_avg                     1 2632330298 7.1443e+11 97564
    ## - n_tokens_content               1 7773037607 7.1957e+11 97602
    ## 
    ## Step:  AIC=97547.05
    ## shares ~ n_tokens_content + n_unique_tokens + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_world + kw_min_min + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + LDA_00 + 
    ##     LDA_03 + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## <none>                                        7.1207e+11 97547
    ## - num_videos                     1  282754576 7.1235e+11 97547
    ## - kw_min_min                     1  362966316 7.1243e+11 97548
    ## - num_imgs                       1  371295209 7.1244e+11 97548
    ## - global_rate_negative_words     1  389368261 7.1245e+11 97548
    ## - LDA_03                         1  394009849 7.1246e+11 97548
    ## - global_sentiment_polarity      1  435663152 7.1250e+11 97548
    ## - title_subjectivity             1  446424578 7.1251e+11 97548
    ## - max_negative_polarity          1  462212688 7.1253e+11 97548
    ## - average_token_length           1  471815102 7.1254e+11 97548
    ## - global_rate_positive_words     1  526549809 7.1259e+11 97549
    ## - num_keywords                   1  565135918 7.1263e+11 97549
    ## - LDA_00                         1  656309485 7.1272e+11 97550
    ## - max_positive_polarity          1  712714429 7.1278e+11 97550
    ## - data_channel_is_lifestyle      1  732924605 7.1280e+11 97550
    ## - n_unique_tokens                1  768730601 7.1283e+11 97551
    ## - data_channel_is_world          1  771454195 7.1284e+11 97551
    ## - num_hrefs                      1  776593587 7.1284e+11 97551
    ## - kw_max_avg                     1  922724662 7.1299e+11 97552
    ## - data_channel_is_entertainment  1 1009938590 7.1308e+11 97552
    ## - num_self_hrefs                 1 1093239338 7.1316e+11 97553
    ## - data_channel_is_bus            1 1111898326 7.1318e+11 97553
    ## - self_reference_min_shares      1 1151890044 7.1322e+11 97553
    ## - avg_negative_polarity          1 1905066822 7.1397e+11 97559
    ## - min_negative_polarity          1 2424563781 7.1449e+11 97563
    ## - kw_avg_avg                     1 2655572698 7.1472e+11 97564
    ## - n_tokens_content               1 7812539997 7.1988e+11 97602

``` r
#summary the model
summary(lm.step)
```

    ## 
    ## Call:
    ## lm(formula = shares ~ n_tokens_content + n_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_world + kw_min_min + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + LDA_00 + 
    ##     LDA_03 + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity, 
    ##     data = train)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -28304  -2465  -1028    431 648519 
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   -2.028e+03  1.461e+03  -1.388  0.16531    
    ## n_tokens_content               4.961e+00  6.583e-01   7.537 5.66e-14 ***
    ## n_unique_tokens                5.947e+03  2.515e+03   2.364  0.01811 *  
    ## num_hrefs                      4.644e+01  1.954e+01   2.376  0.01753 *  
    ## num_self_hrefs                -1.590e+02  5.640e+01  -2.819  0.00483 ** 
    ## num_imgs                      -4.267e+01  2.597e+01  -1.643  0.10044    
    ## num_videos                    -6.589e+01  4.596e+01  -1.434  0.15169    
    ## average_token_length          -7.140e+02  3.855e+02  -1.852  0.06407 .  
    ## num_keywords                   1.839e+02  9.074e+01   2.027  0.04271 *  
    ## data_channel_is_lifestyle     -1.800e+03  7.798e+02  -2.308  0.02102 *  
    ## data_channel_is_entertainment -1.343e+03  4.955e+02  -2.710  0.00676 ** 
    ## data_channel_is_bus           -2.009e+03  7.066e+02  -2.843  0.00448 ** 
    ## data_channel_is_world         -1.250e+03  5.277e+02  -2.368  0.01791 *  
    ## kw_min_min                     4.013e+00  2.470e+00   1.624  0.10434    
    ## kw_max_avg                    -1.404e-01  5.420e-02  -2.590  0.00962 ** 
    ## kw_avg_avg                     1.226e+00  2.790e-01   4.394 1.14e-05 ***
    ## self_reference_min_shares      3.494e-02  1.207e-02   2.894  0.00382 ** 
    ## LDA_00                         2.170e+03  9.932e+02   2.184  0.02898 *  
    ## LDA_03                         1.323e+03  7.815e+02   1.693  0.09061 .  
    ## global_sentiment_polarity      6.195e+03  3.481e+03   1.780  0.07518 .  
    ## global_rate_positive_words    -2.787e+04  1.424e+04  -1.957  0.05045 .  
    ## global_rate_negative_words     4.056e+04  2.410e+04   1.683  0.09253 .  
    ## max_positive_polarity         -2.381e+03  1.046e+03  -2.276  0.02287 *  
    ## avg_negative_polarity         -1.261e+04  3.389e+03  -3.722  0.00020 ***
    ## min_negative_polarity          5.456e+03  1.299e+03   4.199 2.73e-05 ***
    ## max_negative_polarity          5.335e+03  2.910e+03   1.833  0.06684 .  
    ## title_subjectivity             9.321e+02  5.174e+02   1.802  0.07167 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 11730 on 5177 degrees of freedom
    ## Multiple R-squared:  0.03656,    Adjusted R-squared:  0.03172 
    ## F-statistic: 7.556 on 26 and 5177 DF,  p-value: < 2.2e-16

Calculate the predicted mean square error on the train set:

``` r
train.pred <- predict(lm.step, train[,-52])
mean((train.pred - train$shares)^2)
```

    ## [1] 136830375

So, the predicted mean square error on the training dataset is
264157333.

### On test set

Calculate the predicted mean square error on the test set:

``` r
test.pred <- predict(lm.step, test[,-52])
mean((test.pred - test$shares)^2)
```

    ## [1] 376750747

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
