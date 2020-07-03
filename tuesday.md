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

    ## [1] 7390

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

    ## [1] 5173

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
    ##   timedelta                   Min.   :  9.0      1st Qu.:163.0     
    ## n_tokens_title                Min.   : 4.00      1st Qu.: 9.00     
    ## n_tokens_content              Min.   :   0       1st Qu.: 247      
    ## n_unique_tokens               Min.   :  0.0000   1st Qu.:  0.4752  
    ## n_non_stop_words              Min.   :   0.000   1st Qu.:   1.000  
    ## n_non_stop_unique_tokens      Min.   :  0.0000   1st Qu.:  0.6291  
    ##   num_hrefs                   Min.   :  0.00     1st Qu.:  4.00    
    ## num_self_hrefs                Min.   : 0.000     1st Qu.: 1.000    
    ##    num_imgs                   Min.   : 0.000     1st Qu.: 1.000    
    ##   num_videos                  Min.   : 0.000     1st Qu.: 0.000    
    ## average_token_length          Min.   :0.000      1st Qu.:4.473     
    ##  num_keywords                 Min.   : 1.000     1st Qu.: 6.000    
    ## data_channel_is_lifestyle     Min.   :0.00000    1st Qu.:0.00000   
    ## data_channel_is_entertainment Min.   :0.0000     1st Qu.:0.0000    
    ## data_channel_is_bus           Min.   :0.0000     1st Qu.:0.0000    
    ## data_channel_is_socmed        Min.   :0.00000    1st Qu.:0.00000   
    ## data_channel_is_tech          Min.   :0.000      1st Qu.:0.000     
    ## data_channel_is_world         Min.   :0.0000     1st Qu.:0.0000    
    ##   kw_min_min                  Min.   : -1.00     1st Qu.: -1.00    
    ##   kw_max_min                  Min.   :     0     1st Qu.:   442    
    ##   kw_avg_min                  Min.   :   -1.0    1st Qu.:  141.3   
    ##   kw_min_max                  Min.   :     0     1st Qu.:     0    
    ##   kw_max_max                  Min.   : 17100     1st Qu.:843300    
    ##   kw_avg_max                  Min.   :  3460     1st Qu.:172771    
    ##   kw_min_avg                  Min.   :  -1.0     1st Qu.:   0.0    
    ##   kw_max_avg                  Min.   :  2019     1st Qu.:  3540    
    ##   kw_avg_avg                  Min.   :  728.2    1st Qu.: 2372.5   
    ## self_reference_min_shares     Min.   :     0     1st Qu.:   647    
    ## self_reference_max_shares     Min.   :     0     1st Qu.:  1100    
    ## self_reference_avg_sharess    Min.   :     0     1st Qu.:  1000    
    ##     LDA_00                    Min.   :0.00000    1st Qu.:0.02506   
    ##     LDA_01                    Min.   :0.00000    1st Qu.:0.02501   
    ##     LDA_02                    Min.   :0.00000    1st Qu.:0.02857   
    ##     LDA_03                    Min.   :0.00000    1st Qu.:0.02857   
    ##     LDA_04                    Min.   :0.00000    1st Qu.:0.02858   
    ## global_subjectivity           Min.   :0.0000     1st Qu.:0.3952    
    ## global_sentiment_polarity     Min.   :-0.3088    1st Qu.: 0.0603   
    ## global_rate_positive_words    Min.   :0.00000    1st Qu.:0.02882   
    ## global_rate_negative_words    Min.   :0.000000   1st Qu.:0.009346  
    ## rate_positive_words           Min.   :0.0000     1st Qu.:0.6087    
    ## rate_negative_words           Min.   :0.0000     1st Qu.:0.1818    
    ## avg_positive_polarity         Min.   :0.0000     1st Qu.:0.3044    
    ## min_positive_polarity         Min.   :0.00000    1st Qu.:0.05000   
    ## max_positive_polarity         Min.   :0.0000     1st Qu.:0.6000    
    ## avg_negative_polarity         Min.   :-1.0000    1st Qu.:-0.3242   
    ## min_negative_polarity         Min.   :-1.0000    1st Qu.:-0.7000   
    ## max_negative_polarity         Min.   :-1.0000    1st Qu.:-0.1250   
    ## title_subjectivity            Min.   :0.0000     1st Qu.:0.0000    
    ## title_sentiment_polarity      Min.   :-1.00000   1st Qu.: 0.00000  
    ## abs_title_subjectivity        Min.   :0.0000     1st Qu.:0.1667    
    ## abs_title_sentiment_polarity  Min.   :0.0000     1st Qu.:0.0000    
    ##     shares                    Min.   :    42     1st Qu.:   899    
    ##                                                                    
    ##   timedelta                   Median :338.0      Mean   :353.7     
    ## n_tokens_title                Median :10.00      Mean   :10.45     
    ## n_tokens_content              Median : 401       Mean   : 543      
    ## n_unique_tokens               Median :  0.5413   Mean   :  0.6677  
    ## n_non_stop_words              Median :   1.000   Mean   :   1.174  
    ## n_non_stop_unique_tokens      Median :  0.6905   Mean   :  0.8002  
    ##   num_hrefs                   Median :  7.00     Mean   : 10.68    
    ## num_self_hrefs                Median : 3.000     Mean   : 3.319    
    ##    num_imgs                   Median : 1.000     Mean   : 4.476    
    ##   num_videos                  Median : 0.000     Mean   : 1.385    
    ## average_token_length          Median :4.656      Mean   :4.551     
    ##  num_keywords                 Median : 7.000     Mean   : 7.196    
    ## data_channel_is_lifestyle     Median :0.00000    Mean   :0.04717   
    ## data_channel_is_entertainment Median :0.0000     Mean   :0.1755    
    ## data_channel_is_bus           Median :0.0000     Mean   :0.1608    
    ## data_channel_is_socmed        Median :0.00000    Mean   :0.06205   
    ## data_channel_is_tech          Median :0.000      Mean   :0.198     
    ## data_channel_is_world         Median :0.0000     Mean   :0.2051    
    ##   kw_min_min                  Median : -1.00     Mean   : 24.94    
    ##   kw_max_min                  Median :   664     Mean   :  1143    
    ##   kw_avg_min                  Median :  234.1    Mean   :  309.2   
    ##   kw_min_max                  Median :  1300     Mean   : 13309    
    ##   kw_max_max                  Median :843300     Mean   :756142    
    ##   kw_avg_max                  Median :242957     Mean   :261045    
    ##   kw_min_avg                  Median : 991.8     Mean   :1102.5    
    ##   kw_max_avg                  Median :  4300     Mean   :  5605    
    ##   kw_avg_avg                  Median : 2840.3    Mean   : 3124.4   
    ## self_reference_min_shares     Median :  1200     Mean   :  4274    
    ## self_reference_max_shares     Median :  3000     Mean   : 10721    
    ## self_reference_avg_sharess    Median :  2311     Mean   :  6748    
    ##     LDA_00                    Median :0.03344    Mean   :0.18465   
    ##     LDA_01                    Median :0.03334    Mean   :0.13470   
    ##     LDA_02                    Median :0.04001    Mean   :0.21404   
    ##     LDA_03                    Median :0.04000    Mean   :0.22227   
    ##     LDA_04                    Median :0.05000    Mean   :0.24413   
    ## global_subjectivity           Median :0.4511     Mean   :0.4423    
    ## global_sentiment_polarity     Median : 0.1207    Mean   : 0.1207   
    ## global_rate_positive_words    Median :0.03930    Mean   :0.03986   
    ## global_rate_negative_words    Median :0.015164   Mean   :0.016405  
    ## rate_positive_words           Median :0.7143     Mean   :0.6898    
    ## rate_negative_words           Median :0.2727     Mean   :0.2831    
    ## avg_positive_polarity         Median :0.3559     Mean   :0.3520    
    ## min_positive_polarity         Median :0.10000    Mean   :0.09546   
    ## max_positive_polarity         Median :0.8000     Mean   :0.7554    
    ## avg_negative_polarity         Median :-0.2500    Mean   :-0.2565   
    ## min_negative_polarity         Median :-0.5000    Mean   :-0.5127   
    ## max_negative_polarity         Median :-0.1000    Mean   :-0.1083   
    ## title_subjectivity            Median :0.1000     Mean   :0.2792    
    ## title_sentiment_polarity      Median : 0.00000   Mean   : 0.07262  
    ## abs_title_subjectivity        Median :0.5000     Mean   :0.3462    
    ## abs_title_sentiment_polarity  Median :0.0000     Mean   :0.1551    
    ##     shares                    Median :  1300     Mean   :  3257    
    ##                                                                    
    ##   timedelta                   3rd Qu.:541.0      Max.   :730.0     
    ## n_tokens_title                3rd Qu.:12.00      Max.   :19.00     
    ## n_tokens_content              3rd Qu.: 691       Max.   :5530      
    ## n_unique_tokens               3rd Qu.:  0.6092   Max.   :701.0000  
    ## n_non_stop_words              3rd Qu.:   1.000   Max.   :1042.000  
    ## n_non_stop_unique_tokens      3rd Qu.:  0.7544   Max.   :650.0000  
    ##   num_hrefs                   3rd Qu.: 13.00     Max.   :304.00    
    ## num_self_hrefs                3rd Qu.: 4.000     Max.   :62.000    
    ##    num_imgs                   3rd Qu.: 4.000     Max.   :99.000    
    ##   num_videos                  3rd Qu.: 1.000     Max.   :73.000    
    ## average_token_length          3rd Qu.:4.848      Max.   :6.419     
    ##  num_keywords                 3rd Qu.: 9.000     Max.   :10.000    
    ## data_channel_is_lifestyle     3rd Qu.:0.00000    Max.   :1.00000   
    ## data_channel_is_entertainment 3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_bus           3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_socmed        3rd Qu.:0.00000    Max.   :1.00000   
    ## data_channel_is_tech          3rd Qu.:0.000      Max.   :1.000     
    ## data_channel_is_world         3rd Qu.:0.0000     Max.   :1.0000    
    ##   kw_min_min                  3rd Qu.:  4.00     Max.   :217.00    
    ##   kw_max_min                  3rd Qu.:  1000     Max.   :139600    
    ##   kw_avg_min                  3rd Qu.:  355.9    Max.   :15851.2   
    ##   kw_min_max                  3rd Qu.:  8300     Max.   :843300    
    ##   kw_max_max                  3rd Qu.:843300     Max.   :843300    
    ##   kw_avg_max                  3rd Qu.:333657     Max.   :843300    
    ##   kw_min_avg                  3rd Qu.:2051.2     Max.   :3609.7    
    ##   kw_max_avg                  3rd Qu.:  6002     Max.   :178675    
    ##   kw_avg_avg                  3rd Qu.: 3569.8    Max.   :29240.8   
    ## self_reference_min_shares     3rd Qu.:  2700     Max.   :690400    
    ## self_reference_max_shares     3rd Qu.:  8200     Max.   :843300    
    ## self_reference_avg_sharess    3rd Qu.:  5350     Max.   :690400    
    ##     LDA_00                    3rd Qu.:0.24470    Max.   :0.91998   
    ##     LDA_01                    3rd Qu.:0.13484    Max.   :0.91994   
    ##     LDA_02                    3rd Qu.:0.32257    Max.   :0.92000   
    ##     LDA_03                    3rd Qu.:0.36863    Max.   :0.91997   
    ##     LDA_04                    3rd Qu.:0.43001    Max.   :0.92719   
    ## global_subjectivity           3rd Qu.:0.5058     Max.   :1.0000    
    ## global_sentiment_polarity     3rd Qu.: 0.1779    Max.   : 0.6192   
    ## global_rate_positive_words    3rd Qu.:0.05012    Max.   :0.11458   
    ## global_rate_negative_words    3rd Qu.:0.021390   Max.   :0.135294  
    ## rate_positive_words           3rd Qu.:0.8000     Max.   :1.0000    
    ## rate_negative_words           3rd Qu.:0.3750     Max.   :1.0000    
    ## avg_positive_polarity         3rd Qu.:0.4088     Max.   :0.8333    
    ## min_positive_polarity         3rd Qu.:0.10000    Max.   :0.70000   
    ## max_positive_polarity         3rd Qu.:1.0000     Max.   :1.0000    
    ## avg_negative_polarity         3rd Qu.:-0.1833    Max.   : 0.0000   
    ## min_negative_polarity         3rd Qu.:-0.3000    Max.   : 0.0000   
    ## max_negative_polarity         3rd Qu.:-0.0500    Max.   : 0.0000   
    ## title_subjectivity            3rd Qu.:0.5000     Max.   :1.0000    
    ## title_sentiment_polarity      3rd Qu.: 0.13636   Max.   : 1.00000  
    ## abs_title_subjectivity        3rd Qu.:0.5000     Max.   :0.5000    
    ## abs_title_sentiment_polarity  3rd Qu.:0.2500     Max.   :1.0000    
    ##     shares                    3rd Qu.:  2500     Max.   :441000

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
    ##           Mean of squared residuals: 106993941
    ##                     % Var explained: -2.58

``` r
#variable importance measures
importance(rf)
```

    ##                                  %IncMSE IncNodePurity
    ## timedelta                      3.7389822    4789287647
    ## n_tokens_title                -1.5137120   21816840169
    ## n_tokens_content               4.9508830   11653619126
    ## n_unique_tokens                3.0753860    9928341101
    ## n_non_stop_words               3.7473138    9088520486
    ## n_non_stop_unique_tokens       2.7127122   14024118692
    ## num_hrefs                      2.8691146    9075035664
    ## num_self_hrefs                 3.1097270   17986556353
    ## num_imgs                       1.4318914    8814858271
    ## num_videos                     0.4476985    2830198018
    ## average_token_length           0.6354104   42534910770
    ## num_keywords                   0.2555097    2301634537
    ## data_channel_is_lifestyle      2.6287885    5014492226
    ## data_channel_is_entertainment  1.3419134     227269939
    ## data_channel_is_bus            3.8198534     523725839
    ## data_channel_is_socmed         2.6285330    1268763331
    ## data_channel_is_tech           4.1482500     356089232
    ## data_channel_is_world          4.7214719     660576003
    ## kw_min_min                     2.9938323     326785443
    ## kw_max_min                     3.0961780    4750688307
    ## kw_avg_min                     3.8129688    8665749176
    ## kw_min_max                     1.7908076    8290632179
    ## kw_max_max                     1.5155251     456750160
    ## kw_avg_max                     4.6349210   14990124279
    ## kw_min_avg                     1.3840786    6440557784
    ## kw_max_avg                     3.8034821   18585420305
    ## kw_avg_avg                     5.5612222   28373649176
    ## self_reference_min_shares      2.8161331   15784975672
    ## self_reference_max_shares      4.7861600    8592253864
    ## self_reference_avg_sharess     5.3888444   13492838698
    ## LDA_00                         1.8951386    9669004873
    ## LDA_01                         1.9866409   17179675007
    ## LDA_02                         3.9817433   12075845961
    ## LDA_03                         2.1149105    8426554266
    ## LDA_04                         4.7991625   13368459520
    ## global_subjectivity            4.7701153    8587491252
    ## global_sentiment_polarity      2.3914137    7165497754
    ## global_rate_positive_words     2.2591722   12960409746
    ## global_rate_negative_words     2.2555935    5937591596
    ## rate_positive_words            2.9025561    5257720121
    ## rate_negative_words            5.0093784    5191414035
    ## avg_positive_polarity          1.8545412    7718263440
    ## min_positive_polarity          0.3967559    3210975848
    ## max_positive_polarity          2.5885957    3524194574
    ## avg_negative_polarity          1.3802479   16723755261
    ## min_negative_polarity          3.4741826    7080441362
    ## max_negative_polarity          1.6766194    8650544671
    ## title_subjectivity             0.5977224    2852451953
    ## title_sentiment_polarity       3.5415564    3952074221
    ## abs_title_subjectivity         2.2439921    2746534864
    ## abs_title_sentiment_polarity   2.2099229    2326932477

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

    ## [1] 24589997

So, the predicted mean square error on the training dataset is 68724773.

### On test set

``` r
rf.test <- predict(rf, newdata = test[,-52])
mean((test$shares-rf.test)^2)
```

    ## [1] 78929010

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

    ## Start:  AIC=95428.41
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
    ## Step:  AIC=95428.41
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
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - LDA_04                         1     213380 5.2094e+11 95426
    ## - LDA_01                         1     214494 5.2094e+11 95426
    ## - LDA_00                         1     215230 5.2094e+11 95426
    ## - LDA_03                         1     215600 5.2094e+11 95426
    ## - LDA_02                         1     217512 5.2094e+11 95426
    ## - kw_avg_max                     1    2144231 5.2094e+11 95426
    ## - average_token_length           1    2211766 5.2094e+11 95426
    ## - num_self_hrefs                 1    3667890 5.2094e+11 95426
    ## - kw_max_max                     1    3818375 5.2094e+11 95426
    ## - data_channel_is_tech           1    5133209 5.2094e+11 95426
    ## - self_reference_min_shares      1    7411073 5.2094e+11 95426
    ## - global_rate_positive_words     1    7848675 5.2094e+11 95426
    ## - n_tokens_content               1   12582621 5.2095e+11 95427
    ## - min_negative_polarity          1   12862651 5.2095e+11 95427
    ## - kw_min_min                     1   12865444 5.2095e+11 95427
    ## - avg_negative_polarity          1   17388867 5.2095e+11 95427
    ## - max_negative_polarity          1   24791971 5.2096e+11 95427
    ## - rate_positive_words            1   27849052 5.2096e+11 95427
    ## - n_non_stop_words               1   31468202 5.2097e+11 95427
    ## - n_non_stop_unique_tokens       1   31782576 5.2097e+11 95427
    ## - kw_max_min                     1   34659866 5.2097e+11 95427
    ## - avg_positive_polarity          1   52019533 5.2099e+11 95427
    ## - data_channel_is_lifestyle      1   54261270 5.2099e+11 95427
    ## - title_subjectivity             1   56083834 5.2099e+11 95427
    ## - abs_title_sentiment_polarity   1   61652067 5.2100e+11 95427
    ## - num_keywords                   1   67883205 5.2100e+11 95427
    ## - num_videos                     1   68539516 5.2100e+11 95427
    ## - self_reference_avg_sharess     1   71664448 5.2101e+11 95427
    ## - data_channel_is_bus            1   75959372 5.2101e+11 95427
    ## - data_channel_is_entertainment  1   76966623 5.2101e+11 95427
    ## - kw_avg_min                     1   79134274 5.2101e+11 95427
    ## - self_reference_max_shares      1   93760663 5.2103e+11 95427
    ## - num_hrefs                      1  108268430 5.2104e+11 95427
    ## - global_rate_negative_words     1  110604189 5.2105e+11 95428
    ## - n_unique_tokens                1  120483475 5.2106e+11 95428
    ## - abs_title_subjectivity         1  121232992 5.2106e+11 95428
    ## - title_sentiment_polarity       1  125330439 5.2106e+11 95428
    ## - min_positive_polarity          1  146498677 5.2108e+11 95428
    ## - global_sentiment_polarity      1  151665205 5.2109e+11 95428
    ## - timedelta                      1  162798850 5.2110e+11 95428
    ## - data_channel_is_socmed         1  185168968 5.2112e+11 95428
    ## - data_channel_is_world          1  189457638 5.2112e+11 95428
    ## <none>                                        5.2094e+11 95428
    ## - kw_min_max                     1  255402977 5.2119e+11 95429
    ## - global_subjectivity            1  263642614 5.2120e+11 95429
    ## - n_tokens_title                 1  475665491 5.2141e+11 95431
    ## - num_imgs                       1  527714191 5.2146e+11 95432
    ## - max_positive_polarity          1  617937252 5.2155e+11 95433
    ## - kw_min_avg                     1 1471419448 5.2241e+11 95441
    ## - kw_max_avg                     1 2091174508 5.2303e+11 95447
    ## - kw_avg_avg                     1 3740712839 5.2468e+11 95463
    ## 
    ## Step:  AIC=95426.41
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
    ## - kw_avg_max                     1    2124745 5.2094e+11 95424
    ## - num_self_hrefs                 1    3793389 5.2094e+11 95424
    ## - kw_max_max                     1    3835806 5.2094e+11 95424
    ## - data_channel_is_tech           1    5037578 5.2094e+11 95424
    ## - self_reference_min_shares      1    7509609 5.2094e+11 95424
    ## - global_rate_positive_words     1    7639736 5.2094e+11 95424
    ## - average_token_length           1    9138933 5.2094e+11 95424
    ## - n_tokens_content               1   12806428 5.2095e+11 95425
    ## - min_negative_polarity          1   12816024 5.2095e+11 95425
    ## - kw_min_min                     1   12925706 5.2095e+11 95425
    ## - LDA_01                         1   17107555 5.2095e+11 95425
    ## - avg_negative_polarity          1   17310056 5.2095e+11 95425
    ## - max_negative_polarity          1   24623708 5.2096e+11 95425
    ## - kw_max_min                     1   34823734 5.2097e+11 95425
    ## - n_non_stop_unique_tokens       1   37551615 5.2097e+11 95425
    ## - rate_positive_words            1   41682670 5.2098e+11 95425
    ## - avg_positive_polarity          1   53233132 5.2099e+11 95425
    ## - data_channel_is_lifestyle      1   54118604 5.2099e+11 95425
    ## - title_subjectivity             1   55971768 5.2099e+11 95425
    ## - LDA_00                         1   60689418 5.2100e+11 95425
    ## - abs_title_sentiment_polarity   1   61660801 5.2100e+11 95425
    ## - num_keywords                   1   68445172 5.2100e+11 95425
    ## - num_videos                     1   68626085 5.2100e+11 95425
    ## - self_reference_avg_sharess     1   71469775 5.2101e+11 95425
    ## - data_channel_is_bus            1   75765995 5.2101e+11 95425
    ## - LDA_03                         1   76471753 5.2101e+11 95425
    ## - data_channel_is_entertainment  1   77400957 5.2101e+11 95425
    ## - kw_avg_min                     1   79329328 5.2101e+11 95425
    ## - self_reference_max_shares      1   93562321 5.2103e+11 95425
    ## - num_hrefs                      1  110146467 5.2105e+11 95426
    ## - abs_title_subjectivity         1  121157597 5.2106e+11 95426
    ## - title_sentiment_polarity       1  125148685 5.2106e+11 95426
    ## - global_rate_negative_words     1  137186598 5.2107e+11 95426
    ## - n_unique_tokens                1  138397082 5.2107e+11 95426
    ## - min_positive_polarity          1  147644011 5.2108e+11 95426
    ## - global_sentiment_polarity      1  153751272 5.2109e+11 95426
    ## - timedelta                      1  162645737 5.2110e+11 95426
    ## - n_non_stop_words               1  174123545 5.2111e+11 95426
    ## - data_channel_is_socmed         1  185011147 5.2112e+11 95426
    ## - data_channel_is_world          1  189486756 5.2112e+11 95426
    ## <none>                                        5.2094e+11 95426
    ## - kw_min_max                     1  255198442 5.2119e+11 95427
    ## - global_subjectivity            1  267059663 5.2120e+11 95427
    ## - LDA_02                         1  299991928 5.2124e+11 95427
    ## - n_tokens_title                 1  477323898 5.2141e+11 95429
    ## - num_imgs                       1  527829492 5.2146e+11 95430
    ## - max_positive_polarity          1  617958375 5.2155e+11 95431
    ## - kw_min_avg                     1 1474428233 5.2241e+11 95439
    ## - kw_max_avg                     1 2101121789 5.2304e+11 95445
    ## - kw_avg_avg                     1 3756016815 5.2469e+11 95462
    ## 
    ## Step:  AIC=95424.43
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_self_hrefs                 1    3686748 5.2094e+11 95422
    ## - data_channel_is_tech           1    4640992 5.2094e+11 95422
    ## - kw_max_max                     1    6273559 5.2094e+11 95422
    ## - self_reference_min_shares      1    7231763 5.2094e+11 95423
    ## - global_rate_positive_words     1    7997150 5.2095e+11 95423
    ## - average_token_length           1    9656650 5.2095e+11 95423
    ## - min_negative_polarity          1   12936032 5.2095e+11 95423
    ## - n_tokens_content               1   12947354 5.2095e+11 95423
    ## - kw_min_min                     1   12968971 5.2095e+11 95423
    ## - LDA_01                         1   17255502 5.2095e+11 95423
    ## - avg_negative_polarity          1   17541306 5.2096e+11 95423
    ## - max_negative_polarity          1   24871825 5.2096e+11 95423
    ## - n_non_stop_unique_tokens       1   37750126 5.2098e+11 95423
    ## - kw_max_min                     1   40348408 5.2098e+11 95423
    ## - rate_positive_words            1   40868816 5.2098e+11 95423
    ## - data_channel_is_lifestyle      1   52241669 5.2099e+11 95423
    ## - avg_positive_polarity          1   53147202 5.2099e+11 95423
    ## - title_subjectivity             1   55646857 5.2099e+11 95423
    ## - LDA_00                         1   59972395 5.2100e+11 95423
    ## - abs_title_sentiment_polarity   1   61208408 5.2100e+11 95423
    ## - num_keywords                   1   68611065 5.2101e+11 95423
    ## - self_reference_avg_sharess     1   72236588 5.2101e+11 95423
    ## - num_videos                     1   72276285 5.2101e+11 95423
    ## - LDA_03                         1   74753710 5.2101e+11 95423
    ## - data_channel_is_bus            1   77660505 5.2102e+11 95423
    ## - data_channel_is_entertainment  1   86107738 5.2102e+11 95423
    ## - kw_avg_min                     1   90919645 5.2103e+11 95423
    ## - self_reference_max_shares      1   93814359 5.2103e+11 95423
    ## - num_hrefs                      1  108790928 5.2105e+11 95424
    ## - abs_title_subjectivity         1  121823701 5.2106e+11 95424
    ## - title_sentiment_polarity       1  126460451 5.2106e+11 95424
    ## - global_rate_negative_words     1  136307953 5.2107e+11 95424
    ## - n_unique_tokens                1  140199013 5.2108e+11 95424
    ## - min_positive_polarity          1  146566985 5.2108e+11 95424
    ## - global_sentiment_polarity      1  154597258 5.2109e+11 95424
    ## - timedelta                      1  161018604 5.2110e+11 95424
    ## - n_non_stop_words               1  177523643 5.2112e+11 95424
    ## - data_channel_is_socmed         1  183476571 5.2112e+11 95424
    ## - data_channel_is_world          1  187374786 5.2112e+11 95424
    ## <none>                                        5.2094e+11 95424
    ## - global_subjectivity            1  266812905 5.2120e+11 95425
    ## - kw_min_max                     1  269684805 5.2121e+11 95425
    ## - LDA_02                         1  297909641 5.2124e+11 95425
    ## - n_tokens_title                 1  481697120 5.2142e+11 95427
    ## - num_imgs                       1  526006009 5.2146e+11 95428
    ## - max_positive_polarity          1  617925353 5.2156e+11 95429
    ## - kw_min_avg                     1 1487711445 5.2243e+11 95437
    ## - kw_max_avg                     1 2275828527 5.2321e+11 95445
    ## - kw_avg_avg                     1 4241752796 5.2518e+11 95464
    ## 
    ## Step:  AIC=95422.47
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - data_channel_is_tech           1    4267343 5.2095e+11 95421
    ## - self_reference_min_shares      1    5740497 5.2095e+11 95421
    ## - kw_max_max                     1    6088446 5.2095e+11 95421
    ## - global_rate_positive_words     1    7981439 5.2095e+11 95421
    ## - average_token_length           1    9469216 5.2095e+11 95421
    ## - n_tokens_content               1   11478508 5.2095e+11 95421
    ## - kw_min_min                     1   13216618 5.2095e+11 95421
    ## - min_negative_polarity          1   13331496 5.2095e+11 95421
    ## - LDA_01                         1   16659583 5.2096e+11 95421
    ## - avg_negative_polarity          1   17562759 5.2096e+11 95421
    ## - max_negative_polarity          1   24692594 5.2097e+11 95421
    ## - n_non_stop_unique_tokens       1   37702476 5.2098e+11 95421
    ## - kw_max_min                     1   40059569 5.2098e+11 95421
    ## - rate_positive_words            1   42318711 5.2098e+11 95421
    ## - avg_positive_polarity          1   53044103 5.2099e+11 95421
    ## - data_channel_is_lifestyle      1   53307901 5.2099e+11 95421
    ## - title_subjectivity             1   55342312 5.2100e+11 95421
    ## - LDA_00                         1   59009735 5.2100e+11 95421
    ## - abs_title_sentiment_polarity   1   61097052 5.2100e+11 95421
    ## - num_keywords                   1   67700243 5.2101e+11 95421
    ## - num_videos                     1   71840469 5.2101e+11 95421
    ## - LDA_03                         1   74146745 5.2102e+11 95421
    ## - data_channel_is_bus            1   77523267 5.2102e+11 95421
    ## - self_reference_avg_sharess     1   83059793 5.2102e+11 95421
    ## - data_channel_is_entertainment  1   87522748 5.2103e+11 95421
    ## - kw_avg_min                     1   90661316 5.2103e+11 95421
    ## - num_hrefs                      1  105693776 5.2105e+11 95422
    ## - self_reference_max_shares      1  109554352 5.2105e+11 95422
    ## - abs_title_subjectivity         1  121241201 5.2106e+11 95422
    ## - title_sentiment_polarity       1  125729468 5.2107e+11 95422
    ## - global_rate_negative_words     1  137961059 5.2108e+11 95422
    ## - n_unique_tokens                1  139580756 5.2108e+11 95422
    ## - min_positive_polarity          1  147190533 5.2109e+11 95422
    ## - global_sentiment_polarity      1  153798671 5.2109e+11 95422
    ## - timedelta                      1  159155377 5.2110e+11 95422
    ## - n_non_stop_words               1  176416263 5.2112e+11 95422
    ## - data_channel_is_socmed         1  181544194 5.2112e+11 95422
    ## - data_channel_is_world          1  189028813 5.2113e+11 95422
    ## <none>                                        5.2094e+11 95422
    ## - global_subjectivity            1  266973409 5.2121e+11 95423
    ## - kw_min_max                     1  269437897 5.2121e+11 95423
    ## - LDA_02                         1  296354615 5.2124e+11 95423
    ## - n_tokens_title                 1  479623297 5.2142e+11 95425
    ## - num_imgs                       1  522544428 5.2146e+11 95426
    ## - max_positive_polarity          1  618670875 5.2156e+11 95427
    ## - kw_min_avg                     1 1500289202 5.2244e+11 95435
    ## - kw_max_avg                     1 2277695140 5.2322e+11 95443
    ## - kw_avg_avg                     1 4259680069 5.2520e+11 95463
    ## 
    ## Step:  AIC=95420.51
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_max_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - self_reference_min_shares      1    5655093 5.2095e+11 95419
    ## - kw_max_max                     1    6201546 5.2095e+11 95419
    ## - global_rate_positive_words     1    7769151 5.2095e+11 95419
    ## - average_token_length           1    9557346 5.2096e+11 95419
    ## - n_tokens_content               1   11088684 5.2096e+11 95419
    ## - kw_min_min                     1   12618204 5.2096e+11 95419
    ## - min_negative_polarity          1   13101874 5.2096e+11 95419
    ## - avg_negative_polarity          1   17797609 5.2096e+11 95419
    ## - max_negative_polarity          1   24894940 5.2097e+11 95419
    ## - n_non_stop_unique_tokens       1   35621874 5.2098e+11 95419
    ## - kw_max_min                     1   39004079 5.2098e+11 95419
    ## - LDA_01                         1   41868641 5.2099e+11 95419
    ## - rate_positive_words            1   42219495 5.2099e+11 95419
    ## - avg_positive_polarity          1   52442956 5.2100e+11 95419
    ## - title_subjectivity             1   55721885 5.2100e+11 95419
    ## - abs_title_sentiment_polarity   1   60739711 5.2101e+11 95419
    ## - data_channel_is_lifestyle      1   65622369 5.2101e+11 95419
    ## - num_keywords                   1   67734204 5.2101e+11 95419
    ## - num_videos                     1   72054341 5.2102e+11 95419
    ## - LDA_00                         1   77423277 5.2102e+11 95419
    ## - self_reference_avg_sharess     1   83091933 5.2103e+11 95419
    ## - kw_avg_min                     1   88917155 5.2103e+11 95419
    ## - data_channel_is_bus            1   93473697 5.2104e+11 95419
    ## - num_hrefs                      1  104536085 5.2105e+11 95420
    ## - self_reference_max_shares      1  109131175 5.2105e+11 95420
    ## - abs_title_subjectivity         1  120826636 5.2107e+11 95420
    ## - title_sentiment_polarity       1  125917190 5.2107e+11 95420
    ## - data_channel_is_entertainment  1  127804832 5.2107e+11 95420
    ## - n_unique_tokens                1  135808366 5.2108e+11 95420
    ## - global_rate_negative_words     1  138694218 5.2108e+11 95420
    ## - min_positive_polarity          1  146249663 5.2109e+11 95420
    ## - global_sentiment_polarity      1  153620708 5.2110e+11 95420
    ## - timedelta                      1  157365787 5.2110e+11 95420
    ## - n_non_stop_words               1  173237028 5.2112e+11 95420
    ## <none>                                        5.2095e+11 95421
    ## - LDA_03                         1  202034061 5.2115e+11 95421
    ## - data_channel_is_socmed         1  236186312 5.2118e+11 95421
    ## - global_subjectivity            1  265705133 5.2121e+11 95421
    ## - kw_min_max                     1  267895622 5.2121e+11 95421
    ## - data_channel_is_world          1  268851798 5.2121e+11 95421
    ## - LDA_02                         1  357538272 5.2130e+11 95422
    ## - n_tokens_title                 1  477659151 5.2142e+11 95423
    ## - num_imgs                       1  521666406 5.2147e+11 95424
    ## - max_positive_polarity          1  615549270 5.2156e+11 95425
    ## - kw_min_avg                     1 1496021972 5.2244e+11 95433
    ## - kw_max_avg                     1 2295977203 5.2324e+11 95441
    ## - kw_avg_avg                     1 4364265628 5.2531e+11 95462
    ## 
    ## Step:  AIC=95418.56
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_max_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_max_max                     1    5965336 5.2096e+11 95417
    ## - global_rate_positive_words     1    7588495 5.2096e+11 95417
    ## - average_token_length           1    9883569 5.2096e+11 95417
    ## - n_tokens_content               1   11626620 5.2096e+11 95417
    ## - kw_min_min                     1   12306731 5.2096e+11 95417
    ## - min_negative_polarity          1   12985429 5.2096e+11 95417
    ## - avg_negative_polarity          1   17763272 5.2097e+11 95417
    ## - max_negative_polarity          1   25116200 5.2098e+11 95417
    ## - n_non_stop_unique_tokens       1   34509393 5.2099e+11 95417
    ## - kw_max_min                     1   39082418 5.2099e+11 95417
    ## - rate_positive_words            1   42367346 5.2099e+11 95417
    ## - LDA_01                         1   42494067 5.2099e+11 95417
    ## - avg_positive_polarity          1   52311653 5.2100e+11 95417
    ## - title_subjectivity             1   56878753 5.2101e+11 95417
    ## - abs_title_sentiment_polarity   1   61027295 5.2101e+11 95417
    ## - data_channel_is_lifestyle      1   65485535 5.2102e+11 95417
    ## - num_keywords                   1   67461068 5.2102e+11 95417
    ## - num_videos                     1   71465506 5.2102e+11 95417
    ## - LDA_00                         1   78154577 5.2103e+11 95417
    ## - kw_avg_min                     1   88489863 5.2104e+11 95417
    ## - data_channel_is_bus            1   94052897 5.2105e+11 95417
    ## - num_hrefs                      1  110202505 5.2106e+11 95418
    ## - abs_title_subjectivity         1  120399100 5.2107e+11 95418
    ## - title_sentiment_polarity       1  125367022 5.2108e+11 95418
    ## - data_channel_is_entertainment  1  127327171 5.2108e+11 95418
    ## - n_unique_tokens                1  134433201 5.2109e+11 95418
    ## - global_rate_negative_words     1  138626226 5.2109e+11 95418
    ## - min_positive_polarity          1  146243933 5.2110e+11 95418
    ## - global_sentiment_polarity      1  153267131 5.2110e+11 95418
    ## - timedelta                      1  156419977 5.2111e+11 95418
    ## - n_non_stop_words               1  173901036 5.2113e+11 95418
    ## <none>                                        5.2095e+11 95419
    ## - LDA_03                         1  203245715 5.2115e+11 95419
    ## - data_channel_is_socmed         1  237026019 5.2119e+11 95419
    ## - global_subjectivity            1  265548748 5.2122e+11 95419
    ## - kw_min_max                     1  265596914 5.2122e+11 95419
    ## - data_channel_is_world          1  267146119 5.2122e+11 95419
    ## - LDA_02                         1  356077550 5.2131e+11 95420
    ## - n_tokens_title                 1  475394310 5.2143e+11 95421
    ## - num_imgs                       1  524715572 5.2148e+11 95422
    ## - self_reference_max_shares      1  599399147 5.2155e+11 95423
    ## - max_positive_polarity          1  612981639 5.2156e+11 95423
    ## - self_reference_avg_sharess     1 1308398765 5.2226e+11 95430
    ## - kw_min_avg                     1 1495874367 5.2245e+11 95431
    ## - kw_max_avg                     1 2316407769 5.2327e+11 95440
    ## - kw_avg_avg                     1 4367726370 5.2532e+11 95460
    ## 
    ## Step:  AIC=95416.62
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_min_min                     1    6540780 5.2096e+11 95415
    ## - global_rate_positive_words     1    7794274 5.2096e+11 95415
    ## - average_token_length           1    9978388 5.2097e+11 95415
    ## - n_tokens_content               1   11394900 5.2097e+11 95415
    ## - min_negative_polarity          1   13254713 5.2097e+11 95415
    ## - avg_negative_polarity          1   18466345 5.2098e+11 95415
    ## - max_negative_polarity          1   25779940 5.2098e+11 95415
    ## - n_non_stop_unique_tokens       1   33968042 5.2099e+11 95415
    ## - rate_positive_words            1   41628534 5.2100e+11 95415
    ## - kw_max_min                     1   42458199 5.2100e+11 95415
    ## - LDA_01                         1   43375672 5.2100e+11 95415
    ## - avg_positive_polarity          1   52334566 5.2101e+11 95415
    ## - title_subjectivity             1   56654124 5.2101e+11 95415
    ## - abs_title_sentiment_polarity   1   60489858 5.2102e+11 95415
    ## - data_channel_is_lifestyle      1   64816513 5.2102e+11 95415
    ## - num_keywords                   1   70513907 5.2103e+11 95415
    ## - num_videos                     1   71893215 5.2103e+11 95415
    ## - LDA_00                         1   78780610 5.2104e+11 95415
    ## - kw_avg_min                     1   94592962 5.2105e+11 95416
    ## - data_channel_is_bus            1   95055467 5.2105e+11 95416
    ## - num_hrefs                      1  110800360 5.2107e+11 95416
    ## - abs_title_subjectivity         1  121587869 5.2108e+11 95416
    ## - title_sentiment_polarity       1  125066818 5.2108e+11 95416
    ## - data_channel_is_entertainment  1  125438182 5.2108e+11 95416
    ## - n_unique_tokens                1  133117517 5.2109e+11 95416
    ## - global_rate_negative_words     1  137418079 5.2109e+11 95416
    ## - min_positive_polarity          1  146296728 5.2110e+11 95416
    ## - timedelta                      1  151043965 5.2111e+11 95416
    ## - global_sentiment_polarity      1  153249884 5.2111e+11 95416
    ## - n_non_stop_words               1  172781353 5.2113e+11 95416
    ## <none>                                        5.2096e+11 95417
    ## - LDA_03                         1  209895353 5.2117e+11 95417
    ## - data_channel_is_socmed         1  238818767 5.2120e+11 95417
    ## - global_subjectivity            1  265833072 5.2122e+11 95417
    ## - kw_min_max                     1  266650382 5.2122e+11 95417
    ## - data_channel_is_world          1  270467981 5.2123e+11 95417
    ## - LDA_02                         1  357802152 5.2131e+11 95418
    ## - n_tokens_title                 1  475495305 5.2143e+11 95419
    ## - num_imgs                       1  529218728 5.2149e+11 95420
    ## - self_reference_max_shares      1  601926402 5.2156e+11 95421
    ## - max_positive_polarity          1  614064488 5.2157e+11 95421
    ## - self_reference_avg_sharess     1 1311531164 5.2227e+11 95428
    ## - kw_min_avg                     1 1505745961 5.2246e+11 95430
    ## - kw_max_avg                     1 2363760080 5.2332e+11 95438
    ## - kw_avg_avg                     1 4491350463 5.2545e+11 95459
    ## 
    ## Step:  AIC=95414.69
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - global_rate_positive_words     1    7798346 5.2097e+11 95413
    ## - average_token_length           1   10315325 5.2097e+11 95413
    ## - n_tokens_content               1   11428288 5.2098e+11 95413
    ## - min_negative_polarity          1   12980234 5.2098e+11 95413
    ## - avg_negative_polarity          1   18308417 5.2098e+11 95413
    ## - max_negative_polarity          1   26098333 5.2099e+11 95413
    ## - n_non_stop_unique_tokens       1   33581832 5.2100e+11 95413
    ## - kw_max_min                     1   37549307 5.2100e+11 95413
    ## - rate_positive_words            1   42323256 5.2101e+11 95413
    ## - LDA_01                         1   42500162 5.2101e+11 95413
    ## - avg_positive_polarity          1   52344178 5.2102e+11 95413
    ## - title_subjectivity             1   56567256 5.2102e+11 95413
    ## - abs_title_sentiment_polarity   1   60986513 5.2102e+11 95413
    ## - data_channel_is_lifestyle      1   65088754 5.2103e+11 95413
    ## - num_keywords                   1   68341424 5.2103e+11 95413
    ## - num_videos                     1   73504703 5.2104e+11 95413
    ## - LDA_00                         1   78356042 5.2104e+11 95413
    ## - kw_avg_min                     1   88281227 5.2105e+11 95414
    ## - data_channel_is_bus            1   93845916 5.2106e+11 95414
    ## - num_hrefs                      1  109550165 5.2107e+11 95414
    ## - abs_title_subjectivity         1  121967244 5.2109e+11 95414
    ## - title_sentiment_polarity       1  126265089 5.2109e+11 95414
    ## - data_channel_is_entertainment  1  129637701 5.2109e+11 95414
    ## - n_unique_tokens                1  132521472 5.2110e+11 95414
    ## - global_rate_negative_words     1  137747457 5.2110e+11 95414
    ## - min_positive_polarity          1  145464311 5.2111e+11 95414
    ## - global_sentiment_polarity      1  152497574 5.2112e+11 95414
    ## - n_non_stop_words               1  172753363 5.2114e+11 95414
    ## <none>                                        5.2096e+11 95415
    ## - LDA_03                         1  204796147 5.2117e+11 95415
    ## - timedelta                      1  223367407 5.2119e+11 95415
    ## - data_channel_is_socmed         1  236449352 5.2120e+11 95415
    ## - kw_min_max                     1  265974864 5.2123e+11 95415
    ## - global_subjectivity            1  266126654 5.2123e+11 95415
    ## - data_channel_is_world          1  270945563 5.2123e+11 95415
    ## - LDA_02                         1  359683459 5.2132e+11 95416
    ## - n_tokens_title                 1  477966975 5.2144e+11 95417
    ## - num_imgs                       1  529815171 5.2149e+11 95418
    ## - self_reference_max_shares      1  601914746 5.2157e+11 95419
    ## - max_positive_polarity          1  612092042 5.2158e+11 95419
    ## - self_reference_avg_sharess     1 1311646067 5.2228e+11 95426
    ## - kw_min_avg                     1 1499289606 5.2246e+11 95428
    ## - kw_max_avg                     1 2373618771 5.2334e+11 95436
    ## - kw_avg_avg                     1 4579509762 5.2554e+11 95458
    ## 
    ## Step:  AIC=95412.77
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - average_token_length           1    6487521 5.2098e+11 95411
    ## - n_tokens_content               1   10062564 5.2098e+11 95411
    ## - min_negative_polarity          1   12300457 5.2098e+11 95411
    ## - avg_negative_polarity          1   22978022 5.2099e+11 95411
    ## - max_negative_polarity          1   29296515 5.2100e+11 95411
    ## - n_non_stop_unique_tokens       1   31455023 5.2100e+11 95411
    ## - kw_max_min                     1   37762704 5.2101e+11 95411
    ## - LDA_01                         1   42100158 5.2101e+11 95411
    ## - avg_positive_polarity          1   44797048 5.2102e+11 95411
    ## - title_subjectivity             1   58129828 5.2103e+11 95411
    ## - abs_title_sentiment_polarity   1   60860470 5.2103e+11 95411
    ## - data_channel_is_lifestyle      1   65482756 5.2104e+11 95411
    ## - num_keywords                   1   67775162 5.2104e+11 95411
    ## - num_videos                     1   75722181 5.2105e+11 95412
    ## - LDA_00                         1   80815521 5.2105e+11 95412
    ## - kw_avg_min                     1   88446113 5.2106e+11 95412
    ## - rate_positive_words            1   89441392 5.2106e+11 95412
    ## - data_channel_is_bus            1   94558980 5.2107e+11 95412
    ## - num_hrefs                      1  111446898 5.2108e+11 95412
    ## - title_sentiment_polarity       1  125495358 5.2110e+11 95412
    ## - n_unique_tokens                1  126428182 5.2110e+11 95412
    ## - abs_title_subjectivity         1  127478571 5.2110e+11 95412
    ## - data_channel_is_entertainment  1  128609557 5.2110e+11 95412
    ## - min_positive_polarity          1  151391182 5.2112e+11 95412
    ## - n_non_stop_words               1  165904479 5.2114e+11 95412
    ## <none>                                        5.2097e+11 95413
    ## - LDA_03                         1  203902259 5.2118e+11 95413
    ## - timedelta                      1  220507300 5.2119e+11 95413
    ## - data_channel_is_socmed         1  234239384 5.2121e+11 95413
    ## - global_sentiment_polarity      1  240204545 5.2121e+11 95413
    ## - kw_min_max                     1  265108252 5.2124e+11 95413
    ## - global_subjectivity            1  268739870 5.2124e+11 95413
    ## - data_channel_is_world          1  271882119 5.2124e+11 95413
    ## - global_rate_negative_words     1  348523660 5.2132e+11 95414
    ## - LDA_02                         1  361143699 5.2133e+11 95414
    ## - n_tokens_title                 1  482186835 5.2145e+11 95416
    ## - num_imgs                       1  536638682 5.2151e+11 95416
    ## - max_positive_polarity          1  604364573 5.2158e+11 95417
    ## - self_reference_max_shares      1  606648284 5.2158e+11 95417
    ## - self_reference_avg_sharess     1 1321471876 5.2229e+11 95424
    ## - kw_min_avg                     1 1500355827 5.2247e+11 95426
    ## - kw_max_avg                     1 2370330984 5.2334e+11 95434
    ## - kw_avg_avg                     1 4576238959 5.2555e+11 95456
    ## 
    ## Step:  AIC=95410.83
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_world + kw_max_min + kw_avg_min + kw_min_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - n_tokens_content               1   11775057 5.2099e+11 95409
    ## - min_negative_polarity          1   11791696 5.2099e+11 95409
    ## - avg_negative_polarity          1   20962859 5.2100e+11 95409
    ## - max_negative_polarity          1   27200234 5.2101e+11 95409
    ## - n_non_stop_unique_tokens       1   31877099 5.2101e+11 95409
    ## - kw_max_min                     1   37262152 5.2102e+11 95409
    ## - LDA_01                         1   40928252 5.2102e+11 95409
    ## - title_subjectivity             1   57077542 5.2103e+11 95409
    ## - avg_positive_polarity          1   60431304 5.2104e+11 95409
    ## - abs_title_sentiment_polarity   1   61347841 5.2104e+11 95409
    ## - data_channel_is_lifestyle      1   64957692 5.2104e+11 95409
    ## - num_keywords                   1   69937755 5.2105e+11 95410
    ## - num_videos                     1   78791981 5.2106e+11 95410
    ## - LDA_00                         1   80689330 5.2106e+11 95410
    ## - kw_avg_min                     1   88000990 5.2107e+11 95410
    ## - data_channel_is_bus            1   92351645 5.2107e+11 95410
    ## - num_hrefs                      1  104963846 5.2108e+11 95410
    ## - title_sentiment_polarity       1  124966833 5.2110e+11 95410
    ## - abs_title_subjectivity         1  125503512 5.2110e+11 95410
    ## - n_unique_tokens                1  128360080 5.2111e+11 95410
    ## - data_channel_is_entertainment  1  130291432 5.2111e+11 95410
    ## - min_positive_polarity          1  148180060 5.2113e+11 95410
    ## - n_non_stop_words               1  168693607 5.2115e+11 95411
    ## <none>                                        5.2098e+11 95411
    ## - LDA_03                         1  202005845 5.2118e+11 95411
    ## - timedelta                      1  220936766 5.2120e+11 95411
    ## - rate_positive_words            1  227804280 5.2121e+11 95411
    ## - data_channel_is_socmed         1  232906878 5.2121e+11 95411
    ## - global_sentiment_polarity      1  256963036 5.2123e+11 95411
    ## - global_subjectivity            1  262308041 5.2124e+11 95411
    ## - kw_min_max                     1  265242375 5.2124e+11 95411
    ## - data_channel_is_world          1  266315449 5.2124e+11 95411
    ## - LDA_02                         1  370313595 5.2135e+11 95413
    ## - global_rate_negative_words     1  436891236 5.2141e+11 95413
    ## - n_tokens_title                 1  490070336 5.2147e+11 95414
    ## - num_imgs                       1  532545004 5.2151e+11 95414
    ## - self_reference_max_shares      1  605856794 5.2158e+11 95415
    ## - max_positive_polarity          1  608817992 5.2159e+11 95415
    ## - self_reference_avg_sharess     1 1318824985 5.2230e+11 95422
    ## - kw_min_avg                     1 1498546665 5.2248e+11 95424
    ## - kw_max_avg                     1 2376812734 5.2335e+11 95432
    ## - kw_avg_avg                     1 4592164744 5.2557e+11 95454
    ## 
    ## Step:  AIC=95408.95
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - min_negative_polarity          1   18916500 5.2101e+11 95407
    ## - n_non_stop_unique_tokens       1   24915235 5.2101e+11 95407
    ## - avg_negative_polarity          1   25822235 5.2102e+11 95407
    ## - max_negative_polarity          1   28640535 5.2102e+11 95407
    ## - kw_max_min                     1   38194679 5.2103e+11 95407
    ## - LDA_01                         1   40525602 5.2103e+11 95407
    ## - title_subjectivity             1   56500080 5.2105e+11 95408
    ## - abs_title_sentiment_polarity   1   60926873 5.2105e+11 95408
    ## - avg_positive_polarity          1   63556841 5.2105e+11 95408
    ## - data_channel_is_lifestyle      1   63756527 5.2105e+11 95408
    ## - num_keywords                   1   72666838 5.2106e+11 95408
    ## - LDA_00                         1   82066917 5.2107e+11 95408
    ## - kw_avg_min                     1   89830023 5.2108e+11 95408
    ## - num_videos                     1   90088540 5.2108e+11 95408
    ## - data_channel_is_bus            1   94501788 5.2108e+11 95408
    ## - n_unique_tokens                1  122316229 5.2111e+11 95408
    ## - title_sentiment_polarity       1  123912374 5.2111e+11 95408
    ## - abs_title_subjectivity         1  126027676 5.2112e+11 95408
    ## - data_channel_is_entertainment  1  126845464 5.2112e+11 95408
    ## - num_hrefs                      1  128540941 5.2112e+11 95408
    ## - min_positive_polarity          1  147018501 5.2114e+11 95408
    ## - n_non_stop_words               1  181971612 5.2117e+11 95409
    ## <none>                                        5.2099e+11 95409
    ## - LDA_03                         1  202857070 5.2119e+11 95409
    ## - timedelta                      1  223572281 5.2121e+11 95409
    ## - rate_positive_words            1  232992632 5.2122e+11 95409
    ## - data_channel_is_socmed         1  239253039 5.2123e+11 95409
    ## - global_sentiment_polarity      1  251062706 5.2124e+11 95409
    ## - global_subjectivity            1  258197120 5.2125e+11 95410
    ## - kw_min_max                     1  262735242 5.2125e+11 95410
    ## - data_channel_is_world          1  266776024 5.2126e+11 95410
    ## - LDA_02                         1  369686309 5.2136e+11 95411
    ## - global_rate_negative_words     1  441212684 5.2143e+11 95411
    ## - n_tokens_title                 1  487562565 5.2148e+11 95412
    ## - num_imgs                       1  585264548 5.2157e+11 95413
    ## - self_reference_max_shares      1  601212322 5.2159e+11 95413
    ## - max_positive_polarity          1  621914093 5.2161e+11 95413
    ## - self_reference_avg_sharess     1 1312560734 5.2230e+11 95420
    ## - kw_min_avg                     1 1493656784 5.2248e+11 95422
    ## - kw_max_avg                     1 2374729364 5.2336e+11 95430
    ## - kw_avg_avg                     1 4586817250 5.2558e+11 95452
    ## 
    ## Step:  AIC=95407.14
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - avg_negative_polarity          1    7512559 5.2102e+11 95405
    ## - max_negative_polarity          1   13667942 5.2102e+11 95405
    ## - n_non_stop_unique_tokens       1   18537966 5.2103e+11 95405
    ## - kw_max_min                     1   38094005 5.2105e+11 95406
    ## - LDA_01                         1   42207928 5.2105e+11 95406
    ## - title_subjectivity             1   58271216 5.2107e+11 95406
    ## - abs_title_sentiment_polarity   1   61474358 5.2107e+11 95406
    ## - data_channel_is_lifestyle      1   65953137 5.2107e+11 95406
    ## - avg_positive_polarity          1   66091160 5.2107e+11 95406
    ## - num_keywords                   1   73442144 5.2108e+11 95406
    ## - LDA_00                         1   80465033 5.2109e+11 95406
    ## - kw_avg_min                     1   89140414 5.2110e+11 95406
    ## - num_videos                     1   91415968 5.2110e+11 95406
    ## - data_channel_is_bus            1   94075892 5.2110e+11 95406
    ## - n_unique_tokens                1  105944189 5.2111e+11 95406
    ## - abs_title_subjectivity         1  124936506 5.2113e+11 95406
    ## - title_sentiment_polarity       1  126946243 5.2114e+11 95406
    ## - data_channel_is_entertainment  1  128188101 5.2114e+11 95406
    ## - num_hrefs                      1  139410813 5.2115e+11 95407
    ## - min_positive_polarity          1  144073095 5.2115e+11 95407
    ## - n_non_stop_words               1  165476912 5.2117e+11 95407
    ## <none>                                        5.2101e+11 95407
    ## - LDA_03                         1  203174916 5.2121e+11 95407
    ## - timedelta                      1  221077449 5.2123e+11 95407
    ## - data_channel_is_socmed         1  238112240 5.2125e+11 95407
    ## - global_sentiment_polarity      1  244669553 5.2125e+11 95408
    ## - rate_positive_words            1  253330874 5.2126e+11 95408
    ## - global_subjectivity            1  258841856 5.2127e+11 95408
    ## - kw_min_max                     1  261723726 5.2127e+11 95408
    ## - data_channel_is_world          1  270028419 5.2128e+11 95408
    ## - LDA_02                         1  365244244 5.2137e+11 95409
    ## - global_rate_negative_words     1  426117312 5.2143e+11 95409
    ## - n_tokens_title                 1  490781567 5.2150e+11 95410
    ## - num_imgs                       1  595476872 5.2160e+11 95411
    ## - self_reference_max_shares      1  603368287 5.2161e+11 95411
    ## - max_positive_polarity          1  636199455 5.2164e+11 95411
    ## - self_reference_avg_sharess     1 1313395903 5.2232e+11 95418
    ## - kw_min_avg                     1 1503913646 5.2251e+11 95420
    ## - kw_max_avg                     1 2386772076 5.2340e+11 95429
    ## - kw_avg_avg                     1 4598368081 5.2561e+11 95451
    ## 
    ## Step:  AIC=95405.21
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - max_negative_polarity          1    6295077 5.2102e+11 95403
    ## - n_non_stop_unique_tokens       1   21188278 5.2104e+11 95403
    ## - kw_max_min                     1   37666812 5.2105e+11 95404
    ## - LDA_01                         1   44912242 5.2106e+11 95404
    ## - title_subjectivity             1   57715477 5.2107e+11 95404
    ## - abs_title_sentiment_polarity   1   58518107 5.2107e+11 95404
    ## - data_channel_is_lifestyle      1   64758954 5.2108e+11 95404
    ## - num_keywords                   1   74234716 5.2109e+11 95404
    ## - avg_positive_polarity          1   75576830 5.2109e+11 95404
    ## - LDA_00                         1   82377569 5.2110e+11 95404
    ## - kw_avg_min                     1   88245848 5.2110e+11 95404
    ## - num_videos                     1   89539149 5.2111e+11 95404
    ## - data_channel_is_bus            1   95075168 5.2111e+11 95404
    ## - n_unique_tokens                1  118450748 5.2113e+11 95404
    ## - abs_title_subjectivity         1  123734099 5.2114e+11 95404
    ## - data_channel_is_entertainment  1  129779447 5.2115e+11 95404
    ## - title_sentiment_polarity       1  131722555 5.2115e+11 95405
    ## - num_hrefs                      1  136149668 5.2115e+11 95405
    ## - min_positive_polarity          1  159819211 5.2118e+11 95405
    ## - n_non_stop_words               1  179988599 5.2120e+11 95405
    ## <none>                                        5.2102e+11 95405
    ## - LDA_03                         1  211807620 5.2123e+11 95405
    ## - timedelta                      1  220136991 5.2124e+11 95405
    ## - data_channel_is_socmed         1  239318579 5.2126e+11 95406
    ## - global_sentiment_polarity      1  252205359 5.2127e+11 95406
    ## - rate_positive_words            1  253631257 5.2127e+11 95406
    ## - global_subjectivity            1  256300159 5.2127e+11 95406
    ## - kw_min_max                     1  263387851 5.2128e+11 95406
    ## - data_channel_is_world          1  267969643 5.2128e+11 95406
    ## - LDA_02                         1  365900877 5.2138e+11 95407
    ## - global_rate_negative_words     1  425518904 5.2144e+11 95407
    ## - n_tokens_title                 1  489413152 5.2151e+11 95408
    ## - num_imgs                       1  597454455 5.2161e+11 95409
    ## - self_reference_max_shares      1  604874066 5.2162e+11 95409
    ## - max_positive_polarity          1  628783118 5.2164e+11 95409
    ## - self_reference_avg_sharess     1 1315786244 5.2233e+11 95416
    ## - kw_min_avg                     1 1500544368 5.2252e+11 95418
    ## - kw_max_avg                     1 2381153218 5.2340e+11 95427
    ## - kw_avg_avg                     1 4591015531 5.2561e+11 95449
    ## 
    ## Step:  AIC=95403.27
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - n_non_stop_unique_tokens       1   24965156 5.2105e+11 95402
    ## - kw_max_min                     1   37546188 5.2106e+11 95402
    ## - LDA_01                         1   45013126 5.2107e+11 95402
    ## - title_subjectivity             1   58203997 5.2108e+11 95402
    ## - abs_title_sentiment_polarity   1   59088741 5.2108e+11 95402
    ## - data_channel_is_lifestyle      1   65906095 5.2109e+11 95402
    ## - avg_positive_polarity          1   72272925 5.2109e+11 95402
    ## - num_keywords                   1   73048227 5.2110e+11 95402
    ## - LDA_00                         1   81792885 5.2110e+11 95402
    ## - num_videos                     1   87563069 5.2111e+11 95402
    ## - kw_avg_min                     1   87968637 5.2111e+11 95402
    ## - data_channel_is_bus            1   94515196 5.2112e+11 95402
    ## - abs_title_subjectivity         1  124190577 5.2115e+11 95403
    ## - data_channel_is_entertainment  1  129709577 5.2115e+11 95403
    ## - title_sentiment_polarity       1  132741962 5.2116e+11 95403
    ## - num_hrefs                      1  134046350 5.2116e+11 95403
    ## - n_unique_tokens                1  136812676 5.2116e+11 95403
    ## - min_positive_polarity          1  160409445 5.2118e+11 95403
    ## - n_non_stop_words               1  199241511 5.2122e+11 95403
    ## <none>                                        5.2102e+11 95403
    ## - LDA_03                         1  209632989 5.2123e+11 95403
    ## - timedelta                      1  219643377 5.2124e+11 95403
    ## - data_channel_is_socmed         1  240561948 5.2126e+11 95404
    ## - rate_positive_words            1  248185359 5.2127e+11 95404
    ## - kw_min_max                     1  263911061 5.2129e+11 95404
    ## - data_channel_is_world          1  267363649 5.2129e+11 95404
    ## - global_subjectivity            1  275488684 5.2130e+11 95404
    ## - global_sentiment_polarity      1  298629794 5.2132e+11 95404
    ## - LDA_02                         1  365157992 5.2139e+11 95405
    ## - global_rate_negative_words     1  446261706 5.2147e+11 95406
    ## - n_tokens_title                 1  491129130 5.2151e+11 95406
    ## - num_imgs                       1  594952434 5.2162e+11 95407
    ## - self_reference_max_shares      1  611399188 5.2163e+11 95407
    ## - max_positive_polarity          1  630036737 5.2165e+11 95408
    ## - self_reference_avg_sharess     1 1330303579 5.2235e+11 95414
    ## - kw_min_avg                     1 1505221587 5.2253e+11 95416
    ## - kw_max_avg                     1 2384248762 5.2341e+11 95425
    ## - kw_avg_avg                     1 4594269845 5.2562e+11 95447
    ## 
    ## Step:  AIC=95401.52
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_world + kw_max_min + kw_avg_min + kw_min_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_max_min                     1   38843186 5.2109e+11 95400
    ## - LDA_01                         1   40089054 5.2109e+11 95400
    ## - title_subjectivity             1   57910569 5.2111e+11 95400
    ## - data_channel_is_lifestyle      1   59628496 5.2111e+11 95400
    ## - abs_title_sentiment_polarity   1   60325742 5.2111e+11 95400
    ## - num_keywords                   1   76154109 5.2112e+11 95400
    ## - LDA_00                         1   79117825 5.2113e+11 95400
    ## - avg_positive_polarity          1   84545147 5.2113e+11 95400
    ## - kw_avg_min                     1   90540553 5.2114e+11 95400
    ## - data_channel_is_bus            1   90659214 5.2114e+11 95400
    ## - num_videos                     1   94678754 5.2114e+11 95400
    ## - abs_title_subjectivity         1  121805203 5.2117e+11 95401
    ## - title_sentiment_polarity       1  131014937 5.2118e+11 95401
    ## - data_channel_is_entertainment  1  137704285 5.2119e+11 95401
    ## - num_hrefs                      1  141720497 5.2119e+11 95401
    ## - min_positive_polarity          1  176369300 5.2122e+11 95401
    ## - LDA_03                         1  191717934 5.2124e+11 95401
    ## <none>                                        5.2105e+11 95402
    ## - n_non_stop_words               1  210840350 5.2126e+11 95402
    ## - n_unique_tokens                1  211217397 5.2126e+11 95402
    ## - timedelta                      1  231890334 5.2128e+11 95402
    ## - data_channel_is_socmed         1  235105945 5.2128e+11 95402
    ## - data_channel_is_world          1  258379403 5.2131e+11 95402
    ## - kw_min_max                     1  265533885 5.2131e+11 95402
    ## - global_subjectivity            1  267670279 5.2132e+11 95402
    ## - global_sentiment_polarity      1  277209036 5.2132e+11 95402
    ## - rate_positive_words            1  323259464 5.2137e+11 95403
    ## - LDA_02                         1  358299698 5.2141e+11 95403
    ## - n_tokens_title                 1  489933759 5.2154e+11 95404
    ## - global_rate_negative_words     1  498692012 5.2155e+11 95404
    ## - max_positive_polarity          1  609415995 5.2166e+11 95406
    ## - self_reference_max_shares      1  615782770 5.2166e+11 95406
    ## - num_imgs                       1  708099569 5.2176e+11 95407
    ## - self_reference_avg_sharess     1 1334731106 5.2238e+11 95413
    ## - kw_min_avg                     1 1501720338 5.2255e+11 95414
    ## - kw_max_avg                     1 2405635167 5.2345e+11 95423
    ## - kw_avg_avg                     1 4635556148 5.2568e+11 95445
    ## 
    ## Step:  AIC=95399.91
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_world + kw_avg_min + kw_min_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - LDA_01                         1   39024951 5.2113e+11 95398
    ## - data_channel_is_lifestyle      1   56462726 5.2114e+11 95398
    ## - title_subjectivity             1   59596983 5.2115e+11 95398
    ## - abs_title_sentiment_polarity   1   61582256 5.2115e+11 95399
    ## - LDA_00                         1   77333986 5.2116e+11 95399
    ## - num_keywords                   1   79978019 5.2117e+11 95399
    ## - avg_positive_polarity          1   85555372 5.2117e+11 95399
    ## - data_channel_is_bus            1   85903235 5.2117e+11 95399
    ## - num_videos                     1   98731386 5.2118e+11 95399
    ## - kw_avg_min                     1  118378811 5.2120e+11 95399
    ## - abs_title_subjectivity         1  122039970 5.2121e+11 95399
    ## - title_sentiment_polarity       1  129227329 5.2122e+11 95399
    ## - data_channel_is_entertainment  1  141055463 5.2123e+11 95399
    ## - num_hrefs                      1  141090942 5.2123e+11 95399
    ## - min_positive_polarity          1  179689272 5.2127e+11 95400
    ## - LDA_03                         1  183727453 5.2127e+11 95400
    ## - timedelta                      1  193288336 5.2128e+11 95400
    ## <none>                                        5.2109e+11 95400
    ## - n_non_stop_words               1  221150145 5.2131e+11 95400
    ## - n_unique_tokens                1  221542603 5.2131e+11 95400
    ## - data_channel_is_socmed         1  223642256 5.2131e+11 95400
    ## - kw_min_max                     1  252953372 5.2134e+11 95400
    ## - data_channel_is_world          1  255937322 5.2134e+11 95400
    ## - global_subjectivity            1  263197063 5.2135e+11 95401
    ## - global_sentiment_polarity      1  276535744 5.2136e+11 95401
    ## - rate_positive_words            1  319581541 5.2141e+11 95401
    ## - LDA_02                         1  358484302 5.2144e+11 95401
    ## - n_tokens_title                 1  484030551 5.2157e+11 95403
    ## - global_rate_negative_words     1  496891245 5.2158e+11 95403
    ## - max_positive_polarity          1  617901802 5.2170e+11 95404
    ## - self_reference_max_shares      1  622822996 5.2171e+11 95404
    ## - num_imgs                       1  718545878 5.2180e+11 95405
    ## - self_reference_avg_sharess     1 1340918518 5.2243e+11 95411
    ## - kw_min_avg                     1 1465074732 5.2255e+11 95412
    ## - kw_max_avg                     1 2394065069 5.2348e+11 95422
    ## - kw_avg_avg                     1 4623394035 5.2571e+11 95444
    ## 
    ## Step:  AIC=95398.29
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_world + kw_avg_min + kw_min_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - LDA_00                         1   54814341 5.2118e+11 95397
    ## - abs_title_sentiment_polarity   1   60616840 5.2119e+11 95397
    ## - title_subjectivity             1   62282701 5.2119e+11 95397
    ## - data_channel_is_lifestyle      1   64876232 5.2119e+11 95397
    ## - data_channel_is_bus            1   75140745 5.2120e+11 95397
    ## - avg_positive_polarity          1   84369836 5.2121e+11 95397
    ## - num_keywords                   1   89558927 5.2121e+11 95397
    ## - num_videos                     1   96753358 5.2122e+11 95397
    ## - kw_avg_min                     1  117548458 5.2124e+11 95397
    ## - abs_title_subjectivity         1  121527349 5.2125e+11 95397
    ## - title_sentiment_polarity       1  129473014 5.2125e+11 95398
    ## - num_hrefs                      1  136834615 5.2126e+11 95398
    ## - LDA_03                         1  145748656 5.2127e+11 95398
    ## - min_positive_polarity          1  182019888 5.2131e+11 95398
    ## - timedelta                      1  184958920 5.2131e+11 95398
    ## - n_non_stop_words               1  199360506 5.2132e+11 95398
    ## - n_unique_tokens                1  199785003 5.2133e+11 95398
    ## <none>                                        5.2113e+11 95398
    ## - data_channel_is_socmed         1  211078808 5.2134e+11 95398
    ## - data_channel_is_world          1  237230103 5.2136e+11 95399
    ## - kw_min_max                     1  247171284 5.2137e+11 95399
    ## - global_subjectivity            1  251828610 5.2138e+11 95399
    ## - global_sentiment_polarity      1  275341724 5.2140e+11 95399
    ## - rate_positive_words            1  310831793 5.2144e+11 95399
    ## - LDA_02                         1  321037488 5.2145e+11 95399
    ## - data_channel_is_entertainment  1  409041984 5.2153e+11 95400
    ## - n_tokens_title                 1  483860298 5.2161e+11 95401
    ## - global_rate_negative_words     1  508423031 5.2163e+11 95401
    ## - max_positive_polarity          1  612356467 5.2174e+11 95402
    ## - self_reference_max_shares      1  620669741 5.2175e+11 95402
    ## - num_imgs                       1  707973191 5.2183e+11 95403
    ## - self_reference_avg_sharess     1 1344943894 5.2247e+11 95410
    ## - kw_min_avg                     1 1427001927 5.2255e+11 95410
    ## - kw_max_avg                     1 2357570605 5.2348e+11 95420
    ## - kw_avg_avg                     1 4666526299 5.2579e+11 95442
    ## 
    ## Step:  AIC=95396.84
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_world + kw_avg_min + kw_min_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - data_channel_is_bus            1   22110436 5.2120e+11 95395
    ## - data_channel_is_lifestyle      1   50368948 5.2123e+11 95395
    ## - abs_title_sentiment_polarity   1   60736564 5.2124e+11 95395
    ## - title_subjectivity             1   61031251 5.2124e+11 95395
    ## - avg_positive_polarity          1   88274237 5.2127e+11 95396
    ## - num_keywords                   1   97837357 5.2128e+11 95396
    ## - num_videos                     1   99161954 5.2128e+11 95396
    ## - kw_avg_min                     1  117848431 5.2130e+11 95396
    ## - LDA_03                         1  120163134 5.2130e+11 95396
    ## - abs_title_subjectivity         1  122551706 5.2130e+11 95396
    ## - title_sentiment_polarity       1  127033165 5.2131e+11 95396
    ## - num_hrefs                      1  132211670 5.2131e+11 95396
    ## - data_channel_is_socmed         1  158554700 5.2134e+11 95396
    ## - timedelta                      1  180301586 5.2136e+11 95397
    ## - n_non_stop_words               1  194653780 5.2137e+11 95397
    ## - n_unique_tokens                1  195082318 5.2138e+11 95397
    ## - min_positive_polarity          1  199628159 5.2138e+11 95397
    ## <none>                                        5.2118e+11 95397
    ## - data_channel_is_world          1  204002865 5.2138e+11 95397
    ## - kw_min_max                     1  247653759 5.2143e+11 95397
    ## - global_subjectivity            1  254447023 5.2143e+11 95397
    ## - global_sentiment_polarity      1  267134345 5.2145e+11 95397
    ## - LDA_02                         1  273613877 5.2145e+11 95398
    ## - rate_positive_words            1  317077813 5.2150e+11 95398
    ## - data_channel_is_entertainment  1  410988646 5.2159e+11 95399
    ## - n_tokens_title                 1  484402994 5.2166e+11 95400
    ## - global_rate_negative_words     1  502862928 5.2168e+11 95400
    ## - max_positive_polarity          1  609530665 5.2179e+11 95401
    ## - self_reference_max_shares      1  617259716 5.2180e+11 95401
    ## - num_imgs                       1  732813294 5.2191e+11 95402
    ## - self_reference_avg_sharess     1 1341375878 5.2252e+11 95408
    ## - kw_min_avg                     1 1406216131 5.2259e+11 95409
    ## - kw_max_avg                     1 2345240588 5.2353e+11 95418
    ## - kw_avg_avg                     1 4626172873 5.2581e+11 95441
    ## 
    ## Step:  AIC=95395.06
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_socmed + 
    ##     data_channel_is_world + kw_avg_min + kw_min_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - data_channel_is_lifestyle      1   38473933 5.2124e+11 95393
    ## - abs_title_sentiment_polarity   1   62653933 5.2126e+11 95394
    ## - title_subjectivity             1   63542951 5.2127e+11 95394
    ## - avg_positive_polarity          1   83195174 5.2129e+11 95394
    ## - num_keywords                   1   84555251 5.2129e+11 95394
    ## - num_videos                     1   97093620 5.2130e+11 95394
    ## - kw_avg_min                     1  118141495 5.2132e+11 95394
    ## - abs_title_subjectivity         1  122812044 5.2132e+11 95394
    ## - title_sentiment_polarity       1  127290063 5.2133e+11 95394
    ## - num_hrefs                      1  136064530 5.2134e+11 95394
    ## - data_channel_is_socmed         1  138703922 5.2134e+11 95394
    ## - LDA_03                         1  147067186 5.2135e+11 95395
    ## - timedelta                      1  177999046 5.2138e+11 95395
    ## - data_channel_is_world          1  184711520 5.2139e+11 95395
    ## - min_positive_polarity          1  192120188 5.2139e+11 95395
    ## - n_non_stop_words               1  193304344 5.2140e+11 95395
    ## - n_unique_tokens                1  193735558 5.2140e+11 95395
    ## <none>                                        5.2120e+11 95395
    ## - global_subjectivity            1  243640846 5.2145e+11 95395
    ## - kw_min_max                     1  243831498 5.2145e+11 95395
    ## - global_sentiment_polarity      1  276536278 5.2148e+11 95396
    ## - LDA_02                         1  288666422 5.2149e+11 95396
    ## - rate_positive_words            1  308075921 5.2151e+11 95396
    ## - n_tokens_title                 1  487555266 5.2169e+11 95398
    ## - global_rate_negative_words     1  502260934 5.2170e+11 95398
    ## - data_channel_is_entertainment  1  525010898 5.2173e+11 95398
    ## - max_positive_polarity          1  605088759 5.2181e+11 95399
    ## - self_reference_max_shares      1  607722020 5.2181e+11 95399
    ## - num_imgs                       1  714777868 5.2192e+11 95400
    ## - self_reference_avg_sharess     1 1327508839 5.2253e+11 95406
    ## - kw_min_avg                     1 1413311610 5.2262e+11 95407
    ## - kw_max_avg                     1 2326840919 5.2353e+11 95416
    ## - kw_avg_avg                     1 4606421393 5.2581e+11 95439
    ## 
    ## Step:  AIC=95393.44
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_entertainment + 
    ##     data_channel_is_socmed + data_channel_is_world + kw_avg_min + 
    ##     kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - title_subjectivity             1   63262464 5.2130e+11 95392
    ## - abs_title_sentiment_polarity   1   63661418 5.2130e+11 95392
    ## - avg_positive_polarity          1   79264093 5.2132e+11 95392
    ## - num_keywords                   1   92546651 5.2133e+11 95392
    ## - num_videos                     1   93081561 5.2133e+11 95392
    ## - kw_avg_min                     1  111406345 5.2135e+11 95393
    ## - abs_title_subjectivity         1  121439523 5.2136e+11 95393
    ## - data_channel_is_socmed         1  126030174 5.2137e+11 95393
    ## - title_sentiment_polarity       1  128625639 5.2137e+11 95393
    ## - num_hrefs                      1  139422056 5.2138e+11 95393
    ## - LDA_03                         1  164681816 5.2141e+11 95393
    ## - data_channel_is_world          1  173382190 5.2141e+11 95393
    ## - timedelta                      1  185364298 5.2143e+11 95393
    ## - min_positive_polarity          1  186598780 5.2143e+11 95393
    ## - n_non_stop_words               1  188833243 5.2143e+11 95393
    ## - n_unique_tokens                1  189261927 5.2143e+11 95393
    ## <none>                                        5.2124e+11 95393
    ## - kw_min_max                     1  241757367 5.2148e+11 95394
    ## - global_subjectivity            1  242826442 5.2148e+11 95394
    ## - global_sentiment_polarity      1  274094774 5.2151e+11 95394
    ## - LDA_02                         1  292321421 5.2153e+11 95394
    ## - rate_positive_words            1  315281995 5.2156e+11 95395
    ## - n_tokens_title                 1  480340338 5.2172e+11 95396
    ## - global_rate_negative_words     1  498351719 5.2174e+11 95396
    ## - data_channel_is_entertainment  1  559278351 5.2180e+11 95397
    ## - max_positive_polarity          1  602282971 5.2184e+11 95397
    ## - self_reference_max_shares      1  617121311 5.2186e+11 95398
    ## - num_imgs                       1  700500315 5.2194e+11 95398
    ## - self_reference_avg_sharess     1 1335436287 5.2258e+11 95405
    ## - kw_min_avg                     1 1444375811 5.2269e+11 95406
    ## - kw_max_avg                     1 2372044655 5.2361e+11 95415
    ## - kw_avg_avg                     1 4687578882 5.2593e+11 95438
    ## 
    ## Step:  AIC=95392.07
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_entertainment + 
    ##     data_channel_is_socmed + data_channel_is_world + kw_avg_min + 
    ##     kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - abs_title_sentiment_polarity   1   14619124 5.2132e+11 95390
    ## - avg_positive_polarity          1   73317949 5.2138e+11 95391
    ## - num_videos                     1   89066322 5.2139e+11 95391
    ## - num_keywords                   1   93594789 5.2140e+11 95391
    ## - kw_avg_min                     1  117791509 5.2142e+11 95391
    ## - data_channel_is_socmed         1  127114537 5.2143e+11 95391
    ## - num_hrefs                      1  137989101 5.2144e+11 95391
    ## - title_sentiment_polarity       1  148780628 5.2145e+11 95392
    ## - LDA_03                         1  165181831 5.2147e+11 95392
    ## - data_channel_is_world          1  175878360 5.2148e+11 95392
    ## - min_positive_polarity          1  183205653 5.2149e+11 95392
    ## - timedelta                      1  185320871 5.2149e+11 95392
    ## - n_non_stop_words               1  185664898 5.2149e+11 95392
    ## - n_unique_tokens                1  186119363 5.2149e+11 95392
    ## <none>                                        5.2130e+11 95392
    ## - abs_title_subjectivity         1  203960604 5.2151e+11 95392
    ## - global_subjectivity            1  221996725 5.2153e+11 95392
    ## - kw_min_max                     1  241550647 5.2155e+11 95392
    ## - global_sentiment_polarity      1  273187243 5.2158e+11 95393
    ## - LDA_02                         1  296329486 5.2160e+11 95393
    ## - rate_positive_words            1  307471220 5.2161e+11 95393
    ## - n_tokens_title                 1  467940717 5.2177e+11 95395
    ## - global_rate_negative_words     1  497066749 5.2180e+11 95395
    ## - data_channel_is_entertainment  1  564373573 5.2187e+11 95396
    ## - max_positive_polarity          1  596909123 5.2190e+11 95396
    ## - self_reference_max_shares      1  621181745 5.2193e+11 95396
    ## - num_imgs                       1  692889414 5.2200e+11 95397
    ## - self_reference_avg_sharess     1 1338567031 5.2264e+11 95403
    ## - kw_min_avg                     1 1426511080 5.2273e+11 95404
    ## - kw_max_avg                     1 2342543984 5.2365e+11 95413
    ## - kw_avg_avg                     1 4643822862 5.2595e+11 95436
    ## 
    ## Step:  AIC=95390.21
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_entertainment + 
    ##     data_channel_is_socmed + data_channel_is_world + kw_avg_min + 
    ##     kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - avg_positive_polarity          1   68306788 5.2139e+11 95389
    ## - num_videos                     1   88708463 5.2141e+11 95389
    ## - num_keywords                   1   95682663 5.2141e+11 95389
    ## - kw_avg_min                     1  119470930 5.2144e+11 95389
    ## - data_channel_is_socmed         1  127104607 5.2145e+11 95389
    ## - num_hrefs                      1  140006820 5.2146e+11 95390
    ## - LDA_03                         1  160733113 5.2148e+11 95390
    ## - data_channel_is_world          1  177080178 5.2150e+11 95390
    ## - min_positive_polarity          1  179091469 5.2150e+11 95390
    ## - timedelta                      1  183572375 5.2150e+11 95390
    ## - n_non_stop_words               1  187550505 5.2151e+11 95390
    ## - n_unique_tokens                1  187990904 5.2151e+11 95390
    ## - abs_title_subjectivity         1  190233700 5.2151e+11 95390
    ## <none>                                        5.2132e+11 95390
    ## - title_sentiment_polarity       1  214243818 5.2153e+11 95390
    ## - global_subjectivity            1  230549083 5.2155e+11 95390
    ## - kw_min_max                     1  239362309 5.2156e+11 95391
    ## - global_sentiment_polarity      1  275765808 5.2159e+11 95391
    ## - LDA_02                         1  298435686 5.2162e+11 95391
    ## - rate_positive_words            1  323117415 5.2164e+11 95391
    ## - n_tokens_title                 1  467366226 5.2179e+11 95393
    ## - global_rate_negative_words     1  503315432 5.2182e+11 95393
    ## - data_channel_is_entertainment  1  563596626 5.2188e+11 95394
    ## - max_positive_polarity          1  595327016 5.2191e+11 95394
    ## - self_reference_max_shares      1  623574729 5.2194e+11 95394
    ## - num_imgs                       1  696140569 5.2201e+11 95395
    ## - self_reference_avg_sharess     1 1338436179 5.2266e+11 95401
    ## - kw_min_avg                     1 1424198143 5.2274e+11 95402
    ## - kw_max_avg                     1 2338896523 5.2366e+11 95411
    ## - kw_avg_avg                     1 4646922631 5.2597e+11 95434
    ## 
    ## Step:  AIC=95388.89
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + num_keywords + data_channel_is_entertainment + 
    ##     data_channel_is_socmed + data_channel_is_world + kw_avg_min + 
    ##     kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + min_positive_polarity + max_positive_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_videos                     1   86701136 5.2147e+11 95388
    ## - num_keywords                   1  102273596 5.2149e+11 95388
    ## - min_positive_polarity          1  111516629 5.2150e+11 95388
    ## - kw_avg_min                     1  121395663 5.2151e+11 95388
    ## - data_channel_is_socmed         1  130497536 5.2152e+11 95388
    ## - num_hrefs                      1  131439710 5.2152e+11 95388
    ## - LDA_03                         1  160658927 5.2155e+11 95388
    ## - data_channel_is_world          1  174116277 5.2156e+11 95389
    ## - abs_title_subjectivity         1  174440485 5.2156e+11 95389
    ## - timedelta                      1  179435082 5.2157e+11 95389
    ## - n_non_stop_words               1  181014954 5.2157e+11 95389
    ## - global_subjectivity            1  181373073 5.2157e+11 95389
    ## - n_unique_tokens                1  181465898 5.2157e+11 95389
    ## <none>                                        5.2139e+11 95389
    ## - title_sentiment_polarity       1  214089527 5.2160e+11 95389
    ## - kw_min_max                     1  240728826 5.2163e+11 95389
    ## - rate_positive_words            1  294919351 5.2168e+11 95390
    ## - LDA_02                         1  304226181 5.2169e+11 95390
    ## - global_sentiment_polarity      1  467621194 5.2185e+11 95392
    ## - n_tokens_title                 1  468521247 5.2186e+11 95392
    ## - max_positive_polarity          1  561942031 5.2195e+11 95392
    ## - data_channel_is_entertainment  1  570324994 5.2196e+11 95393
    ## - self_reference_max_shares      1  628880232 5.2202e+11 95393
    ## - global_rate_negative_words     1  630831011 5.2202e+11 95393
    ## - num_imgs                       1  686109117 5.2207e+11 95394
    ## - self_reference_avg_sharess     1 1335133513 5.2272e+11 95400
    ## - kw_min_avg                     1 1429726372 5.2282e+11 95401
    ## - kw_max_avg                     1 2349010249 5.2374e+11 95410
    ## - kw_avg_avg                     1 4674831727 5.2606e+11 95433
    ## 
    ## Step:  AIC=95387.75
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_keywords + data_channel_is_entertainment + 
    ##     data_channel_is_socmed + data_channel_is_world + kw_avg_min + 
    ##     kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + min_positive_polarity + max_positive_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - min_positive_polarity          1  101097522 5.2157e+11 95387
    ## - num_keywords                   1  101419202 5.2157e+11 95387
    ## - LDA_03                         1  117792502 5.2159e+11 95387
    ## - kw_avg_min                     1  123764268 5.2160e+11 95387
    ## - data_channel_is_socmed         1  127543152 5.2160e+11 95387
    ## - num_hrefs                      1  160619511 5.2163e+11 95387
    ## - data_channel_is_world          1  169437715 5.2164e+11 95387
    ## - global_subjectivity            1  176491184 5.2165e+11 95387
    ## - timedelta                      1  178744901 5.2165e+11 95388
    ## - abs_title_subjectivity         1  179940509 5.2165e+11 95388
    ## - n_non_stop_words               1  183709334 5.2166e+11 95388
    ## - n_unique_tokens                1  184214351 5.2166e+11 95388
    ## <none>                                        5.2147e+11 95388
    ## - title_sentiment_polarity       1  220364015 5.2169e+11 95388
    ## - kw_min_max                     1  231935900 5.2171e+11 95388
    ## - rate_positive_words            1  283922327 5.2176e+11 95389
    ## - LDA_02                         1  302042261 5.2178e+11 95389
    ## - global_sentiment_polarity      1  472741572 5.2195e+11 95390
    ## - n_tokens_title                 1  491912931 5.2197e+11 95391
    ## - data_channel_is_entertainment  1  547338401 5.2202e+11 95391
    ## - global_rate_negative_words     1  580739598 5.2205e+11 95392
    ## - max_positive_polarity          1  600930148 5.2207e+11 95392
    ## - self_reference_max_shares      1  603759147 5.2208e+11 95392
    ## - num_imgs                       1  617550713 5.2209e+11 95392
    ## - self_reference_avg_sharess     1 1305793175 5.2278e+11 95399
    ## - kw_min_avg                     1 1423798606 5.2290e+11 95400
    ## - kw_max_avg                     1 2345126242 5.2382e+11 95409
    ## - kw_avg_avg                     1 4675804839 5.2615e+11 95432
    ## 
    ## Step:  AIC=95386.75
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_keywords + data_channel_is_entertainment + 
    ##     data_channel_is_socmed + data_channel_is_world + kw_avg_min + 
    ##     kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + max_positive_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_keywords                   1  101181254 5.2168e+11 95386
    ## - LDA_03                         1  108234751 5.2168e+11 95386
    ## - data_channel_is_socmed         1  115933520 5.2169e+11 95386
    ## - kw_avg_min                     1  122637609 5.2170e+11 95386
    ## - num_hrefs                      1  149187787 5.2172e+11 95386
    ## - data_channel_is_world          1  163019350 5.2174e+11 95386
    ## - timedelta                      1  175428232 5.2175e+11 95386
    ## - abs_title_subjectivity         1  182970816 5.2176e+11 95387
    ## <none>                                        5.2157e+11 95387
    ## - title_sentiment_polarity       1  224845486 5.2180e+11 95387
    ## - kw_min_max                     1  230801970 5.2181e+11 95387
    ## - global_subjectivity            1  252300331 5.2183e+11 95387
    ## - n_non_stop_words               1  281138328 5.2186e+11 95388
    ## - n_unique_tokens                1  281773948 5.2186e+11 95388
    ## - LDA_02                         1  287654619 5.2186e+11 95388
    ## - rate_positive_words            1  298624899 5.2187e+11 95388
    ## - global_sentiment_polarity      1  449647365 5.2202e+11 95389
    ## - n_tokens_title                 1  490787715 5.2207e+11 95390
    ## - global_rate_negative_words     1  569342324 5.2214e+11 95390
    ## - data_channel_is_entertainment  1  574093770 5.2215e+11 95390
    ## - max_positive_polarity          1  587219154 5.2216e+11 95391
    ## - self_reference_max_shares      1  614564983 5.2219e+11 95391
    ## - num_imgs                       1  651942927 5.2223e+11 95391
    ## - self_reference_avg_sharess     1 1319918292 5.2289e+11 95398
    ## - kw_min_avg                     1 1403423086 5.2298e+11 95399
    ## - kw_max_avg                     1 2317537222 5.2389e+11 95408
    ## - kw_avg_avg                     1 4626717224 5.2620e+11 95430
    ## 
    ## Step:  AIC=95385.76
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + data_channel_is_entertainment + data_channel_is_socmed + 
    ##     data_channel_is_world + kw_avg_min + kw_min_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + max_positive_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - data_channel_is_socmed         1   95041677 5.2177e+11 95385
    ## - LDA_03                         1  105313576 5.2178e+11 95385
    ## - kw_avg_min                     1  115162430 5.2179e+11 95385
    ## - num_hrefs                      1  166249484 5.2184e+11 95385
    ## - data_channel_is_world          1  169335683 5.2185e+11 95385
    ## - timedelta                      1  177267694 5.2185e+11 95386
    ## - abs_title_subjectivity         1  179116909 5.2185e+11 95386
    ## <none>                                        5.2168e+11 95386
    ## - title_sentiment_polarity       1  224067077 5.2190e+11 95386
    ## - global_subjectivity            1  252573019 5.2193e+11 95386
    ## - n_non_stop_words               1  257161319 5.2193e+11 95386
    ## - n_unique_tokens                1  257775091 5.2193e+11 95386
    ## - LDA_02                         1  293888414 5.2197e+11 95387
    ## - kw_min_max                     1  299742090 5.2198e+11 95387
    ## - rate_positive_words            1  311602592 5.2199e+11 95387
    ## - global_sentiment_polarity      1  429488697 5.2211e+11 95388
    ## - n_tokens_title                 1  500808395 5.2218e+11 95389
    ## - global_rate_negative_words     1  588377806 5.2226e+11 95390
    ## - max_positive_polarity          1  593684412 5.2227e+11 95390
    ## - self_reference_max_shares      1  617003814 5.2229e+11 95390
    ## - data_channel_is_entertainment  1  629007549 5.2230e+11 95390
    ## - num_imgs                       1  662599008 5.2234e+11 95390
    ## - self_reference_avg_sharess     1 1319496606 5.2300e+11 95397
    ## - kw_min_avg                     1 1670817181 5.2335e+11 95400
    ## - kw_max_avg                     1 2344986554 5.2402e+11 95407
    ## - kw_avg_avg                     1 4788455342 5.2646e+11 95431
    ## 
    ## Step:  AIC=95384.7
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + data_channel_is_entertainment + data_channel_is_world + 
    ##     kw_avg_min + kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + max_positive_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - LDA_03                         1  108431874 5.2188e+11 95384
    ## - kw_avg_min                     1  109276101 5.2188e+11 95384
    ## - data_channel_is_world          1  121054598 5.2189e+11 95384
    ## - abs_title_subjectivity         1  173562212 5.2194e+11 95384
    ## - num_hrefs                      1  178948206 5.2195e+11 95384
    ## - timedelta                      1  184627380 5.2196e+11 95385
    ## <none>                                        5.2177e+11 95385
    ## - title_sentiment_polarity       1  223352644 5.2199e+11 95385
    ## - global_subjectivity            1  238719319 5.2201e+11 95385
    ## - n_non_stop_words               1  239552440 5.2201e+11 95385
    ## - n_unique_tokens                1  240151511 5.2201e+11 95385
    ## - LDA_02                         1  241515113 5.2201e+11 95385
    ## - kw_min_max                     1  287694307 5.2206e+11 95386
    ## - rate_positive_words            1  297152223 5.2207e+11 95386
    ## - global_sentiment_polarity      1  415378813 5.2219e+11 95387
    ## - n_tokens_title                 1  479588649 5.2225e+11 95387
    ## - global_rate_negative_words     1  566218566 5.2234e+11 95388
    ## - max_positive_polarity          1  579271574 5.2235e+11 95388
    ## - self_reference_max_shares      1  603320636 5.2237e+11 95389
    ## - num_imgs                       1  649616594 5.2242e+11 95389
    ## - data_channel_is_entertainment  1  698040016 5.2247e+11 95390
    ## - self_reference_avg_sharess     1 1304963791 5.2308e+11 95396
    ## - kw_min_avg                     1 1677160266 5.2345e+11 95399
    ## - kw_max_avg                     1 2392195752 5.2416e+11 95406
    ## - kw_avg_avg                     1 4839204683 5.2661e+11 95430
    ## 
    ## Step:  AIC=95383.77
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + data_channel_is_entertainment + data_channel_is_world + 
    ##     kw_avg_min + kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_02 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + max_positive_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_avg_min                     1   98051050 5.2198e+11 95383
    ## - data_channel_is_world          1  104296050 5.2198e+11 95383
    ## - num_hrefs                      1  154042410 5.2203e+11 95383
    ## - timedelta                      1  155778114 5.2204e+11 95383
    ## - abs_title_subjectivity         1  172033623 5.2205e+11 95383
    ## - n_non_stop_words               1  185434554 5.2206e+11 95384
    ## - n_unique_tokens                1  186031070 5.2207e+11 95384
    ## <none>                                        5.2188e+11 95384
    ## - LDA_02                         1  206641089 5.2209e+11 95384
    ## - title_sentiment_polarity       1  213649704 5.2209e+11 95384
    ## - global_subjectivity            1  214514452 5.2209e+11 95384
    ## - rate_positive_words            1  243052759 5.2212e+11 95384
    ## - kw_min_max                     1  288164105 5.2217e+11 95385
    ## - global_sentiment_polarity      1  431630575 5.2231e+11 95386
    ## - n_tokens_title                 1  452778900 5.2233e+11 95386
    ## - global_rate_negative_words     1  540113984 5.2242e+11 95387
    ## - max_positive_polarity          1  557621327 5.2244e+11 95387
    ## - self_reference_max_shares      1  585858499 5.2247e+11 95388
    ## - num_imgs                       1  592930217 5.2247e+11 95388
    ## - data_channel_is_entertainment  1  867331765 5.2275e+11 95390
    ## - self_reference_avg_sharess     1 1298391591 5.2318e+11 95395
    ## - kw_min_avg                     1 1578489615 5.2346e+11 95397
    ## - kw_max_avg                     1 2297032826 5.2418e+11 95404
    ## - kw_avg_avg                     1 5097227346 5.2698e+11 95432
    ## 
    ## Step:  AIC=95382.74
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + data_channel_is_entertainment + data_channel_is_world + 
    ##     kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_02 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + max_positive_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - data_channel_is_world          1  107059612 5.2208e+11 95382
    ## - timedelta                      1  117631313 5.2209e+11 95382
    ## - num_hrefs                      1  162073217 5.2214e+11 95382
    ## - abs_title_subjectivity         1  172451412 5.2215e+11 95382
    ## - n_non_stop_words               1  199508476 5.2218e+11 95383
    ## - n_unique_tokens                1  200126632 5.2218e+11 95383
    ## <none>                                        5.2198e+11 95383
    ## - LDA_02                         1  209493838 5.2219e+11 95383
    ## - title_sentiment_polarity       1  214468595 5.2219e+11 95383
    ## - global_subjectivity            1  222779816 5.2220e+11 95383
    ## - rate_positive_words            1  250619735 5.2223e+11 95383
    ## - kw_min_max                     1  260934296 5.2224e+11 95383
    ## - global_sentiment_polarity      1  429702097 5.2241e+11 95385
    ## - n_tokens_title                 1  453266651 5.2243e+11 95385
    ## - global_rate_negative_words     1  544808509 5.2252e+11 95386
    ## - self_reference_max_shares      1  558785922 5.2254e+11 95386
    ## - max_positive_polarity          1  560885942 5.2254e+11 95386
    ## - num_imgs                       1  597436093 5.2257e+11 95387
    ## - data_channel_is_entertainment  1  846782066 5.2282e+11 95389
    ## - self_reference_avg_sharess     1 1304337019 5.2328e+11 95394
    ## - kw_min_avg                     1 1602704993 5.2358e+11 95397
    ## - kw_max_avg                     1 2736555242 5.2471e+11 95408
    ## - kw_avg_avg                     1 5132823023 5.2711e+11 95431
    ## 
    ## Step:  AIC=95381.81
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + data_channel_is_entertainment + kw_min_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_02 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + max_positive_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - timedelta                      1   97204403 5.2218e+11 95381
    ## - LDA_02                         1  104405768 5.2219e+11 95381
    ## - num_hrefs                      1  152906124 5.2224e+11 95381
    ## - abs_title_subjectivity         1  170104611 5.2225e+11 95381
    ## - n_non_stop_words               1  198829884 5.2228e+11 95382
    ## - n_unique_tokens                1  199458789 5.2228e+11 95382
    ## <none>                                        5.2208e+11 95382
    ## - title_sentiment_polarity       1  211812537 5.2230e+11 95382
    ## - global_subjectivity            1  234246578 5.2232e+11 95382
    ## - kw_min_max                     1  259015717 5.2234e+11 95382
    ## - rate_positive_words            1  271567757 5.2236e+11 95382
    ## - global_sentiment_polarity      1  456086444 5.2254e+11 95384
    ## - n_tokens_title                 1  464006676 5.2255e+11 95384
    ## - self_reference_max_shares      1  570789055 5.2266e+11 95385
    ## - num_imgs                       1  575802545 5.2266e+11 95386
    ## - global_rate_negative_words     1  577322011 5.2266e+11 95386
    ## - max_positive_polarity          1  585836654 5.2267e+11 95386
    ## - data_channel_is_entertainment  1  928256631 5.2301e+11 95389
    ## - self_reference_avg_sharess     1 1319695301 5.2340e+11 95393
    ## - kw_min_avg                     1 1594444725 5.2368e+11 95396
    ## - kw_max_avg                     1 2659000887 5.2474e+11 95406
    ## - kw_avg_avg                     1 5037177837 5.2712e+11 95429
    ## 
    ## Step:  AIC=95380.77
    ## shares ~ n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + data_channel_is_entertainment + kw_min_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_02 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + max_positive_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - LDA_02                         1  144182457 5.2233e+11 95380
    ## - num_hrefs                      1  159640429 5.2234e+11 95380
    ## - abs_title_subjectivity         1  168815838 5.2235e+11 95380
    ## <none>                                        5.2218e+11 95381
    ## - title_sentiment_polarity       1  212981598 5.2239e+11 95381
    ## - global_subjectivity            1  234786215 5.2242e+11 95381
    ## - rate_positive_words            1  247152157 5.2243e+11 95381
    ## - n_non_stop_words               1  247532693 5.2243e+11 95381
    ## - n_unique_tokens                1  248221619 5.2243e+11 95381
    ## - kw_min_max                     1  257713714 5.2244e+11 95381
    ## - n_tokens_title                 1  389627424 5.2257e+11 95383
    ## - global_sentiment_polarity      1  448879920 5.2263e+11 95383
    ## - global_rate_negative_words     1  542111815 5.2272e+11 95384
    ## - self_reference_max_shares      1  560815599 5.2274e+11 95384
    ## - num_imgs                       1  597137934 5.2278e+11 95385
    ## - max_positive_polarity          1  603205041 5.2278e+11 95385
    ## - data_channel_is_entertainment  1  963296335 5.2314e+11 95388
    ## - self_reference_avg_sharess     1 1304852685 5.2349e+11 95392
    ## - kw_min_avg                     1 1597257589 5.2378e+11 95395
    ## - kw_max_avg                     1 2564491900 5.2475e+11 95404
    ## - kw_avg_avg                     1 4957582308 5.2714e+11 95428
    ## 
    ## Step:  AIC=95380.2
    ## shares ~ n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_hrefs + num_imgs + data_channel_is_entertainment + kw_min_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + max_positive_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_hrefs                      1  138229912 5.2246e+11 95380
    ## - abs_title_subjectivity         1  157784518 5.2248e+11 95380
    ## <none>                                        5.2233e+11 95380
    ## - title_sentiment_polarity       1  218028889 5.2254e+11 95380
    ## - rate_positive_words            1  227422118 5.2255e+11 95380
    ## - global_subjectivity            1  254626259 5.2258e+11 95381
    ## - kw_min_max                     1  257756258 5.2258e+11 95381
    ## - n_non_stop_words               1  266560782 5.2259e+11 95381
    ## - n_unique_tokens                1  267325979 5.2259e+11 95381
    ## - n_tokens_title                 1  356796041 5.2268e+11 95382
    ## - global_sentiment_polarity      1  409599359 5.2274e+11 95382
    ## - global_rate_negative_words     1  532459099 5.2286e+11 95383
    ## - self_reference_max_shares      1  564613143 5.2289e+11 95384
    ## - max_positive_polarity          1  593735396 5.2292e+11 95384
    ## - num_imgs                       1  613848423 5.2294e+11 95384
    ## - data_channel_is_entertainment  1  839691094 5.2317e+11 95387
    ## - self_reference_avg_sharess     1 1313391434 5.2364e+11 95391
    ## - kw_min_avg                     1 1785978571 5.2411e+11 95396
    ## - kw_max_avg                     1 3084960806 5.2541e+11 95409
    ## - kw_avg_avg                     1 6204565385 5.2853e+11 95439
    ## 
    ## Step:  AIC=95379.57
    ## shares ~ n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_imgs + data_channel_is_entertainment + kw_min_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + max_positive_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - abs_title_subjectivity         1  159873987 5.2262e+11 95379
    ## <none>                                        5.2246e+11 95380
    ## - n_non_stop_words               1  209951713 5.2267e+11 95380
    ## - n_unique_tokens                1  210612999 5.2267e+11 95380
    ## - title_sentiment_polarity       1  228361916 5.2269e+11 95380
    ## - rate_positive_words            1  241788960 5.2271e+11 95380
    ## - kw_min_max                     1  268658903 5.2273e+11 95380
    ## - global_subjectivity            1  304083095 5.2277e+11 95381
    ## - n_tokens_title                 1  336217137 5.2280e+11 95381
    ## - global_sentiment_polarity      1  434158201 5.2290e+11 95382
    ## - self_reference_max_shares      1  522307955 5.2299e+11 95383
    ## - global_rate_negative_words     1  586074369 5.2305e+11 95383
    ## - max_positive_polarity          1  668844923 5.2313e+11 95384
    ## - num_imgs                       1  795987061 5.2326e+11 95385
    ## - data_channel_is_entertainment  1  851692192 5.2332e+11 95386
    ## - self_reference_avg_sharess     1 1255931659 5.2372e+11 95390
    ## - kw_min_avg                     1 1784488254 5.2425e+11 95395
    ## - kw_max_avg                     1 3122284189 5.2559e+11 95408
    ## - kw_avg_avg                     1 6343063950 5.2881e+11 95440
    ## 
    ## Step:  AIC=95379.15
    ## shares ~ n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_imgs + data_channel_is_entertainment + kw_min_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + max_positive_polarity + title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - title_sentiment_polarity       1  157120019 5.2278e+11 95379
    ## - n_non_stop_words               1  200483023 5.2282e+11 95379
    ## - n_unique_tokens                1  201034938 5.2282e+11 95379
    ## <none>                                        5.2262e+11 95379
    ## - rate_positive_words            1  262620255 5.2289e+11 95380
    ## - n_tokens_title                 1  274483818 5.2290e+11 95380
    ## - kw_min_max                     1  277060047 5.2290e+11 95380
    ## - global_subjectivity            1  320189755 5.2294e+11 95380
    ## - global_sentiment_polarity      1  439617444 5.2306e+11 95381
    ## - self_reference_max_shares      1  522938704 5.2315e+11 95382
    ## - global_rate_negative_words     1  647149658 5.2327e+11 95384
    ## - max_positive_polarity          1  694828079 5.2332e+11 95384
    ## - num_imgs                       1  793177428 5.2342e+11 95385
    ## - data_channel_is_entertainment  1  858747198 5.2348e+11 95386
    ## - self_reference_avg_sharess     1 1262370110 5.2389e+11 95390
    ## - kw_min_avg                     1 1773439843 5.2440e+11 95395
    ## - kw_max_avg                     1 3132214153 5.2576e+11 95408
    ## - kw_avg_avg                     1 6337884294 5.2896e+11 95440
    ## 
    ## Step:  AIC=95378.7
    ## shares ~ n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     num_imgs + data_channel_is_entertainment + kw_min_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + max_positive_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - n_non_stop_words               1  186692090 5.2297e+11 95379
    ## - n_unique_tokens                1  187210216 5.2297e+11 95379
    ## <none>                                        5.2278e+11 95379
    ## - rate_positive_words            1  273286992 5.2305e+11 95379
    ## - kw_min_max                     1  281116861 5.2306e+11 95379
    ## - n_tokens_title                 1  283849523 5.2306e+11 95380
    ## - global_subjectivity            1  303297507 5.2308e+11 95380
    ## - global_sentiment_polarity      1  359680875 5.2314e+11 95380
    ## - self_reference_max_shares      1  516820555 5.2330e+11 95382
    ## - global_rate_negative_words     1  637766409 5.2342e+11 95383
    ## - max_positive_polarity          1  683180052 5.2346e+11 95383
    ## - num_imgs                       1  801615039 5.2358e+11 95385
    ## - data_channel_is_entertainment  1  862845548 5.2364e+11 95385
    ## - self_reference_avg_sharess     1 1258867708 5.2404e+11 95389
    ## - kw_min_avg                     1 1783177943 5.2456e+11 95394
    ## - kw_max_avg                     1 3174471256 5.2596e+11 95408
    ## - kw_avg_avg                     1 6419308772 5.2920e+11 95440
    ## 
    ## Step:  AIC=95378.55
    ## shares ~ n_tokens_title + n_unique_tokens + num_imgs + data_channel_is_entertainment + 
    ##     kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + max_positive_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - n_unique_tokens                1    5004389 5.2297e+11 95377
    ## <none>                                        5.2297e+11 95379
    ## - n_tokens_title                 1  269666053 5.2324e+11 95379
    ## - kw_min_max                     1  277559755 5.2325e+11 95379
    ## - global_sentiment_polarity      1  279333667 5.2325e+11 95379
    ## - rate_positive_words            1  295614807 5.2326e+11 95379
    ## - global_subjectivity            1  330051492 5.2330e+11 95380
    ## - max_positive_polarity          1  514528137 5.2348e+11 95382
    ## - self_reference_max_shares      1  539095989 5.2351e+11 95382
    ## - global_rate_negative_words     1  600483373 5.2357e+11 95382
    ## - num_imgs                       1  634520894 5.2360e+11 95383
    ## - data_channel_is_entertainment  1  799481669 5.2377e+11 95384
    ## - self_reference_avg_sharess     1 1305784374 5.2427e+11 95389
    ## - kw_min_avg                     1 1867952305 5.2484e+11 95395
    ## - kw_max_avg                     1 3423805566 5.2639e+11 95410
    ## - kw_avg_avg                     1 7043775856 5.3001e+11 95446
    ## 
    ## Step:  AIC=95376.6
    ## shares ~ n_tokens_title + num_imgs + data_channel_is_entertainment + 
    ##     kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + max_positive_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## <none>                                        5.2297e+11 95377
    ## - n_tokens_title                 1  268630859 5.2324e+11 95377
    ## - kw_min_max                     1  277612564 5.2325e+11 95377
    ## - global_sentiment_polarity      1  277690771 5.2325e+11 95377
    ## - rate_positive_words            1  298734418 5.2327e+11 95378
    ## - global_subjectivity            1  329164117 5.2330e+11 95378
    ## - max_positive_polarity          1  512220894 5.2348e+11 95380
    ## - self_reference_max_shares      1  540211725 5.2351e+11 95380
    ## - global_rate_negative_words     1  602177729 5.2357e+11 95381
    ## - num_imgs                       1  647468613 5.2362e+11 95381
    ## - data_channel_is_entertainment  1  796864985 5.2377e+11 95382
    ## - self_reference_avg_sharess     1 1308066188 5.2428e+11 95388
    ## - kw_min_avg                     1 1864057946 5.2484e+11 95393
    ## - kw_max_avg                     1 3419678849 5.2639e+11 95408
    ## - kw_avg_avg                     1 7038909771 5.3001e+11 95444

``` r
#summary the model
summary(lm.step)
```

    ## 
    ## Call:
    ## lm(formula = shares ~ n_tokens_title + num_imgs + data_channel_is_entertainment + 
    ##     kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_max_shares + 
    ##     self_reference_avg_sharess + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + max_positive_polarity, 
    ##     data = train)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -20074  -2244  -1211    -45 435700 
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   -2.167e+03  1.201e+03  -1.804 0.071311 .  
    ## n_tokens_title                 1.094e+02  6.721e+01   1.628 0.103646    
    ## num_imgs                       4.475e+01  1.771e+01   2.527 0.011533 *  
    ## data_channel_is_entertainment -1.057e+03  3.771e+02  -2.803 0.005075 ** 
    ## kw_min_max                    -4.468e-03  2.700e-03  -1.655 0.098045 .  
    ## kw_min_avg                    -7.286e-01  1.699e-01  -4.288 1.84e-05 ***
    ## kw_max_avg                    -2.855e-01  4.915e-02  -5.808 6.72e-09 ***
    ## kw_avg_avg                     2.171e+00  2.606e-01   8.332  < 2e-16 ***
    ## self_reference_max_shares     -1.699e-02  7.361e-03  -2.308 0.021025 *  
    ## self_reference_avg_sharess     4.168e-02  1.161e-02   3.592 0.000331 ***
    ## global_subjectivity            3.184e+03  1.767e+03   1.802 0.071635 .  
    ## global_sentiment_polarity     -4.096e+03  2.475e+03  -1.655 0.097998 .  
    ## global_rate_negative_words    -4.847e+04  1.989e+04  -2.437 0.014841 *  
    ## rate_positive_words           -2.299e+03  1.339e+03  -1.716 0.086131 .  
    ## max_positive_polarity          1.739e+03  7.737e+02   2.248 0.024640 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10070 on 5158 degrees of freedom
    ## Multiple R-squared:  0.0307, Adjusted R-squared:  0.02807 
    ## F-statistic: 11.67 on 14 and 5158 DF,  p-value: < 2.2e-16

Calculate the predicted mean square error on the train set:

``` r
train.pred <- predict(lm.step, train[,-52])
mean((train.pred - train$shares)^2)
```

    ## [1] 101096596

So, the predicted mean square error on the training dataset is
264157333.

### On test set

Calculate the predicted mean square error on the test set:

``` r
test.pred <- predict(lm.step, test[,-52])
mean((test.pred - test$shares)^2)
```

    ## [1] 74304954

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
