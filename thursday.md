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

    ## [1] 7267

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

    ## [1] 5086

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
    ##   timedelta                   Min.   : 14.0      1st Qu.:168.0     
    ## n_tokens_title                Min.   : 3.00      1st Qu.: 9.00     
    ## n_tokens_content              Min.   :   0.0     1st Qu.: 244.0    
    ## n_unique_tokens               Min.   :0.0000     1st Qu.:0.4708    
    ## n_non_stop_words              Min.   :0.0000     1st Qu.:1.0000    
    ## n_non_stop_unique_tokens      Min.   :0.0000     1st Qu.:0.6254    
    ##   num_hrefs                   Min.   :  0.00     1st Qu.:  4.00    
    ## num_self_hrefs                Min.   : 0.00      1st Qu.: 1.00     
    ##    num_imgs                   Min.   :  0.000    1st Qu.:  1.000   
    ##   num_videos                  Min.   : 0.000     1st Qu.: 0.000    
    ## average_token_length          Min.   :0.000      1st Qu.:4.484     
    ##  num_keywords                 Min.   : 1.000     1st Qu.: 6.000    
    ## data_channel_is_lifestyle     Min.   :0.0000     1st Qu.:0.0000    
    ## data_channel_is_entertainment Min.   :0.0000     1st Qu.:0.0000    
    ## data_channel_is_bus           Min.   :0.0000     1st Qu.:0.0000    
    ## data_channel_is_socmed        Min.   :0.00000    1st Qu.:0.00000   
    ## data_channel_is_tech          Min.   :0.0000     1st Qu.:0.0000    
    ## data_channel_is_world         Min.   :0.0000     1st Qu.:0.0000    
    ##   kw_min_min                  Min.   : -1.00     1st Qu.: -1.00    
    ##   kw_max_min                  Min.   :    0      1st Qu.:  444     
    ##   kw_avg_min                  Min.   :   -1.0    1st Qu.:  143.0   
    ##   kw_min_max                  Min.   :     0     1st Qu.:     0    
    ##   kw_max_max                  Min.   : 11100     1st Qu.:843300    
    ##   kw_avg_max                  Min.   :  3120     1st Qu.:172692    
    ##   kw_min_avg                  Min.   :   0       1st Qu.:   0      
    ##   kw_max_avg                  Min.   :  2241     1st Qu.:  3573    
    ##   kw_avg_avg                  Min.   :  489      1st Qu.: 2377     
    ## self_reference_min_shares     Min.   :     0.0   1st Qu.:   600.2  
    ## self_reference_max_shares     Min.   :     0     1st Qu.:  1000    
    ## self_reference_avg_sharess    Min.   :     0.0   1st Qu.:   940.8  
    ##     LDA_00                    Min.   :0.01818    1st Qu.:0.02521   
    ##     LDA_01                    Min.   :0.01818    1st Qu.:0.02501   
    ##     LDA_02                    Min.   :0.01818    1st Qu.:0.02857   
    ##     LDA_03                    Min.   :0.01818    1st Qu.:0.02857   
    ##     LDA_04                    Min.   :0.01819    1st Qu.:0.02857   
    ## global_subjectivity           Min.   :0.0000     1st Qu.:0.3950    
    ## global_sentiment_polarity     Min.   :-0.3429    1st Qu.: 0.0560   
    ## global_rate_positive_words    Min.   :0.00000    1st Qu.:0.02866   
    ## global_rate_negative_words    Min.   :0.00000    1st Qu.:0.00939   
    ## rate_positive_words           Min.   :0.0000     1st Qu.:0.6000    
    ## rate_negative_words           Min.   :0.0000     1st Qu.:0.1818    
    ## avg_positive_polarity         Min.   :0.0000     1st Qu.:0.3034    
    ## min_positive_polarity         Min.   :0.00000    1st Qu.:0.05000   
    ## max_positive_polarity         Min.   :0.0000     1st Qu.:0.6000    
    ## avg_negative_polarity         Min.   :-1.0000    1st Qu.:-0.3250   
    ## min_negative_polarity         Min.   :-1.0000    1st Qu.:-0.7000   
    ## max_negative_polarity         Min.   :-1.0000    1st Qu.:-0.1250   
    ## title_subjectivity            Min.   :0.0000     1st Qu.:0.0000    
    ## title_sentiment_polarity      Min.   :-1.00000   1st Qu.: 0.00000  
    ## abs_title_subjectivity        Min.   :0.0000     1st Qu.:0.1667    
    ## abs_title_sentiment_polarity  Min.   :0.000000   1st Qu.:0.000000  
    ##     shares                    Min.   :     5     1st Qu.:   903    
    ##                                                                    
    ##   timedelta                   Median :336.0      Mean   :357.3     
    ## n_tokens_title                Median :10.00      Mean   :10.34     
    ## n_tokens_content              Median : 397.0     Mean   : 541.1    
    ## n_unique_tokens               Median :0.5413     Mean   :0.5309    
    ## n_non_stop_words              Median :1.0000     Mean   :0.9693    
    ## n_non_stop_unique_tokens      Median :0.6921     Mean   :0.6729    
    ##   num_hrefs                   Median :  7.00     Mean   : 10.57    
    ## num_self_hrefs                Median : 2.00      Mean   : 3.14     
    ##    num_imgs                   Median :  1.000    Mean   :  4.549   
    ##   num_videos                  Median : 0.000     Mean   : 1.164    
    ## average_token_length          Median :4.673      Mean   :4.549     
    ##  num_keywords                 Median : 7.000     Mean   : 7.154    
    ## data_channel_is_lifestyle     Median :0.0000     Mean   :0.0466    
    ## data_channel_is_entertainment Median :0.0000     Mean   :0.1683    
    ## data_channel_is_bus           Median :0.0000     Mean   :0.1677    
    ## data_channel_is_socmed        Median :0.00000    Mean   :0.06626   
    ## data_channel_is_tech          Median :0.0000     Mean   :0.1773    
    ## data_channel_is_world         Median :0.0000     Mean   :0.2216    
    ##   kw_min_min                  Median : -1.00     Mean   : 27.12    
    ##   kw_max_min                  Median :  652      Mean   : 1142     
    ##   kw_avg_min                  Median :  239.0    Mean   :  310.9   
    ##   kw_min_max                  Median :  1400     Mean   : 14267    
    ##   kw_max_max                  Median :843300     Mean   :751353    
    ##   kw_avg_max                  Median :245478     Mean   :261997    
    ##   kw_min_avg                  Median : 999       Mean   :1106      
    ##   kw_max_avg                  Median :  4350     Mean   :  5621    
    ##   kw_avg_avg                  Median : 2861      Mean   : 3124     
    ## self_reference_min_shares     Median :  1200.0   Mean   :  3992.6  
    ## self_reference_max_shares     Median :  2800     Mean   :  9969    
    ## self_reference_avg_sharess    Median :  2200.0   Mean   :  6303.1  
    ##     LDA_00                    Median :0.03395    Mean   :0.19462   
    ##     LDA_01                    Median :0.03334    Mean   :0.13516   
    ##     LDA_02                    Median :0.04005    Mean   :0.21914   
    ##     LDA_03                    Median :0.04000    Mean   :0.22147   
    ##     LDA_04                    Median :0.05000    Mean   :0.22961   
    ## global_subjectivity           Median :0.4523     Mean   :0.4424    
    ## global_sentiment_polarity     Median : 0.1184    Mean   : 0.1192   
    ## global_rate_positive_words    Median :0.03905    Mean   :0.03941   
    ## global_rate_negative_words    Median :0.01530    Mean   :0.01655   
    ## rate_positive_words           Median :0.7111     Mean   :0.6814    
    ## rate_negative_words           Median :0.2792     Mean   :0.2879    
    ## avg_positive_polarity         Median :0.3574     Mean   :0.3517    
    ## min_positive_polarity         Median :0.10000    Mean   :0.09635   
    ## max_positive_polarity         Median :0.8000     Mean   :0.7492    
    ## avg_negative_polarity         Median :-0.2503    Mean   :-0.2553   
    ## min_negative_polarity         Median :-0.5000    Mean   :-0.5163   
    ## max_negative_polarity         Median :-0.1000    Mean   :-0.1056   
    ## title_subjectivity            Median :0.1667     Mean   :0.2876    
    ## title_sentiment_polarity      Median : 0.00000   Mean   : 0.07269  
    ## abs_title_subjectivity        Median :0.5000     Mean   :0.3426    
    ## abs_title_sentiment_polarity  Median :0.008965   Mean   :0.155637  
    ##     shares                    Median :  1400     Mean   :  3101    
    ##                                                                    
    ##   timedelta                   3rd Qu.:546.0      Max.   :728.0     
    ## n_tokens_title                3rd Qu.:12.00      Max.   :18.00     
    ## n_tokens_content              3rd Qu.: 701.8     Max.   :6159.0    
    ## n_unique_tokens               3rd Qu.:0.6088     Max.   :0.9545    
    ## n_non_stop_words              3rd Qu.:1.0000     Max.   :1.0000    
    ## n_non_stop_unique_tokens      3rd Qu.:0.7539     Max.   :1.0000    
    ##   num_hrefs                   3rd Qu.: 13.00     Max.   :140.00    
    ## num_self_hrefs                3rd Qu.: 4.00      Max.   :56.00     
    ##    num_imgs                   3rd Qu.:  4.000    Max.   :100.000   
    ##   num_videos                  3rd Qu.: 1.000     Max.   :74.000    
    ## average_token_length          3rd Qu.:4.865      Max.   :6.198     
    ##  num_keywords                 3rd Qu.: 9.000     Max.   :10.000    
    ## data_channel_is_lifestyle     3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_entertainment 3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_bus           3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_socmed        3rd Qu.:0.00000    Max.   :1.00000   
    ## data_channel_is_tech          3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_world         3rd Qu.:0.0000     Max.   :1.0000    
    ##   kw_min_min                  3rd Qu.:  4.00     Max.   :377.00    
    ##   kw_max_min                  3rd Qu.: 1000      Max.   :80400     
    ##   kw_avg_min                  3rd Qu.:  355.5    Max.   :13744.8   
    ##   kw_min_max                  3rd Qu.:  7900     Max.   :843300    
    ##   kw_max_max                  3rd Qu.:843300     Max.   :843300    
    ##   kw_avg_max                  3rd Qu.:335403     Max.   :843300    
    ##   kw_min_avg                  3rd Qu.:2070       Max.   :3610      
    ##   kw_max_avg                  3rd Qu.:  6033     Max.   :117700    
    ##   kw_avg_avg                  3rd Qu.: 3589      Max.   :19429     
    ## self_reference_min_shares     3rd Qu.:  2500.0   Max.   :690400.0  
    ## self_reference_max_shares     3rd Qu.:  7700     Max.   :690400    
    ## self_reference_avg_sharess    3rd Qu.:  5062.5   Max.   :690400.0  
    ##     LDA_00                    3rd Qu.:0.27073    Max.   :0.92000   
    ##     LDA_01                    3rd Qu.:0.13377    Max.   :0.91997   
    ##     LDA_02                    3rd Qu.:0.33323    Max.   :0.92000   
    ##     LDA_03                    3rd Qu.:0.37092    Max.   :0.91992   
    ##     LDA_04                    3rd Qu.:0.39269    Max.   :0.92645   
    ## global_subjectivity           3rd Qu.:0.5084     Max.   :0.9375    
    ## global_sentiment_polarity     3rd Qu.: 0.1782    Max.   : 0.7278   
    ## global_rate_positive_words    3rd Qu.:0.05000    Max.   :0.15278   
    ## global_rate_negative_words    3rd Qu.:0.02168    Max.   :0.10303   
    ## rate_positive_words           3rd Qu.:0.8000     Max.   :1.0000    
    ## rate_negative_words           3rd Qu.:0.3846     Max.   :1.0000    
    ## avg_positive_polarity         3rd Qu.:0.4112     Max.   :0.8500    
    ## min_positive_polarity         3rd Qu.:0.10000    Max.   :0.80000   
    ## max_positive_polarity         3rd Qu.:1.0000     Max.   :1.0000    
    ## avg_negative_polarity         3rd Qu.:-0.1833    Max.   : 0.0000   
    ## min_negative_polarity         3rd Qu.:-0.3000    Max.   : 0.0000   
    ## max_negative_polarity         3rd Qu.:-0.0500    Max.   : 0.0000   
    ## title_subjectivity            3rd Qu.:0.5000     Max.   :1.0000    
    ## title_sentiment_polarity      3rd Qu.: 0.14256   Max.   : 1.00000  
    ## abs_title_subjectivity        3rd Qu.:0.5000     Max.   :0.5000    
    ## abs_title_sentiment_polarity  3rd Qu.:0.250000   Max.   :1.000000  
    ##     shares                    3rd Qu.:  2600     Max.   :298400

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
    ##           Mean of squared residuals: 72943183
    ##                     % Var explained: -3.28

``` r
#variable importance measures
importance(rf)
```

    ##                                  %IncMSE IncNodePurity
    ## timedelta                      1.8151334    8323049419
    ## n_tokens_title                -1.2559261   11436840037
    ## n_tokens_content               5.4694961    4516372163
    ## n_unique_tokens                4.8420654    6022939364
    ## n_non_stop_words               3.9383626    4372971151
    ## n_non_stop_unique_tokens       7.7518386    4864706294
    ## num_hrefs                      3.3967724    4369902222
    ## num_self_hrefs                 4.4076662    3881294830
    ## num_imgs                       2.2103929    5691434130
    ## num_videos                     1.9312397    2825772766
    ## average_token_length          10.7681160    4492588416
    ## num_keywords                   5.0112015    2164244501
    ## data_channel_is_lifestyle      4.0214237     448119018
    ## data_channel_is_entertainment  1.3347818     219994362
    ## data_channel_is_bus            2.2880749    2570858450
    ## data_channel_is_socmed        -0.3742647     103536236
    ## data_channel_is_tech           5.3133019     347164591
    ## data_channel_is_world          1.8979251    2390013481
    ## kw_min_min                     4.4956628     483175731
    ## kw_max_min                     3.0671788    6002889883
    ## kw_avg_min                     8.5141072    5033879247
    ## kw_min_max                     2.6253029    2548619452
    ## kw_max_max                     3.5238635     372750542
    ## kw_avg_max                     3.2401777    8720446982
    ## kw_min_avg                     6.1975982    2641374169
    ## kw_max_avg                     3.3526457   10758219317
    ## kw_avg_avg                     5.4211155   12583584008
    ## self_reference_min_shares      3.6559011    5889752523
    ## self_reference_max_shares      5.0878624   16502988607
    ## self_reference_avg_sharess     4.7403036   18648884619
    ## LDA_00                         5.5294771   15532314259
    ## LDA_01                         3.5330208    6538910545
    ## LDA_02                         3.6853020    7048346934
    ## LDA_03                         3.4678818    7954176756
    ## LDA_04                         2.7903037    7195811576
    ## global_subjectivity            3.5753076    9772234259
    ## global_sentiment_polarity      3.2581805   10332651317
    ## global_rate_positive_words     2.4035535    6022953443
    ## global_rate_negative_words     5.3372115    4540488252
    ## rate_positive_words            2.7765299    3930405432
    ## rate_negative_words            3.7844240    2971859416
    ## avg_positive_polarity          4.4280174   19296558577
    ## min_positive_polarity          3.6970096    1503290379
    ## max_positive_polarity          2.7498258    3561302975
    ## avg_negative_polarity          4.0839545   15014421775
    ## min_negative_polarity          3.6250466    8165630904
    ## max_negative_polarity          4.4060945   12601992573
    ## title_subjectivity             2.5531444    3427020247
    ## title_sentiment_polarity       1.8906939   10486590498
    ## abs_title_subjectivity         0.9965748    3929631167
    ## abs_title_sentiment_polarity   1.9832207    4364622556

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

    ## [1] 15203361

So, the predicted mean square error on the training dataset is 68724773.

### On test set

``` r
rf.test <- predict(rf, newdata = test[,-52])
mean((test$shares-rf.test)^2)
```

    ## [1] 126919798

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

    ## Start:  AIC=91764.69
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
    ## Step:  AIC=91764.69
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
    ## Step:  AIC=91764.69
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
    ## - min_positive_polarity          1     616424 3.4170e+11 91763
    ## - max_positive_polarity          1    1080170 3.4170e+11 91763
    ## - title_subjectivity             1    1745425 3.4170e+11 91763
    ## - abs_title_subjectivity         1    4800178 3.4170e+11 91763
    ## - data_channel_is_lifestyle      1    4916389 3.4170e+11 91763
    ## - num_self_hrefs                 1    5826813 3.4170e+11 91763
    ## - n_tokens_content               1    7610394 3.4170e+11 91763
    ## - title_sentiment_polarity       1   11129145 3.4171e+11 91763
    ## - LDA_01                         1   16845758 3.4171e+11 91763
    ## - n_non_stop_words               1   16994232 3.4171e+11 91763
    ## - num_hrefs                      1   22146830 3.4172e+11 91763
    ## - data_channel_is_tech           1   24022604 3.4172e+11 91763
    ## - global_rate_positive_words     1   25043672 3.4172e+11 91763
    ## - kw_max_min                     1   25798741 3.4172e+11 91763
    ## - global_rate_negative_words     1   27648749 3.4172e+11 91763
    ## - kw_avg_min                     1   31497934 3.4173e+11 91763
    ## - kw_max_max                     1   37592215 3.4173e+11 91763
    ## - num_videos                     1   38217224 3.4174e+11 91763
    ## - avg_positive_polarity          1   39295405 3.4174e+11 91763
    ## - LDA_00                         1   39344969 3.4174e+11 91763
    ## - kw_avg_max                     1   42619921 3.4174e+11 91763
    ## - LDA_03                         1   52481667 3.4175e+11 91763
    ## - avg_negative_polarity          1   54834558 3.4175e+11 91764
    ## - data_channel_is_bus            1   57317566 3.4175e+11 91764
    ## - kw_min_min                     1   61307516 3.4176e+11 91764
    ## - abs_title_sentiment_polarity   1   64322813 3.4176e+11 91764
    ## - num_keywords                   1   70443906 3.4177e+11 91764
    ## - average_token_length           1   89219796 3.4179e+11 91764
    ## - data_channel_is_socmed         1   92443819 3.4179e+11 91764
    ## - global_sentiment_polarity      1  106006757 3.4180e+11 91764
    ## - min_negative_polarity          1  108783187 3.4181e+11 91764
    ## - num_imgs                       1  119160721 3.4182e+11 91764
    ## - data_channel_is_world          1  131711420 3.4183e+11 91765
    ## - kw_min_max                     1  133489862 3.4183e+11 91765
    ## <none>                                        3.4170e+11 91765
    ## - rate_positive_words            1  143725466 3.4184e+11 91765
    ## - timedelta                      1  158085109 3.4186e+11 91765
    ## - max_negative_polarity          1  162702054 3.4186e+11 91765
    ## - data_channel_is_entertainment  1  173327846 3.4187e+11 91765
    ## - n_unique_tokens                1  209256997 3.4191e+11 91766
    ## - n_non_stop_unique_tokens       1  229931696 3.4193e+11 91766
    ## - LDA_02                         1  274740649 3.4197e+11 91767
    ## - n_tokens_title                 1  485258239 3.4218e+11 91770
    ## - global_subjectivity            1  681812757 3.4238e+11 91773
    ## - kw_min_avg                     1 1216421125 3.4291e+11 91781
    ## - kw_max_avg                     1 1218162416 3.4292e+11 91781
    ## - self_reference_min_shares      1 1481815623 3.4318e+11 91785
    ## - kw_avg_avg                     1 2552023066 3.4425e+11 91801
    ## - self_reference_avg_sharess     1 2568208055 3.4427e+11 91801
    ## - self_reference_max_shares      1 4154500545 3.4585e+11 91824
    ## 
    ## Step:  AIC=91762.7
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
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - max_positive_polarity          1     744429 3.4170e+11 91761
    ## - title_subjectivity             1    1710225 3.4170e+11 91761
    ## - abs_title_subjectivity         1    4844521 3.4170e+11 91761
    ## - data_channel_is_lifestyle      1    4957609 3.4170e+11 91761
    ## - num_self_hrefs                 1    5844859 3.4170e+11 91761
    ## - n_tokens_content               1    7649917 3.4171e+11 91761
    ## - title_sentiment_polarity       1   11021399 3.4171e+11 91761
    ## - n_non_stop_words               1   16458706 3.4171e+11 91761
    ## - LDA_01                         1   16953854 3.4171e+11 91761
    ## - num_hrefs                      1   21887476 3.4172e+11 91761
    ## - data_channel_is_tech           1   24216733 3.4172e+11 91761
    ## - global_rate_positive_words     1   25276202 3.4172e+11 91761
    ## - kw_max_min                     1   26020319 3.4172e+11 91761
    ## - global_rate_negative_words     1   27085367 3.4172e+11 91761
    ## - kw_avg_min                     1   31690161 3.4173e+11 91761
    ## - kw_max_max                     1   37597309 3.4174e+11 91761
    ## - num_videos                     1   38136032 3.4174e+11 91761
    ## - LDA_00                         1   39034198 3.4174e+11 91761
    ## - kw_avg_max                     1   42719655 3.4174e+11 91761
    ## - avg_positive_polarity          1   45179738 3.4174e+11 91761
    ## - LDA_03                         1   53029252 3.4175e+11 91761
    ## - avg_negative_polarity          1   55548089 3.4175e+11 91762
    ## - data_channel_is_bus            1   57594904 3.4176e+11 91762
    ## - kw_min_min                     1   61487796 3.4176e+11 91762
    ## - abs_title_sentiment_polarity   1   64172622 3.4176e+11 91762
    ## - num_keywords                   1   70459342 3.4177e+11 91762
    ## - average_token_length           1   92525706 3.4179e+11 91762
    ## - data_channel_is_socmed         1   93209632 3.4179e+11 91762
    ## - min_negative_polarity          1  108356193 3.4181e+11 91762
    ## - global_sentiment_polarity      1  109859746 3.4181e+11 91762
    ## - num_imgs                       1  118961347 3.4182e+11 91762
    ## - data_channel_is_world          1  131522013 3.4183e+11 91763
    ## - kw_min_max                     1  133430599 3.4183e+11 91763
    ## <none>                                        3.4170e+11 91763
    ## - rate_positive_words            1  143109536 3.4184e+11 91763
    ## - timedelta                      1  157893456 3.4186e+11 91763
    ## - max_negative_polarity          1  163655150 3.4186e+11 91763
    ## - data_channel_is_entertainment  1  173910641 3.4187e+11 91763
    ## - n_unique_tokens                1  221715970 3.4192e+11 91764
    ## - n_non_stop_unique_tokens       1  239766498 3.4194e+11 91764
    ## - LDA_02                         1  275430204 3.4197e+11 91765
    ## - n_tokens_title                 1  484643804 3.4218e+11 91768
    ## - global_subjectivity            1  684985777 3.4238e+11 91771
    ## - kw_max_avg                     1 1220128201 3.4292e+11 91779
    ## - kw_min_avg                     1 1220311700 3.4292e+11 91779
    ## - self_reference_min_shares      1 1486625483 3.4318e+11 91783
    ## - kw_avg_avg                     1 2555230627 3.4425e+11 91799
    ## - self_reference_avg_sharess     1 2574430176 3.4427e+11 91799
    ## - self_reference_max_shares      1 4157966672 3.4586e+11 91822
    ## 
    ## Step:  AIC=91760.71
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
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - title_subjectivity             1    1675447 3.4170e+11 91759
    ## - abs_title_subjectivity         1    4818433 3.4170e+11 91759
    ## - data_channel_is_lifestyle      1    4947515 3.4170e+11 91759
    ## - num_self_hrefs                 1    6023732 3.4170e+11 91759
    ## - n_tokens_content               1    7277060 3.4171e+11 91759
    ## - title_sentiment_polarity       1   10966164 3.4171e+11 91759
    ## - n_non_stop_words               1   16216366 3.4171e+11 91759
    ## - LDA_01                         1   16984888 3.4172e+11 91759
    ## - num_hrefs                      1   22491796 3.4172e+11 91759
    ## - data_channel_is_tech           1   24164346 3.4172e+11 91759
    ## - global_rate_positive_words     1   24624908 3.4172e+11 91759
    ## - kw_max_min                     1   26127702 3.4172e+11 91759
    ## - global_rate_negative_words     1   28049465 3.4173e+11 91759
    ## - kw_avg_min                     1   31790632 3.4173e+11 91759
    ## - kw_max_max                     1   37608361 3.4174e+11 91759
    ## - num_videos                     1   38905201 3.4174e+11 91759
    ## - LDA_00                         1   39250003 3.4174e+11 91759
    ## - kw_avg_max                     1   42704322 3.4174e+11 91759
    ## - avg_positive_polarity          1   50425106 3.4175e+11 91759
    ## - LDA_03                         1   52922243 3.4175e+11 91759
    ## - avg_negative_polarity          1   54974583 3.4175e+11 91760
    ## - data_channel_is_bus            1   57714996 3.4176e+11 91760
    ## - kw_min_min                     1   61455576 3.4176e+11 91760
    ## - abs_title_sentiment_polarity   1   63988459 3.4176e+11 91760
    ## - num_keywords                   1   70083989 3.4177e+11 91760
    ## - average_token_length           1   92356477 3.4179e+11 91760
    ## - data_channel_is_socmed         1   93496294 3.4179e+11 91760
    ## - min_negative_polarity          1  108676038 3.4181e+11 91760
    ## - global_sentiment_polarity      1  109240392 3.4181e+11 91760
    ## - num_imgs                       1  119520985 3.4182e+11 91760
    ## - data_channel_is_world          1  131904735 3.4183e+11 91761
    ## - kw_min_max                     1  133447793 3.4183e+11 91761
    ## <none>                                        3.4170e+11 91761
    ## - rate_positive_words            1  146436105 3.4184e+11 91761
    ## - timedelta                      1  158024948 3.4186e+11 91761
    ## - max_negative_polarity          1  162931170 3.4186e+11 91761
    ## - data_channel_is_entertainment  1  173336001 3.4187e+11 91761
    ## - n_unique_tokens                1  224591626 3.4192e+11 91762
    ## - n_non_stop_unique_tokens       1  239580384 3.4194e+11 91762
    ## - LDA_02                         1  276424587 3.4197e+11 91763
    ## - n_tokens_title                 1  484755821 3.4218e+11 91766
    ## - global_subjectivity            1  687687959 3.4239e+11 91769
    ## - kw_min_avg                     1 1221126239 3.4292e+11 91777
    ## - kw_max_avg                     1 1221760255 3.4292e+11 91777
    ## - self_reference_min_shares      1 1487392921 3.4319e+11 91781
    ## - kw_avg_avg                     1 2557576985 3.4426e+11 91797
    ## - self_reference_avg_sharess     1 2576340286 3.4427e+11 91797
    ## - self_reference_max_shares      1 4160672356 3.4586e+11 91820
    ## 
    ## Step:  AIC=91758.74
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
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - abs_title_subjectivity         1    3582747 3.4170e+11 91757
    ## - data_channel_is_lifestyle      1    4880295 3.4170e+11 91757
    ## - num_self_hrefs                 1    5972332 3.4171e+11 91757
    ## - n_tokens_content               1    7336716 3.4171e+11 91757
    ## - title_sentiment_polarity       1   12190051 3.4171e+11 91757
    ## - n_non_stop_words               1   16009650 3.4172e+11 91757
    ## - LDA_01                         1   17151225 3.4172e+11 91757
    ## - num_hrefs                      1   22566944 3.4172e+11 91757
    ## - data_channel_is_tech           1   24065035 3.4172e+11 91757
    ## - global_rate_positive_words     1   24944914 3.4173e+11 91757
    ## - kw_max_min                     1   26091274 3.4173e+11 91757
    ## - global_rate_negative_words     1   28299102 3.4173e+11 91757
    ## - kw_avg_min                     1   31668951 3.4173e+11 91757
    ## - kw_max_max                     1   37416905 3.4174e+11 91757
    ## - num_videos                     1   39016732 3.4174e+11 91757
    ## - LDA_00                         1   39680668 3.4174e+11 91757
    ## - kw_avg_max                     1   42785368 3.4174e+11 91757
    ## - avg_positive_polarity          1   50143132 3.4175e+11 91757
    ## - LDA_03                         1   53266708 3.4175e+11 91758
    ## - avg_negative_polarity          1   54943528 3.4176e+11 91758
    ## - data_channel_is_bus            1   57945103 3.4176e+11 91758
    ## - kw_min_min                     1   61566244 3.4176e+11 91758
    ## - num_keywords                   1   69841838 3.4177e+11 91758
    ## - abs_title_sentiment_polarity   1   83162428 3.4178e+11 91758
    ## - average_token_length           1   92598170 3.4179e+11 91758
    ## - data_channel_is_socmed         1   93538265 3.4179e+11 91758
    ## - global_sentiment_polarity      1  108589120 3.4181e+11 91758
    ## - min_negative_polarity          1  108932976 3.4181e+11 91758
    ## - num_imgs                       1  119121228 3.4182e+11 91759
    ## - data_channel_is_world          1  131631478 3.4183e+11 91759
    ## - kw_min_max                     1  134162307 3.4183e+11 91759
    ## <none>                                        3.4170e+11 91759
    ## - rate_positive_words            1  146736935 3.4185e+11 91759
    ## - timedelta                      1  157314203 3.4186e+11 91759
    ## - max_negative_polarity          1  162789384 3.4186e+11 91759
    ## - data_channel_is_entertainment  1  173143276 3.4187e+11 91759
    ## - n_unique_tokens                1  224490780 3.4192e+11 91760
    ## - n_non_stop_unique_tokens       1  239350058 3.4194e+11 91760
    ## - LDA_02                         1  276194461 3.4198e+11 91761
    ## - n_tokens_title                 1  484315838 3.4218e+11 91764
    ## - global_subjectivity            1  688937869 3.4239e+11 91767
    ## - kw_min_avg                     1 1219992339 3.4292e+11 91775
    ## - kw_max_avg                     1 1222556374 3.4292e+11 91775
    ## - self_reference_min_shares      1 1486216160 3.4319e+11 91779
    ## - kw_avg_avg                     1 2557557691 3.4426e+11 91795
    ## - self_reference_avg_sharess     1 2574842560 3.4427e+11 91795
    ## - self_reference_max_shares      1 4159183902 3.4586e+11 91818
    ## 
    ## Step:  AIC=91756.79
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
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - data_channel_is_lifestyle      1    4779041 3.4171e+11 91755
    ## - num_self_hrefs                 1    6147609 3.4171e+11 91755
    ## - n_tokens_content               1    7573267 3.4171e+11 91755
    ## - title_sentiment_polarity       1   13038885 3.4172e+11 91755
    ## - n_non_stop_words               1   16068094 3.4172e+11 91755
    ## - LDA_01                         1   16819986 3.4172e+11 91755
    ## - num_hrefs                      1   22783030 3.4173e+11 91755
    ## - global_rate_positive_words     1   23308545 3.4173e+11 91755
    ## - data_channel_is_tech           1   23904310 3.4173e+11 91755
    ## - kw_max_min                     1   26648600 3.4173e+11 91755
    ## - global_rate_negative_words     1   28762993 3.4173e+11 91755
    ## - kw_avg_min                     1   32231999 3.4174e+11 91755
    ## - kw_max_max                     1   38188626 3.4174e+11 91755
    ## - num_videos                     1   38921318 3.4174e+11 91755
    ## - LDA_00                         1   40027603 3.4174e+11 91755
    ## - kw_avg_max                     1   42588722 3.4175e+11 91755
    ## - avg_positive_polarity          1   50547156 3.4175e+11 91756
    ## - LDA_03                         1   53605221 3.4176e+11 91756
    ## - avg_negative_polarity          1   55755463 3.4176e+11 91756
    ## - data_channel_is_bus            1   57762975 3.4176e+11 91756
    ## - kw_min_min                     1   62567750 3.4177e+11 91756
    ## - num_keywords                   1   69169640 3.4177e+11 91756
    ## - average_token_length           1   92072708 3.4180e+11 91756
    ## - data_channel_is_socmed         1   93596400 3.4180e+11 91756
    ## - abs_title_sentiment_polarity   1  106417404 3.4181e+11 91756
    ## - min_negative_polarity          1  109160895 3.4181e+11 91756
    ## - global_sentiment_polarity      1  110409354 3.4181e+11 91756
    ## - num_imgs                       1  119689853 3.4182e+11 91757
    ## - data_channel_is_world          1  131554554 3.4184e+11 91757
    ## - kw_min_max                     1  133848342 3.4184e+11 91757
    ## <none>                                        3.4170e+11 91757
    ## - rate_positive_words            1  147178938 3.4185e+11 91757
    ## - timedelta                      1  157198367 3.4186e+11 91757
    ## - max_negative_polarity          1  163569860 3.4187e+11 91757
    ## - data_channel_is_entertainment  1  172740856 3.4188e+11 91757
    ## - n_unique_tokens                1  226657597 3.4193e+11 91758
    ## - n_non_stop_unique_tokens       1  241550459 3.4195e+11 91758
    ## - LDA_02                         1  275890903 3.4198e+11 91759
    ## - n_tokens_title                 1  506076740 3.4221e+11 91762
    ## - global_subjectivity            1  685709726 3.4239e+11 91765
    ## - kw_max_avg                     1 1221492593 3.4293e+11 91773
    ## - kw_min_avg                     1 1222430426 3.4293e+11 91773
    ## - self_reference_min_shares      1 1487109260 3.4319e+11 91777
    ## - kw_avg_avg                     1 2556549217 3.4426e+11 91793
    ## - self_reference_avg_sharess     1 2576506948 3.4428e+11 91793
    ## - self_reference_max_shares      1 4162505995 3.4587e+11 91816
    ## 
    ## Step:  AIC=91754.86
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_entertainment + data_channel_is_bus + 
    ##     data_channel_is_socmed + data_channel_is_tech + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_self_hrefs                 1    5764360 3.4171e+11 91753
    ## - n_tokens_content               1    7768711 3.4172e+11 91753
    ## - LDA_01                         1   12300901 3.4172e+11 91753
    ## - title_sentiment_polarity       1   12873332 3.4172e+11 91753
    ## - n_non_stop_words               1   16084719 3.4172e+11 91753
    ## - data_channel_is_tech           1   21108718 3.4173e+11 91753
    ## - global_rate_positive_words     1   22826137 3.4173e+11 91753
    ## - num_hrefs                      1   23024156 3.4173e+11 91753
    ## - kw_max_min                     1   27378887 3.4174e+11 91753
    ## - global_rate_negative_words     1   28412979 3.4174e+11 91753
    ## - kw_avg_min                     1   33152069 3.4174e+11 91753
    ## - kw_max_max                     1   37694020 3.4175e+11 91753
    ## - kw_avg_max                     1   39686776 3.4175e+11 91753
    ## - num_videos                     1   39803056 3.4175e+11 91753
    ## - LDA_00                         1   43710315 3.4175e+11 91754
    ## - avg_positive_polarity          1   49895869 3.4176e+11 91754
    ## - LDA_03                         1   52447642 3.4176e+11 91754
    ## - avg_negative_polarity          1   55041262 3.4176e+11 91754
    ## - data_channel_is_bus            1   59000904 3.4177e+11 91754
    ## - kw_min_min                     1   63684655 3.4177e+11 91754
    ## - num_keywords                   1   68262094 3.4178e+11 91754
    ## - average_token_length           1   92121013 3.4180e+11 91754
    ## - data_channel_is_socmed         1  100266317 3.4181e+11 91754
    ## - abs_title_sentiment_polarity   1  105931190 3.4181e+11 91754
    ## - min_negative_polarity          1  108486073 3.4182e+11 91754
    ## - global_sentiment_polarity      1  111381672 3.4182e+11 91755
    ## - num_imgs                       1  121251024 3.4183e+11 91755
    ## <none>                                        3.4171e+11 91755
    ## - kw_min_max                     1  137580859 3.4185e+11 91755
    ## - rate_positive_words            1  146857575 3.4186e+11 91755
    ## - timedelta                      1  160653844 3.4187e+11 91755
    ## - max_negative_polarity          1  162768247 3.4187e+11 91755
    ## - data_channel_is_entertainment  1  175407767 3.4188e+11 91755
    ## - data_channel_is_world          1  222689896 3.4193e+11 91756
    ## - n_unique_tokens                1  232509615 3.4194e+11 91756
    ## - n_non_stop_unique_tokens       1  248133929 3.4196e+11 91757
    ## - LDA_02                         1  273292816 3.4198e+11 91757
    ## - n_tokens_title                 1  507400651 3.4222e+11 91760
    ## - global_subjectivity            1  685914387 3.4239e+11 91763
    ## - kw_min_avg                     1 1230707006 3.4294e+11 91771
    ## - kw_max_avg                     1 1234594805 3.4294e+11 91771
    ## - self_reference_min_shares      1 1486409464 3.4319e+11 91775
    ## - kw_avg_avg                     1 2575324996 3.4428e+11 91791
    ## - self_reference_avg_sharess     1 2577063338 3.4429e+11 91791
    ## - self_reference_max_shares      1 4164656277 3.4587e+11 91814
    ## 
    ## Step:  AIC=91752.95
    ## shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - n_tokens_content               1    9279889 3.4172e+11 91751
    ## - LDA_01                         1   12251984 3.4173e+11 91751
    ## - title_sentiment_polarity       1   12958451 3.4173e+11 91751
    ## - n_non_stop_words               1   17379571 3.4173e+11 91751
    ## - num_hrefs                      1   18059774 3.4173e+11 91751
    ## - global_rate_positive_words     1   23320837 3.4174e+11 91751
    ## - data_channel_is_tech           1   23731961 3.4174e+11 91751
    ## - kw_max_min                     1   27708171 3.4174e+11 91751
    ## - global_rate_negative_words     1   28506166 3.4174e+11 91751
    ## - kw_avg_min                     1   33582140 3.4175e+11 91751
    ## - kw_max_max                     1   37529381 3.4175e+11 91752
    ## - num_videos                     1   38073700 3.4175e+11 91752
    ## - kw_avg_max                     1   40695914 3.4175e+11 91752
    ## - LDA_00                         1   45055259 3.4176e+11 91752
    ## - avg_positive_polarity          1   49526817 3.4176e+11 91752
    ## - LDA_03                         1   51737244 3.4177e+11 91752
    ## - avg_negative_polarity          1   55551556 3.4177e+11 91752
    ## - data_channel_is_bus            1   60786935 3.4178e+11 91752
    ## - kw_min_min                     1   64095969 3.4178e+11 91752
    ## - num_keywords                   1   69582028 3.4178e+11 91752
    ## - average_token_length           1   89166263 3.4180e+11 91752
    ## - abs_title_sentiment_polarity   1  106846685 3.4182e+11 91753
    ## - data_channel_is_socmed         1  107822827 3.4182e+11 91753
    ## - min_negative_polarity          1  110163296 3.4182e+11 91753
    ## - global_sentiment_polarity      1  111198064 3.4183e+11 91753
    ## - num_imgs                       1  117710688 3.4183e+11 91753
    ## <none>                                        3.4171e+11 91753
    ## - kw_min_max                     1  135495094 3.4185e+11 91753
    ## - rate_positive_words            1  147014220 3.4186e+11 91753
    ## - timedelta                      1  158852948 3.4187e+11 91753
    ## - max_negative_polarity          1  162965615 3.4188e+11 91753
    ## - data_channel_is_entertainment  1  180220081 3.4189e+11 91754
    ## - data_channel_is_world          1  222167067 3.4194e+11 91754
    ## - n_unique_tokens                1  230471063 3.4194e+11 91754
    ## - n_non_stop_unique_tokens       1  246875905 3.4196e+11 91755
    ## - LDA_02                         1  272398315 3.4199e+11 91755
    ## - n_tokens_title                 1  506002552 3.4222e+11 91758
    ## - global_subjectivity            1  689200995 3.4240e+11 91761
    ## - kw_max_avg                     1 1241575336 3.4296e+11 91769
    ## - kw_min_avg                     1 1247712012 3.4296e+11 91769
    ## - self_reference_min_shares      1 1480713315 3.4319e+11 91773
    ## - self_reference_avg_sharess     1 2583252248 3.4430e+11 91789
    ## - kw_avg_avg                     1 2596782883 3.4431e+11 91789
    ## - self_reference_max_shares      1 4220086334 3.4593e+11 91813
    ## 
    ## Step:  AIC=91751.08
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     average_token_length + num_keywords + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_hrefs                      1   12016966 3.4174e+11 91749
    ## - LDA_01                         1   12377237 3.4174e+11 91749
    ## - title_sentiment_polarity       1   12632543 3.4174e+11 91749
    ## - n_non_stop_words               1   23113005 3.4175e+11 91749
    ## - data_channel_is_tech           1   24465058 3.4175e+11 91749
    ## - global_rate_positive_words     1   25294166 3.4175e+11 91749
    ## - kw_max_min                     1   27500036 3.4175e+11 91749
    ## - global_rate_negative_words     1   28691336 3.4175e+11 91750
    ## - kw_avg_min                     1   33219532 3.4176e+11 91750
    ## - num_videos                     1   33372850 3.4176e+11 91750
    ## - kw_max_max                     1   37780119 3.4176e+11 91750
    ## - kw_avg_max                     1   41350002 3.4176e+11 91750
    ## - LDA_00                         1   44096259 3.4177e+11 91750
    ## - avg_negative_polarity          1   50275382 3.4177e+11 91750
    ## - LDA_03                         1   51078916 3.4177e+11 91750
    ## - avg_positive_polarity          1   51658431 3.4178e+11 91750
    ## - data_channel_is_bus            1   60764604 3.4178e+11 91750
    ## - kw_min_min                     1   64348742 3.4179e+11 91750
    ## - num_keywords                   1   71342636 3.4179e+11 91750
    ## - average_token_length           1   88531672 3.4181e+11 91750
    ## - min_negative_polarity          1  101493503 3.4182e+11 91751
    ## - abs_title_sentiment_polarity   1  106852981 3.4183e+11 91751
    ## - global_sentiment_polarity      1  108426675 3.4183e+11 91751
    ## - data_channel_is_socmed         1  109127418 3.4183e+11 91751
    ## - num_imgs                       1  109418365 3.4183e+11 91751
    ## <none>                                        3.4172e+11 91751
    ## - kw_min_max                     1  135389749 3.4186e+11 91751
    ## - rate_positive_words            1  145196589 3.4187e+11 91751
    ## - timedelta                      1  158985833 3.4188e+11 91751
    ## - max_negative_polarity          1  160134423 3.4188e+11 91751
    ## - data_channel_is_entertainment  1  187393142 3.4191e+11 91752
    ## - data_channel_is_world          1  223172455 3.4195e+11 91752
    ## - LDA_02                         1  275853169 3.4200e+11 91753
    ## - n_non_stop_unique_tokens       1  286799407 3.4201e+11 91753
    ## - n_unique_tokens                1  370890634 3.4209e+11 91755
    ## - n_tokens_title                 1  505038865 3.4223e+11 91757
    ## - global_subjectivity            1  693064020 3.4242e+11 91759
    ## - kw_max_avg                     1 1250925453 3.4297e+11 91768
    ## - kw_min_avg                     1 1254678228 3.4298e+11 91768
    ## - self_reference_min_shares      1 1478708322 3.4320e+11 91771
    ## - self_reference_avg_sharess     1 2578590290 3.4430e+11 91787
    ## - kw_avg_avg                     1 2616184790 3.4434e+11 91788
    ## - self_reference_max_shares      1 4213730408 3.4594e+11 91811
    ## 
    ## Step:  AIC=91749.26
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_entertainment + data_channel_is_bus + 
    ##     data_channel_is_socmed + data_channel_is_tech + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - LDA_01                         1   12019055 3.4175e+11 91747
    ## - title_sentiment_polarity       1   12439065 3.4175e+11 91747
    ## - n_non_stop_words               1   23832597 3.4176e+11 91748
    ## - global_rate_positive_words     1   25817104 3.4176e+11 91748
    ## - data_channel_is_tech           1   26515517 3.4176e+11 91748
    ## - global_rate_negative_words     1   26840458 3.4176e+11 91748
    ## - kw_max_min                     1   27643179 3.4176e+11 91748
    ## - kw_avg_min                     1   33611539 3.4177e+11 91748
    ## - num_videos                     1   36627337 3.4177e+11 91748
    ## - kw_max_max                     1   40242746 3.4178e+11 91748
    ## - kw_avg_max                     1   42666716 3.4178e+11 91748
    ## - LDA_00                         1   46066642 3.4178e+11 91748
    ## - LDA_03                         1   48584420 3.4178e+11 91748
    ## - avg_negative_polarity          1   51357206 3.4179e+11 91748
    ## - avg_positive_polarity          1   52013609 3.4179e+11 91748
    ## - data_channel_is_bus            1   63379034 3.4180e+11 91748
    ## - kw_min_min                     1   65270971 3.4180e+11 91748
    ## - num_keywords                   1   66654098 3.4180e+11 91748
    ## - average_token_length           1   79986754 3.4182e+11 91748
    ## - global_sentiment_polarity      1  105627737 3.4184e+11 91749
    ## - abs_title_sentiment_polarity   1  109265054 3.4184e+11 91749
    ## - data_channel_is_socmed         1  110164571 3.4185e+11 91749
    ## - min_negative_polarity          1  112016138 3.4185e+11 91749
    ## - num_imgs                       1  128056065 3.4186e+11 91749
    ## <none>                                        3.4174e+11 91749
    ## - kw_min_max                     1  136278863 3.4187e+11 91749
    ## - rate_positive_words            1  143516231 3.4188e+11 91749
    ## - max_negative_polarity          1  157539683 3.4189e+11 91750
    ## - timedelta                      1  165326077 3.4190e+11 91750
    ## - data_channel_is_entertainment  1  194337815 3.4193e+11 91750
    ## - data_channel_is_world          1  215603739 3.4195e+11 91750
    ## - LDA_02                         1  268544578 3.4200e+11 91751
    ## - n_non_stop_unique_tokens       1  290936050 3.4203e+11 91752
    ## - n_unique_tokens                1  363814439 3.4210e+11 91753
    ## - n_tokens_title                 1  498444583 3.4223e+11 91755
    ## - global_subjectivity            1  700933467 3.4244e+11 91758
    ## - kw_min_avg                     1 1246076656 3.4298e+11 91766
    ## - kw_max_avg                     1 1254183179 3.4299e+11 91766
    ## - self_reference_min_shares      1 1500883733 3.4324e+11 91770
    ## - self_reference_avg_sharess     1 2626443907 3.4436e+11 91786
    ## - kw_avg_avg                     1 2627316086 3.4436e+11 91786
    ## - self_reference_max_shares      1 4300507253 3.4604e+11 91811
    ## 
    ## Step:  AIC=91747.44
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_entertainment + data_channel_is_bus + 
    ##     data_channel_is_socmed + data_channel_is_tech + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - title_sentiment_polarity       1   11756480 3.4176e+11 91746
    ## - data_channel_is_tech           1   16378288 3.4176e+11 91746
    ## - n_non_stop_words               1   21573609 3.4177e+11 91746
    ## - global_rate_positive_words     1   26004168 3.4177e+11 91746
    ## - global_rate_negative_words     1   26372830 3.4177e+11 91746
    ## - kw_max_min                     1   28365757 3.4178e+11 91746
    ## - kw_avg_min                     1   34790660 3.4178e+11 91746
    ## - num_videos                     1   37546506 3.4179e+11 91746
    ## - LDA_03                         1   38008194 3.4179e+11 91746
    ## - kw_max_max                     1   43014989 3.4179e+11 91746
    ## - kw_avg_max                     1   45900707 3.4179e+11 91746
    ## - avg_positive_polarity          1   52211778 3.4180e+11 91746
    ## - avg_negative_polarity          1   53640374 3.4180e+11 91746
    ## - data_channel_is_bus            1   58217429 3.4181e+11 91746
    ## - num_keywords                   1   62637595 3.4181e+11 91746
    ## - kw_min_min                     1   65536804 3.4181e+11 91746
    ## - LDA_00                         1   77122413 3.4182e+11 91747
    ## - average_token_length           1   83346582 3.4183e+11 91747
    ## - data_channel_is_socmed         1  104277260 3.4185e+11 91747
    ## - global_sentiment_polarity      1  105829937 3.4185e+11 91747
    ## - abs_title_sentiment_polarity   1  109090146 3.4186e+11 91747
    ## - min_negative_polarity          1  113247867 3.4186e+11 91747
    ## - num_imgs                       1  125117981 3.4187e+11 91747
    ## <none>                                        3.4175e+11 91747
    ## - kw_min_max                     1  134683547 3.4188e+11 91747
    ## - rate_positive_words            1  143356053 3.4189e+11 91748
    ## - max_negative_polarity          1  158453884 3.4191e+11 91748
    ## - timedelta                      1  166709608 3.4191e+11 91748
    ## - data_channel_is_world          1  231613493 3.4198e+11 91749
    ## - LDA_02                         1  263429520 3.4201e+11 91749
    ## - n_non_stop_unique_tokens       1  289356966 3.4204e+11 91750
    ## - data_channel_is_entertainment  1  294603199 3.4204e+11 91750
    ## - n_unique_tokens                1  357917079 3.4211e+11 91751
    ## - n_tokens_title                 1  495806187 3.4224e+11 91753
    ## - global_subjectivity            1  699541005 3.4245e+11 91756
    ## - kw_min_avg                     1 1234079665 3.4298e+11 91764
    ## - kw_max_avg                     1 1242884347 3.4299e+11 91764
    ## - self_reference_min_shares      1 1498786882 3.4325e+11 91768
    ## - kw_avg_avg                     1 2616719003 3.4436e+11 91784
    ## - self_reference_avg_sharess     1 2628055762 3.4438e+11 91784
    ## - self_reference_max_shares      1 4308222770 3.4606e+11 91809
    ## 
    ## Step:  AIC=91745.62
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_entertainment + data_channel_is_bus + 
    ##     data_channel_is_socmed + data_channel_is_tech + data_channel_is_world + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - data_channel_is_tech           1   16505415 3.4178e+11 91744
    ## - n_non_stop_words               1   20257105 3.4178e+11 91744
    ## - global_rate_negative_words     1   23958074 3.4178e+11 91744
    ## - global_rate_positive_words     1   25269441 3.4178e+11 91744
    ## - kw_max_min                     1   28850237 3.4179e+11 91744
    ## - kw_avg_min                     1   35198814 3.4179e+11 91744
    ## - LDA_03                         1   38897281 3.4180e+11 91744
    ## - num_videos                     1   40194869 3.4180e+11 91744
    ## - kw_max_max                     1   43125627 3.4180e+11 91744
    ## - kw_avg_max                     1   47820537 3.4181e+11 91744
    ## - avg_positive_polarity          1   53821208 3.4181e+11 91744
    ## - avg_negative_polarity          1   56406119 3.4182e+11 91744
    ## - data_channel_is_bus            1   59045903 3.4182e+11 91744
    ## - num_keywords                   1   62040996 3.4182e+11 91745
    ## - kw_min_min                     1   64745809 3.4182e+11 91745
    ## - LDA_00                         1   79055751 3.4184e+11 91745
    ## - average_token_length           1   84704989 3.4184e+11 91745
    ## - global_sentiment_polarity      1  100941432 3.4186e+11 91745
    ## - data_channel_is_socmed         1  105679830 3.4186e+11 91745
    ## - min_negative_polarity          1  116189792 3.4188e+11 91745
    ## - num_imgs                       1  127199719 3.4189e+11 91746
    ## - kw_min_max                     1  134379667 3.4189e+11 91746
    ## <none>                                        3.4176e+11 91746
    ## - rate_positive_words            1  140205195 3.4190e+11 91746
    ## - max_negative_polarity          1  161457690 3.4192e+11 91746
    ## - timedelta                      1  167441900 3.4193e+11 91746
    ## - abs_title_sentiment_polarity   1  179278877 3.4194e+11 91746
    ## - data_channel_is_world          1  228009990 3.4199e+11 91747
    ## - LDA_02                         1  260255284 3.4202e+11 91747
    ## - n_non_stop_unique_tokens       1  290354957 3.4205e+11 91748
    ## - data_channel_is_entertainment  1  294817089 3.4205e+11 91748
    ## - n_unique_tokens                1  358569715 3.4212e+11 91749
    ## - n_tokens_title                 1  493376000 3.4225e+11 91751
    ## - global_subjectivity            1  698506659 3.4246e+11 91754
    ## - kw_min_avg                     1 1227157633 3.4299e+11 91762
    ## - kw_max_avg                     1 1241726716 3.4300e+11 91762
    ## - self_reference_min_shares      1 1496304207 3.4326e+11 91766
    ## - kw_avg_avg                     1 2617883538 3.4438e+11 91782
    ## - self_reference_avg_sharess     1 2623942567 3.4438e+11 91783
    ## - self_reference_max_shares      1 4302833782 3.4606e+11 91807
    ## 
    ## Step:  AIC=91743.86
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_entertainment + data_channel_is_bus + 
    ##     data_channel_is_socmed + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - n_non_stop_words               1   23741116 3.4180e+11 91742
    ## - global_rate_positive_words     1   24526629 3.4180e+11 91742
    ## - global_rate_negative_words     1   24624634 3.4180e+11 91742
    ## - LDA_03                         1   25026861 3.4180e+11 91742
    ## - kw_max_min                     1   29682945 3.4181e+11 91742
    ## - kw_avg_min                     1   35889747 3.4181e+11 91742
    ## - num_videos                     1   39731697 3.4182e+11 91742
    ## - kw_max_max                     1   41398344 3.4182e+11 91742
    ## - data_channel_is_bus            1   43409976 3.4182e+11 91743
    ## - kw_avg_max                     1   48443956 3.4182e+11 91743
    ## - avg_positive_polarity          1   51828224 3.4183e+11 91743
    ## - avg_negative_polarity          1   54162342 3.4183e+11 91743
    ## - num_keywords                   1   61910733 3.4184e+11 91743
    ## - kw_min_min                     1   66273087 3.4184e+11 91743
    ## - average_token_length           1   79737980 3.4186e+11 91743
    ## - data_channel_is_socmed         1   89177328 3.4186e+11 91743
    ## - LDA_00                         1   95710378 3.4187e+11 91743
    ## - global_sentiment_polarity      1   99437805 3.4188e+11 91743
    ## - min_negative_polarity          1  116851668 3.4189e+11 91744
    ## - num_imgs                       1  130259485 3.4191e+11 91744
    ## <none>                                        3.4178e+11 91744
    ## - kw_min_max                     1  135104424 3.4191e+11 91744
    ## - rate_positive_words            1  137046982 3.4191e+11 91744
    ## - max_negative_polarity          1  157913407 3.4193e+11 91744
    ## - timedelta                      1  168228915 3.4194e+11 91744
    ## - abs_title_sentiment_polarity   1  180418392 3.4196e+11 91745
    ## - LDA_02                         1  248999218 3.4202e+11 91746
    ## - n_non_stop_unique_tokens       1  292073101 3.4207e+11 91746
    ## - data_channel_is_entertainment  1  326087957 3.4210e+11 91747
    ## - n_unique_tokens                1  367350860 3.4214e+11 91747
    ## - data_channel_is_world          1  379577469 3.4216e+11 91748
    ## - n_tokens_title                 1  493667111 3.4227e+11 91749
    ## - global_subjectivity            1  697530118 3.4247e+11 91752
    ## - kw_min_avg                     1 1289122766 3.4306e+11 91761
    ## - kw_max_avg                     1 1322518093 3.4310e+11 91762
    ## - self_reference_min_shares      1 1512300381 3.4329e+11 91764
    ## - self_reference_avg_sharess     1 2640876040 3.4442e+11 91781
    ## - kw_avg_avg                     1 2860552377 3.4464e+11 91784
    ## - self_reference_max_shares      1 4309658655 3.4609e+11 91806
    ## 
    ## Step:  AIC=91742.22
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - global_rate_negative_words     1   12532258 3.4181e+11 91740
    ## - global_rate_positive_words     1   16307766 3.4182e+11 91740
    ## - LDA_03                         1   22447902 3.4182e+11 91741
    ## - kw_max_min                     1   30044383 3.4183e+11 91741
    ## - kw_avg_min                     1   36042378 3.4184e+11 91741
    ## - kw_max_max                     1   39075427 3.4184e+11 91741
    ## - num_videos                     1   39184201 3.4184e+11 91741
    ## - data_channel_is_bus            1   42878396 3.4184e+11 91741
    ## - kw_avg_max                     1   46182949 3.4185e+11 91741
    ## - avg_negative_polarity          1   49797743 3.4185e+11 91741
    ## - num_keywords                   1   56980111 3.4186e+11 91741
    ## - kw_min_min                     1   64575623 3.4186e+11 91741
    ## - avg_positive_polarity          1   69018050 3.4187e+11 91741
    ## - global_sentiment_polarity      1   85789693 3.4189e+11 91741
    ## - data_channel_is_socmed         1   91108055 3.4189e+11 91742
    ## - LDA_00                         1  101625072 3.4190e+11 91742
    ## - min_negative_polarity          1  108165817 3.4191e+11 91742
    ## - rate_positive_words            1  114202217 3.4191e+11 91742
    ## - num_imgs                       1  119972582 3.4192e+11 91742
    ## <none>                                        3.4180e+11 91742
    ## - kw_min_max                     1  135612124 3.4194e+11 91742
    ## - max_negative_polarity          1  151175996 3.4195e+11 91742
    ## - timedelta                      1  167723284 3.4197e+11 91743
    ## - abs_title_sentiment_polarity   1  186495159 3.4199e+11 91743
    ## - LDA_02                         1  239452718 3.4204e+11 91744
    ## - data_channel_is_entertainment  1  328235324 3.4213e+11 91745
    ## - average_token_length           1  376399311 3.4218e+11 91746
    ## - data_channel_is_world          1  389954059 3.4219e+11 91746
    ## - n_non_stop_unique_tokens       1  403058731 3.4220e+11 91746
    ## - n_unique_tokens                1  435621461 3.4224e+11 91747
    ## - n_tokens_title                 1  479695056 3.4228e+11 91747
    ## - global_subjectivity            1  675018712 3.4247e+11 91750
    ## - kw_min_avg                     1 1303181702 3.4310e+11 91760
    ## - kw_max_avg                     1 1356529795 3.4316e+11 91760
    ## - self_reference_min_shares      1 1505916526 3.4331e+11 91763
    ## - self_reference_avg_sharess     1 2637013399 3.4444e+11 91779
    ## - kw_avg_avg                     1 2914142120 3.4471e+11 91783
    ## - self_reference_max_shares      1 4306484185 3.4611e+11 91804
    ## 
    ## Step:  AIC=91740.4
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - global_rate_positive_words     1    4932384 3.4182e+11 91738
    ## - LDA_03                         1   22829086 3.4183e+11 91739
    ## - kw_max_min                     1   30050659 3.4184e+11 91739
    ## - kw_avg_min                     1   36201041 3.4185e+11 91739
    ## - kw_max_max                     1   39852783 3.4185e+11 91739
    ## - data_channel_is_bus            1   44115387 3.4186e+11 91739
    ## - kw_avg_max                     1   45139454 3.4186e+11 91739
    ## - num_videos                     1   46506037 3.4186e+11 91739
    ## - num_keywords                   1   56887103 3.4187e+11 91739
    ## - avg_negative_polarity          1   56973659 3.4187e+11 91739
    ## - avg_positive_polarity          1   57458687 3.4187e+11 91739
    ## - kw_min_min                     1   63351145 3.4188e+11 91739
    ## - data_channel_is_socmed         1   90233101 3.4190e+11 91740
    ## - LDA_00                         1   99940875 3.4191e+11 91740
    ## - min_negative_polarity          1  114838100 3.4193e+11 91740
    ## - num_imgs                       1  122215078 3.4193e+11 91740
    ## - rate_positive_words            1  123234915 3.4194e+11 91740
    ## <none>                                        3.4181e+11 91740
    ## - kw_min_max                     1  136333289 3.4195e+11 91740
    ## - global_sentiment_polarity      1  136971052 3.4195e+11 91740
    ## - max_negative_polarity          1  151099557 3.4196e+11 91741
    ## - timedelta                      1  174399824 3.4199e+11 91741
    ## - abs_title_sentiment_polarity   1  188253704 3.4200e+11 91741
    ## - LDA_02                         1  242696023 3.4205e+11 91742
    ## - data_channel_is_entertainment  1  323596839 3.4214e+11 91743
    ## - average_token_length           1  375772167 3.4219e+11 91744
    ## - data_channel_is_world          1  383547221 3.4220e+11 91744
    ## - n_non_stop_unique_tokens       1  391287238 3.4220e+11 91744
    ## - n_unique_tokens                1  427698137 3.4224e+11 91745
    ## - n_tokens_title                 1  489625341 3.4230e+11 91746
    ## - global_subjectivity            1  752465645 3.4256e+11 91750
    ## - kw_min_avg                     1 1299937144 3.4311e+11 91758
    ## - kw_max_avg                     1 1349154169 3.4316e+11 91758
    ## - self_reference_min_shares      1 1508979052 3.4332e+11 91761
    ## - self_reference_avg_sharess     1 2634487587 3.4445e+11 91777
    ## - kw_avg_avg                     1 2902741922 3.4471e+11 91781
    ## - self_reference_max_shares      1 4302547908 3.4611e+11 91802
    ## 
    ## Step:  AIC=91738.48
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + rate_positive_words + avg_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - LDA_03                         1   22251670 3.4184e+11 91737
    ## - kw_max_min                     1   29874069 3.4185e+11 91737
    ## - kw_avg_min                     1   35926267 3.4185e+11 91737
    ## - kw_max_max                     1   39258959 3.4186e+11 91737
    ## - num_videos                     1   43925591 3.4186e+11 91737
    ## - kw_avg_max                     1   44857054 3.4186e+11 91737
    ## - data_channel_is_bus            1   45017252 3.4186e+11 91737
    ## - avg_positive_polarity          1   53082021 3.4187e+11 91737
    ## - num_keywords                   1   58068976 3.4188e+11 91737
    ## - avg_negative_polarity          1   60003902 3.4188e+11 91737
    ## - kw_min_min                     1   62367256 3.4188e+11 91737
    ## - data_channel_is_socmed         1   93554262 3.4191e+11 91738
    ## - LDA_00                         1  100107783 3.4192e+11 91738
    ## - min_negative_polarity          1  110531141 3.4193e+11 91738
    ## - rate_positive_words            1  118315648 3.4194e+11 91738
    ## - num_imgs                       1  125750994 3.4194e+11 91738
    ## <none>                                        3.4182e+11 91738
    ## - kw_min_max                     1  136590306 3.4195e+11 91739
    ## - max_negative_polarity          1  161406492 3.4198e+11 91739
    ## - global_sentiment_polarity      1  165209196 3.4198e+11 91739
    ## - timedelta                      1  171348402 3.4199e+11 91739
    ## - abs_title_sentiment_polarity   1  184586446 3.4200e+11 91739
    ## - LDA_02                         1  240750411 3.4206e+11 91740
    ## - data_channel_is_entertainment  1  326221092 3.4214e+11 91741
    ## - average_token_length           1  372613780 3.4219e+11 91742
    ## - data_channel_is_world          1  387896436 3.4220e+11 91742
    ## - n_non_stop_unique_tokens       1  401955215 3.4222e+11 91742
    ## - n_unique_tokens                1  429699137 3.4225e+11 91743
    ## - n_tokens_title                 1  489081469 3.4231e+11 91744
    ## - global_subjectivity            1  748109275 3.4257e+11 91748
    ## - kw_min_avg                     1 1301089847 3.4312e+11 91756
    ## - kw_max_avg                     1 1347533067 3.4316e+11 91756
    ## - self_reference_min_shares      1 1511909318 3.4333e+11 91759
    ## - self_reference_avg_sharess     1 2636541735 3.4445e+11 91776
    ## - kw_avg_avg                     1 2899835530 3.4472e+11 91779
    ## - self_reference_max_shares      1 4307295978 3.4612e+11 91800
    ## 
    ## Step:  AIC=91736.81
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_02 + global_subjectivity + 
    ##     global_sentiment_polarity + rate_positive_words + avg_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_max_min                     1   29528524 3.4187e+11 91735
    ## - num_videos                     1   34119227 3.4187e+11 91735
    ## - kw_avg_min                     1   35905657 3.4188e+11 91735
    ## - data_channel_is_bus            1   44143215 3.4188e+11 91735
    ## - kw_max_max                     1   48116664 3.4189e+11 91736
    ## - avg_positive_polarity          1   53837931 3.4189e+11 91736
    ## - kw_min_min                     1   60802542 3.4190e+11 91736
    ## - num_keywords                   1   62170492 3.4190e+11 91736
    ## - kw_avg_max                     1   62757058 3.4190e+11 91736
    ## - avg_negative_polarity          1   64068028 3.4190e+11 91736
    ## - data_channel_is_socmed         1   98917354 3.4194e+11 91736
    ## - min_negative_polarity          1  111359883 3.4195e+11 91736
    ## - num_imgs                       1  116900813 3.4196e+11 91737
    ## - LDA_00                         1  124864532 3.4196e+11 91737
    ## - kw_min_max                     1  128670780 3.4197e+11 91737
    ## - rate_positive_words            1  131015344 3.4197e+11 91737
    ## <none>                                        3.4184e+11 91737
    ## - max_negative_polarity          1  165073547 3.4200e+11 91737
    ## - timedelta                      1  165605103 3.4200e+11 91737
    ## - global_sentiment_polarity      1  174379663 3.4201e+11 91737
    ## - abs_title_sentiment_polarity   1  181695465 3.4202e+11 91738
    ## - LDA_02                         1  221401831 3.4206e+11 91738
    ## - data_channel_is_entertainment  1  348237858 3.4219e+11 91740
    ## - average_token_length           1  371813459 3.4221e+11 91740
    ## - data_channel_is_world          1  374713196 3.4221e+11 91740
    ## - n_non_stop_unique_tokens       1  382405731 3.4222e+11 91740
    ## - n_unique_tokens                1  409438944 3.4225e+11 91741
    ## - n_tokens_title                 1  483121340 3.4232e+11 91742
    ## - global_subjectivity            1  742495276 3.4258e+11 91746
    ## - kw_min_avg                     1 1280769035 3.4312e+11 91754
    ## - kw_max_avg                     1 1332748790 3.4317e+11 91755
    ## - self_reference_min_shares      1 1499010400 3.4334e+11 91757
    ## - self_reference_avg_sharess     1 2622259771 3.4446e+11 91774
    ## - kw_avg_avg                     1 2992920257 3.4483e+11 91779
    ## - self_reference_max_shares      1 4298834466 3.4614e+11 91798
    ## 
    ## Step:  AIC=91735.25
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_world + kw_min_min + kw_avg_min + kw_min_max + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_02 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + avg_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_avg_min                     1    7266016 3.4188e+11 91733
    ## - num_videos                     1   34419049 3.4190e+11 91734
    ## - kw_avg_max                     1   47374710 3.4192e+11 91734
    ## - data_channel_is_bus            1   47714596 3.4192e+11 91734
    ## - kw_max_max                     1   50451946 3.4192e+11 91734
    ## - avg_positive_polarity          1   53576496 3.4192e+11 91734
    ## - kw_min_min                     1   54371584 3.4192e+11 91734
    ## - num_keywords                   1   55194708 3.4192e+11 91734
    ## - avg_negative_polarity          1   61185515 3.4193e+11 91734
    ## - data_channel_is_socmed         1  101146795 3.4197e+11 91735
    ## - min_negative_polarity          1  106873819 3.4198e+11 91735
    ## - num_imgs                       1  119822233 3.4199e+11 91735
    ## - LDA_00                         1  121149487 3.4199e+11 91735
    ## - kw_min_max                     1  128534655 3.4200e+11 91735
    ## - rate_positive_words            1  131196468 3.4200e+11 91735
    ## <none>                                        3.4187e+11 91735
    ## - timedelta                      1  147989042 3.4202e+11 91735
    ## - max_negative_polarity          1  160885209 3.4203e+11 91736
    ## - global_sentiment_polarity      1  174684139 3.4204e+11 91736
    ## - abs_title_sentiment_polarity   1  180254491 3.4205e+11 91736
    ## - LDA_02                         1  223357355 3.4209e+11 91737
    ## - data_channel_is_entertainment  1  345066017 3.4221e+11 91738
    ## - average_token_length           1  372072182 3.4224e+11 91739
    ## - data_channel_is_world          1  376923179 3.4225e+11 91739
    ## - n_non_stop_unique_tokens       1  383465240 3.4225e+11 91739
    ## - n_unique_tokens                1  410914932 3.4228e+11 91739
    ## - n_tokens_title                 1  482657373 3.4235e+11 91740
    ## - global_subjectivity            1  744005278 3.4261e+11 91744
    ## - kw_min_avg                     1 1254250793 3.4312e+11 91752
    ## - kw_max_avg                     1 1311125938 3.4318e+11 91753
    ## - self_reference_min_shares      1 1501286846 3.4337e+11 91756
    ## - self_reference_avg_sharess     1 2619987516 3.4449e+11 91772
    ## - kw_avg_avg                     1 2992346656 3.4486e+11 91778
    ## - self_reference_max_shares      1 4284773559 3.4615e+11 91797
    ## 
    ## Step:  AIC=91733.35
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_world + kw_min_min + kw_min_max + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_02 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + avg_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_videos                     1   34258708 3.4191e+11 91732
    ## - kw_avg_max                     1   43635489 3.4192e+11 91732
    ## - data_channel_is_bus            1   46740896 3.4192e+11 91732
    ## - kw_max_max                     1   50090763 3.4193e+11 91732
    ## - kw_min_min                     1   53520104 3.4193e+11 91732
    ## - avg_positive_polarity          1   54122617 3.4193e+11 91732
    ## - num_keywords                   1   55571785 3.4193e+11 91732
    ## - avg_negative_polarity          1   60138420 3.4194e+11 91732
    ## - data_channel_is_socmed         1  100078879 3.4198e+11 91733
    ## - min_negative_polarity          1  106636504 3.4198e+11 91733
    ## - LDA_00                         1  119148396 3.4200e+11 91733
    ## - num_imgs                       1  121302106 3.4200e+11 91733
    ## - kw_min_max                     1  127903473 3.4200e+11 91733
    ## - rate_positive_words            1  130287825 3.4201e+11 91733
    ## <none>                                        3.4188e+11 91733
    ## - timedelta                      1  143543117 3.4202e+11 91733
    ## - max_negative_polarity          1  160042631 3.4204e+11 91734
    ## - global_sentiment_polarity      1  173120603 3.4205e+11 91734
    ## - abs_title_sentiment_polarity   1  180376401 3.4206e+11 91734
    ## - LDA_02                         1  225825664 3.4210e+11 91735
    ## - data_channel_is_entertainment  1  344028468 3.4222e+11 91736
    ## - average_token_length           1  371380313 3.4225e+11 91737
    ## - data_channel_is_world          1  382152457 3.4226e+11 91737
    ## - n_non_stop_unique_tokens       1  384129059 3.4226e+11 91737
    ## - n_unique_tokens                1  412880206 3.4229e+11 91737
    ## - n_tokens_title                 1  481346893 3.4236e+11 91739
    ## - global_subjectivity            1  741825665 3.4262e+11 91742
    ## - kw_min_avg                     1 1276804799 3.4315e+11 91750
    ## - self_reference_min_shares      1 1502783200 3.4338e+11 91754
    ## - kw_max_avg                     1 1506772096 3.4338e+11 91754
    ## - self_reference_avg_sharess     1 2620124391 3.4450e+11 91770
    ## - kw_avg_avg                     1 3027737614 3.4490e+11 91776
    ## - self_reference_max_shares      1 4283418846 3.4616e+11 91795
    ## 
    ## Step:  AIC=91731.86
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + average_token_length + num_keywords + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_02 + global_subjectivity + 
    ##     global_sentiment_polarity + rate_positive_words + avg_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_avg_max                     1   33707138 3.4194e+11 91730
    ## - kw_max_max                     1   45799829 3.4196e+11 91731
    ## - avg_positive_polarity          1   46328653 3.4196e+11 91731
    ## - data_channel_is_bus            1   49374573 3.4196e+11 91731
    ## - num_keywords                   1   53317585 3.4196e+11 91731
    ## - kw_min_min                     1   55801492 3.4197e+11 91731
    ## - avg_negative_polarity          1   56026474 3.4197e+11 91731
    ## - data_channel_is_socmed         1   97962321 3.4201e+11 91731
    ## - num_imgs                       1  105915141 3.4202e+11 91731
    ## - min_negative_polarity          1  110826210 3.4202e+11 91732
    ## - LDA_00                         1  115394178 3.4203e+11 91732
    ## - kw_min_max                     1  133373671 3.4204e+11 91732
    ## <none>                                        3.4191e+11 91732
    ## - rate_positive_words            1  135027879 3.4205e+11 91732
    ## - timedelta                      1  147965872 3.4206e+11 91732
    ## - max_negative_polarity          1  147977755 3.4206e+11 91732
    ## - abs_title_sentiment_polarity   1  179393309 3.4209e+11 91733
    ## - global_sentiment_polarity      1  180716899 3.4209e+11 91733
    ## - LDA_02                         1  227422717 3.4214e+11 91733
    ## - data_channel_is_entertainment  1  323446512 3.4223e+11 91735
    ## - data_channel_is_world          1  378305019 3.4229e+11 91735
    ## - average_token_length           1  381955064 3.4229e+11 91736
    ## - n_non_stop_unique_tokens       1  396732747 3.4231e+11 91736
    ## - n_unique_tokens                1  423159766 3.4233e+11 91736
    ## - n_tokens_title                 1  491366599 3.4240e+11 91737
    ## - global_subjectivity            1  743219209 3.4265e+11 91741
    ## - kw_min_avg                     1 1280683754 3.4319e+11 91749
    ## - kw_max_avg                     1 1513029594 3.4342e+11 91752
    ## - self_reference_min_shares      1 1514025035 3.4342e+11 91752
    ## - self_reference_avg_sharess     1 2648632501 3.4456e+11 91769
    ## - kw_avg_avg                     1 3038841730 3.4495e+11 91775
    ## - self_reference_max_shares      1 4340946902 3.4625e+11 91794
    ## 
    ## Step:  AIC=91730.36
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + average_token_length + num_keywords + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_min_max + kw_max_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_02 + global_subjectivity + 
    ##     global_sentiment_polarity + rate_positive_words + avg_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_max_max                     1   30239592 3.4197e+11 91729
    ## - num_keywords                   1   31834844 3.4198e+11 91729
    ## - avg_positive_polarity          1   47125129 3.4199e+11 91729
    ## - avg_negative_polarity          1   60154725 3.4200e+11 91729
    ## - kw_min_min                     1   60558556 3.4200e+11 91729
    ## - data_channel_is_bus            1   65354034 3.4201e+11 91729
    ## - data_channel_is_socmed         1   84599736 3.4203e+11 91730
    ## - num_imgs                       1  109308363 3.4205e+11 91730
    ## - LDA_00                         1  113286880 3.4206e+11 91730
    ## - min_negative_polarity          1  113307140 3.4206e+11 91730
    ## - rate_positive_words            1  134250017 3.4208e+11 91730
    ## <none>                                        3.4194e+11 91730
    ## - max_negative_polarity          1  154650084 3.4210e+11 91731
    ## - global_sentiment_polarity      1  178102406 3.4212e+11 91731
    ## - abs_title_sentiment_polarity   1  180554106 3.4212e+11 91731
    ## - timedelta                      1  193230239 3.4214e+11 91731
    ## - kw_min_max                     1  217522312 3.4216e+11 91732
    ## - LDA_02                         1  239320925 3.4218e+11 91732
    ## - data_channel_is_entertainment  1  302632276 3.4225e+11 91733
    ## - average_token_length           1  365091306 3.4231e+11 91734
    ## - n_non_stop_unique_tokens       1  379646664 3.4232e+11 91734
    ## - data_channel_is_world          1  395718960 3.4234e+11 91734
    ## - n_unique_tokens                1  400054349 3.4234e+11 91734
    ## - n_tokens_title                 1  484063025 3.4243e+11 91736
    ## - global_subjectivity            1  738943527 3.4268e+11 91739
    ## - kw_min_avg                     1 1253283416 3.4320e+11 91747
    ## - kw_max_avg                     1 1517080192 3.4346e+11 91751
    ## - self_reference_min_shares      1 1520763002 3.4346e+11 91751
    ## - self_reference_avg_sharess     1 2652523471 3.4460e+11 91768
    ## - kw_avg_avg                     1 3313588533 3.4526e+11 91777
    ## - self_reference_max_shares      1 4331603485 3.4628e+11 91792
    ## 
    ## Step:  AIC=91728.81
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + average_token_length + num_keywords + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_02 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + avg_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_keywords                   1   29641108 3.4200e+11 91727
    ## - kw_min_min                     1   31288479 3.4201e+11 91727
    ## - avg_positive_polarity          1   49074479 3.4202e+11 91728
    ## - avg_negative_polarity          1   59819460 3.4203e+11 91728
    ## - data_channel_is_bus            1   64087865 3.4204e+11 91728
    ## - data_channel_is_socmed         1   85353866 3.4206e+11 91728
    ## - min_negative_polarity          1  111973783 3.4209e+11 91728
    ## - num_imgs                       1  112184517 3.4209e+11 91728
    ## - LDA_00                         1  115842472 3.4209e+11 91729
    ## <none>                                        3.4197e+11 91729
    ## - rate_positive_words            1  135783808 3.4211e+11 91729
    ## - max_negative_polarity          1  152916116 3.4213e+11 91729
    ## - timedelta                      1  165874128 3.4214e+11 91729
    ## - global_sentiment_polarity      1  178423741 3.4215e+11 91729
    ## - abs_title_sentiment_polarity   1  179807211 3.4215e+11 91729
    ## - kw_min_max                     1  218389106 3.4219e+11 91730
    ## - LDA_02                         1  231596330 3.4221e+11 91730
    ## - data_channel_is_entertainment  1  301282527 3.4228e+11 91731
    ## - average_token_length           1  362916420 3.4234e+11 91732
    ## - n_non_stop_unique_tokens       1  371335057 3.4235e+11 91732
    ## - data_channel_is_world          1  391894394 3.4237e+11 91733
    ## - n_unique_tokens                1  392975719 3.4237e+11 91733
    ## - n_tokens_title                 1  482349251 3.4246e+11 91734
    ## - global_subjectivity            1  740467091 3.4271e+11 91738
    ## - kw_min_avg                     1 1279447439 3.4325e+11 91746
    ## - self_reference_min_shares      1 1518462998 3.4349e+11 91749
    ## - kw_max_avg                     1 1564674549 3.4354e+11 91750
    ## - self_reference_avg_sharess     1 2646889298 3.4462e+11 91766
    ## - kw_avg_avg                     1 3437034992 3.4541e+11 91778
    ## - self_reference_max_shares      1 4326575512 3.4630e+11 91791
    ## 
    ## Step:  AIC=91727.26
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + average_token_length + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_min + kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_02 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + avg_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - kw_min_min                     1   35441276 3.4204e+11 91726
    ## - avg_positive_polarity          1   48690743 3.4205e+11 91726
    ## - data_channel_is_bus            1   55914811 3.4206e+11 91726
    ## - avg_negative_polarity          1   61879677 3.4207e+11 91726
    ## - data_channel_is_socmed         1   75668262 3.4208e+11 91726
    ## - num_imgs                       1  112445520 3.4212e+11 91727
    ## - min_negative_polarity          1  113234314 3.4212e+11 91727
    ## - LDA_00                         1  120432303 3.4212e+11 91727
    ## <none>                                        3.4200e+11 91727
    ## - rate_positive_words            1  138582452 3.4214e+11 91727
    ## - max_negative_polarity          1  155967496 3.4216e+11 91728
    ## - timedelta                      1  157850162 3.4216e+11 91728
    ## - abs_title_sentiment_polarity   1  177604520 3.4218e+11 91728
    ## - global_sentiment_polarity      1  188064650 3.4219e+11 91728
    ## - kw_min_max                     1  197684286 3.4220e+11 91728
    ## - LDA_02                         1  224167147 3.4223e+11 91729
    ## - data_channel_is_entertainment  1  281195996 3.4229e+11 91729
    ## - n_non_stop_unique_tokens       1  369057117 3.4237e+11 91731
    ## - average_token_length           1  374466300 3.4238e+11 91731
    ## - data_channel_is_world          1  394323251 3.4240e+11 91731
    ## - n_unique_tokens                1  403018381 3.4241e+11 91731
    ## - n_tokens_title                 1  472216095 3.4248e+11 91732
    ## - global_subjectivity            1  740749542 3.4274e+11 91736
    ## - kw_min_avg                     1 1256852139 3.4326e+11 91744
    ## - self_reference_min_shares      1 1529559560 3.4353e+11 91748
    ## - kw_max_avg                     1 1579187528 3.4358e+11 91749
    ## - self_reference_avg_sharess     1 2663234638 3.4467e+11 91765
    ## - kw_avg_avg                     1 3419075284 3.4542e+11 91776
    ## - self_reference_max_shares      1 4349279677 3.4635e+11 91790
    ## 
    ## Step:  AIC=91725.78
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + average_token_length + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_02 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + avg_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - avg_positive_polarity          1   49688759 3.4209e+11 91725
    ## - data_channel_is_bus            1   57355708 3.4210e+11 91725
    ## - avg_negative_polarity          1   60512929 3.4210e+11 91725
    ## - data_channel_is_socmed         1   78550243 3.4212e+11 91725
    ## - num_imgs                       1  114385170 3.4215e+11 91725
    ## - min_negative_polarity          1  114573304 3.4215e+11 91725
    ## - LDA_00                         1  117946712 3.4216e+11 91726
    ## - rate_positive_words            1  133935116 3.4217e+11 91726
    ## <none>                                        3.4204e+11 91726
    ## - max_negative_polarity          1  153876912 3.4219e+11 91726
    ## - abs_title_sentiment_polarity   1  175317602 3.4221e+11 91726
    ## - global_sentiment_polarity      1  182926468 3.4222e+11 91727
    ## - kw_min_max                     1  197663741 3.4224e+11 91727
    ## - LDA_02                         1  231723912 3.4227e+11 91727
    ## - data_channel_is_entertainment  1  293434569 3.4233e+11 91728
    ## - timedelta                      1  353877992 3.4239e+11 91729
    ## - n_non_stop_unique_tokens       1  372481010 3.4241e+11 91729
    ## - average_token_length           1  381868305 3.4242e+11 91729
    ## - data_channel_is_world          1  390272114 3.4243e+11 91730
    ## - n_unique_tokens                1  410175004 3.4245e+11 91730
    ## - n_tokens_title                 1  478183233 3.4252e+11 91731
    ## - global_subjectivity            1  735916024 3.4278e+11 91735
    ## - kw_min_avg                     1 1235215102 3.4327e+11 91742
    ## - self_reference_min_shares      1 1525350842 3.4356e+11 91746
    ## - kw_max_avg                     1 1546243590 3.4359e+11 91747
    ## - self_reference_avg_sharess     1 2655641636 3.4469e+11 91763
    ## - kw_avg_avg                     1 3396532009 3.4544e+11 91774
    ## - self_reference_max_shares      1 4338295998 3.4638e+11 91788
    ## 
    ## Step:  AIC=91724.52
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + average_token_length + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_world + 
    ##     kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_02 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - data_channel_is_bus            1   58664808 3.4215e+11 91723
    ## - data_channel_is_socmed         1   79816982 3.4217e+11 91724
    ## - avg_negative_polarity          1   82798555 3.4217e+11 91724
    ## - num_imgs                       1  101949059 3.4219e+11 91724
    ## - min_negative_polarity          1  116008792 3.4221e+11 91724
    ## - LDA_00                         1  125213056 3.4221e+11 91724
    ## <none>                                        3.4209e+11 91725
    ## - abs_title_sentiment_polarity   1  168239794 3.4226e+11 91725
    ## - max_negative_polarity          1  171929008 3.4226e+11 91725
    ## - kw_min_max                     1  200624912 3.4229e+11 91726
    ## - LDA_02                         1  225192755 3.4231e+11 91726
    ## - data_channel_is_entertainment  1  292420612 3.4238e+11 91727
    ## - rate_positive_words            1  297316149 3.4239e+11 91727
    ## - timedelta                      1  348482829 3.4244e+11 91728
    ## - data_channel_is_world          1  401727203 3.4249e+11 91728
    ## - n_non_stop_unique_tokens       1  438332851 3.4253e+11 91729
    ## - n_unique_tokens                1  441751889 3.4253e+11 91729
    ## - n_tokens_title                 1  471070561 3.4256e+11 91730
    ## - global_sentiment_polarity      1  550873069 3.4264e+11 91731
    ## - average_token_length           1  576859233 3.4267e+11 91731
    ## - global_subjectivity            1  693270542 3.4278e+11 91733
    ## - kw_min_avg                     1 1234590040 3.4332e+11 91741
    ## - self_reference_min_shares      1 1528405482 3.4362e+11 91745
    ## - kw_max_avg                     1 1543334010 3.4363e+11 91745
    ## - self_reference_avg_sharess     1 2658244248 3.4475e+11 91762
    ## - kw_avg_avg                     1 3395059174 3.4548e+11 91773
    ## - self_reference_max_shares      1 4335522843 3.4642e+11 91787
    ## 
    ## Step:  AIC=91723.39
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + average_token_length + data_channel_is_entertainment + 
    ##     data_channel_is_socmed + data_channel_is_world + kw_min_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_02 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - data_channel_is_socmed         1   35354526 3.4218e+11 91722
    ## - LDA_00                         1   68460005 3.4222e+11 91722
    ## - avg_negative_polarity          1   80886718 3.4223e+11 91723
    ## - num_imgs                       1  102366309 3.4225e+11 91723
    ## - min_negative_polarity          1  114334126 3.4226e+11 91723
    ## <none>                                        3.4215e+11 91723
    ## - max_negative_polarity          1  168851842 3.4232e+11 91724
    ## - abs_title_sentiment_polarity   1  171968560 3.4232e+11 91724
    ## - kw_min_max                     1  211705533 3.4236e+11 91725
    ## - data_channel_is_entertainment  1  255279745 3.4240e+11 91725
    ## - LDA_02                         1  259533692 3.4241e+11 91725
    ## - rate_positive_words            1  303692918 3.4245e+11 91726
    ## - timedelta                      1  361270705 3.4251e+11 91727
    ## - n_unique_tokens                1  436063217 3.4258e+11 91728
    ## - n_non_stop_unique_tokens       1  439948139 3.4259e+11 91728
    ## - n_tokens_title                 1  455904253 3.4260e+11 91728
    ## - data_channel_is_world          1  517758793 3.4267e+11 91729
    ## - global_sentiment_polarity      1  549925135 3.4270e+11 91730
    ## - average_token_length           1  580058727 3.4273e+11 91730
    ## - global_subjectivity            1  708454515 3.4286e+11 91732
    ## - kw_min_avg                     1 1275168088 3.4342e+11 91740
    ## - self_reference_min_shares      1 1517583137 3.4367e+11 91744
    ## - kw_max_avg                     1 1621474974 3.4377e+11 91745
    ## - self_reference_avg_sharess     1 2643656885 3.4479e+11 91761
    ## - kw_avg_avg                     1 3583227463 3.4573e+11 91774
    ## - self_reference_max_shares      1 4312646045 3.4646e+11 91785
    ## 
    ## Step:  AIC=91721.92
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + average_token_length + data_channel_is_entertainment + 
    ##     data_channel_is_world + kw_min_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_02 + global_subjectivity + 
    ##     global_sentiment_polarity + rate_positive_words + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - LDA_00                         1   53612923 3.4224e+11 91721
    ## - avg_negative_polarity          1   77022845 3.4226e+11 91721
    ## - num_imgs                       1  100683819 3.4228e+11 91721
    ## - min_negative_polarity          1  109513028 3.4229e+11 91722
    ## <none>                                        3.4218e+11 91722
    ## - max_negative_polarity          1  167607173 3.4235e+11 91722
    ## - abs_title_sentiment_polarity   1  174593629 3.4236e+11 91723
    ## - kw_min_max                     1  213953799 3.4240e+11 91723
    ## - data_channel_is_entertainment  1  242247786 3.4243e+11 91724
    ## - rate_positive_words            1  294659041 3.4248e+11 91724
    ## - LDA_02                         1  305428542 3.4249e+11 91724
    ## - timedelta                      1  348547107 3.4253e+11 91725
    ## - n_unique_tokens                1  434418195 3.4262e+11 91726
    ## - n_non_stop_unique_tokens       1  439233812 3.4262e+11 91726
    ## - n_tokens_title                 1  475586068 3.4266e+11 91727
    ## - global_sentiment_polarity      1  541659462 3.4272e+11 91728
    ## - average_token_length           1  567250532 3.4275e+11 91728
    ## - data_channel_is_world          1  587866111 3.4277e+11 91729
    ## - global_subjectivity            1  695072850 3.4288e+11 91730
    ## - kw_min_avg                     1 1275938064 3.4346e+11 91739
    ## - self_reference_min_shares      1 1507148833 3.4369e+11 91742
    ## - kw_max_avg                     1 1599261811 3.4378e+11 91744
    ## - self_reference_avg_sharess     1 2625617613 3.4481e+11 91759
    ## - kw_avg_avg                     1 3555304949 3.4574e+11 91772
    ## - self_reference_max_shares      1 4287858027 3.4647e+11 91783
    ## 
    ## Step:  AIC=91720.72
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + average_token_length + data_channel_is_entertainment + 
    ##     data_channel_is_world + kw_min_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_02 + global_subjectivity + 
    ##     global_sentiment_polarity + rate_positive_words + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - avg_negative_polarity          1   76665497 3.4231e+11 91720
    ## - num_imgs                       1   83690994 3.4232e+11 91720
    ## - min_negative_polarity          1  112628847 3.4235e+11 91720
    ## <none>                                        3.4224e+11 91721
    ## - max_negative_polarity          1  170190062 3.4241e+11 91721
    ## - abs_title_sentiment_polarity   1  176137242 3.4241e+11 91721
    ## - kw_min_max                     1  201713863 3.4244e+11 91722
    ## - rate_positive_words            1  305184076 3.4254e+11 91723
    ## - timedelta                      1  343608008 3.4258e+11 91724
    ## - data_channel_is_entertainment  1  344158660 3.4258e+11 91724
    ## - LDA_02                         1  370411128 3.4261e+11 91724
    ## - n_unique_tokens                1  407267969 3.4264e+11 91725
    ## - n_non_stop_unique_tokens       1  416368883 3.4265e+11 91725
    ## - n_tokens_title                 1  468092276 3.4270e+11 91726
    ## - average_token_length           1  534026575 3.4277e+11 91727
    ## - global_sentiment_polarity      1  536999864 3.4277e+11 91727
    ## - data_channel_is_world          1  555381374 3.4279e+11 91727
    ## - global_subjectivity            1  648963311 3.4289e+11 91728
    ## - kw_min_avg                     1 1245538534 3.4348e+11 91737
    ## - self_reference_min_shares      1 1507613415 3.4374e+11 91741
    ## - kw_max_avg                     1 1560510727 3.4380e+11 91742
    ## - self_reference_avg_sharess     1 2625337258 3.4486e+11 91758
    ## - kw_avg_avg                     1 3501692433 3.4574e+11 91770
    ## - self_reference_max_shares      1 4285485972 3.4652e+11 91782
    ## 
    ## Step:  AIC=91719.85
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + average_token_length + data_channel_is_entertainment + 
    ##     data_channel_is_world + kw_min_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_02 + global_subjectivity + 
    ##     global_sentiment_polarity + rate_positive_words + min_negative_polarity + 
    ##     max_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - min_negative_polarity          1   36632181 3.4235e+11 91718
    ## - num_imgs                       1   86439236 3.4240e+11 91719
    ## - max_negative_polarity          1   95044385 3.4241e+11 91719
    ## <none>                                        3.4231e+11 91720
    ## - abs_title_sentiment_polarity   1  164852390 3.4248e+11 91720
    ## - kw_min_max                     1  199960816 3.4251e+11 91721
    ## - rate_positive_words            1  282787795 3.4260e+11 91722
    ## - timedelta                      1  331798727 3.4265e+11 91723
    ## - LDA_02                         1  359226869 3.4267e+11 91723
    ## - n_unique_tokens                1  376234894 3.4269e+11 91723
    ## - data_channel_is_entertainment  1  380821520 3.4269e+11 91724
    ## - n_non_stop_unique_tokens       1  394676055 3.4271e+11 91724
    ## - global_sentiment_polarity      1  470735036 3.4278e+11 91725
    ## - n_tokens_title                 1  471705231 3.4279e+11 91725
    ## - average_token_length           1  507881100 3.4282e+11 91725
    ## - data_channel_is_world          1  552455288 3.4287e+11 91726
    ## - global_subjectivity            1  581047868 3.4289e+11 91726
    ## - kw_min_avg                     1 1225085736 3.4354e+11 91736
    ## - self_reference_min_shares      1 1526244022 3.4384e+11 91740
    ## - kw_max_avg                     1 1536277889 3.4385e+11 91741
    ## - self_reference_avg_sharess     1 2640041287 3.4495e+11 91757
    ## - kw_avg_avg                     1 3457175156 3.4577e+11 91769
    ## - self_reference_max_shares      1 4299249038 3.4661e+11 91781
    ## 
    ## Step:  AIC=91718.4
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     num_imgs + average_token_length + data_channel_is_entertainment + 
    ##     data_channel_is_world + kw_min_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_02 + global_subjectivity + 
    ##     global_sentiment_polarity + rate_positive_words + max_negative_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - num_imgs                       1   95637251 3.4245e+11 91718
    ## - max_negative_polarity          1  111804130 3.4246e+11 91718
    ## <none>                                        3.4235e+11 91718
    ## - abs_title_sentiment_polarity   1  178235666 3.4253e+11 91719
    ## - kw_min_max                     1  200188329 3.4255e+11 91719
    ## - rate_positive_words            1  262723514 3.4261e+11 91720
    ## - timedelta                      1  330759559 3.4268e+11 91721
    ## - n_unique_tokens                1  346527753 3.4270e+11 91722
    ## - data_channel_is_entertainment  1  356822345 3.4271e+11 91722
    ## - n_non_stop_unique_tokens       1  358058969 3.4271e+11 91722
    ## - LDA_02                         1  362117113 3.4271e+11 91722
    ## - average_token_length           1  472021854 3.4282e+11 91723
    ## - n_tokens_title                 1  476197336 3.4283e+11 91723
    ## - data_channel_is_world          1  556938015 3.4291e+11 91725
    ## - global_sentiment_polarity      1  590187231 3.4294e+11 91725
    ## - global_subjectivity            1  830218264 3.4318e+11 91729
    ## - kw_min_avg                     1 1229687211 3.4358e+11 91735
    ## - self_reference_min_shares      1 1526407912 3.4388e+11 91739
    ## - kw_max_avg                     1 1550194381 3.4390e+11 91739
    ## - self_reference_avg_sharess     1 2650777741 3.4500e+11 91756
    ## - kw_avg_avg                     1 3516856256 3.4587e+11 91768
    ## - self_reference_max_shares      1 4310298243 3.4666e+11 91780
    ## 
    ## Step:  AIC=91717.82
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     average_token_length + data_channel_is_entertainment + data_channel_is_world + 
    ##     kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_02 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + max_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## - max_negative_polarity          1  104317012 3.4255e+11 91717
    ## <none>                                        3.4245e+11 91718
    ## - abs_title_sentiment_polarity   1  182168268 3.4263e+11 91719
    ## - kw_min_max                     1  210178874 3.4266e+11 91719
    ## - rate_positive_words            1  261011610 3.4271e+11 91720
    ## - data_channel_is_entertainment  1  320261998 3.4277e+11 91721
    ## - timedelta                      1  343451031 3.4279e+11 91721
    ## - LDA_02                         1  353744485 3.4280e+11 91721
    ## - n_unique_tokens                1  377124789 3.4282e+11 91721
    ## - average_token_length           1  394598485 3.4284e+11 91722
    ## - n_tokens_title                 1  481377998 3.4293e+11 91723
    ## - n_non_stop_unique_tokens       1  511781785 3.4296e+11 91723
    ## - data_channel_is_world          1  536747717 3.4298e+11 91724
    ## - global_sentiment_polarity      1  593682485 3.4304e+11 91725
    ## - global_subjectivity            1  901287643 3.4335e+11 91729
    ## - kw_min_avg                     1 1256613374 3.4370e+11 91734
    ## - self_reference_min_shares      1 1557532160 3.4400e+11 91739
    ## - kw_max_avg                     1 1649401989 3.4409e+11 91740
    ## - self_reference_avg_sharess     1 2701437335 3.4515e+11 91756
    ## - kw_avg_avg                     1 3759326544 3.4620e+11 91771
    ## - self_reference_max_shares      1 4382660784 3.4683e+11 91780
    ## 
    ## Step:  AIC=91717.37
    ## shares ~ timedelta + n_tokens_title + n_unique_tokens + n_non_stop_unique_tokens + 
    ##     average_token_length + data_channel_is_entertainment + data_channel_is_world + 
    ##     kw_min_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_02 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS   AIC
    ## <none>                                        3.4255e+11 91717
    ## - abs_title_sentiment_polarity   1  184319282 3.4273e+11 91718
    ## - kw_min_max                     1  209471099 3.4276e+11 91718
    ## - data_channel_is_entertainment  1  315508338 3.4287e+11 91720
    ## - rate_positive_words            1  331762340 3.4288e+11 91720
    ## - timedelta                      1  334859819 3.4288e+11 91720
    ## - LDA_02                         1  357891648 3.4291e+11 91721
    ## - average_token_length           1  438023752 3.4299e+11 91722
    ## - n_tokens_title                 1  481433698 3.4303e+11 91723
    ## - n_unique_tokens                1  527287005 3.4308e+11 91723
    ## - data_channel_is_world          1  539905814 3.4309e+11 91723
    ## - n_non_stop_unique_tokens       1  615530598 3.4317e+11 91724
    ## - global_sentiment_polarity      1  701376575 3.4325e+11 91726
    ## - global_subjectivity            1 1003832203 3.4355e+11 91730
    ## - kw_min_avg                     1 1264448954 3.4381e+11 91734
    ## - self_reference_min_shares      1 1547510595 3.4410e+11 91738
    ## - kw_max_avg                     1 1645554033 3.4420e+11 91740
    ## - self_reference_avg_sharess     1 2694426503 3.4524e+11 91755
    ## - kw_avg_avg                     1 3760925212 3.4631e+11 91771
    ## - self_reference_max_shares      1 4377111865 3.4693e+11 91780

``` r
#summary the model
summary(lm.step)
```

    ## 
    ## Call:
    ## lm(formula = shares ~ timedelta + n_tokens_title + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + average_token_length + data_channel_is_entertainment + 
    ##     data_channel_is_world + kw_min_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_02 + global_subjectivity + 
    ##     global_sentiment_polarity + rate_positive_words + abs_title_sentiment_polarity, 
    ##     data = train)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -22002  -2126  -1061     71 292592 
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   -1.864e+03  1.137e+03  -1.640 0.101159    
    ## timedelta                      1.350e+00  6.066e-01   2.225 0.026100 *  
    ## n_tokens_title                 1.508e+02  5.651e+01   2.668 0.007647 ** 
    ## n_unique_tokens                7.191e+03  2.575e+03   2.793 0.005250 ** 
    ## n_non_stop_unique_tokens      -7.442e+03  2.467e+03  -3.017 0.002564 ** 
    ## average_token_length          -6.422e+02  2.523e+02  -2.545 0.010951 *  
    ## data_channel_is_entertainment -7.042e+02  3.260e+02  -2.160 0.030811 *  
    ## data_channel_is_world          1.457e+03  5.155e+02   2.826 0.004736 ** 
    ## kw_min_max                    -3.437e-03  1.953e-03  -1.760 0.078454 .  
    ## kw_min_avg                    -6.126e-01  1.417e-01  -4.324 1.56e-05 ***
    ## kw_max_avg                    -2.219e-01  4.498e-02  -4.933 8.35e-07 ***
    ## kw_avg_avg                     1.770e+00  2.374e-01   7.458 1.03e-13 ***
    ## self_reference_min_shares      7.296e-02  1.525e-02   4.784 1.77e-06 ***
    ## self_reference_max_shares      6.961e-02  8.652e-03   8.046 1.06e-15 ***
    ## self_reference_avg_sharess    -1.377e-01  2.181e-02  -6.313 2.98e-10 ***
    ## LDA_02                        -1.735e+03  7.540e+02  -2.301 0.021453 *  
    ## global_subjectivity            5.315e+03  1.379e+03   3.853 0.000118 ***
    ## global_sentiment_polarity     -6.260e+03  1.944e+03  -3.221 0.001287 ** 
    ## rate_positive_words            2.635e+03  1.190e+03   2.215 0.026801 *  
    ## abs_title_sentiment_polarity   8.680e+02  5.257e+02   1.651 0.098794 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 8223 on 5066 degrees of freedom
    ## Multiple R-squared:  0.04641,    Adjusted R-squared:  0.04283 
    ## F-statistic: 12.98 on 19 and 5066 DF,  p-value: < 2.2e-16

Calculate the predicted mean square error on the train set:

``` r
train.pred <- predict(lm.step, train[,-52])
mean((train.pred - train$shares)^2)
```

    ## [1] 67351536

So, the predicted mean square error on the training dataset is
264157333.

### On test set

Calculate the predicted mean square error on the test set:

``` r
test.pred <- predict(lm.step, test[,-52])
mean((test.pred - test$shares)^2)
```

    ## [1] 127767821

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
