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

The analysis for [Monday is available here](https://github.com/yxie27/Project2/blob/master/weekday_is_monday.md).  
The analysis for [Tuesday is available here](https://github.com/yxie27/Project2/blob/master/weekday_is_tuesday.md).  
The analysis for [Wednesday is available here](https://github.com/yxie27/Project2/blob/master/weekday_is_wednesday.md).  
The analysis for [Thursday is available here](https://github.com/yxie27/Project2/blob/master/weekday_is_thursday.md).  
The analysis for [Friday is available here](https://github.com/yxie27/Project2/blob/master/weekday_is_friday.md).  
The analysis for [Saturday is available here](https://github.com/yxie27/Project2/blob/master/weekday_is_saturday.md).  
The analysis for [Sunday is available here](https://github.com/yxie27/Project2/blob/master/weekday_is_sunday.md).  

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
#Remove the useless colomns
news1 <- select(news, -url, -timedelta, 
                -weekday_is_monday, -weekday_is_tuesday,
                -weekday_is_wednesday, -weekday_is_thursday, 
                -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, 
                -is_weekend, params$weekday)
r = nrow(news1)
c = ncol(news1)
r
```

    ## [1] 39644

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

    ## [1] 27750

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
    ## n_tokens_title                Min.   : 3.00      1st Qu.: 9.00     
    ## n_tokens_content              Min.   :   0.0     1st Qu.: 246.0    
    ## n_unique_tokens               Min.   :  0.0000   1st Qu.:  0.4707  
    ## n_non_stop_words              Min.   :   0.000   1st Qu.:   1.000  
    ## n_non_stop_unique_tokens      Min.   :  0.0000   1st Qu.:  0.6253  
    ##   num_hrefs                   Min.   :  0.00     1st Qu.:  4.00    
    ## num_self_hrefs                Min.   :  0.000    1st Qu.:  1.000   
    ##    num_imgs                   Min.   :  0.000    1st Qu.:  1.000   
    ##   num_videos                  Min.   : 0.000     1st Qu.: 0.000    
    ## average_token_length          Min.   :0.000      1st Qu.:4.477     
    ##  num_keywords                 Min.   : 1.000     1st Qu.: 6.000    
    ## data_channel_is_lifestyle     Min.   :0.00000    1st Qu.:0.00000   
    ## data_channel_is_entertainment Min.   :0.0000     1st Qu.:0.0000    
    ## data_channel_is_bus           Min.   :0.0000     1st Qu.:0.0000    
    ## data_channel_is_socmed        Min.   :0.00000    1st Qu.:0.00000   
    ## data_channel_is_tech          Min.   :0.0000     1st Qu.:0.0000    
    ## data_channel_is_world         Min.   :0.0000     1st Qu.:0.0000    
    ##   kw_min_min                  Min.   : -1.00     1st Qu.: -1.00    
    ##   kw_max_min                  Min.   :     0     1st Qu.:   444    
    ##   kw_avg_min                  Min.   :   -1.0    1st Qu.:  141.6   
    ##   kw_min_max                  Min.   :     0     1st Qu.:     0    
    ##   kw_max_max                  Min.   :     0     1st Qu.:843300    
    ##   kw_avg_max                  Min.   :     0     1st Qu.:173184    
    ##   kw_min_avg                  Min.   :  -1       1st Qu.:   0      
    ##   kw_max_avg                  Min.   :     0     1st Qu.:  3564    
    ##   kw_avg_avg                  Min.   :    0      1st Qu.: 2383     
    ## self_reference_min_shares     Min.   :     0.0   1st Qu.:   640.2  
    ## self_reference_max_shares     Min.   :     0     1st Qu.:  1000    
    ## self_reference_avg_sharess    Min.   :     0.0   1st Qu.:   975.4  
    ##     LDA_00                    Min.   :0.00000    1st Qu.:0.02505   
    ##     LDA_01                    Min.   :0.00000    1st Qu.:0.02501   
    ##     LDA_02                    Min.   :0.00000    1st Qu.:0.02857   
    ##     LDA_03                    Min.   :0.00000    1st Qu.:0.02857   
    ##     LDA_04                    Min.   :0.00000    1st Qu.:0.02857   
    ## global_subjectivity           Min.   :0.0000     1st Qu.:0.3959    
    ## global_sentiment_polarity     Min.   :-0.39375   1st Qu.: 0.05761  
    ## global_rate_positive_words    Min.   :0.00000    1st Qu.:0.02830   
    ## global_rate_negative_words    Min.   :0.000000   1st Qu.:0.009615  
    ## rate_positive_words           Min.   :0.0000     1st Qu.:0.6000    
    ## rate_negative_words           Min.   :0.0000     1st Qu.:0.1860    
    ## avg_positive_polarity         Min.   :0.0000     1st Qu.:0.3063    
    ## min_positive_polarity         Min.   :0.00000    1st Qu.:0.05000   
    ## max_positive_polarity         Min.   :0.0000     1st Qu.:0.6000    
    ## avg_negative_polarity         Min.   :-1.0000    1st Qu.:-0.3280   
    ## min_negative_polarity         Min.   :-1.0000    1st Qu.:-0.7000   
    ## max_negative_polarity         Min.   :-1.0000    1st Qu.:-0.1250   
    ## title_subjectivity            Min.   :0.0000     1st Qu.:0.0000    
    ## title_sentiment_polarity      Min.   :-1.00000   1st Qu.: 0.00000  
    ## abs_title_subjectivity        Min.   :0.0000     1st Qu.:0.1667    
    ## abs_title_sentiment_polarity  Min.   :0.000000   1st Qu.:0.000000  
    ##     shares                    Min.   :     1     1st Qu.:   943    
    ## weekday_is_monday             Min.   :0.0000     1st Qu.:0.0000    
    ##                                                                    
    ## n_tokens_title                Median :10.00      Mean   :10.41     
    ## n_tokens_content              Median : 409.0     Mean   : 546.6    
    ## n_unique_tokens               Median :  0.5391   Mean   :  0.5554  
    ## n_non_stop_words              Median :   1.000   Mean   :   1.007  
    ## n_non_stop_unique_tokens      Median :  0.6905   Mean   :  0.6958  
    ##   num_hrefs                   Median :  8.00     Mean   : 10.96    
    ## num_self_hrefs                Median :  3.000    Mean   :  3.314   
    ##    num_imgs                   Median :  1.000    Mean   :  4.533   
    ##   num_videos                  Median : 0.000     Mean   : 1.248    
    ## average_token_length          Median :4.664      Mean   :4.545     
    ##  num_keywords                 Median : 7.000     Mean   : 7.219    
    ## data_channel_is_lifestyle     Median :0.00000    Mean   :0.05359   
    ## data_channel_is_entertainment Median :0.0000     Mean   :0.1767    
    ## data_channel_is_bus           Median :0.0000     Mean   :0.1595    
    ## data_channel_is_socmed        Median :0.00000    Mean   :0.05888   
    ## data_channel_is_tech          Median :0.0000     Mean   :0.1843    
    ## data_channel_is_world         Median :0.0000     Mean   :0.2134    
    ##   kw_min_min                  Median : -1.00     Mean   : 26.16    
    ##   kw_max_min                  Median :   659     Mean   :  1154    
    ##   kw_avg_min                  Median :  235.2    Mean   :  312.5   
    ##   kw_min_max                  Median :  1400     Mean   : 13874    
    ##   kw_max_max                  Median :843300     Mean   :751872    
    ##   kw_avg_max                  Median :244999     Mean   :259451    
    ##   kw_min_avg                  Median :1028       Mean   :1119      
    ##   kw_max_avg                  Median :  4358     Mean   :  5653    
    ##   kw_avg_avg                  Median : 2876      Mean   : 3134     
    ## self_reference_min_shares     Median :  1200.0   Mean   :  3972.7  
    ## self_reference_max_shares     Median :  2800     Mean   : 10287    
    ## self_reference_avg_sharess    Median :  2200.0   Mean   :  6350.6  
    ##     LDA_00                    Median :0.03339    Mean   :0.18615   
    ##     LDA_01                    Median :0.03334    Mean   :0.14033   
    ##     LDA_02                    Median :0.04000    Mean   :0.21606   
    ##     LDA_03                    Median :0.04000    Mean   :0.22337   
    ##     LDA_04                    Median :0.04072    Mean   :0.23405   
    ## global_subjectivity           Median :0.4533     Mean   :0.4430    
    ## global_sentiment_polarity     Median : 0.11851   Mean   : 0.11894  
    ## global_rate_positive_words    Median :0.03904    Mean   :0.03957   
    ## global_rate_negative_words    Median :0.015385   Mean   :0.016668  
    ## rate_positive_words           Median :0.7097     Mean   :0.6806    
    ## rate_negative_words           Median :0.2800     Mean   :0.2884    
    ## avg_positive_polarity         Median :0.3590     Mean   :0.3536    
    ## min_positive_polarity         Median :0.10000    Mean   :0.09534   
    ## max_positive_polarity         Median :0.8000     Mean   :0.7559    
    ## avg_negative_polarity         Median :-0.2528    Mean   :-0.2594   
    ## min_negative_polarity         Median :-0.5000    Mean   :-0.5212   
    ## max_negative_polarity         Median :-0.1000    Mean   :-0.1076   
    ## title_subjectivity            Median :0.1500     Mean   :0.2843    
    ## title_sentiment_polarity      Median : 0.00000   Mean   : 0.07096  
    ## abs_title_subjectivity        Median :0.5000     Mean   :0.3414    
    ## abs_title_sentiment_polarity  Median :0.005682   Mean   :0.157838  
    ##     shares                    Median :  1400     Mean   :  3356    
    ## weekday_is_monday             Median :0.0000     Mean   :0.1671    
    ##                                                                    
    ## n_tokens_title                3rd Qu.:12.00      Max.   :23.00     
    ## n_tokens_content              3rd Qu.: 716.0     Max.   :8474.0    
    ## n_unique_tokens               3rd Qu.:  0.6088   Max.   :701.0000  
    ## n_non_stop_words              3rd Qu.:   1.000   Max.   :1042.000  
    ## n_non_stop_unique_tokens      3rd Qu.:  0.7551   Max.   :650.0000  
    ##   num_hrefs                   3rd Qu.: 14.00     Max.   :304.00    
    ## num_self_hrefs                3rd Qu.:  4.000    Max.   :116.000   
    ##    num_imgs                   3rd Qu.:  4.000    Max.   :111.000   
    ##   num_videos                  3rd Qu.: 1.000     Max.   :74.000    
    ## average_token_length          3rd Qu.:4.857      Max.   :8.042     
    ##  num_keywords                 3rd Qu.: 9.000     Max.   :10.000    
    ## data_channel_is_lifestyle     3rd Qu.:0.00000    Max.   :1.00000   
    ## data_channel_is_entertainment 3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_bus           3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_socmed        3rd Qu.:0.00000    Max.   :1.00000   
    ## data_channel_is_tech          3rd Qu.:0.0000     Max.   :1.0000    
    ## data_channel_is_world         3rd Qu.:0.0000     Max.   :1.0000    
    ##   kw_min_min                  3rd Qu.:  4.00     Max.   :377.00    
    ##   kw_max_min                  3rd Qu.:  1000     Max.   :298400    
    ##   kw_avg_min                  3rd Qu.:  356.8    Max.   :42827.9   
    ##   kw_min_max                  3rd Qu.:  7900     Max.   :843300    
    ##   kw_max_max                  3rd Qu.:843300     Max.   :843300    
    ##   kw_avg_max                  3rd Qu.:330479     Max.   :843300    
    ##   kw_min_avg                  3rd Qu.:2063       Max.   :3613      
    ##   kw_max_avg                  3rd Qu.:  6015     Max.   :298400    
    ##   kw_avg_avg                  3rd Qu.: 3598      Max.   :43568     
    ## self_reference_min_shares     3rd Qu.:  2600.0   Max.   :843300.0  
    ## self_reference_max_shares     3rd Qu.:  7900     Max.   :843300    
    ## self_reference_avg_sharess    3rd Qu.:  5100.0   Max.   :843300.0  
    ##     LDA_00                    3rd Qu.:0.24492    Max.   :0.92699   
    ##     LDA_01                    3rd Qu.:0.15002    Max.   :0.91998   
    ##     LDA_02                    3rd Qu.:0.33499    Max.   :0.92000   
    ##     LDA_03                    3rd Qu.:0.37378    Max.   :0.92653   
    ##     LDA_04                    3rd Qu.:0.40010    Max.   :0.92708   
    ## global_subjectivity           3rd Qu.:0.5082     Max.   :1.0000    
    ## global_sentiment_polarity     3rd Qu.: 0.17734   Max.   : 0.72784  
    ## global_rate_positive_words    3rd Qu.:0.05028    Max.   :0.15549   
    ## global_rate_negative_words    3rd Qu.:0.021779   Max.   :0.184932  
    ## rate_positive_words           3rd Qu.:0.8000     Max.   :1.0000    
    ## rate_negative_words           3rd Qu.:0.3846     Max.   :1.0000    
    ## avg_positive_polarity         3rd Qu.:0.4117     Max.   :1.0000    
    ## min_positive_polarity         3rd Qu.:0.10000    Max.   :1.00000   
    ## max_positive_polarity         3rd Qu.:1.0000     Max.   :1.0000    
    ## avg_negative_polarity         3rd Qu.:-0.1863    Max.   : 0.0000   
    ## min_negative_polarity         3rd Qu.:-0.3000    Max.   : 0.0000   
    ## max_negative_polarity         3rd Qu.:-0.0500    Max.   : 0.0000   
    ## title_subjectivity            3rd Qu.:0.5000     Max.   :1.0000    
    ## title_sentiment_polarity      3rd Qu.: 0.15000   Max.   : 1.00000  
    ## abs_title_subjectivity        3rd Qu.:0.5000     Max.   :0.5000    
    ## abs_title_sentiment_polarity  3rd Qu.:0.250000   Max.   :1.000000  
    ##     shares                    3rd Qu.:  2800     Max.   :843300    
    ## weekday_is_monday             3rd Qu.:0.0000     Max.   :1.0000

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
    ##           Mean of squared residuals: 117029469
    ##                     % Var explained: -0.23

``` r
#variable importance measures
importance(rf)
```

    ##                                   %IncMSE IncNodePurity
    ## n_tokens_title                 2.09530898   38717232372
    ## n_tokens_content               3.57172000   43376619991
    ## n_unique_tokens                7.89681977   46077097566
    ## n_non_stop_words               7.83127800   36605780148
    ## n_non_stop_unique_tokens       6.41934151   54500897231
    ## num_hrefs                      5.38680514   72201500770
    ## num_self_hrefs                 3.02551100   44958345870
    ## num_imgs                       3.08050087   41853904687
    ## num_videos                     5.97602549   39085251681
    ## average_token_length           5.39672901   60839160632
    ## num_keywords                   6.14400056   18618931618
    ## data_channel_is_lifestyle      2.89275704    4817375921
    ## data_channel_is_entertainment  2.00192475    4495489092
    ## data_channel_is_bus            4.37763591   23725076020
    ## data_channel_is_socmed         4.20865308    2635575760
    ## data_channel_is_tech           7.02062060    1691304233
    ## data_channel_is_world          6.64030291    2566705817
    ## kw_min_min                     4.98124011   17082201969
    ## kw_max_min                     3.50569175   97678357139
    ## kw_avg_min                     4.85317219  106378912576
    ## kw_min_max                     6.02776526   57478957932
    ## kw_max_max                     4.29493986   68202357461
    ## kw_avg_max                     5.31823273  134905165672
    ## kw_min_avg                     4.35699362   41837149896
    ## kw_max_avg                     5.41099915  183781243743
    ## kw_avg_avg                     4.56404548  184656932814
    ## self_reference_min_shares      7.22657474   55568253406
    ## self_reference_max_shares      6.54292608   65887825619
    ## self_reference_avg_sharess     4.52996978  145610533997
    ## LDA_00                         3.83768590  111616072074
    ## LDA_01                        10.40352642   48793683015
    ## LDA_02                         3.83534375   70414193386
    ## LDA_03                         4.43547958  126258133773
    ## LDA_04                         4.62133727   94073588385
    ## global_subjectivity            5.54999844   82897888332
    ## global_sentiment_polarity      5.45008630   51616774222
    ## global_rate_positive_words     2.62732271   54795655383
    ## global_rate_negative_words     4.58703585   40999766124
    ## rate_positive_words            5.55161485   29860060401
    ## rate_negative_words            9.88090943   29893650696
    ## avg_positive_polarity          2.69563776   49749985752
    ## min_positive_polarity          0.48031805   23204112151
    ## max_positive_polarity          3.75991316   13514393395
    ## avg_negative_polarity          3.38133662   58819335705
    ## min_negative_polarity          1.61300727   31121113401
    ## max_negative_polarity          1.35041612   25444300209
    ## title_subjectivity             3.82099408   62137937848
    ## title_sentiment_polarity       3.89821224   71605739181
    ## abs_title_subjectivity         3.58731869   14284633225
    ## abs_title_sentiment_polarity   3.33395304   33293932065
    ## weekday_is_monday              0.06593767   10328309101

``` r
#draw dotplot of variable importance as measured by Random Forest
varImpPlot(rf)
```

![](README_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Calculate the predicted mean square error on the train set:

``` r
train.pred <- predict(rf, train)
mean((train.pred - train$shares)^2)
```

    ## [1] 25396555

### On test set

``` r
rf.test <- predict(rf, newdata = test)
mean((test$shares-rf.test)^2)
```

    ## [1] 177519651

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

    ## Start:  AIC=514892.4
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
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
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - title_subjectivity             1 2.2798e+05 3.1610e+12 514890
    ## - global_sentiment_polarity      1 4.3288e+05 3.1610e+12 514890
    ## - n_unique_tokens                1 2.3167e+06 3.1610e+12 514890
    ## - global_rate_negative_words     1 3.5858e+06 3.1610e+12 514890
    ## - num_keywords                   1 4.5566e+06 3.1611e+12 514890
    ## - max_positive_polarity          1 6.1711e+06 3.1611e+12 514890
    ## - LDA_00                         1 7.0584e+06 3.1611e+12 514890
    ## - LDA_04                         1 7.0644e+06 3.1611e+12 514890
    ## - LDA_03                         1 7.0668e+06 3.1611e+12 514890
    ## - LDA_02                         1 7.0720e+06 3.1611e+12 514890
    ## - LDA_01                         1 7.0733e+06 3.1611e+12 514890
    ## - avg_negative_polarity          1 7.7354e+06 3.1611e+12 514890
    ## - rate_positive_words            1 9.7750e+06 3.1611e+12 514890
    ## - rate_negative_words            1 1.1284e+07 3.1611e+12 514890
    ## - n_non_stop_words               1 1.5074e+07 3.1611e+12 514891
    ## - n_tokens_content               1 1.6134e+07 3.1611e+12 514891
    ## - avg_positive_polarity          1 2.2956e+07 3.1611e+12 514891
    ## - min_negative_polarity          1 2.6657e+07 3.1611e+12 514891
    ## - title_sentiment_polarity       1 3.6982e+07 3.1611e+12 514891
    ## - global_rate_positive_words     1 4.7807e+07 3.1611e+12 514891
    ## - self_reference_avg_sharess     1 4.9481e+07 3.1611e+12 514891
    ## - max_negative_polarity          1 5.2152e+07 3.1611e+12 514891
    ## - n_non_stop_unique_tokens       1 7.9774e+07 3.1611e+12 514891
    ## - kw_min_max                     1 8.0138e+07 3.1611e+12 514891
    ## - kw_max_max                     1 9.4039e+07 3.1611e+12 514891
    ## - kw_min_min                     1 9.7063e+07 3.1611e+12 514891
    ## - average_token_length           1 1.5556e+08 3.1612e+12 514892
    ## - num_imgs                       1 1.7192e+08 3.1612e+12 514892
    ## <none>                                        3.1610e+12 514892
    ## - num_videos                     1 2.3496e+08 3.1613e+12 514892
    ## - min_positive_polarity          1 2.4069e+08 3.1613e+12 514892
    ## - self_reference_max_shares      1 2.4345e+08 3.1613e+12 514893
    ## - kw_max_min                     1 3.0322e+08 3.1613e+12 514893
    ## - kw_avg_max                     1 3.1783e+08 3.1614e+12 514893
    ## - self_reference_min_shares      1 3.3479e+08 3.1614e+12 514893
    ## - kw_avg_min                     1 3.6837e+08 3.1614e+12 514894
    ## - abs_title_sentiment_polarity   1 4.9591e+08 3.1615e+12 514895
    ## - abs_title_subjectivity         1 7.5351e+08 3.1618e+12 514897
    ## - data_channel_is_world          1 7.9930e+08 3.1618e+12 514897
    ## - n_tokens_title                 1 8.8279e+08 3.1619e+12 514898
    ## - num_self_hrefs                 1 9.2652e+08 3.1620e+12 514899
    ## - weekday_is_monday              1 9.6768e+08 3.1620e+12 514899
    ## - global_subjectivity            1 9.9847e+08 3.1620e+12 514899
    ## - data_channel_is_socmed         1 1.0202e+09 3.1621e+12 514899
    ## - data_channel_is_tech           1 1.0252e+09 3.1621e+12 514899
    ## - data_channel_is_lifestyle      1 1.4752e+09 3.1625e+12 514903
    ## - kw_min_avg                     1 1.8656e+09 3.1629e+12 514907
    ## - data_channel_is_bus            1 1.9444e+09 3.1630e+12 514907
    ## - num_hrefs                      1 2.9787e+09 3.1640e+12 514917
    ## - data_channel_is_entertainment  1 3.3896e+09 3.1644e+12 514920
    ## - kw_max_avg                     1 5.6820e+09 3.1667e+12 514940
    ## - kw_avg_avg                     1 1.1547e+10 3.1726e+12 514992
    ## 
    ## Step:  AIC=514890.4
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
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
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - global_sentiment_polarity      1 4.4748e+05 3.1610e+12 514888
    ## - n_unique_tokens                1 2.3274e+06 3.1610e+12 514888
    ## - global_rate_negative_words     1 3.5812e+06 3.1610e+12 514888
    ## - num_keywords                   1 4.5388e+06 3.1611e+12 514888
    ## - max_positive_polarity          1 6.1555e+06 3.1611e+12 514888
    ## - LDA_00                         1 7.0354e+06 3.1611e+12 514888
    ## - LDA_04                         1 7.0414e+06 3.1611e+12 514888
    ## - LDA_03                         1 7.0437e+06 3.1611e+12 514888
    ## - LDA_02                         1 7.0489e+06 3.1611e+12 514888
    ## - LDA_01                         1 7.0502e+06 3.1611e+12 514888
    ## - avg_negative_polarity          1 7.7055e+06 3.1611e+12 514888
    ## - rate_positive_words            1 9.7676e+06 3.1611e+12 514888
    ## - rate_negative_words            1 1.1273e+07 3.1611e+12 514888
    ## - n_non_stop_words               1 1.5036e+07 3.1611e+12 514889
    ## - n_tokens_content               1 1.6078e+07 3.1611e+12 514889
    ## - avg_positive_polarity          1 2.2867e+07 3.1611e+12 514889
    ## - min_negative_polarity          1 2.6698e+07 3.1611e+12 514889
    ## - title_sentiment_polarity       1 3.8143e+07 3.1611e+12 514889
    ## - global_rate_positive_words     1 4.8023e+07 3.1611e+12 514889
    ## - self_reference_avg_sharess     1 4.9496e+07 3.1611e+12 514889
    ## - max_negative_polarity          1 5.2128e+07 3.1611e+12 514889
    ## - n_non_stop_unique_tokens       1 7.9796e+07 3.1611e+12 514889
    ## - kw_min_max                     1 8.0170e+07 3.1611e+12 514889
    ## - kw_max_max                     1 9.3947e+07 3.1611e+12 514889
    ## - kw_min_min                     1 9.7139e+07 3.1611e+12 514889
    ## - average_token_length           1 1.5558e+08 3.1612e+12 514890
    ## - num_imgs                       1 1.7182e+08 3.1612e+12 514890
    ## <none>                                        3.1610e+12 514890
    ## - num_videos                     1 2.3491e+08 3.1613e+12 514890
    ## - min_positive_polarity          1 2.4088e+08 3.1613e+12 514891
    ## - self_reference_max_shares      1 2.4345e+08 3.1613e+12 514891
    ## - kw_max_min                     1 3.0333e+08 3.1613e+12 514891
    ## - kw_avg_max                     1 3.1794e+08 3.1614e+12 514891
    ## - self_reference_min_shares      1 3.3488e+08 3.1614e+12 514891
    ## - kw_avg_min                     1 3.6857e+08 3.1614e+12 514892
    ## - data_channel_is_world          1 7.9929e+08 3.1618e+12 514895
    ## - abs_title_sentiment_polarity   1 8.1929e+08 3.1619e+12 514896
    ## - abs_title_subjectivity         1 8.4389e+08 3.1619e+12 514896
    ## - n_tokens_title                 1 8.8259e+08 3.1619e+12 514896
    ## - num_self_hrefs                 1 9.2663e+08 3.1620e+12 514897
    ## - weekday_is_monday              1 9.6802e+08 3.1620e+12 514897
    ## - global_subjectivity            1 1.0083e+09 3.1621e+12 514897
    ## - data_channel_is_socmed         1 1.0200e+09 3.1621e+12 514897
    ## - data_channel_is_tech           1 1.0253e+09 3.1621e+12 514897
    ## - data_channel_is_lifestyle      1 1.4752e+09 3.1625e+12 514901
    ## - kw_min_avg                     1 1.8653e+09 3.1629e+12 514905
    ## - data_channel_is_bus            1 1.9443e+09 3.1630e+12 514905
    ## - num_hrefs                      1 2.9786e+09 3.1640e+12 514915
    ## - data_channel_is_entertainment  1 3.3894e+09 3.1644e+12 514918
    ## - kw_max_avg                     1 5.6818e+09 3.1667e+12 514938
    ## - kw_avg_avg                     1 1.1546e+10 3.1726e+12 514990
    ## 
    ## Step:  AIC=514888.4
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     LDA_04 + global_subjectivity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_unique_tokens                1 2.3317e+06 3.1610e+12 514886
    ## - num_keywords                   1 4.5191e+06 3.1611e+12 514886
    ## - global_rate_negative_words     1 4.9424e+06 3.1611e+12 514886
    ## - max_positive_polarity          1 6.1962e+06 3.1611e+12 514886
    ## - LDA_00                         1 7.0499e+06 3.1611e+12 514886
    ## - LDA_04                         1 7.0559e+06 3.1611e+12 514886
    ## - LDA_03                         1 7.0583e+06 3.1611e+12 514886
    ## - LDA_02                         1 7.0635e+06 3.1611e+12 514886
    ## - LDA_01                         1 7.0648e+06 3.1611e+12 514886
    ## - rate_positive_words            1 9.7823e+06 3.1611e+12 514886
    ## - avg_negative_polarity          1 1.0106e+07 3.1611e+12 514886
    ## - rate_negative_words            1 1.1086e+07 3.1611e+12 514886
    ## - n_non_stop_words               1 1.5036e+07 3.1611e+12 514887
    ## - n_tokens_content               1 1.5897e+07 3.1611e+12 514887
    ## - min_negative_polarity          1 2.6885e+07 3.1611e+12 514887
    ## - avg_positive_polarity          1 2.9709e+07 3.1611e+12 514887
    ## - title_sentiment_polarity       1 3.9170e+07 3.1611e+12 514887
    ## - self_reference_avg_sharess     1 4.9470e+07 3.1611e+12 514887
    ## - global_rate_positive_words     1 5.3390e+07 3.1611e+12 514887
    ## - max_negative_polarity          1 5.5186e+07 3.1611e+12 514887
    ## - n_non_stop_unique_tokens       1 7.9588e+07 3.1611e+12 514887
    ## - kw_min_max                     1 8.0317e+07 3.1611e+12 514887
    ## - kw_max_max                     1 9.4000e+07 3.1611e+12 514887
    ## - kw_min_min                     1 9.7072e+07 3.1611e+12 514887
    ## - average_token_length           1 1.5587e+08 3.1612e+12 514888
    ## - num_imgs                       1 1.7193e+08 3.1612e+12 514888
    ## <none>                                        3.1610e+12 514888
    ## - num_videos                     1 2.3470e+08 3.1613e+12 514888
    ## - self_reference_max_shares      1 2.4333e+08 3.1613e+12 514889
    ## - min_positive_polarity          1 2.4778e+08 3.1613e+12 514889
    ## - kw_max_min                     1 3.0316e+08 3.1613e+12 514889
    ## - kw_avg_max                     1 3.1780e+08 3.1614e+12 514889
    ## - self_reference_min_shares      1 3.3487e+08 3.1614e+12 514889
    ## - kw_avg_min                     1 3.6837e+08 3.1614e+12 514890
    ## - data_channel_is_world          1 7.9889e+08 3.1618e+12 514893
    ## - abs_title_sentiment_polarity   1 8.1888e+08 3.1619e+12 514894
    ## - abs_title_subjectivity         1 8.4593e+08 3.1619e+12 514894
    ## - n_tokens_title                 1 8.8263e+08 3.1619e+12 514894
    ## - num_self_hrefs                 1 9.2667e+08 3.1620e+12 514895
    ## - weekday_is_monday              1 9.6795e+08 3.1620e+12 514895
    ## - data_channel_is_socmed         1 1.0199e+09 3.1621e+12 514895
    ## - data_channel_is_tech           1 1.0250e+09 3.1621e+12 514895
    ## - global_subjectivity            1 1.0531e+09 3.1621e+12 514896
    ## - data_channel_is_lifestyle      1 1.4747e+09 3.1625e+12 514899
    ## - kw_min_avg                     1 1.8649e+09 3.1629e+12 514903
    ## - data_channel_is_bus            1 1.9442e+09 3.1630e+12 514903
    ## - num_hrefs                      1 2.9879e+09 3.1640e+12 514913
    ## - data_channel_is_entertainment  1 3.3893e+09 3.1644e+12 514916
    ## - kw_max_avg                     1 5.6820e+09 3.1667e+12 514936
    ## - kw_avg_avg                     1 1.1548e+10 3.1726e+12 514988
    ## 
    ## Step:  AIC=514886.4
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + rate_negative_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - num_keywords                   1 4.6032e+06 3.1611e+12 514884
    ## - global_rate_negative_words     1 5.1413e+06 3.1611e+12 514884
    ## - LDA_00                         1 6.4443e+06 3.1611e+12 514884
    ## - LDA_04                         1 6.4500e+06 3.1611e+12 514884
    ## - LDA_03                         1 6.4523e+06 3.1611e+12 514884
    ## - LDA_02                         1 6.4573e+06 3.1611e+12 514884
    ## - LDA_01                         1 6.4585e+06 3.1611e+12 514884
    ## - max_positive_polarity          1 7.5349e+06 3.1611e+12 514884
    ## - rate_positive_words            1 9.9616e+06 3.1611e+12 514885
    ## - avg_negative_polarity          1 1.0104e+07 3.1611e+12 514885
    ## - rate_negative_words            1 1.1279e+07 3.1611e+12 514885
    ## - n_non_stop_words               1 1.4911e+07 3.1611e+12 514885
    ## - n_tokens_content               1 2.7686e+07 3.1611e+12 514885
    ## - min_negative_polarity          1 2.9209e+07 3.1611e+12 514885
    ## - avg_positive_polarity          1 2.9829e+07 3.1611e+12 514885
    ## - title_sentiment_polarity       1 3.9547e+07 3.1611e+12 514885
    ## - self_reference_avg_sharess     1 4.9790e+07 3.1611e+12 514885
    ## - max_negative_polarity          1 5.3347e+07 3.1611e+12 514885
    ## - global_rate_positive_words     1 5.5231e+07 3.1611e+12 514885
    ## - kw_min_max                     1 8.0041e+07 3.1611e+12 514885
    ## - kw_max_max                     1 9.3284e+07 3.1611e+12 514885
    ## - kw_min_min                     1 9.7032e+07 3.1611e+12 514885
    ## - num_imgs                       1 1.7056e+08 3.1612e+12 514886
    ## - average_token_length           1 1.8048e+08 3.1612e+12 514886
    ## <none>                                        3.1610e+12 514886
    ## - num_videos                     1 2.3238e+08 3.1613e+12 514886
    ## - n_non_stop_unique_tokens       1 2.3560e+08 3.1613e+12 514886
    ## - self_reference_max_shares      1 2.4415e+08 3.1613e+12 514887
    ## - min_positive_polarity          1 2.6392e+08 3.1613e+12 514887
    ## - kw_max_min                     1 3.0308e+08 3.1614e+12 514887
    ## - kw_avg_max                     1 3.1990e+08 3.1614e+12 514887
    ## - self_reference_min_shares      1 3.3554e+08 3.1614e+12 514887
    ## - kw_avg_min                     1 3.6809e+08 3.1614e+12 514888
    ## - data_channel_is_world          1 7.9762e+08 3.1618e+12 514891
    ## - abs_title_sentiment_polarity   1 8.1726e+08 3.1619e+12 514892
    ## - abs_title_subjectivity         1 8.4807e+08 3.1619e+12 514892
    ## - n_tokens_title                 1 8.8260e+08 3.1619e+12 514892
    ## - num_self_hrefs                 1 9.2779e+08 3.1620e+12 514893
    ## - weekday_is_monday              1 9.6763e+08 3.1620e+12 514893
    ## - data_channel_is_socmed         1 1.0176e+09 3.1621e+12 514893
    ## - data_channel_is_tech           1 1.0232e+09 3.1621e+12 514893
    ## - global_subjectivity            1 1.0519e+09 3.1621e+12 514894
    ## - data_channel_is_lifestyle      1 1.4751e+09 3.1625e+12 514897
    ## - kw_min_avg                     1 1.8686e+09 3.1629e+12 514901
    ## - data_channel_is_bus            1 1.9437e+09 3.1630e+12 514901
    ## - num_hrefs                      1 3.0015e+09 3.1641e+12 514911
    ## - data_channel_is_entertainment  1 3.3905e+09 3.1644e+12 514914
    ## - kw_max_avg                     1 5.6864e+09 3.1667e+12 514934
    ## - kw_avg_avg                     1 1.1552e+10 3.1726e+12 514986
    ## 
    ## Step:  AIC=514884.5
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + rate_negative_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - global_rate_negative_words     1 5.2299e+06 3.1611e+12 514882
    ## - LDA_00                         1 6.3114e+06 3.1611e+12 514883
    ## - LDA_04                         1 6.3172e+06 3.1611e+12 514883
    ## - LDA_03                         1 6.3194e+06 3.1611e+12 514883
    ## - LDA_02                         1 6.3243e+06 3.1611e+12 514883
    ## - LDA_01                         1 6.3255e+06 3.1611e+12 514883
    ## - max_positive_polarity          1 7.4210e+06 3.1611e+12 514883
    ## - rate_positive_words            1 9.9801e+06 3.1611e+12 514883
    ## - avg_negative_polarity          1 1.0116e+07 3.1611e+12 514883
    ## - rate_negative_words            1 1.1345e+07 3.1611e+12 514883
    ## - n_non_stop_words               1 1.4731e+07 3.1611e+12 514883
    ## - n_tokens_content               1 2.7388e+07 3.1611e+12 514883
    ## - min_negative_polarity          1 2.9312e+07 3.1611e+12 514883
    ## - avg_positive_polarity          1 3.0180e+07 3.1611e+12 514883
    ## - title_sentiment_polarity       1 3.9344e+07 3.1611e+12 514883
    ## - self_reference_avg_sharess     1 5.0289e+07 3.1611e+12 514883
    ## - max_negative_polarity          1 5.3250e+07 3.1611e+12 514883
    ## - global_rate_positive_words     1 5.5943e+07 3.1611e+12 514883
    ## - kw_min_max                     1 7.8663e+07 3.1611e+12 514883
    ## - kw_min_min                     1 9.7275e+07 3.1612e+12 514883
    ## - kw_max_max                     1 1.0184e+08 3.1612e+12 514883
    ## - num_imgs                       1 1.7099e+08 3.1612e+12 514884
    ## - average_token_length           1 1.8246e+08 3.1612e+12 514884
    ## <none>                                        3.1611e+12 514884
    ## - num_videos                     1 2.3113e+08 3.1613e+12 514884
    ## - n_non_stop_unique_tokens       1 2.3660e+08 3.1613e+12 514885
    ## - self_reference_max_shares      1 2.4496e+08 3.1613e+12 514885
    ## - min_positive_polarity          1 2.6346e+08 3.1613e+12 514885
    ## - kw_max_min                     1 2.9903e+08 3.1614e+12 514885
    ## - kw_avg_max                     1 3.3630e+08 3.1614e+12 514885
    ## - self_reference_min_shares      1 3.3703e+08 3.1614e+12 514885
    ## - kw_avg_min                     1 3.6355e+08 3.1614e+12 514886
    ## - data_channel_is_world          1 7.9490e+08 3.1618e+12 514889
    ## - abs_title_sentiment_polarity   1 8.1639e+08 3.1619e+12 514890
    ## - abs_title_subjectivity         1 8.4784e+08 3.1619e+12 514890
    ## - n_tokens_title                 1 8.7902e+08 3.1619e+12 514890
    ## - num_self_hrefs                 1 9.3969e+08 3.1620e+12 514891
    ## - weekday_is_monday              1 9.6932e+08 3.1620e+12 514891
    ## - data_channel_is_socmed         1 1.0157e+09 3.1621e+12 514891
    ## - data_channel_is_tech           1 1.0203e+09 3.1621e+12 514891
    ## - global_subjectivity            1 1.0506e+09 3.1621e+12 514892
    ## - data_channel_is_lifestyle      1 1.4730e+09 3.1625e+12 514895
    ## - kw_min_avg                     1 1.9209e+09 3.1630e+12 514899
    ## - data_channel_is_bus            1 1.9391e+09 3.1630e+12 514899
    ## - num_hrefs                      1 2.9969e+09 3.1641e+12 514909
    ## - data_channel_is_entertainment  1 3.4042e+09 3.1645e+12 514912
    ## - kw_max_avg                     1 5.6896e+09 3.1667e+12 514932
    ## - kw_avg_avg                     1 1.1749e+10 3.1728e+12 514985
    ## 
    ## Step:  AIC=514882.5
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - LDA_00                         1 6.3521e+06 3.1611e+12 514881
    ## - LDA_04                         1 6.3579e+06 3.1611e+12 514881
    ## - LDA_03                         1 6.3602e+06 3.1611e+12 514881
    ## - LDA_02                         1 6.3651e+06 3.1611e+12 514881
    ## - LDA_01                         1 6.3663e+06 3.1611e+12 514881
    ## - max_positive_polarity          1 6.9812e+06 3.1611e+12 514881
    ## - rate_negative_words            1 9.6740e+06 3.1611e+12 514881
    ## - avg_negative_polarity          1 9.9541e+06 3.1611e+12 514881
    ## - rate_positive_words            1 1.0621e+07 3.1611e+12 514881
    ## - n_non_stop_words               1 1.4769e+07 3.1611e+12 514881
    ## - n_tokens_content               1 2.8058e+07 3.1611e+12 514881
    ## - min_negative_polarity          1 2.8694e+07 3.1611e+12 514881
    ## - avg_positive_polarity          1 3.0470e+07 3.1611e+12 514881
    ## - title_sentiment_polarity       1 4.1572e+07 3.1611e+12 514881
    ## - self_reference_avg_sharess     1 5.0448e+07 3.1611e+12 514881
    ## - max_negative_polarity          1 5.5029e+07 3.1611e+12 514881
    ## - kw_min_max                     1 7.8738e+07 3.1611e+12 514881
    ## - kw_min_min                     1 9.6540e+07 3.1612e+12 514881
    ## - kw_max_max                     1 1.0256e+08 3.1612e+12 514881
    ## - global_rate_positive_words     1 1.3478e+08 3.1612e+12 514882
    ## - num_imgs                       1 1.6938e+08 3.1612e+12 514882
    ## - average_token_length           1 1.8030e+08 3.1612e+12 514882
    ## - num_videos                     1 2.2603e+08 3.1613e+12 514882
    ## <none>                                        3.1611e+12 514882
    ## - n_non_stop_unique_tokens       1 2.3556e+08 3.1613e+12 514883
    ## - self_reference_max_shares      1 2.4559e+08 3.1613e+12 514883
    ## - min_positive_polarity          1 2.5987e+08 3.1613e+12 514883
    ## - kw_max_min                     1 2.9902e+08 3.1614e+12 514883
    ## - kw_avg_max                     1 3.3655e+08 3.1614e+12 514883
    ## - self_reference_min_shares      1 3.3686e+08 3.1614e+12 514883
    ## - kw_avg_min                     1 3.6402e+08 3.1614e+12 514884
    ## - data_channel_is_world          1 7.9041e+08 3.1618e+12 514887
    ## - abs_title_sentiment_polarity   1 8.1406e+08 3.1619e+12 514888
    ## - abs_title_subjectivity         1 8.5408e+08 3.1619e+12 514888
    ## - n_tokens_title                 1 8.8158e+08 3.1619e+12 514888
    ## - num_self_hrefs                 1 9.3914e+08 3.1620e+12 514889
    ## - weekday_is_monday              1 9.6744e+08 3.1620e+12 514889
    ## - data_channel_is_socmed         1 1.0122e+09 3.1621e+12 514889
    ## - data_channel_is_tech           1 1.0167e+09 3.1621e+12 514889
    ## - global_subjectivity            1 1.0464e+09 3.1621e+12 514890
    ## - data_channel_is_lifestyle      1 1.4691e+09 3.1625e+12 514893
    ## - kw_min_avg                     1 1.9202e+09 3.1630e+12 514897
    ## - data_channel_is_bus            1 1.9343e+09 3.1630e+12 514897
    ## - num_hrefs                      1 3.0283e+09 3.1641e+12 514907
    ## - data_channel_is_entertainment  1 3.4011e+09 3.1645e+12 514910
    ## - kw_max_avg                     1 5.6868e+09 3.1667e+12 514930
    ## - kw_avg_avg                     1 1.1747e+10 3.1728e+12 514983
    ## 
    ## Step:  AIC=514880.6
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - max_positive_polarity          1 6.7883e+06 3.1611e+12 514879
    ## - rate_negative_words            1 8.2668e+06 3.1611e+12 514879
    ## - avg_negative_polarity          1 9.8924e+06 3.1611e+12 514879
    ## - rate_positive_words            1 1.2372e+07 3.1611e+12 514879
    ## - n_tokens_content               1 2.6994e+07 3.1611e+12 514879
    ## - min_negative_polarity          1 2.8645e+07 3.1611e+12 514879
    ## - avg_positive_polarity          1 3.0038e+07 3.1611e+12 514879
    ## - title_sentiment_polarity       1 4.1440e+07 3.1611e+12 514879
    ## - self_reference_avg_sharess     1 5.0723e+07 3.1611e+12 514879
    ## - max_negative_polarity          1 5.5078e+07 3.1611e+12 514879
    ## - kw_min_max                     1 7.8532e+07 3.1611e+12 514879
    ## - kw_min_min                     1 9.6640e+07 3.1612e+12 514879
    ## - kw_max_max                     1 1.0251e+08 3.1612e+12 514879
    ## - global_rate_positive_words     1 1.3464e+08 3.1612e+12 514880
    ## - num_imgs                       1 1.6998e+08 3.1612e+12 514880
    ## - average_token_length           1 2.0011e+08 3.1613e+12 514880
    ## - LDA_04                         1 2.2017e+08 3.1613e+12 514880
    ## - num_videos                     1 2.2585e+08 3.1613e+12 514881
    ## <none>                                        3.1611e+12 514881
    ## - n_non_stop_words               1 2.2988e+08 3.1613e+12 514881
    ## - n_non_stop_unique_tokens       1 2.3271e+08 3.1613e+12 514881
    ## - self_reference_max_shares      1 2.4601e+08 3.1613e+12 514881
    ## - min_positive_polarity          1 2.6032e+08 3.1613e+12 514881
    ## - kw_max_min                     1 2.9958e+08 3.1614e+12 514881
    ## - kw_avg_max                     1 3.3646e+08 3.1614e+12 514882
    ## - self_reference_min_shares      1 3.3749e+08 3.1614e+12 514882
    ## - LDA_03                         1 3.5076e+08 3.1614e+12 514882
    ## - kw_avg_min                     1 3.6466e+08 3.1614e+12 514882
    ## - data_channel_is_world          1 7.8998e+08 3.1619e+12 514885
    ## - abs_title_sentiment_polarity   1 8.1488e+08 3.1619e+12 514886
    ## - abs_title_subjectivity         1 8.5337e+08 3.1619e+12 514886
    ## - n_tokens_title                 1 8.8150e+08 3.1619e+12 514886
    ## - LDA_02                         1 9.2663e+08 3.1620e+12 514887
    ## - num_self_hrefs                 1 9.4244e+08 3.1620e+12 514887
    ## - weekday_is_monday              1 9.6655e+08 3.1620e+12 514887
    ## - LDA_01                         1 9.8644e+08 3.1621e+12 514887
    ## - data_channel_is_socmed         1 1.0166e+09 3.1621e+12 514887
    ## - data_channel_is_tech           1 1.0191e+09 3.1621e+12 514888
    ## - global_subjectivity            1 1.0406e+09 3.1621e+12 514888
    ## - data_channel_is_lifestyle      1 1.4717e+09 3.1625e+12 514891
    ## - kw_min_avg                     1 1.9244e+09 3.1630e+12 514895
    ## - data_channel_is_bus            1 1.9384e+09 3.1630e+12 514896
    ## - num_hrefs                      1 3.0464e+09 3.1641e+12 514905
    ## - data_channel_is_entertainment  1 3.4063e+09 3.1645e+12 514908
    ## - kw_max_avg                     1 5.6912e+09 3.1668e+12 514928
    ## - kw_avg_avg                     1 1.1757e+10 3.1728e+12 514982
    ## 
    ## Step:  AIC=514878.6
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - rate_negative_words            1 9.4703e+06 3.1611e+12 514877
    ## - avg_negative_polarity          1 9.6417e+06 3.1611e+12 514877
    ## - rate_positive_words            1 1.4677e+07 3.1611e+12 514877
    ## - avg_positive_polarity          1 2.4436e+07 3.1611e+12 514877
    ## - min_negative_polarity          1 3.0368e+07 3.1611e+12 514877
    ## - n_tokens_content               1 3.2309e+07 3.1611e+12 514877
    ## - title_sentiment_polarity       1 4.1205e+07 3.1611e+12 514877
    ## - self_reference_avg_sharess     1 5.0949e+07 3.1611e+12 514877
    ## - max_negative_polarity          1 5.3217e+07 3.1611e+12 514877
    ## - kw_min_max                     1 7.8542e+07 3.1612e+12 514877
    ## - kw_min_min                     1 9.6750e+07 3.1612e+12 514877
    ## - kw_max_max                     1 1.0167e+08 3.1612e+12 514878
    ## - global_rate_positive_words     1 1.2831e+08 3.1612e+12 514878
    ## - num_imgs                       1 1.6990e+08 3.1612e+12 514878
    ## - average_token_length           1 2.0422e+08 3.1613e+12 514878
    ## - LDA_04                         1 2.2014e+08 3.1613e+12 514879
    ## - n_non_stop_words               1 2.2518e+08 3.1613e+12 514879
    ## <none>                                        3.1611e+12 514879
    ## - n_non_stop_unique_tokens       1 2.2799e+08 3.1613e+12 514879
    ## - num_videos                     1 2.2873e+08 3.1613e+12 514879
    ## - self_reference_max_shares      1 2.4634e+08 3.1613e+12 514879
    ## - kw_max_min                     1 2.9883e+08 3.1614e+12 514879
    ## - min_positive_polarity          1 3.1653e+08 3.1614e+12 514879
    ## - kw_avg_max                     1 3.3675e+08 3.1614e+12 514880
    ## - self_reference_min_shares      1 3.3776e+08 3.1614e+12 514880
    ## - LDA_03                         1 3.5115e+08 3.1614e+12 514880
    ## - kw_avg_min                     1 3.6328e+08 3.1614e+12 514880
    ## - data_channel_is_world          1 7.9151e+08 3.1619e+12 514884
    ## - abs_title_sentiment_polarity   1 8.1381e+08 3.1619e+12 514884
    ## - abs_title_subjectivity         1 8.5201e+08 3.1619e+12 514884
    ## - n_tokens_title                 1 8.8411e+08 3.1620e+12 514884
    ## - LDA_02                         1 9.2843e+08 3.1620e+12 514885
    ## - num_self_hrefs                 1 9.4402e+08 3.1620e+12 514885
    ## - weekday_is_monday              1 9.6759e+08 3.1620e+12 514885
    ## - LDA_01                         1 9.8816e+08 3.1621e+12 514885
    ## - data_channel_is_tech           1 1.0220e+09 3.1621e+12 514886
    ## - data_channel_is_socmed         1 1.0231e+09 3.1621e+12 514886
    ## - global_subjectivity            1 1.0343e+09 3.1621e+12 514886
    ## - data_channel_is_lifestyle      1 1.4742e+09 3.1625e+12 514890
    ## - kw_min_avg                     1 1.9264e+09 3.1630e+12 514894
    ## - data_channel_is_bus            1 1.9440e+09 3.1630e+12 514894
    ## - num_hrefs                      1 3.0564e+09 3.1641e+12 514903
    ## - data_channel_is_entertainment  1 3.4061e+09 3.1645e+12 514906
    ## - kw_max_avg                     1 5.6923e+09 3.1668e+12 514927
    ## - kw_avg_avg                     1 1.1755e+10 3.1728e+12 514980
    ## 
    ## Step:  AIC=514876.7
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - rate_positive_words            1 6.0574e+06 3.1611e+12 514875
    ## - avg_negative_polarity          1 8.9318e+06 3.1611e+12 514875
    ## - avg_positive_polarity          1 2.1413e+07 3.1611e+12 514875
    ## - min_negative_polarity          1 3.3451e+07 3.1611e+12 514875
    ## - n_tokens_content               1 3.9566e+07 3.1611e+12 514875
    ## - title_sentiment_polarity       1 4.0343e+07 3.1611e+12 514875
    ## - self_reference_avg_sharess     1 5.0513e+07 3.1611e+12 514875
    ## - max_negative_polarity          1 5.1330e+07 3.1611e+12 514875
    ## - kw_min_max                     1 7.8700e+07 3.1612e+12 514875
    ## - kw_min_min                     1 9.8042e+07 3.1612e+12 514876
    ## - kw_max_max                     1 1.0106e+08 3.1612e+12 514876
    ## - global_rate_positive_words     1 1.2883e+08 3.1612e+12 514876
    ## - num_imgs                       1 1.6749e+08 3.1612e+12 514876
    ## - LDA_04                         1 2.1628e+08 3.1613e+12 514877
    ## <none>                                        3.1611e+12 514877
    ## - num_videos                     1 2.3069e+08 3.1613e+12 514877
    ## - n_non_stop_words               1 2.4290e+08 3.1613e+12 514877
    ## - self_reference_max_shares      1 2.4552e+08 3.1613e+12 514877
    ## - n_non_stop_unique_tokens       1 2.4555e+08 3.1613e+12 514877
    ## - kw_max_min                     1 2.9694e+08 3.1614e+12 514877
    ## - min_positive_polarity          1 3.1005e+08 3.1614e+12 514877
    ## - kw_avg_max                     1 3.3700e+08 3.1614e+12 514878
    ## - self_reference_min_shares      1 3.3758e+08 3.1614e+12 514878
    ## - LDA_03                         1 3.4989e+08 3.1614e+12 514878
    ## - kw_avg_min                     1 3.6119e+08 3.1614e+12 514878
    ## - average_token_length           1 5.6803e+08 3.1616e+12 514880
    ## - data_channel_is_world          1 7.8961e+08 3.1619e+12 514882
    ## - abs_title_sentiment_polarity   1 8.1040e+08 3.1619e+12 514882
    ## - abs_title_subjectivity         1 8.5056e+08 3.1619e+12 514882
    ## - n_tokens_title                 1 9.0728e+08 3.1620e+12 514883
    ## - LDA_02                         1 9.3169e+08 3.1620e+12 514883
    ## - num_self_hrefs                 1 9.3524e+08 3.1620e+12 514883
    ## - weekday_is_monday              1 9.6859e+08 3.1620e+12 514883
    ## - LDA_01                         1 9.8571e+08 3.1621e+12 514883
    ## - data_channel_is_tech           1 1.0143e+09 3.1621e+12 514884
    ## - data_channel_is_socmed         1 1.0150e+09 3.1621e+12 514884
    ## - global_subjectivity            1 1.1123e+09 3.1622e+12 514884
    ## - data_channel_is_lifestyle      1 1.4672e+09 3.1625e+12 514888
    ## - kw_min_avg                     1 1.9208e+09 3.1630e+12 514892
    ## - data_channel_is_bus            1 1.9360e+09 3.1630e+12 514892
    ## - num_hrefs                      1 3.1129e+09 3.1642e+12 514902
    ## - data_channel_is_entertainment  1 3.4002e+09 3.1645e+12 514905
    ## - kw_max_avg                     1 5.6832e+09 3.1668e+12 514925
    ## - kw_avg_avg                     1 1.1747e+10 3.1728e+12 514978
    ## 
    ## Step:  AIC=514874.8
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - avg_negative_polarity          1 1.0016e+07 3.1611e+12 514873
    ## - avg_positive_polarity          1 1.9767e+07 3.1611e+12 514873
    ## - min_negative_polarity          1 2.8935e+07 3.1611e+12 514873
    ## - n_tokens_content               1 4.3119e+07 3.1611e+12 514873
    ## - title_sentiment_polarity       1 4.4398e+07 3.1611e+12 514873
    ## - self_reference_avg_sharess     1 5.0587e+07 3.1611e+12 514873
    ## - max_negative_polarity          1 6.0220e+07 3.1611e+12 514873
    ## - kw_min_max                     1 7.8517e+07 3.1612e+12 514873
    ## - kw_min_min                     1 9.8255e+07 3.1612e+12 514874
    ## - kw_max_max                     1 1.0097e+08 3.1612e+12 514874
    ## - global_rate_positive_words     1 1.3573e+08 3.1612e+12 514874
    ## - num_imgs                       1 1.6529e+08 3.1613e+12 514874
    ## - LDA_04                         1 2.1595e+08 3.1613e+12 514875
    ## - num_videos                     1 2.2717e+08 3.1613e+12 514875
    ## <none>                                        3.1611e+12 514875
    ## - n_non_stop_words               1 2.4071e+08 3.1613e+12 514875
    ## - n_non_stop_unique_tokens       1 2.4331e+08 3.1613e+12 514875
    ## - self_reference_max_shares      1 2.4573e+08 3.1613e+12 514875
    ## - kw_max_min                     1 2.9699e+08 3.1614e+12 514875
    ## - min_positive_polarity          1 3.2017e+08 3.1614e+12 514876
    ## - kw_avg_max                     1 3.3605e+08 3.1614e+12 514876
    ## - self_reference_min_shares      1 3.3802e+08 3.1614e+12 514876
    ## - LDA_03                         1 3.5203e+08 3.1614e+12 514876
    ## - kw_avg_min                     1 3.6107e+08 3.1614e+12 514876
    ## - average_token_length           1 6.3974e+08 3.1617e+12 514878
    ## - data_channel_is_world          1 7.8671e+08 3.1619e+12 514880
    ## - abs_title_sentiment_polarity   1 8.0510e+08 3.1619e+12 514880
    ## - abs_title_subjectivity         1 8.6126e+08 3.1619e+12 514880
    ## - n_tokens_title                 1 9.1270e+08 3.1620e+12 514881
    ## - num_self_hrefs                 1 9.3192e+08 3.1620e+12 514881
    ## - LDA_02                         1 9.4007e+08 3.1620e+12 514881
    ## - weekday_is_monday              1 9.6872e+08 3.1621e+12 514881
    ## - LDA_01                         1 9.9043e+08 3.1621e+12 514881
    ## - data_channel_is_tech           1 1.0091e+09 3.1621e+12 514882
    ## - data_channel_is_socmed         1 1.0100e+09 3.1621e+12 514882
    ## - global_subjectivity            1 1.1538e+09 3.1622e+12 514883
    ## - data_channel_is_lifestyle      1 1.4626e+09 3.1625e+12 514886
    ## - kw_min_avg                     1 1.9225e+09 3.1630e+12 514890
    ## - data_channel_is_bus            1 1.9308e+09 3.1630e+12 514890
    ## - num_hrefs                      1 3.1214e+09 3.1642e+12 514900
    ## - data_channel_is_entertainment  1 3.3956e+09 3.1645e+12 514903
    ## - kw_max_avg                     1 5.6779e+09 3.1668e+12 514923
    ## - kw_avg_avg                     1 1.1741e+10 3.1728e+12 514976
    ## 
    ## Step:  AIC=514872.8
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - avg_positive_polarity          1 1.9524e+07 3.1611e+12 514871
    ## - min_negative_polarity          1 2.3900e+07 3.1611e+12 514871
    ## - title_sentiment_polarity       1 4.9571e+07 3.1611e+12 514871
    ## - self_reference_avg_sharess     1 5.0973e+07 3.1611e+12 514871
    ## - n_tokens_content               1 5.1661e+07 3.1611e+12 514871
    ## - max_negative_polarity          1 6.7168e+07 3.1612e+12 514871
    ## - kw_min_max                     1 7.7951e+07 3.1612e+12 514872
    ## - kw_min_min                     1 9.8537e+07 3.1612e+12 514872
    ## - kw_max_max                     1 1.0005e+08 3.1612e+12 514872
    ## - global_rate_positive_words     1 1.3257e+08 3.1612e+12 514872
    ## - num_imgs                       1 1.6309e+08 3.1613e+12 514872
    ## - LDA_04                         1 2.1499e+08 3.1613e+12 514873
    ## - num_videos                     1 2.2228e+08 3.1613e+12 514873
    ## <none>                                        3.1611e+12 514873
    ## - n_non_stop_words               1 2.4101e+08 3.1613e+12 514873
    ## - n_non_stop_unique_tokens       1 2.4361e+08 3.1613e+12 514873
    ## - self_reference_max_shares      1 2.4634e+08 3.1613e+12 514873
    ## - kw_max_min                     1 2.9652e+08 3.1614e+12 514873
    ## - min_positive_polarity          1 3.1639e+08 3.1614e+12 514874
    ## - kw_avg_max                     1 3.3859e+08 3.1614e+12 514874
    ## - self_reference_min_shares      1 3.3904e+08 3.1614e+12 514874
    ## - LDA_03                         1 3.5258e+08 3.1614e+12 514874
    ## - kw_avg_min                     1 3.6022e+08 3.1615e+12 514874
    ## - average_token_length           1 6.4309e+08 3.1617e+12 514876
    ## - data_channel_is_world          1 7.8412e+08 3.1619e+12 514878
    ## - abs_title_sentiment_polarity   1 7.9606e+08 3.1619e+12 514878
    ## - abs_title_subjectivity         1 8.6001e+08 3.1620e+12 514878
    ## - n_tokens_title                 1 9.1221e+08 3.1620e+12 514879
    ## - num_self_hrefs                 1 9.3468e+08 3.1620e+12 514879
    ## - LDA_02                         1 9.3714e+08 3.1620e+12 514879
    ## - weekday_is_monday              1 9.6683e+08 3.1621e+12 514879
    ## - LDA_01                         1 9.9305e+08 3.1621e+12 514880
    ## - data_channel_is_tech           1 1.0056e+09 3.1621e+12 514880
    ## - data_channel_is_socmed         1 1.0067e+09 3.1621e+12 514880
    ## - global_subjectivity            1 1.1513e+09 3.1622e+12 514881
    ## - data_channel_is_lifestyle      1 1.4595e+09 3.1626e+12 514884
    ## - kw_min_avg                     1 1.9217e+09 3.1630e+12 514888
    ## - data_channel_is_bus            1 1.9261e+09 3.1630e+12 514888
    ## - num_hrefs                      1 3.1229e+09 3.1642e+12 514898
    ## - data_channel_is_entertainment  1 3.4071e+09 3.1645e+12 514901
    ## - kw_max_avg                     1 5.6830e+09 3.1668e+12 514921
    ## - kw_avg_avg                     1 1.1745e+10 3.1728e+12 514974
    ## 
    ## Step:  AIC=514871
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + min_positive_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - min_negative_polarity          1 2.2672e+07 3.1611e+12 514869
    ## - n_tokens_content               1 4.5570e+07 3.1612e+12 514869
    ## - title_sentiment_polarity       1 4.6158e+07 3.1612e+12 514869
    ## - self_reference_avg_sharess     1 5.1332e+07 3.1612e+12 514869
    ## - max_negative_polarity          1 7.1260e+07 3.1612e+12 514870
    ## - kw_min_max                     1 7.8469e+07 3.1612e+12 514870
    ## - kw_min_min                     1 9.9131e+07 3.1612e+12 514870
    ## - kw_max_max                     1 9.9474e+07 3.1612e+12 514870
    ## - global_rate_positive_words     1 1.5052e+08 3.1613e+12 514870
    ## - num_imgs                       1 1.6029e+08 3.1613e+12 514870
    ## - LDA_04                         1 2.1211e+08 3.1613e+12 514871
    ## - num_videos                     1 2.1622e+08 3.1613e+12 514871
    ## <none>                                        3.1611e+12 514871
    ## - n_non_stop_words               1 2.3887e+08 3.1614e+12 514871
    ## - n_non_stop_unique_tokens       1 2.4149e+08 3.1614e+12 514871
    ## - self_reference_max_shares      1 2.4676e+08 3.1614e+12 514871
    ## - kw_max_min                     1 2.9765e+08 3.1614e+12 514872
    ## - kw_avg_max                     1 3.3597e+08 3.1615e+12 514872
    ## - self_reference_min_shares      1 3.4005e+08 3.1615e+12 514872
    ## - LDA_03                         1 3.5271e+08 3.1615e+12 514872
    ## - kw_avg_min                     1 3.6038e+08 3.1615e+12 514872
    ## - min_positive_polarity          1 4.8282e+08 3.1616e+12 514873
    ## - average_token_length           1 7.3266e+08 3.1618e+12 514875
    ## - data_channel_is_world          1 7.8073e+08 3.1619e+12 514876
    ## - abs_title_sentiment_polarity   1 7.8253e+08 3.1619e+12 514876
    ## - abs_title_subjectivity         1 8.4499e+08 3.1620e+12 514876
    ## - n_tokens_title                 1 9.0925e+08 3.1620e+12 514877
    ## - LDA_02                         1 9.3095e+08 3.1620e+12 514877
    ## - num_self_hrefs                 1 9.3179e+08 3.1620e+12 514877
    ## - weekday_is_monday              1 9.6347e+08 3.1621e+12 514877
    ## - LDA_01                         1 9.8963e+08 3.1621e+12 514878
    ## - data_channel_is_socmed         1 1.0054e+09 3.1621e+12 514878
    ## - data_channel_is_tech           1 1.0057e+09 3.1621e+12 514878
    ## - global_subjectivity            1 1.1664e+09 3.1623e+12 514879
    ## - data_channel_is_lifestyle      1 1.4697e+09 3.1626e+12 514882
    ## - kw_min_avg                     1 1.9232e+09 3.1630e+12 514886
    ## - data_channel_is_bus            1 1.9273e+09 3.1630e+12 514886
    ## - num_hrefs                      1 3.1084e+09 3.1642e+12 514896
    ## - data_channel_is_entertainment  1 3.4209e+09 3.1645e+12 514899
    ## - kw_max_avg                     1 5.6869e+09 3.1668e+12 514919
    ## - kw_avg_avg                     1 1.1742e+10 3.1729e+12 514972
    ## 
    ## Step:  AIC=514869.2
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + min_positive_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - title_sentiment_polarity       1 3.8971e+07 3.1612e+12 514868
    ## - self_reference_avg_sharess     1 5.1599e+07 3.1612e+12 514868
    ## - kw_min_max                     1 7.7951e+07 3.1612e+12 514868
    ## - n_tokens_content               1 8.2113e+07 3.1612e+12 514868
    ## - max_negative_polarity          1 8.3871e+07 3.1612e+12 514868
    ## - kw_max_max                     1 9.8692e+07 3.1612e+12 514868
    ## - kw_min_min                     1 9.8881e+07 3.1612e+12 514868
    ## - num_imgs                       1 1.5563e+08 3.1613e+12 514869
    ## - global_rate_positive_words     1 1.6186e+08 3.1613e+12 514869
    ## - LDA_04                         1 2.1511e+08 3.1614e+12 514869
    ## - num_videos                     1 2.2515e+08 3.1614e+12 514869
    ## <none>                                        3.1611e+12 514869
    ## - n_non_stop_words               1 2.4240e+08 3.1614e+12 514869
    ## - n_non_stop_unique_tokens       1 2.4501e+08 3.1614e+12 514869
    ## - self_reference_max_shares      1 2.4812e+08 3.1614e+12 514869
    ## - kw_max_min                     1 2.9714e+08 3.1614e+12 514870
    ## - kw_avg_max                     1 3.3547e+08 3.1615e+12 514870
    ## - self_reference_min_shares      1 3.4040e+08 3.1615e+12 514870
    ## - LDA_03                         1 3.5205e+08 3.1615e+12 514870
    ## - kw_avg_min                     1 3.6016e+08 3.1615e+12 514870
    ## - min_positive_polarity          1 5.1457e+08 3.1617e+12 514872
    ## - average_token_length           1 7.1945e+08 3.1619e+12 514874
    ## - data_channel_is_world          1 7.7942e+08 3.1619e+12 514874
    ## - abs_title_sentiment_polarity   1 8.0808e+08 3.1619e+12 514874
    ## - abs_title_subjectivity         1 8.4646e+08 3.1620e+12 514875
    ## - n_tokens_title                 1 9.1452e+08 3.1621e+12 514875
    ## - LDA_02                         1 9.2872e+08 3.1621e+12 514875
    ## - num_self_hrefs                 1 9.3751e+08 3.1621e+12 514875
    ## - weekday_is_monday              1 9.6492e+08 3.1621e+12 514876
    ## - LDA_01                         1 9.8581e+08 3.1621e+12 514876
    ## - data_channel_is_socmed         1 1.0150e+09 3.1622e+12 514876
    ## - data_channel_is_tech           1 1.0213e+09 3.1622e+12 514876
    ## - global_subjectivity            1 1.3280e+09 3.1625e+12 514879
    ## - data_channel_is_lifestyle      1 1.4707e+09 3.1626e+12 514880
    ## - kw_min_avg                     1 1.9252e+09 3.1631e+12 514884
    ## - data_channel_is_bus            1 1.9476e+09 3.1631e+12 514884
    ## - num_hrefs                      1 3.1168e+09 3.1643e+12 514895
    ## - data_channel_is_entertainment  1 3.4190e+09 3.1646e+12 514897
    ## - kw_max_avg                     1 5.6882e+09 3.1668e+12 514917
    ## - kw_avg_avg                     1 1.1749e+10 3.1729e+12 514970
    ## 
    ## Step:  AIC=514867.5
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + min_positive_polarity + max_negative_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - self_reference_avg_sharess     1 5.2290e+07 3.1612e+12 514866
    ## - kw_min_max                     1 7.7400e+07 3.1613e+12 514866
    ## - n_tokens_content               1 8.0612e+07 3.1613e+12 514866
    ## - max_negative_polarity          1 8.4034e+07 3.1613e+12 514866
    ## - kw_max_max                     1 9.8826e+07 3.1613e+12 514866
    ## - kw_min_min                     1 9.9939e+07 3.1613e+12 514866
    ## - global_rate_positive_words     1 1.4779e+08 3.1613e+12 514867
    ## - num_imgs                       1 1.5901e+08 3.1613e+12 514867
    ## - LDA_04                         1 2.1741e+08 3.1614e+12 514867
    ## - num_videos                     1 2.2656e+08 3.1614e+12 514868
    ## <none>                                        3.1612e+12 514868
    ## - n_non_stop_words               1 2.3733e+08 3.1614e+12 514868
    ## - n_non_stop_unique_tokens       1 2.3992e+08 3.1614e+12 514868
    ## - self_reference_max_shares      1 2.4944e+08 3.1614e+12 514868
    ## - kw_max_min                     1 2.9763e+08 3.1615e+12 514868
    ## - kw_avg_max                     1 3.3616e+08 3.1615e+12 514869
    ## - self_reference_min_shares      1 3.4241e+08 3.1615e+12 514869
    ## - LDA_03                         1 3.5427e+08 3.1615e+12 514869
    ## - kw_avg_min                     1 3.6099e+08 3.1615e+12 514869
    ## - min_positive_polarity          1 5.0495e+08 3.1617e+12 514870
    ## - average_token_length           1 7.2562e+08 3.1619e+12 514872
    ## - data_channel_is_world          1 7.8183e+08 3.1620e+12 514872
    ## - abs_title_subjectivity         1 8.2245e+08 3.1620e+12 514873
    ## - n_tokens_title                 1 9.1088e+08 3.1621e+12 514874
    ## - LDA_02                         1 9.2808e+08 3.1621e+12 514874
    ## - num_self_hrefs                 1 9.3582e+08 3.1621e+12 514874
    ## - weekday_is_monday              1 9.6349e+08 3.1621e+12 514874
    ## - LDA_01                         1 9.9324e+08 3.1622e+12 514874
    ## - data_channel_is_socmed         1 1.0087e+09 3.1622e+12 514874
    ## - data_channel_is_tech           1 1.0124e+09 3.1622e+12 514874
    ## - abs_title_sentiment_polarity   1 1.0485e+09 3.1622e+12 514875
    ## - global_subjectivity            1 1.3122e+09 3.1625e+12 514877
    ## - data_channel_is_lifestyle      1 1.4585e+09 3.1626e+12 514878
    ## - kw_min_avg                     1 1.9248e+09 3.1631e+12 514882
    ## - data_channel_is_bus            1 1.9406e+09 3.1631e+12 514883
    ## - num_hrefs                      1 3.1258e+09 3.1643e+12 514893
    ## - data_channel_is_entertainment  1 3.4163e+09 3.1646e+12 514896
    ## - kw_max_avg                     1 5.6904e+09 3.1669e+12 514915
    ## - kw_avg_avg                     1 1.1755e+10 3.1729e+12 514969
    ## 
    ## Step:  AIC=514866
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + global_rate_positive_words + min_positive_polarity + 
    ##     max_negative_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - kw_min_max                     1 7.5869e+07 3.1613e+12 514865
    ## - n_tokens_content               1 8.1141e+07 3.1613e+12 514865
    ## - max_negative_polarity          1 8.3559e+07 3.1613e+12 514865
    ## - kw_max_max                     1 9.9447e+07 3.1613e+12 514865
    ## - kw_min_min                     1 9.9552e+07 3.1613e+12 514865
    ## - global_rate_positive_words     1 1.4601e+08 3.1614e+12 514865
    ## - num_imgs                       1 1.5696e+08 3.1614e+12 514865
    ## - LDA_04                         1 2.1777e+08 3.1614e+12 514866
    ## <none>                                        3.1612e+12 514866
    ## - num_videos                     1 2.3329e+08 3.1615e+12 514866
    ## - n_non_stop_words               1 2.3967e+08 3.1615e+12 514866
    ## - n_non_stop_unique_tokens       1 2.4223e+08 3.1615e+12 514866
    ## - kw_max_min                     1 3.0324e+08 3.1615e+12 514867
    ## - kw_avg_max                     1 3.3851e+08 3.1616e+12 514867
    ## - LDA_03                         1 3.5402e+08 3.1616e+12 514867
    ## - kw_avg_min                     1 3.6552e+08 3.1616e+12 514867
    ## - min_positive_polarity          1 5.0257e+08 3.1617e+12 514868
    ## - self_reference_max_shares      1 5.3812e+08 3.1618e+12 514869
    ## - average_token_length           1 7.3456e+08 3.1620e+12 514870
    ## - self_reference_min_shares      1 7.6478e+08 3.1620e+12 514871
    ## - data_channel_is_world          1 7.7710e+08 3.1620e+12 514871
    ## - abs_title_subjectivity         1 8.2285e+08 3.1621e+12 514871
    ## - num_self_hrefs                 1 8.9279e+08 3.1621e+12 514872
    ## - n_tokens_title                 1 9.0466e+08 3.1621e+12 514872
    ## - LDA_02                         1 9.2727e+08 3.1622e+12 514872
    ## - weekday_is_monday              1 9.5974e+08 3.1622e+12 514872
    ## - LDA_01                         1 9.9312e+08 3.1622e+12 514873
    ## - data_channel_is_socmed         1 1.0074e+09 3.1622e+12 514873
    ## - data_channel_is_tech           1 1.0142e+09 3.1622e+12 514873
    ## - abs_title_sentiment_polarity   1 1.0496e+09 3.1623e+12 514873
    ## - global_subjectivity            1 1.3032e+09 3.1625e+12 514875
    ## - data_channel_is_lifestyle      1 1.4540e+09 3.1627e+12 514877
    ## - data_channel_is_bus            1 1.9335e+09 3.1632e+12 514881
    ## - kw_min_avg                     1 1.9350e+09 3.1632e+12 514881
    ## - num_hrefs                      1 3.1535e+09 3.1644e+12 514892
    ## - data_channel_is_entertainment  1 3.4117e+09 3.1646e+12 514894
    ## - kw_max_avg                     1 5.7115e+09 3.1669e+12 514914
    ## - kw_avg_avg                     1 1.1763e+10 3.1730e+12 514967
    ## 
    ## Step:  AIC=514864.7
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + min_positive_polarity + max_negative_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - kw_max_max                     1 7.7061e+07 3.1614e+12 514863
    ## - n_tokens_content               1 7.8995e+07 3.1614e+12 514863
    ## - max_negative_polarity          1 8.5273e+07 3.1614e+12 514863
    ## - kw_min_min                     1 9.6630e+07 3.1614e+12 514864
    ## - global_rate_positive_words     1 1.4966e+08 3.1615e+12 514864
    ## - num_imgs                       1 1.5459e+08 3.1615e+12 514864
    ## - LDA_04                         1 2.2439e+08 3.1615e+12 514865
    ## <none>                                        3.1613e+12 514865
    ## - num_videos                     1 2.3802e+08 3.1615e+12 514865
    ## - n_non_stop_words               1 2.4137e+08 3.1615e+12 514865
    ## - n_non_stop_unique_tokens       1 2.4396e+08 3.1616e+12 514865
    ## - kw_max_min                     1 3.1121e+08 3.1616e+12 514865
    ## - LDA_03                         1 3.5117e+08 3.1617e+12 514866
    ## - kw_avg_min                     1 3.7210e+08 3.1617e+12 514866
    ## - min_positive_polarity          1 5.0847e+08 3.1618e+12 514867
    ## - self_reference_max_shares      1 5.4462e+08 3.1619e+12 514867
    ## - kw_avg_max                     1 5.7886e+08 3.1619e+12 514868
    ## - average_token_length           1 7.4666e+08 3.1621e+12 514869
    ## - self_reference_min_shares      1 7.6110e+08 3.1621e+12 514869
    ## - data_channel_is_world          1 8.1281e+08 3.1621e+12 514870
    ## - abs_title_subjectivity         1 8.2762e+08 3.1621e+12 514870
    ## - num_self_hrefs                 1 8.7978e+08 3.1622e+12 514870
    ## - n_tokens_title                 1 9.1544e+08 3.1622e+12 514871
    ## - LDA_02                         1 9.2206e+08 3.1622e+12 514871
    ## - weekday_is_monday              1 9.6976e+08 3.1623e+12 514871
    ## - LDA_01                         1 1.0064e+09 3.1623e+12 514872
    ## - data_channel_is_tech           1 1.0400e+09 3.1623e+12 514872
    ## - abs_title_sentiment_polarity   1 1.0500e+09 3.1624e+12 514872
    ## - data_channel_is_socmed         1 1.0828e+09 3.1624e+12 514872
    ## - global_subjectivity            1 1.3080e+09 3.1626e+12 514874
    ## - data_channel_is_lifestyle      1 1.5016e+09 3.1628e+12 514876
    ## - data_channel_is_bus            1 1.9431e+09 3.1632e+12 514880
    ## - kw_min_avg                     1 2.1672e+09 3.1635e+12 514882
    ## - num_hrefs                      1 3.1467e+09 3.1645e+12 514890
    ## - data_channel_is_entertainment  1 3.5233e+09 3.1648e+12 514894
    ## - kw_max_avg                     1 5.8060e+09 3.1671e+12 514914
    ## - kw_avg_avg                     1 1.1937e+10 3.1732e+12 514967
    ## 
    ## Step:  AIC=514863.3
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + min_positive_polarity + max_negative_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_tokens_content               1 7.5690e+07 3.1615e+12 514862
    ## - max_negative_polarity          1 8.5590e+07 3.1615e+12 514862
    ## - global_rate_positive_words     1 1.4736e+08 3.1615e+12 514863
    ## - num_imgs                       1 1.5101e+08 3.1615e+12 514863
    ## <none>                                        3.1614e+12 514863
    ## - LDA_04                         1 2.3069e+08 3.1616e+12 514863
    ## - n_non_stop_words               1 2.4497e+08 3.1616e+12 514864
    ## - num_videos                     1 2.4509e+08 3.1616e+12 514864
    ## - n_non_stop_unique_tokens       1 2.4759e+08 3.1616e+12 514864
    ## - kw_max_min                     1 2.9505e+08 3.1617e+12 514864
    ## - LDA_03                         1 3.3490e+08 3.1617e+12 514864
    ## - kw_avg_min                     1 3.5344e+08 3.1617e+12 514864
    ## - min_positive_polarity          1 5.0963e+08 3.1619e+12 514866
    ## - self_reference_max_shares      1 5.5320e+08 3.1619e+12 514866
    ## - kw_min_min                     1 6.4214e+08 3.1620e+12 514867
    ## - self_reference_min_shares      1 7.5421e+08 3.1621e+12 514868
    ## - average_token_length           1 7.5518e+08 3.1621e+12 514868
    ## - kw_avg_max                     1 7.6423e+08 3.1621e+12 514868
    ## - abs_title_subjectivity         1 8.3385e+08 3.1622e+12 514869
    ## - data_channel_is_world          1 8.6492e+08 3.1622e+12 514869
    ## - num_self_hrefs                 1 8.7033e+08 3.1623e+12 514869
    ## - n_tokens_title                 1 9.0348e+08 3.1623e+12 514869
    ## - LDA_02                         1 9.2095e+08 3.1623e+12 514869
    ## - weekday_is_monday              1 9.8108e+08 3.1624e+12 514870
    ## - LDA_01                         1 1.0072e+09 3.1624e+12 514870
    ## - abs_title_sentiment_polarity   1 1.0554e+09 3.1624e+12 514871
    ## - data_channel_is_tech           1 1.0713e+09 3.1625e+12 514871
    ## - data_channel_is_socmed         1 1.1206e+09 3.1625e+12 514871
    ## - global_subjectivity            1 1.3063e+09 3.1627e+12 514873
    ## - data_channel_is_lifestyle      1 1.5306e+09 3.1629e+12 514875
    ## - data_channel_is_bus            1 1.9543e+09 3.1633e+12 514879
    ## - kw_min_avg                     1 2.0989e+09 3.1635e+12 514880
    ## - num_hrefs                      1 3.1188e+09 3.1645e+12 514889
    ## - data_channel_is_entertainment  1 3.6846e+09 3.1651e+12 514894
    ## - kw_max_avg                     1 5.7310e+09 3.1671e+12 514912
    ## - kw_avg_avg                     1 1.1897e+10 3.1733e+12 514966
    ## 
    ## Step:  AIC=514862
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + global_rate_positive_words + min_positive_polarity + 
    ##     max_negative_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - max_negative_polarity          1 6.2255e+07 3.1615e+12 514861
    ## - global_rate_positive_words     1 1.2815e+08 3.1616e+12 514861
    ## - n_non_stop_words               1 1.7494e+08 3.1616e+12 514862
    ## - n_non_stop_unique_tokens       1 1.7738e+08 3.1616e+12 514862
    ## - num_imgs                       1 1.7981e+08 3.1616e+12 514862
    ## <none>                                        3.1615e+12 514862
    ## - LDA_04                         1 2.3497e+08 3.1617e+12 514862
    ## - num_videos                     1 2.8960e+08 3.1617e+12 514863
    ## - kw_max_min                     1 2.9515e+08 3.1618e+12 514863
    ## - kw_avg_min                     1 3.5209e+08 3.1618e+12 514863
    ## - LDA_03                         1 3.5904e+08 3.1618e+12 514863
    ## - self_reference_max_shares      1 5.5054e+08 3.1620e+12 514865
    ## - min_positive_polarity          1 5.7893e+08 3.1620e+12 514865
    ## - kw_min_min                     1 6.3607e+08 3.1621e+12 514866
    ## - average_token_length           1 7.2624e+08 3.1622e+12 514866
    ## - self_reference_min_shares      1 7.5307e+08 3.1622e+12 514867
    ## - kw_avg_max                     1 7.5770e+08 3.1622e+12 514867
    ## - num_self_hrefs                 1 8.2808e+08 3.1623e+12 514867
    ## - data_channel_is_world          1 8.3946e+08 3.1623e+12 514867
    ## - abs_title_subjectivity         1 8.4737e+08 3.1623e+12 514867
    ## - LDA_02                         1 9.1370e+08 3.1624e+12 514868
    ## - n_tokens_title                 1 9.2260e+08 3.1624e+12 514868
    ## - weekday_is_monday              1 9.7694e+08 3.1624e+12 514869
    ## - LDA_01                         1 1.0186e+09 3.1625e+12 514869
    ## - data_channel_is_tech           1 1.0490e+09 3.1625e+12 514869
    ## - abs_title_sentiment_polarity   1 1.0543e+09 3.1625e+12 514869
    ## - data_channel_is_socmed         1 1.0991e+09 3.1626e+12 514870
    ## - global_subjectivity            1 1.3896e+09 3.1628e+12 514872
    ## - data_channel_is_lifestyle      1 1.4918e+09 3.1630e+12 514873
    ## - data_channel_is_bus            1 1.9213e+09 3.1634e+12 514877
    ## - kw_min_avg                     1 2.0941e+09 3.1636e+12 514878
    ## - num_hrefs                      1 3.4583e+09 3.1649e+12 514890
    ## - data_channel_is_entertainment  1 3.6110e+09 3.1651e+12 514892
    ## - kw_max_avg                     1 5.7086e+09 3.1672e+12 514910
    ## - kw_avg_avg                     1 1.1850e+10 3.1733e+12 514964
    ## 
    ## Step:  AIC=514860.6
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + global_rate_positive_words + min_positive_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - global_rate_positive_words     1 1.3311e+08 3.1617e+12 514860
    ## - num_imgs                       1 1.7817e+08 3.1617e+12 514860
    ## - n_non_stop_words               1 2.0033e+08 3.1617e+12 514860
    ## - n_non_stop_unique_tokens       1 2.0295e+08 3.1617e+12 514860
    ## <none>                                        3.1615e+12 514861
    ## - LDA_04                         1 2.3710e+08 3.1618e+12 514861
    ## - num_videos                     1 2.7457e+08 3.1618e+12 514861
    ## - kw_max_min                     1 2.9435e+08 3.1618e+12 514861
    ## - kw_avg_min                     1 3.5196e+08 3.1619e+12 514862
    ## - LDA_03                         1 3.5365e+08 3.1619e+12 514862
    ## - self_reference_max_shares      1 5.5084e+08 3.1621e+12 514863
    ## - min_positive_polarity          1 5.5784e+08 3.1621e+12 514863
    ## - kw_min_min                     1 6.4300e+08 3.1622e+12 514864
    ## - average_token_length           1 6.9006e+08 3.1622e+12 514865
    ## - kw_avg_max                     1 7.5195e+08 3.1623e+12 514865
    ## - self_reference_min_shares      1 7.6283e+08 3.1623e+12 514865
    ## - num_self_hrefs                 1 8.3168e+08 3.1624e+12 514866
    ## - abs_title_subjectivity         1 8.4735e+08 3.1624e+12 514866
    ## - data_channel_is_world          1 8.5070e+08 3.1624e+12 514866
    ## - LDA_02                         1 9.2020e+08 3.1624e+12 514867
    ## - n_tokens_title                 1 9.2521e+08 3.1624e+12 514867
    ## - weekday_is_monday              1 9.7449e+08 3.1625e+12 514867
    ## - LDA_01                         1 1.0153e+09 3.1625e+12 514867
    ## - abs_title_sentiment_polarity   1 1.0571e+09 3.1626e+12 514868
    ## - data_channel_is_tech           1 1.0595e+09 3.1626e+12 514868
    ## - data_channel_is_socmed         1 1.1019e+09 3.1626e+12 514868
    ## - global_subjectivity            1 1.4600e+09 3.1630e+12 514871
    ## - data_channel_is_lifestyle      1 1.5083e+09 3.1630e+12 514872
    ## - data_channel_is_bus            1 1.9362e+09 3.1635e+12 514876
    ## - kw_min_avg                     1 2.0992e+09 3.1636e+12 514877
    ## - num_hrefs                      1 3.4179e+09 3.1649e+12 514889
    ## - data_channel_is_entertainment  1 3.6329e+09 3.1652e+12 514890
    ## - kw_max_avg                     1 5.7134e+09 3.1672e+12 514909
    ## - kw_avg_avg                     1 1.1868e+10 3.1734e+12 514963
    ## 
    ## Step:  AIC=514859.7
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_non_stop_words               1 1.7384e+08 3.1618e+12 514859
    ## - n_non_stop_unique_tokens       1 1.7633e+08 3.1618e+12 514859
    ## - num_imgs                       1 1.9105e+08 3.1618e+12 514859
    ## - LDA_04                         1 2.2575e+08 3.1619e+12 514860
    ## <none>                                        3.1617e+12 514860
    ## - num_videos                     1 2.5619e+08 3.1619e+12 514860
    ## - kw_max_min                     1 2.9775e+08 3.1620e+12 514860
    ## - LDA_03                         1 3.3153e+08 3.1620e+12 514861
    ## - kw_avg_min                     1 3.5538e+08 3.1620e+12 514861
    ## - min_positive_polarity          1 4.4889e+08 3.1621e+12 514862
    ## - self_reference_max_shares      1 5.5722e+08 3.1622e+12 514863
    ## - kw_min_min                     1 6.3438e+08 3.1623e+12 514863
    ## - kw_avg_max                     1 7.1592e+08 3.1624e+12 514864
    ## - self_reference_min_shares      1 7.7326e+08 3.1624e+12 514865
    ## - average_token_length           1 7.7682e+08 3.1624e+12 514865
    ## - data_channel_is_world          1 8.0861e+08 3.1625e+12 514865
    ## - LDA_02                         1 8.8112e+08 3.1625e+12 514865
    ## - num_self_hrefs                 1 8.8333e+08 3.1625e+12 514865
    ## - n_tokens_title                 1 9.5362e+08 3.1626e+12 514866
    ## - abs_title_subjectivity         1 9.5505e+08 3.1626e+12 514866
    ## - weekday_is_monday              1 9.8032e+08 3.1626e+12 514866
    ## - LDA_01                         1 1.0017e+09 3.1627e+12 514867
    ## - abs_title_sentiment_polarity   1 1.0357e+09 3.1627e+12 514867
    ## - data_channel_is_tech           1 1.0587e+09 3.1627e+12 514867
    ## - data_channel_is_socmed         1 1.1173e+09 3.1628e+12 514868
    ## - global_subjectivity            1 1.3311e+09 3.1630e+12 514869
    ## - data_channel_is_lifestyle      1 1.5112e+09 3.1632e+12 514871
    ## - data_channel_is_bus            1 1.9250e+09 3.1636e+12 514875
    ## - kw_min_avg                     1 2.1148e+09 3.1638e+12 514876
    ## - num_hrefs                      1 3.5325e+09 3.1652e+12 514889
    ## - data_channel_is_entertainment  1 3.6050e+09 3.1653e+12 514889
    ## - kw_max_avg                     1 5.7187e+09 3.1674e+12 514908
    ## - kw_avg_avg                     1 1.1867e+10 3.1735e+12 514962
    ## 
    ## Step:  AIC=514859.3
    ## shares ~ n_tokens_title + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_non_stop_unique_tokens       1 2.1731e+07 3.1619e+12 514857
    ## - num_imgs                       1 8.4414e+07 3.1619e+12 514858
    ## - LDA_04                         1 2.2753e+08 3.1621e+12 514859
    ## <none>                                        3.1618e+12 514859
    ## - num_videos                     1 2.3525e+08 3.1621e+12 514859
    ## - kw_max_min                     1 2.9700e+08 3.1621e+12 514860
    ## - LDA_03                         1 3.1005e+08 3.1621e+12 514860
    ## - kw_avg_min                     1 3.5520e+08 3.1622e+12 514860
    ## - min_positive_polarity          1 3.6704e+08 3.1622e+12 514860
    ## - self_reference_max_shares      1 5.7042e+08 3.1624e+12 514862
    ## - kw_min_min                     1 6.7098e+08 3.1625e+12 514863
    ## - kw_avg_max                     1 6.9444e+08 3.1625e+12 514863
    ## - average_token_length           1 7.1162e+08 3.1625e+12 514864
    ## - self_reference_min_shares      1 7.6965e+08 3.1626e+12 514864
    ## - data_channel_is_world          1 8.2902e+08 3.1627e+12 514865
    ## - LDA_02                         1 8.8566e+08 3.1627e+12 514865
    ## - num_self_hrefs                 1 8.9471e+08 3.1627e+12 514865
    ## - n_tokens_title                 1 9.3022e+08 3.1628e+12 514865
    ## - abs_title_subjectivity         1 9.4390e+08 3.1628e+12 514866
    ## - LDA_01                         1 9.5813e+08 3.1628e+12 514866
    ## - weekday_is_monday              1 9.8450e+08 3.1628e+12 514866
    ## - abs_title_sentiment_polarity   1 1.0148e+09 3.1628e+12 514866
    ## - data_channel_is_tech           1 1.0799e+09 3.1629e+12 514867
    ## - data_channel_is_socmed         1 1.1300e+09 3.1630e+12 514867
    ## - global_subjectivity            1 1.3828e+09 3.1632e+12 514869
    ## - data_channel_is_lifestyle      1 1.5155e+09 3.1633e+12 514871
    ## - data_channel_is_bus            1 1.9442e+09 3.1638e+12 514874
    ## - kw_min_avg                     1 2.1319e+09 3.1640e+12 514876
    ## - num_hrefs                      1 3.3587e+09 3.1652e+12 514887
    ## - data_channel_is_entertainment  1 3.6911e+09 3.1655e+12 514890
    ## - kw_max_avg                     1 5.7245e+09 3.1676e+12 514907
    ## - kw_avg_avg                     1 1.1883e+10 3.1737e+12 514961
    ## 
    ## Step:  AIC=514857.5
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - num_imgs                       1 8.6870e+07 3.1619e+12 514856
    ## <none>                                        3.1619e+12 514857
    ## - LDA_04                         1 2.3147e+08 3.1621e+12 514857
    ## - num_videos                     1 2.3583e+08 3.1621e+12 514858
    ## - kw_max_min                     1 2.9705e+08 3.1621e+12 514858
    ## - LDA_03                         1 3.1690e+08 3.1622e+12 514858
    ## - kw_avg_min                     1 3.5521e+08 3.1622e+12 514859
    ## - min_positive_polarity          1 3.6627e+08 3.1622e+12 514859
    ## - self_reference_max_shares      1 5.7011e+08 3.1624e+12 514860
    ## - kw_min_min                     1 6.7183e+08 3.1625e+12 514861
    ## - kw_avg_max                     1 6.9300e+08 3.1625e+12 514862
    ## - average_token_length           1 7.0338e+08 3.1626e+12 514862
    ## - self_reference_min_shares      1 7.6990e+08 3.1626e+12 514862
    ## - data_channel_is_world          1 8.3233e+08 3.1627e+12 514863
    ## - num_self_hrefs                 1 8.9301e+08 3.1627e+12 514863
    ## - LDA_02                         1 8.9498e+08 3.1627e+12 514863
    ## - n_tokens_title                 1 9.2843e+08 3.1628e+12 514864
    ## - abs_title_subjectivity         1 9.4030e+08 3.1628e+12 514864
    ## - LDA_01                         1 9.7098e+08 3.1628e+12 514864
    ## - weekday_is_monday              1 9.8357e+08 3.1628e+12 514864
    ## - abs_title_sentiment_polarity   1 1.0128e+09 3.1629e+12 514864
    ## - data_channel_is_tech           1 1.0847e+09 3.1629e+12 514865
    ## - data_channel_is_socmed         1 1.1371e+09 3.1630e+12 514865
    ## - global_subjectivity            1 1.3751e+09 3.1632e+12 514868
    ## - data_channel_is_lifestyle      1 1.5207e+09 3.1634e+12 514869
    ## - data_channel_is_bus            1 1.9598e+09 3.1638e+12 514873
    ## - kw_min_avg                     1 2.1297e+09 3.1640e+12 514874
    ## - num_hrefs                      1 3.3498e+09 3.1652e+12 514885
    ## - data_channel_is_entertainment  1 3.6873e+09 3.1655e+12 514888
    ## - kw_max_avg                     1 5.7229e+09 3.1676e+12 514906
    ## - kw_avg_avg                     1 1.1879e+10 3.1737e+12 514960
    ## 
    ## Step:  AIC=514856.2
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + num_videos + 
    ##     average_token_length + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - num_videos                     1 1.9391e+08 3.1621e+12 514856
    ## - LDA_04                         1 2.1933e+08 3.1622e+12 514856
    ## <none>                                        3.1619e+12 514856
    ## - LDA_03                         1 2.8707e+08 3.1622e+12 514857
    ## - kw_max_min                     1 3.0242e+08 3.1622e+12 514857
    ## - kw_avg_min                     1 3.6443e+08 3.1623e+12 514857
    ## - min_positive_polarity          1 3.7968e+08 3.1623e+12 514858
    ## - self_reference_max_shares      1 5.6930e+08 3.1625e+12 514859
    ## - kw_min_min                     1 6.5637e+08 3.1626e+12 514860
    ## - average_token_length           1 6.9352e+08 3.1626e+12 514860
    ## - kw_avg_max                     1 7.2221e+08 3.1627e+12 514861
    ## - self_reference_min_shares      1 7.7348e+08 3.1627e+12 514861
    ## - num_self_hrefs                 1 8.3393e+08 3.1628e+12 514862
    ## - data_channel_is_world          1 8.5813e+08 3.1628e+12 514862
    ## - LDA_02                         1 8.7082e+08 3.1628e+12 514862
    ## - n_tokens_title                 1 9.2464e+08 3.1629e+12 514862
    ## - abs_title_subjectivity         1 9.4083e+08 3.1629e+12 514862
    ## - LDA_01                         1 9.4294e+08 3.1629e+12 514862
    ## - weekday_is_monday              1 9.8506e+08 3.1629e+12 514863
    ## - abs_title_sentiment_polarity   1 1.0307e+09 3.1630e+12 514863
    ## - data_channel_is_tech           1 1.0985e+09 3.1630e+12 514864
    ## - data_channel_is_socmed         1 1.1603e+09 3.1631e+12 514864
    ## - global_subjectivity            1 1.3667e+09 3.1633e+12 514866
    ## - data_channel_is_lifestyle      1 1.5346e+09 3.1635e+12 514868
    ## - data_channel_is_bus            1 1.9814e+09 3.1639e+12 514872
    ## - kw_min_avg                     1 2.1286e+09 3.1641e+12 514873
    ## - data_channel_is_entertainment  1 3.6834e+09 3.1656e+12 514887
    ## - num_hrefs                      1 3.8585e+09 3.1658e+12 514888
    ## - kw_max_avg                     1 5.7772e+09 3.1677e+12 514905
    ## - kw_avg_avg                     1 1.2016e+10 3.1740e+12 514959
    ## 
    ## Step:  AIC=514855.9
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - LDA_04                         1 2.1827e+08 3.1623e+12 514856
    ## <none>                                        3.1621e+12 514856
    ## - LDA_03                         1 2.4999e+08 3.1624e+12 514856
    ## - kw_max_min                     1 3.0227e+08 3.1624e+12 514857
    ## - kw_avg_min                     1 3.6059e+08 3.1625e+12 514857
    ## - min_positive_polarity          1 3.9978e+08 3.1625e+12 514857
    ## - self_reference_max_shares      1 6.1577e+08 3.1627e+12 514859
    ## - kw_avg_max                     1 6.4176e+08 3.1628e+12 514860
    ## - average_token_length           1 6.9252e+08 3.1628e+12 514860
    ## - kw_min_min                     1 7.2498e+08 3.1629e+12 514860
    ## - self_reference_min_shares      1 7.4645e+08 3.1629e+12 514860
    ## - num_self_hrefs                 1 8.0109e+08 3.1629e+12 514861
    ## - data_channel_is_world          1 8.5456e+08 3.1630e+12 514861
    ## - LDA_02                         1 8.6804e+08 3.1630e+12 514862
    ## - abs_title_subjectivity         1 9.3565e+08 3.1631e+12 514862
    ## - n_tokens_title                 1 9.4243e+08 3.1631e+12 514862
    ## - LDA_01                         1 9.5058e+08 3.1631e+12 514862
    ## - weekday_is_monday              1 9.9678e+08 3.1631e+12 514863
    ## - abs_title_sentiment_polarity   1 1.0409e+09 3.1632e+12 514863
    ## - data_channel_is_tech           1 1.1023e+09 3.1632e+12 514864
    ## - data_channel_is_socmed         1 1.1552e+09 3.1633e+12 514864
    ## - global_subjectivity            1 1.4219e+09 3.1636e+12 514866
    ## - data_channel_is_lifestyle      1 1.5348e+09 3.1637e+12 514867
    ## - data_channel_is_bus            1 1.9962e+09 3.1641e+12 514871
    ## - kw_min_avg                     1 2.1436e+09 3.1643e+12 514873
    ## - data_channel_is_entertainment  1 3.5753e+09 3.1657e+12 514885
    ## - num_hrefs                      1 3.9641e+09 3.1661e+12 514889
    ## - kw_max_avg                     1 5.7489e+09 3.1679e+12 514904
    ## - kw_avg_avg                     1 1.1938e+10 3.1741e+12 514958
    ## 
    ## Step:  AIC=514855.8
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - LDA_03                         1 9.3262e+07 3.1624e+12 514855
    ## <none>                                        3.1623e+12 514856
    ## - kw_max_min                     1 2.9673e+08 3.1626e+12 514856
    ## - kw_avg_min                     1 3.5246e+08 3.1627e+12 514857
    ## - min_positive_polarity          1 4.4155e+08 3.1628e+12 514858
    ## - kw_avg_max                     1 6.0350e+08 3.1630e+12 514859
    ## - self_reference_max_shares      1 6.1257e+08 3.1630e+12 514859
    ## - LDA_02                         1 6.5228e+08 3.1630e+12 514860
    ## - average_token_length           1 6.6699e+08 3.1630e+12 514860
    ## - LDA_01                         1 7.3232e+08 3.1631e+12 514860
    ## - kw_min_min                     1 7.4674e+08 3.1631e+12 514860
    ## - self_reference_min_shares      1 7.4742e+08 3.1631e+12 514860
    ## - num_self_hrefs                 1 8.3154e+08 3.1632e+12 514861
    ## - data_channel_is_world          1 9.0775e+08 3.1633e+12 514862
    ## - n_tokens_title                 1 9.1423e+08 3.1633e+12 514862
    ## - abs_title_subjectivity         1 9.4064e+08 3.1633e+12 514862
    ## - weekday_is_monday              1 9.9395e+08 3.1633e+12 514863
    ## - data_channel_is_socmed         1 1.0167e+09 3.1634e+12 514863
    ## - abs_title_sentiment_polarity   1 1.0497e+09 3.1634e+12 514863
    ## - global_subjectivity            1 1.4048e+09 3.1638e+12 514866
    ## - data_channel_is_tech           1 1.6023e+09 3.1640e+12 514868
    ## - data_channel_is_lifestyle      1 1.7610e+09 3.1641e+12 514869
    ## - data_channel_is_bus            1 1.7806e+09 3.1641e+12 514869
    ## - kw_min_avg                     1 2.1956e+09 3.1645e+12 514873
    ## - data_channel_is_entertainment  1 3.5353e+09 3.1659e+12 514885
    ## - num_hrefs                      1 4.0215e+09 3.1664e+12 514889
    ## - kw_max_avg                     1 5.7703e+09 3.1681e+12 514904
    ## - kw_avg_avg                     1 1.2019e+10 3.1744e+12 514959
    ## 
    ## Step:  AIC=514854.7
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + global_subjectivity + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_monday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## <none>                                        3.1624e+12 514855
    ## - kw_max_min                     1 2.8892e+08 3.1627e+12 514855
    ## - kw_avg_min                     1 3.4109e+08 3.1628e+12 514856
    ## - min_positive_polarity          1 4.7341e+08 3.1629e+12 514857
    ## - LDA_02                         1 5.6098e+08 3.1630e+12 514858
    ## - self_reference_max_shares      1 6.1232e+08 3.1631e+12 514858
    ## - average_token_length           1 6.3587e+08 3.1631e+12 514858
    ## - kw_avg_max                     1 6.4846e+08 3.1631e+12 514858
    ## - kw_min_min                     1 7.0274e+08 3.1631e+12 514859
    ## - self_reference_min_shares      1 7.5032e+08 3.1632e+12 514859
    ## - LDA_01                         1 7.7415e+08 3.1632e+12 514859
    ## - num_self_hrefs                 1 8.1602e+08 3.1633e+12 514860
    ## - data_channel_is_world          1 8.3095e+08 3.1633e+12 514860
    ## - n_tokens_title                 1 9.0414e+08 3.1633e+12 514861
    ## - abs_title_subjectivity         1 9.2396e+08 3.1634e+12 514861
    ## - data_channel_is_socmed         1 9.8220e+08 3.1634e+12 514861
    ## - weekday_is_monday              1 9.8972e+08 3.1634e+12 514861
    ## - abs_title_sentiment_polarity   1 1.0257e+09 3.1635e+12 514862
    ## - global_subjectivity            1 1.3774e+09 3.1638e+12 514865
    ## - data_channel_is_lifestyle      1 1.9958e+09 3.1644e+12 514870
    ## - kw_min_avg                     1 2.1312e+09 3.1646e+12 514871
    ## - data_channel_is_tech           1 2.1856e+09 3.1646e+12 514872
    ## - data_channel_is_bus            1 2.4998e+09 3.1649e+12 514875
    ## - data_channel_is_entertainment  1 3.4715e+09 3.1659e+12 514883
    ## - num_hrefs                      1 3.9669e+09 3.1664e+12 514887
    ## - kw_max_avg                     1 5.6865e+09 3.1681e+12 514903
    ## - kw_avg_avg                     1 1.1949e+10 3.1744e+12 514957

``` r
#summary the model
summary(lm.step)
```

    ## 
    ## Call:
    ## lm(formula = shares ~ n_tokens_title + num_hrefs + num_self_hrefs + 
    ##     average_token_length + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + global_subjectivity + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_monday, data = train)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -24554  -2152  -1199   -117 836891 
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                    1.081e+02  7.030e+02   0.154 0.877806    
    ## n_tokens_title                 8.864e+01  3.148e+01   2.815 0.004877 ** 
    ## num_hrefs                      3.824e+01  6.485e+00   5.897 3.74e-09 ***
    ## num_self_hrefs                -5.020e+01  1.877e+01  -2.675 0.007486 ** 
    ## average_token_length          -2.479e+02  1.050e+02  -2.361 0.018233 *  
    ## data_channel_is_lifestyle     -1.447e+03  3.460e+02  -4.183 2.89e-05 ***
    ## data_channel_is_entertainment -1.503e+03  2.725e+02  -5.517 3.49e-08 ***
    ## data_channel_is_bus           -1.295e+03  2.767e+02  -4.681 2.87e-06 ***
    ## data_channel_is_socmed        -1.001e+03  3.412e+02  -2.934 0.003346 ** 
    ## data_channel_is_tech          -1.200e+03  2.742e+02  -4.377 1.21e-05 ***
    ## data_channel_is_world         -9.898e+02  3.667e+02  -2.699 0.006960 ** 
    ## kw_min_min                     2.848e+00  1.147e+00   2.482 0.013069 *  
    ## kw_max_min                     7.974e-02  5.010e-02   1.591 0.111514    
    ## kw_avg_min                    -5.285e-01  3.056e-01  -1.729 0.083784 .  
    ## kw_avg_max                    -1.760e-03  7.383e-04  -2.384 0.017121 *  
    ## kw_min_avg                    -3.399e-01  7.864e-02  -4.322 1.55e-05 ***
    ## kw_max_avg                    -1.919e-01  2.717e-02  -7.060 1.70e-12 ***
    ## kw_avg_avg                     1.552e+00  1.517e-01  10.235  < 2e-16 ***
    ## self_reference_min_shares      9.810e-03  3.825e-03   2.565 0.010333 *  
    ## self_reference_max_shares      4.240e-03  1.830e-03   2.317 0.020519 *  
    ## LDA_01                        -9.828e+02  3.773e+02  -2.605 0.009190 ** 
    ## LDA_02                        -9.659e+02  4.355e+02  -2.218 0.026590 *  
    ## global_subjectivity            2.598e+03  7.477e+02   3.475 0.000512 ***
    ## min_positive_polarity         -1.975e+03  9.693e+02  -2.037 0.041643 *  
    ## abs_title_subjectivity         1.069e+03  3.756e+02   2.846 0.004430 ** 
    ## abs_title_sentiment_polarity   9.381e+02  3.128e+02   2.999 0.002714 ** 
    ## weekday_is_monday              5.074e+02  1.722e+02   2.946 0.003227 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10680 on 27723 degrees of freedom
    ## Multiple R-squared:  0.02398,    Adjusted R-squared:  0.02306 
    ## F-statistic:  26.2 on 26 and 27723 DF,  p-value: < 2.2e-16

Calculate the predicted mean square error on the train set:

``` r
train.pred <- predict(lm.step, train)
mean((train.pred - train$shares)^2)
```

    ## [1] 113961890

### On test set

Calculate the predicted mean square error on the test set:

``` r
test.pred <- predict(lm.step, test)
mean((test.pred - test$shares)^2)
```

    ## [1] 175193023

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
