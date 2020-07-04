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

    ##                                                              url timedelta n_tokens_title n_tokens_content
    ## 1   http://mashable.com/2013/01/07/amazon-instant-video-browser/       731             12              219
    ## 2    http://mashable.com/2013/01/07/ap-samsung-sponsored-tweets/       731              9              255
    ## 3 http://mashable.com/2013/01/07/apple-40-billion-app-downloads/       731              9              211
    ## 4       http://mashable.com/2013/01/07/astronaut-notre-dame-bcs/       731              9              531
    ## 5               http://mashable.com/2013/01/07/att-u-verse-apps/       731             13             1072
    ## 6               http://mashable.com/2013/01/07/beewi-smart-toys/       731             10              370
    ##   n_unique_tokens n_non_stop_words n_non_stop_unique_tokens num_hrefs num_self_hrefs num_imgs num_videos
    ## 1       0.6635945                1                0.8153846         4              2        1          0
    ## 2       0.6047431                1                0.7919463         3              1        1          0
    ## 3       0.5751295                1                0.6638655         3              1        1          0
    ## 4       0.5037879                1                0.6656347         9              0        1          0
    ## 5       0.4156456                1                0.5408895        19             19       20          0
    ## 6       0.5598886                1                0.6981982         2              2        0          0
    ##   average_token_length num_keywords data_channel_is_lifestyle data_channel_is_entertainment data_channel_is_bus
    ## 1             4.680365            5                         0                             1                   0
    ## 2             4.913725            4                         0                             0                   1
    ## 3             4.393365            6                         0                             0                   1
    ## 4             4.404896            7                         0                             1                   0
    ## 5             4.682836            7                         0                             0                   0
    ## 6             4.359459            9                         0                             0                   0
    ##   data_channel_is_socmed data_channel_is_tech data_channel_is_world kw_min_min kw_max_min kw_avg_min kw_min_max
    ## 1                      0                    0                     0          0          0          0          0
    ## 2                      0                    0                     0          0          0          0          0
    ## 3                      0                    0                     0          0          0          0          0
    ## 4                      0                    0                     0          0          0          0          0
    ## 5                      0                    1                     0          0          0          0          0
    ## 6                      0                    1                     0          0          0          0          0
    ##   kw_max_max kw_avg_max kw_min_avg kw_max_avg kw_avg_avg self_reference_min_shares self_reference_max_shares
    ## 1          0          0          0          0          0                       496                       496
    ## 2          0          0          0          0          0                         0                         0
    ## 3          0          0          0          0          0                       918                       918
    ## 4          0          0          0          0          0                         0                         0
    ## 5          0          0          0          0          0                       545                     16000
    ## 6          0          0          0          0          0                      8500                      8500
    ##   self_reference_avg_sharess weekday_is_monday weekday_is_tuesday weekday_is_wednesday weekday_is_thursday
    ## 1                    496.000                 1                  0                    0                   0
    ## 2                      0.000                 1                  0                    0                   0
    ## 3                    918.000                 1                  0                    0                   0
    ## 4                      0.000                 1                  0                    0                   0
    ## 5                   3151.158                 1                  0                    0                   0
    ## 6                   8500.000                 1                  0                    0                   0
    ##   weekday_is_friday weekday_is_saturday weekday_is_sunday is_weekend     LDA_00     LDA_01     LDA_02     LDA_03
    ## 1                 0                   0                 0          0 0.50033120 0.37827893 0.04000468 0.04126265
    ## 2                 0                   0                 0          0 0.79975569 0.05004668 0.05009625 0.05010067
    ## 3                 0                   0                 0          0 0.21779229 0.03333446 0.03335142 0.03333354
    ## 4                 0                   0                 0          0 0.02857322 0.41929964 0.49465083 0.02890472
    ## 5                 0                   0                 0          0 0.02863281 0.02879355 0.02857518 0.02857168
    ## 6                 0                   0                 0          0 0.02224528 0.30671758 0.02223128 0.02222429
    ##       LDA_04 global_subjectivity global_sentiment_polarity global_rate_positive_words global_rate_negative_words
    ## 1 0.04012254           0.5216171                0.09256198                 0.04566210                0.013698630
    ## 2 0.05000071           0.3412458                0.14894781                 0.04313725                0.015686275
    ## 3 0.68218829           0.7022222                0.32333333                 0.05687204                0.009478673
    ## 4 0.02857160           0.4298497                0.10070467                 0.04143126                0.020715631
    ## 5 0.88542678           0.5135021                0.28100348                 0.07462687                0.012126866
    ## 6 0.62658158           0.4374086                0.07118419                 0.02972973                0.027027027
    ##   rate_positive_words rate_negative_words avg_positive_polarity min_positive_polarity max_positive_polarity
    ## 1           0.7692308           0.2307692             0.3786364            0.10000000                   0.7
    ## 2           0.7333333           0.2666667             0.2869146            0.03333333                   0.7
    ## 3           0.8571429           0.1428571             0.4958333            0.10000000                   1.0
    ## 4           0.6666667           0.3333333             0.3859652            0.13636364                   0.8
    ## 5           0.8602151           0.1397849             0.4111274            0.03333333                   1.0
    ## 6           0.5238095           0.4761905             0.3506100            0.13636364                   0.6
    ##   avg_negative_polarity min_negative_polarity max_negative_polarity title_subjectivity title_sentiment_polarity
    ## 1            -0.3500000                -0.600            -0.2000000          0.5000000               -0.1875000
    ## 2            -0.1187500                -0.125            -0.1000000          0.0000000                0.0000000
    ## 3            -0.4666667                -0.800            -0.1333333          0.0000000                0.0000000
    ## 4            -0.3696970                -0.600            -0.1666667          0.0000000                0.0000000
    ## 5            -0.2201923                -0.500            -0.0500000          0.4545455                0.1363636
    ## 6            -0.1950000                -0.400            -0.1000000          0.6428571                0.2142857
    ##   abs_title_subjectivity abs_title_sentiment_polarity shares
    ## 1             0.00000000                    0.1875000    593
    ## 2             0.50000000                    0.0000000    711
    ## 3             0.50000000                    0.0000000   1500
    ## 4             0.50000000                    0.0000000   1200
    ## 5             0.04545455                    0.1363636    505
    ## 6             0.14285714                    0.2142857    855

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

![](weekday_is_thursday_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
hist(log(train$shares))
```

![](weekday_is_thursday_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

``` r
#summary the training dataset
t(summary(train))
```

    ##                                                                                                          
    ## n_tokens_title                Min.   : 3.00      1st Qu.: 9.00      Median :10.00      Mean   :10.41     
    ## n_tokens_content              Min.   :   0.0     1st Qu.: 246.0     Median : 409.0     Mean   : 546.6    
    ## n_unique_tokens               Min.   :  0.0000   1st Qu.:  0.4707   Median :  0.5391   Mean   :  0.5554  
    ## n_non_stop_words              Min.   :   0.000   1st Qu.:   1.000   Median :   1.000   Mean   :   1.007  
    ## n_non_stop_unique_tokens      Min.   :  0.0000   1st Qu.:  0.6253   Median :  0.6905   Mean   :  0.6958  
    ##   num_hrefs                   Min.   :  0.00     1st Qu.:  4.00     Median :  8.00     Mean   : 10.96    
    ## num_self_hrefs                Min.   :  0.000    1st Qu.:  1.000    Median :  3.000    Mean   :  3.314   
    ##    num_imgs                   Min.   :  0.000    1st Qu.:  1.000    Median :  1.000    Mean   :  4.533   
    ##   num_videos                  Min.   : 0.000     1st Qu.: 0.000     Median : 0.000     Mean   : 1.248    
    ## average_token_length          Min.   :0.000      1st Qu.:4.477      Median :4.664      Mean   :4.545     
    ##  num_keywords                 Min.   : 1.000     1st Qu.: 6.000     Median : 7.000     Mean   : 7.219    
    ## data_channel_is_lifestyle     Min.   :0.00000    1st Qu.:0.00000    Median :0.00000    Mean   :0.05359   
    ## data_channel_is_entertainment Min.   :0.0000     1st Qu.:0.0000     Median :0.0000     Mean   :0.1767    
    ## data_channel_is_bus           Min.   :0.0000     1st Qu.:0.0000     Median :0.0000     Mean   :0.1595    
    ## data_channel_is_socmed        Min.   :0.00000    1st Qu.:0.00000    Median :0.00000    Mean   :0.05888   
    ## data_channel_is_tech          Min.   :0.0000     1st Qu.:0.0000     Median :0.0000     Mean   :0.1843    
    ## data_channel_is_world         Min.   :0.0000     1st Qu.:0.0000     Median :0.0000     Mean   :0.2134    
    ##   kw_min_min                  Min.   : -1.00     1st Qu.: -1.00     Median : -1.00     Mean   : 26.16    
    ##   kw_max_min                  Min.   :     0     1st Qu.:   444     Median :   659     Mean   :  1154    
    ##   kw_avg_min                  Min.   :   -1.0    1st Qu.:  141.6    Median :  235.2    Mean   :  312.5   
    ##   kw_min_max                  Min.   :     0     1st Qu.:     0     Median :  1400     Mean   : 13874    
    ##   kw_max_max                  Min.   :     0     1st Qu.:843300     Median :843300     Mean   :751872    
    ##   kw_avg_max                  Min.   :     0     1st Qu.:173184     Median :244999     Mean   :259451    
    ##   kw_min_avg                  Min.   :  -1       1st Qu.:   0       Median :1028       Mean   :1119      
    ##   kw_max_avg                  Min.   :     0     1st Qu.:  3564     Median :  4358     Mean   :  5653    
    ##   kw_avg_avg                  Min.   :    0      1st Qu.: 2383      Median : 2876      Mean   : 3134     
    ## self_reference_min_shares     Min.   :     0.0   1st Qu.:   640.2   Median :  1200.0   Mean   :  3972.7  
    ## self_reference_max_shares     Min.   :     0     1st Qu.:  1000     Median :  2800     Mean   : 10287    
    ## self_reference_avg_sharess    Min.   :     0.0   1st Qu.:   975.4   Median :  2200.0   Mean   :  6350.6  
    ##     LDA_00                    Min.   :0.00000    1st Qu.:0.02505    Median :0.03339    Mean   :0.18615   
    ##     LDA_01                    Min.   :0.00000    1st Qu.:0.02501    Median :0.03334    Mean   :0.14033   
    ##     LDA_02                    Min.   :0.00000    1st Qu.:0.02857    Median :0.04000    Mean   :0.21606   
    ##     LDA_03                    Min.   :0.00000    1st Qu.:0.02857    Median :0.04000    Mean   :0.22337   
    ##     LDA_04                    Min.   :0.00000    1st Qu.:0.02857    Median :0.04072    Mean   :0.23405   
    ## global_subjectivity           Min.   :0.0000     1st Qu.:0.3959     Median :0.4533     Mean   :0.4430    
    ## global_sentiment_polarity     Min.   :-0.39375   1st Qu.: 0.05761   Median : 0.11851   Mean   : 0.11894  
    ## global_rate_positive_words    Min.   :0.00000    1st Qu.:0.02830    Median :0.03904    Mean   :0.03957   
    ## global_rate_negative_words    Min.   :0.000000   1st Qu.:0.009615   Median :0.015385   Mean   :0.016668  
    ## rate_positive_words           Min.   :0.0000     1st Qu.:0.6000     Median :0.7097     Mean   :0.6806    
    ## rate_negative_words           Min.   :0.0000     1st Qu.:0.1860     Median :0.2800     Mean   :0.2884    
    ## avg_positive_polarity         Min.   :0.0000     1st Qu.:0.3063     Median :0.3590     Mean   :0.3536    
    ## min_positive_polarity         Min.   :0.00000    1st Qu.:0.05000    Median :0.10000    Mean   :0.09534   
    ## max_positive_polarity         Min.   :0.0000     1st Qu.:0.6000     Median :0.8000     Mean   :0.7559    
    ## avg_negative_polarity         Min.   :-1.0000    1st Qu.:-0.3280    Median :-0.2528    Mean   :-0.2594   
    ## min_negative_polarity         Min.   :-1.0000    1st Qu.:-0.7000    Median :-0.5000    Mean   :-0.5212   
    ## max_negative_polarity         Min.   :-1.0000    1st Qu.:-0.1250    Median :-0.1000    Mean   :-0.1076   
    ## title_subjectivity            Min.   :0.0000     1st Qu.:0.0000     Median :0.1500     Mean   :0.2843    
    ## title_sentiment_polarity      Min.   :-1.00000   1st Qu.: 0.00000   Median : 0.00000   Mean   : 0.07096  
    ## abs_title_subjectivity        Min.   :0.0000     1st Qu.:0.1667     Median :0.5000     Mean   :0.3414    
    ## abs_title_sentiment_polarity  Min.   :0.000000   1st Qu.:0.000000   Median :0.005682   Mean   :0.157838  
    ##     shares                    Min.   :     1     1st Qu.:   943     Median :  1400     Mean   :  3356    
    ## weekday_is_thursday           Min.   :0.000      1st Qu.:0.000      Median :0.000      Mean   :0.184     
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
    ## weekday_is_thursday           3rd Qu.:0.000      Max.   :1.000

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
    ##           Mean of squared residuals: 117031618
    ##                     % Var explained: -0.23

``` r
#variable importance measures
importance(rf)
```

    ##                                  %IncMSE IncNodePurity
    ## n_tokens_title                2.02159217   47794263858
    ## n_tokens_content              4.96393296   38369845240
    ## n_unique_tokens               5.09381712   44175015260
    ## n_non_stop_words              7.23461436   44268533187
    ## n_non_stop_unique_tokens      6.15635217   55694706689
    ## num_hrefs                     5.12211102   71969919064
    ## num_self_hrefs                3.78541631   43605066277
    ## num_imgs                      2.30755896   34497726889
    ## num_videos                    3.27801938   33772154433
    ## average_token_length          4.40698085   61321402839
    ## num_keywords                  5.87716596   16792385619
    ## data_channel_is_lifestyle     1.76455409    4780664510
    ## data_channel_is_entertainment 5.12544723    4181628820
    ## data_channel_is_bus           4.05723921   16537649221
    ## data_channel_is_socmed        3.78508441    2787910302
    ## data_channel_is_tech          2.15094433    1915199863
    ## data_channel_is_world         8.35530612    3213536492
    ## kw_min_min                    3.83405512   10989817654
    ## kw_max_min                    3.42506631   79242896591
    ## kw_avg_min                    4.15223464  104879389018
    ## kw_min_max                    5.07435409   42444948937
    ## kw_max_max                    3.16051305   59346608931
    ## kw_avg_max                    4.86411348  119386818826
    ## kw_min_avg                    3.13190666   55373566402
    ## kw_max_avg                    3.62982692  187090814484
    ## kw_avg_avg                    2.61966153  151733950944
    ## self_reference_min_shares     3.63012323   60294357889
    ## self_reference_max_shares     5.59796587   83796882462
    ## self_reference_avg_sharess    4.58264500  140373181956
    ## LDA_00                        4.35733291   99566129811
    ## LDA_01                        8.17203865   50555006673
    ## LDA_02                        4.36495717   74194263196
    ## LDA_03                        7.70150542  114853583570
    ## LDA_04                        6.34705391   81878036687
    ## global_subjectivity           3.81868658   70174102310
    ## global_sentiment_polarity     2.66013260   48739842642
    ## global_rate_positive_words    3.85094259   71909267058
    ## global_rate_negative_words    4.23678879   53114876737
    ## rate_positive_words           2.07681103   47658943957
    ## rate_negative_words           2.00788782   39521566618
    ## avg_positive_polarity         3.23450084   57556220607
    ## min_positive_polarity         0.84933002   20947455001
    ## max_positive_polarity         4.99700318   14108902564
    ## avg_negative_polarity         2.68402234   60119905861
    ## min_negative_polarity         4.25009298   37036781192
    ## max_negative_polarity         2.44877520   32654268987
    ## title_subjectivity            4.08290233   72298573099
    ## title_sentiment_polarity      4.39997616   66578213913
    ## abs_title_subjectivity        4.46388197   14524192338
    ## abs_title_sentiment_polarity  2.21075388   26972938096
    ## weekday_is_thursday           0.07636273    3868383306

``` r
#draw dotplot of variable importance as measured by Random Forest
varImpPlot(rf)
```

![](weekday_is_thursday_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Calculate the predicted mean square error on the train set:

``` r
train.pred <- predict(rf, train)
mean((train.pred - train$shares)^2)
```

    ## [1] 26764855

### On test set

``` r
rf.test <- predict(rf, newdata = test)
mean((test$shares-rf.test)^2)
```

    ## [1] 177027416

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

    ## Start:  AIC=514897.1
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
    ##     weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - title_subjectivity             1 2.5075e+05 3.1616e+12 514895
    ## - global_sentiment_polarity      1 7.6981e+05 3.1616e+12 514895
    ## - global_rate_negative_words     1 1.4394e+06 3.1616e+12 514895
    ## - n_unique_tokens                1 1.8736e+06 3.1616e+12 514895
    ## - avg_negative_polarity          1 6.6131e+06 3.1616e+12 514895
    ## - LDA_00                         1 6.7130e+06 3.1616e+12 514895
    ## - LDA_04                         1 6.7188e+06 3.1616e+12 514895
    ## - LDA_03                         1 6.7211e+06 3.1616e+12 514895
    ## - LDA_02                         1 6.7264e+06 3.1616e+12 514895
    ## - LDA_01                         1 6.7273e+06 3.1616e+12 514895
    ## - max_positive_polarity          1 6.9376e+06 3.1616e+12 514895
    ## - num_keywords                   1 7.1160e+06 3.1616e+12 514895
    ## - rate_positive_words            1 9.7239e+06 3.1616e+12 514895
    ## - rate_negative_words            1 1.0608e+07 3.1616e+12 514895
    ## - n_non_stop_words               1 1.4561e+07 3.1616e+12 514895
    ## - n_tokens_content               1 1.4776e+07 3.1616e+12 514895
    ## - avg_positive_polarity          1 2.3848e+07 3.1616e+12 514895
    ## - min_negative_polarity          1 2.5523e+07 3.1616e+12 514895
    ## - title_sentiment_polarity       1 3.7717e+07 3.1616e+12 514895
    ## - self_reference_avg_sharess     1 4.7495e+07 3.1616e+12 514896
    ## - max_negative_polarity          1 4.9056e+07 3.1616e+12 514896
    ## - global_rate_positive_words     1 5.6370e+07 3.1616e+12 514896
    ## - n_non_stop_unique_tokens       1 7.6214e+07 3.1617e+12 514896
    ## - kw_min_max                     1 9.1750e+07 3.1617e+12 514896
    ## - kw_min_min                     1 9.3767e+07 3.1617e+12 514896
    ## - kw_max_max                     1 1.0524e+08 3.1617e+12 514896
    ## - average_token_length           1 1.5811e+08 3.1617e+12 514897
    ## - num_imgs                       1 1.7734e+08 3.1618e+12 514897
    ## <none>                                        3.1616e+12 514897
    ## - self_reference_max_shares      1 2.3651e+08 3.1618e+12 514897
    ## - min_positive_polarity          1 2.3874e+08 3.1618e+12 514897
    ## - num_videos                     1 2.4470e+08 3.1618e+12 514897
    ## - kw_avg_max                     1 2.9410e+08 3.1619e+12 514898
    ## - kw_max_min                     1 3.2231e+08 3.1619e+12 514898
    ## - self_reference_min_shares      1 3.3213e+08 3.1619e+12 514898
    ## - kw_avg_min                     1 3.8203e+08 3.1620e+12 514898
    ## - weekday_is_thursday            1 4.2887e+08 3.1620e+12 514899
    ## - abs_title_sentiment_polarity   1 4.8855e+08 3.1621e+12 514899
    ## - abs_title_subjectivity         1 7.4109e+08 3.1623e+12 514902
    ## - data_channel_is_world          1 7.7166e+08 3.1624e+12 514902
    ## - n_tokens_title                 1 8.5116e+08 3.1624e+12 514903
    ## - num_self_hrefs                 1 9.0910e+08 3.1625e+12 514903
    ## - global_subjectivity            1 9.8115e+08 3.1626e+12 514904
    ## - data_channel_is_socmed         1 9.9586e+08 3.1626e+12 514904
    ## - data_channel_is_tech           1 1.0057e+09 3.1626e+12 514904
    ## - data_channel_is_lifestyle      1 1.4619e+09 3.1630e+12 514908
    ## - kw_min_avg                     1 1.8856e+09 3.1635e+12 514912
    ## - data_channel_is_bus            1 1.8948e+09 3.1635e+12 514912
    ## - num_hrefs                      1 2.9643e+09 3.1645e+12 514921
    ## - data_channel_is_entertainment  1 3.3146e+09 3.1649e+12 514924
    ## - kw_max_avg                     1 5.6233e+09 3.1672e+12 514944
    ## - kw_avg_avg                     1 1.1424e+10 3.1730e+12 514995
    ## 
    ## Step:  AIC=514895.1
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
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - global_sentiment_polarity      1 7.9042e+05 3.1616e+12 514893
    ## - global_rate_negative_words     1 1.4360e+06 3.1616e+12 514893
    ## - n_unique_tokens                1 1.8836e+06 3.1616e+12 514893
    ## - avg_negative_polarity          1 6.5840e+06 3.1616e+12 514893
    ## - LDA_00                         1 6.6894e+06 3.1616e+12 514893
    ## - LDA_04                         1 6.6952e+06 3.1616e+12 514893
    ## - LDA_03                         1 6.6975e+06 3.1616e+12 514893
    ## - LDA_02                         1 6.7028e+06 3.1616e+12 514893
    ## - LDA_01                         1 6.7037e+06 3.1616e+12 514893
    ## - max_positive_polarity          1 6.9204e+06 3.1616e+12 514893
    ## - num_keywords                   1 7.0933e+06 3.1616e+12 514893
    ## - rate_positive_words            1 9.7164e+06 3.1616e+12 514893
    ## - rate_negative_words            1 1.0597e+07 3.1616e+12 514893
    ## - n_non_stop_words               1 1.4523e+07 3.1616e+12 514893
    ## - n_tokens_content               1 1.4720e+07 3.1616e+12 514893
    ## - avg_positive_polarity          1 2.3753e+07 3.1616e+12 514893
    ## - min_negative_polarity          1 2.5565e+07 3.1616e+12 514893
    ## - title_sentiment_polarity       1 3.8929e+07 3.1616e+12 514893
    ## - self_reference_avg_sharess     1 4.7510e+07 3.1616e+12 514894
    ## - max_negative_polarity          1 4.9031e+07 3.1616e+12 514894
    ## - global_rate_positive_words     1 5.6620e+07 3.1616e+12 514894
    ## - n_non_stop_unique_tokens       1 7.6235e+07 3.1617e+12 514894
    ## - kw_min_max                     1 9.1787e+07 3.1617e+12 514894
    ## - kw_min_min                     1 9.3846e+07 3.1617e+12 514894
    ## - kw_max_max                     1 1.0514e+08 3.1617e+12 514894
    ## - average_token_length           1 1.5813e+08 3.1617e+12 514895
    ## - num_imgs                       1 1.7723e+08 3.1618e+12 514895
    ## <none>                                        3.1616e+12 514895
    ## - self_reference_max_shares      1 2.3652e+08 3.1618e+12 514895
    ## - min_positive_polarity          1 2.3894e+08 3.1618e+12 514895
    ## - num_videos                     1 2.4464e+08 3.1618e+12 514895
    ## - kw_avg_max                     1 2.9420e+08 3.1619e+12 514896
    ## - kw_max_min                     1 3.2244e+08 3.1619e+12 514896
    ## - self_reference_min_shares      1 3.3223e+08 3.1619e+12 514896
    ## - kw_avg_min                     1 3.8225e+08 3.1620e+12 514896
    ## - weekday_is_thursday            1 4.2920e+08 3.1620e+12 514897
    ## - data_channel_is_world          1 7.7165e+08 3.1624e+12 514900
    ## - abs_title_sentiment_polarity   1 8.0596e+08 3.1624e+12 514900
    ## - abs_title_subjectivity         1 8.3049e+08 3.1624e+12 514900
    ## - n_tokens_title                 1 8.5093e+08 3.1624e+12 514901
    ## - num_self_hrefs                 1 9.0921e+08 3.1625e+12 514901
    ## - global_subjectivity            1 9.9060e+08 3.1626e+12 514902
    ## - data_channel_is_socmed         1 9.9565e+08 3.1626e+12 514902
    ## - data_channel_is_tech           1 1.0058e+09 3.1626e+12 514902
    ## - data_channel_is_lifestyle      1 1.4619e+09 3.1630e+12 514906
    ## - kw_min_avg                     1 1.8853e+09 3.1635e+12 514910
    ## - data_channel_is_bus            1 1.8947e+09 3.1635e+12 514910
    ## - num_hrefs                      1 2.9642e+09 3.1645e+12 514919
    ## - data_channel_is_entertainment  1 3.3144e+09 3.1649e+12 514922
    ## - kw_max_avg                     1 5.6231e+09 3.1672e+12 514942
    ## - kw_avg_avg                     1 1.1424e+10 3.1730e+12 514993
    ## 
    ## Step:  AIC=514893.1
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
    ##     weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_unique_tokens                1 1.8888e+06 3.1616e+12 514891
    ## - global_rate_negative_words     1 2.4498e+06 3.1616e+12 514891
    ## - LDA_00                         1 6.7079e+06 3.1616e+12 514891
    ## - LDA_04                         1 6.7138e+06 3.1616e+12 514891
    ## - LDA_03                         1 6.7161e+06 3.1616e+12 514891
    ## - LDA_02                         1 6.7214e+06 3.1616e+12 514891
    ## - LDA_01                         1 6.7223e+06 3.1616e+12 514891
    ## - max_positive_polarity          1 6.9776e+06 3.1616e+12 514891
    ## - num_keywords                   1 7.0599e+06 3.1616e+12 514891
    ## - avg_negative_polarity          1 9.2044e+06 3.1616e+12 514891
    ## - rate_positive_words            1 9.7353e+06 3.1616e+12 514891
    ## - rate_negative_words            1 1.0348e+07 3.1616e+12 514891
    ## - n_tokens_content               1 1.4481e+07 3.1616e+12 514891
    ## - n_non_stop_words               1 1.4522e+07 3.1616e+12 514891
    ## - min_negative_polarity          1 2.5803e+07 3.1616e+12 514891
    ## - avg_positive_polarity          1 2.9186e+07 3.1616e+12 514891
    ## - title_sentiment_polarity       1 4.0213e+07 3.1616e+12 514891
    ## - self_reference_avg_sharess     1 4.7475e+07 3.1616e+12 514892
    ## - max_negative_polarity          1 5.2481e+07 3.1616e+12 514892
    ## - global_rate_positive_words     1 6.1787e+07 3.1616e+12 514892
    ## - n_non_stop_unique_tokens       1 7.5955e+07 3.1617e+12 514892
    ## - kw_min_max                     1 9.1992e+07 3.1617e+12 514892
    ## - kw_min_min                     1 9.3755e+07 3.1617e+12 514892
    ## - kw_max_max                     1 1.0521e+08 3.1617e+12 514892
    ## - average_token_length           1 1.5851e+08 3.1617e+12 514893
    ## - num_imgs                       1 1.7737e+08 3.1618e+12 514893
    ## <none>                                        3.1616e+12 514893
    ## - self_reference_max_shares      1 2.3636e+08 3.1618e+12 514893
    ## - num_videos                     1 2.4435e+08 3.1618e+12 514893
    ## - min_positive_polarity          1 2.4670e+08 3.1618e+12 514893
    ## - kw_avg_max                     1 2.9402e+08 3.1619e+12 514894
    ## - kw_max_min                     1 3.2220e+08 3.1619e+12 514894
    ## - self_reference_min_shares      1 3.3222e+08 3.1619e+12 514894
    ## - kw_avg_min                     1 3.8197e+08 3.1620e+12 514894
    ## - weekday_is_thursday            1 4.2878e+08 3.1620e+12 514895
    ## - data_channel_is_world          1 7.7106e+08 3.1624e+12 514898
    ## - abs_title_sentiment_polarity   1 8.0534e+08 3.1624e+12 514898
    ## - abs_title_subjectivity         1 8.3290e+08 3.1624e+12 514898
    ## - n_tokens_title                 1 8.5099e+08 3.1624e+12 514899
    ## - num_self_hrefs                 1 9.0925e+08 3.1625e+12 514899
    ## - data_channel_is_socmed         1 9.9546e+08 3.1626e+12 514900
    ## - data_channel_is_tech           1 1.0055e+09 3.1626e+12 514900
    ## - global_subjectivity            1 1.0374e+09 3.1626e+12 514900
    ## - data_channel_is_lifestyle      1 1.4611e+09 3.1630e+12 514904
    ## - kw_min_avg                     1 1.8847e+09 3.1635e+12 514908
    ## - data_channel_is_bus            1 1.8946e+09 3.1635e+12 514908
    ## - num_hrefs                      1 2.9746e+09 3.1646e+12 514917
    ## - data_channel_is_entertainment  1 3.3143e+09 3.1649e+12 514920
    ## - kw_max_avg                     1 5.6227e+09 3.1672e+12 514940
    ## - kw_avg_avg                     1 1.1424e+10 3.1730e+12 514991
    ## 
    ## Step:  AIC=514891.1
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
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - global_rate_negative_words     1 2.5757e+06 3.1616e+12 514889
    ## - LDA_00                         1 6.1788e+06 3.1616e+12 514889
    ## - LDA_04                         1 6.1844e+06 3.1616e+12 514889
    ## - LDA_03                         1 6.1867e+06 3.1616e+12 514889
    ## - LDA_02                         1 6.1917e+06 3.1616e+12 514889
    ## - LDA_01                         1 6.1927e+06 3.1616e+12 514889
    ## - num_keywords                   1 7.1545e+06 3.1616e+12 514889
    ## - max_positive_polarity          1 8.2721e+06 3.1616e+12 514889
    ## - avg_negative_polarity          1 9.2033e+06 3.1616e+12 514889
    ## - rate_positive_words            1 9.8969e+06 3.1616e+12 514889
    ## - rate_negative_words            1 1.0515e+07 3.1616e+12 514889
    ## - n_non_stop_words               1 1.4412e+07 3.1616e+12 514889
    ## - n_tokens_content               1 2.4796e+07 3.1616e+12 514889
    ## - min_negative_polarity          1 2.7880e+07 3.1616e+12 514889
    ## - avg_positive_polarity          1 2.9293e+07 3.1616e+12 514889
    ## - title_sentiment_polarity       1 4.0559e+07 3.1616e+12 514889
    ## - self_reference_avg_sharess     1 4.7759e+07 3.1616e+12 514890
    ## - max_negative_polarity          1 5.0916e+07 3.1616e+12 514890
    ## - global_rate_positive_words     1 6.3615e+07 3.1617e+12 514890
    ## - kw_min_max                     1 9.1726e+07 3.1617e+12 514890
    ## - kw_min_min                     1 9.3722e+07 3.1617e+12 514890
    ## - kw_max_max                     1 1.0454e+08 3.1617e+12 514890
    ## - num_imgs                       1 1.7687e+08 3.1618e+12 514891
    ## - average_token_length           1 1.8224e+08 3.1618e+12 514891
    ## <none>                                        3.1616e+12 514891
    ## - n_non_stop_unique_tokens       1 2.3106e+08 3.1618e+12 514891
    ## - self_reference_max_shares      1 2.3709e+08 3.1618e+12 514891
    ## - num_videos                     1 2.4255e+08 3.1618e+12 514891
    ## - min_positive_polarity          1 2.6192e+08 3.1618e+12 514891
    ## - kw_avg_max                     1 2.9586e+08 3.1619e+12 514892
    ## - kw_max_min                     1 3.2212e+08 3.1619e+12 514892
    ## - self_reference_min_shares      1 3.3282e+08 3.1619e+12 514892
    ## - kw_avg_min                     1 3.8171e+08 3.1620e+12 514892
    ## - weekday_is_thursday            1 4.2890e+08 3.1620e+12 514893
    ## - data_channel_is_world          1 7.7046e+08 3.1624e+12 514896
    ## - abs_title_sentiment_polarity   1 8.0394e+08 3.1624e+12 514896
    ## - abs_title_subjectivity         1 8.3485e+08 3.1624e+12 514896
    ## - n_tokens_title                 1 8.5096e+08 3.1624e+12 514897
    ## - num_self_hrefs                 1 9.1027e+08 3.1625e+12 514897
    ## - data_channel_is_socmed         1 9.9358e+08 3.1626e+12 514898
    ## - data_channel_is_tech           1 1.0042e+09 3.1626e+12 514898
    ## - global_subjectivity            1 1.0363e+09 3.1626e+12 514898
    ## - data_channel_is_lifestyle      1 1.4624e+09 3.1630e+12 514902
    ## - kw_min_avg                     1 1.8881e+09 3.1635e+12 514906
    ## - data_channel_is_bus            1 1.8948e+09 3.1635e+12 514906
    ## - num_hrefs                      1 2.9898e+09 3.1646e+12 514915
    ## - data_channel_is_entertainment  1 3.3164e+09 3.1649e+12 514918
    ## - kw_max_avg                     1 5.6267e+09 3.1672e+12 514938
    ## - kw_avg_avg                     1 1.1428e+10 3.1730e+12 514989
    ## 
    ## Step:  AIC=514889.2
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + data_channel_is_lifestyle + 
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
    ##     weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - LDA_00                         1 6.2092e+06 3.1616e+12 514887
    ## - LDA_04                         1 6.2148e+06 3.1616e+12 514887
    ## - LDA_03                         1 6.2170e+06 3.1616e+12 514887
    ## - LDA_02                         1 6.2221e+06 3.1616e+12 514887
    ## - LDA_01                         1 6.2231e+06 3.1616e+12 514887
    ## - num_keywords                   1 7.2319e+06 3.1616e+12 514887
    ## - max_positive_polarity          1 7.9480e+06 3.1616e+12 514887
    ## - avg_negative_polarity          1 9.0971e+06 3.1616e+12 514887
    ## - rate_negative_words            1 9.4165e+06 3.1616e+12 514887
    ## - rate_positive_words            1 1.0349e+07 3.1616e+12 514887
    ## - n_non_stop_words               1 1.4441e+07 3.1616e+12 514887
    ## - n_tokens_content               1 2.5252e+07 3.1616e+12 514887
    ## - min_negative_polarity          1 2.7463e+07 3.1616e+12 514887
    ## - avg_positive_polarity          1 2.9495e+07 3.1616e+12 514887
    ## - title_sentiment_polarity       1 4.2207e+07 3.1616e+12 514888
    ## - self_reference_avg_sharess     1 4.7868e+07 3.1616e+12 514888
    ## - max_negative_polarity          1 5.2176e+07 3.1616e+12 514888
    ## - kw_min_max                     1 9.1787e+07 3.1617e+12 514888
    ## - kw_min_min                     1 9.3230e+07 3.1617e+12 514888
    ## - kw_max_max                     1 1.0499e+08 3.1617e+12 514888
    ## - global_rate_positive_words     1 1.3648e+08 3.1617e+12 514888
    ## - num_imgs                       1 1.7576e+08 3.1618e+12 514889
    ## - average_token_length           1 1.8080e+08 3.1618e+12 514889
    ## <none>                                        3.1616e+12 514889
    ## - n_non_stop_unique_tokens       1 2.3035e+08 3.1618e+12 514889
    ## - self_reference_max_shares      1 2.3753e+08 3.1618e+12 514889
    ## - num_videos                     1 2.4012e+08 3.1618e+12 514889
    ## - min_positive_polarity          1 2.5971e+08 3.1618e+12 514889
    ## - kw_avg_max                     1 2.9621e+08 3.1619e+12 514890
    ## - kw_max_min                     1 3.2214e+08 3.1619e+12 514890
    ## - self_reference_min_shares      1 3.3270e+08 3.1619e+12 514890
    ## - kw_avg_min                     1 3.8210e+08 3.1620e+12 514891
    ## - weekday_is_thursday            1 4.2959e+08 3.1620e+12 514891
    ## - data_channel_is_world          1 7.6793e+08 3.1624e+12 514894
    ## - abs_title_sentiment_polarity   1 8.0242e+08 3.1624e+12 514894
    ## - abs_title_subjectivity         1 8.3957e+08 3.1624e+12 514895
    ## - n_tokens_title                 1 8.5284e+08 3.1624e+12 514895
    ## - num_self_hrefs                 1 9.0988e+08 3.1625e+12 514895
    ## - data_channel_is_socmed         1 9.9149e+08 3.1626e+12 514896
    ## - data_channel_is_tech           1 1.0021e+09 3.1626e+12 514896
    ## - global_subjectivity            1 1.0364e+09 3.1626e+12 514896
    ## - data_channel_is_lifestyle      1 1.4600e+09 3.1630e+12 514900
    ## - kw_min_avg                     1 1.8879e+09 3.1635e+12 514904
    ## - data_channel_is_bus            1 1.8923e+09 3.1635e+12 514904
    ## - num_hrefs                      1 3.0161e+09 3.1646e+12 514914
    ## - data_channel_is_entertainment  1 3.3145e+09 3.1649e+12 514916
    ## - kw_max_avg                     1 5.6250e+09 3.1672e+12 514936
    ## - kw_avg_avg                     1 1.1427e+10 3.1730e+12 514987
    ## 
    ## Step:  AIC=514887.2
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + data_channel_is_lifestyle + 
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
    ##     weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - num_keywords                   1 7.0674e+06 3.1616e+12 514885
    ## - max_positive_polarity          1 7.7425e+06 3.1616e+12 514885
    ## - rate_negative_words            1 7.8920e+06 3.1616e+12 514885
    ## - avg_negative_polarity          1 9.0391e+06 3.1616e+12 514885
    ## - rate_positive_words            1 1.1904e+07 3.1616e+12 514885
    ## - n_tokens_content               1 2.4251e+07 3.1616e+12 514885
    ## - min_negative_polarity          1 2.7417e+07 3.1616e+12 514885
    ## - avg_positive_polarity          1 2.9079e+07 3.1616e+12 514885
    ## - title_sentiment_polarity       1 4.2071e+07 3.1616e+12 514886
    ## - self_reference_avg_sharess     1 4.8141e+07 3.1616e+12 514886
    ## - max_negative_polarity          1 5.2222e+07 3.1616e+12 514886
    ## - kw_min_max                     1 9.1539e+07 3.1617e+12 514886
    ## - kw_min_min                     1 9.3331e+07 3.1617e+12 514886
    ## - kw_max_max                     1 1.0503e+08 3.1617e+12 514886
    ## - global_rate_positive_words     1 1.3636e+08 3.1617e+12 514886
    ## - num_imgs                       1 1.7637e+08 3.1618e+12 514887
    ## - average_token_length           1 2.0053e+08 3.1618e+12 514887
    ## - LDA_04                         1 2.1561e+08 3.1618e+12 514887
    ## - n_non_stop_words               1 2.2487e+08 3.1618e+12 514887
    ## - n_non_stop_unique_tokens       1 2.2758e+08 3.1618e+12 514887
    ## <none>                                        3.1616e+12 514887
    ## - self_reference_max_shares      1 2.3795e+08 3.1618e+12 514887
    ## - num_videos                     1 2.3990e+08 3.1618e+12 514887
    ## - min_positive_polarity          1 2.6014e+08 3.1619e+12 514888
    ## - kw_avg_max                     1 2.9578e+08 3.1619e+12 514888
    ## - kw_max_min                     1 3.2262e+08 3.1619e+12 514888
    ## - self_reference_min_shares      1 3.3335e+08 3.1619e+12 514888
    ## - LDA_03                         1 3.4749e+08 3.1619e+12 514888
    ## - kw_avg_min                     1 3.8262e+08 3.1620e+12 514889
    ## - weekday_is_thursday            1 4.2899e+08 3.1620e+12 514889
    ## - data_channel_is_world          1 7.6748e+08 3.1624e+12 514892
    ## - abs_title_sentiment_polarity   1 8.0322e+08 3.1624e+12 514892
    ## - abs_title_subjectivity         1 8.3888e+08 3.1624e+12 514893
    ## - n_tokens_title                 1 8.5271e+08 3.1624e+12 514893
    ## - num_self_hrefs                 1 9.1323e+08 3.1625e+12 514893
    ## - LDA_02                         1 9.5189e+08 3.1625e+12 514894
    ## - LDA_01                         1 9.7602e+08 3.1626e+12 514894
    ## - data_channel_is_socmed         1 9.9546e+08 3.1626e+12 514894
    ## - data_channel_is_tech           1 1.0044e+09 3.1626e+12 514894
    ## - global_subjectivity            1 1.0307e+09 3.1626e+12 514894
    ## - data_channel_is_lifestyle      1 1.4626e+09 3.1631e+12 514898
    ## - kw_min_avg                     1 1.8912e+09 3.1635e+12 514902
    ## - data_channel_is_bus            1 1.8962e+09 3.1635e+12 514902
    ## - num_hrefs                      1 3.0338e+09 3.1646e+12 514912
    ## - data_channel_is_entertainment  1 3.3191e+09 3.1649e+12 514914
    ## - kw_max_avg                     1 5.6290e+09 3.1672e+12 514935
    ## - kw_avg_avg                     1 1.1435e+10 3.1730e+12 514985
    ## 
    ## Step:  AIC=514885.3
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
    ##     weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - max_positive_polarity          1 7.5984e+06 3.1616e+12 514883
    ## - rate_negative_words            1 8.9765e+06 3.1616e+12 514883
    ## - avg_negative_polarity          1 9.0493e+06 3.1616e+12 514883
    ## - rate_positive_words            1 1.3075e+07 3.1616e+12 514883
    ## - n_tokens_content               1 2.3920e+07 3.1616e+12 514883
    ## - min_negative_polarity          1 2.7532e+07 3.1616e+12 514884
    ## - avg_positive_polarity          1 2.9513e+07 3.1616e+12 514884
    ## - title_sentiment_polarity       1 4.1831e+07 3.1616e+12 514884
    ## - self_reference_avg_sharess     1 4.8739e+07 3.1617e+12 514884
    ## - max_negative_polarity          1 5.2112e+07 3.1617e+12 514884
    ## - kw_min_max                     1 8.9703e+07 3.1617e+12 514884
    ## - kw_min_min                     1 9.3604e+07 3.1617e+12 514884
    ## - kw_max_max                     1 1.1595e+08 3.1617e+12 514884
    ## - global_rate_positive_words     1 1.3875e+08 3.1617e+12 514884
    ## - num_imgs                       1 1.7689e+08 3.1618e+12 514885
    ## - average_token_length           1 2.0291e+08 3.1618e+12 514885
    ## - LDA_04                         1 2.1880e+08 3.1618e+12 514885
    ## - n_non_stop_words               1 2.2610e+08 3.1618e+12 514885
    ## <none>                                        3.1616e+12 514885
    ## - n_non_stop_unique_tokens       1 2.2882e+08 3.1618e+12 514885
    ## - num_videos                     1 2.3822e+08 3.1618e+12 514885
    ## - self_reference_max_shares      1 2.3893e+08 3.1618e+12 514885
    ## - min_positive_polarity          1 2.5953e+08 3.1619e+12 514886
    ## - kw_avg_max                     1 3.0293e+08 3.1619e+12 514886
    ## - kw_max_min                     1 3.1703e+08 3.1619e+12 514886
    ## - self_reference_min_shares      1 3.3514e+08 3.1619e+12 514886
    ## - LDA_03                         1 3.4995e+08 3.1620e+12 514886
    ## - kw_avg_min                     1 3.7608e+08 3.1620e+12 514887
    ## - weekday_is_thursday            1 4.2815e+08 3.1620e+12 514887
    ## - data_channel_is_world          1 7.6396e+08 3.1624e+12 514890
    ## - abs_title_sentiment_polarity   1 8.0210e+08 3.1624e+12 514890
    ## - abs_title_subjectivity         1 8.3863e+08 3.1624e+12 514891
    ## - n_tokens_title                 1 8.4805e+08 3.1625e+12 514891
    ## - num_self_hrefs                 1 9.2682e+08 3.1625e+12 514891
    ## - LDA_02                         1 9.5042e+08 3.1626e+12 514892
    ## - LDA_01                         1 9.7189e+08 3.1626e+12 514892
    ## - data_channel_is_socmed         1 9.8960e+08 3.1626e+12 514892
    ## - data_channel_is_tech           1 1.0006e+09 3.1626e+12 514892
    ## - global_subjectivity            1 1.0291e+09 3.1626e+12 514892
    ## - data_channel_is_lifestyle      1 1.4598e+09 3.1631e+12 514896
    ## - data_channel_is_bus            1 1.8895e+09 3.1635e+12 514900
    ## - kw_min_avg                     1 1.9344e+09 3.1635e+12 514900
    ## - num_hrefs                      1 3.0273e+09 3.1646e+12 514910
    ## - data_channel_is_entertainment  1 3.3259e+09 3.1649e+12 514912
    ## - kw_max_avg                     1 5.6271e+09 3.1672e+12 514933
    ## - kw_avg_avg                     1 1.1614e+10 3.1732e+12 514985
    ## 
    ## Step:  AIC=514883.3
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
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - avg_negative_polarity          1 8.7948e+06 3.1616e+12 514881
    ## - rate_negative_words            1 1.0304e+07 3.1616e+12 514881
    ## - rate_positive_words            1 1.5581e+07 3.1616e+12 514881
    ## - avg_positive_polarity          1 2.2647e+07 3.1616e+12 514882
    ## - n_tokens_content               1 2.9156e+07 3.1616e+12 514882
    ## - min_negative_polarity          1 2.9313e+07 3.1616e+12 514882
    ## - title_sentiment_polarity       1 4.1581e+07 3.1617e+12 514882
    ## - self_reference_avg_sharess     1 4.8971e+07 3.1617e+12 514882
    ## - max_negative_polarity          1 5.0185e+07 3.1617e+12 514882
    ## - kw_min_max                     1 8.9721e+07 3.1617e+12 514882
    ## - kw_min_min                     1 9.3715e+07 3.1617e+12 514882
    ## - kw_max_max                     1 1.1501e+08 3.1617e+12 514882
    ## - global_rate_positive_words     1 1.3179e+08 3.1617e+12 514883
    ## - num_imgs                       1 1.7681e+08 3.1618e+12 514883
    ## - average_token_length           1 2.0727e+08 3.1618e+12 514883
    ## - LDA_04                         1 2.1877e+08 3.1618e+12 514883
    ## - n_non_stop_words               1 2.2108e+08 3.1618e+12 514883
    ## - n_non_stop_unique_tokens       1 2.2378e+08 3.1618e+12 514883
    ## <none>                                        3.1616e+12 514883
    ## - self_reference_max_shares      1 2.3927e+08 3.1619e+12 514883
    ## - num_videos                     1 2.4135e+08 3.1619e+12 514883
    ## - kw_avg_max                     1 3.0320e+08 3.1619e+12 514884
    ## - kw_max_min                     1 3.1622e+08 3.1619e+12 514884
    ## - min_positive_polarity          1 3.1735e+08 3.1619e+12 514884
    ## - self_reference_min_shares      1 3.3542e+08 3.1619e+12 514884
    ## - LDA_03                         1 3.5036e+08 3.1620e+12 514884
    ## - kw_avg_min                     1 3.7459e+08 3.1620e+12 514885
    ## - weekday_is_thursday            1 4.2838e+08 3.1620e+12 514885
    ## - data_channel_is_world          1 7.6553e+08 3.1624e+12 514888
    ## - abs_title_sentiment_polarity   1 8.0096e+08 3.1624e+12 514888
    ## - abs_title_subjectivity         1 8.3719e+08 3.1624e+12 514889
    ## - n_tokens_title                 1 8.5073e+08 3.1625e+12 514889
    ## - num_self_hrefs                 1 9.2846e+08 3.1625e+12 514889
    ## - LDA_02                         1 9.5236e+08 3.1626e+12 514890
    ## - LDA_01                         1 9.7368e+08 3.1626e+12 514890
    ## - data_channel_is_socmed         1 9.9633e+08 3.1626e+12 514890
    ## - data_channel_is_tech           1 1.0036e+09 3.1626e+12 514890
    ## - global_subjectivity            1 1.0223e+09 3.1626e+12 514890
    ## - data_channel_is_lifestyle      1 1.4624e+09 3.1631e+12 514894
    ## - data_channel_is_bus            1 1.8952e+09 3.1635e+12 514898
    ## - kw_min_avg                     1 1.9365e+09 3.1635e+12 514898
    ## - num_hrefs                      1 3.0376e+09 3.1646e+12 514908
    ## - data_channel_is_entertainment  1 3.3256e+09 3.1649e+12 514911
    ## - kw_max_avg                     1 5.6283e+09 3.1672e+12 514931
    ## - kw_avg_avg                     1 1.1612e+10 3.1732e+12 514983
    ## 
    ## Step:  AIC=514881.4
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
    ##     avg_positive_polarity + min_positive_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - rate_negative_words            1 9.5958e+06 3.1616e+12 514880
    ## - rate_positive_words            1 1.5445e+07 3.1616e+12 514880
    ## - avg_positive_polarity          1 2.2439e+07 3.1616e+12 514880
    ## - min_negative_polarity          1 2.4610e+07 3.1616e+12 514880
    ## - n_tokens_content               1 3.5474e+07 3.1617e+12 514880
    ## - title_sentiment_polarity       1 4.5813e+07 3.1617e+12 514880
    ## - self_reference_avg_sharess     1 4.9305e+07 3.1617e+12 514880
    ## - max_negative_polarity          1 5.4435e+07 3.1617e+12 514880
    ## - kw_min_max                     1 8.9167e+07 3.1617e+12 514880
    ## - kw_min_min                     1 9.4003e+07 3.1617e+12 514880
    ## - kw_max_max                     1 1.1407e+08 3.1617e+12 514880
    ## - global_rate_positive_words     1 1.3133e+08 3.1618e+12 514881
    ## - num_imgs                       1 1.7479e+08 3.1618e+12 514881
    ## - average_token_length           1 2.0638e+08 3.1618e+12 514881
    ## - LDA_04                         1 2.1775e+08 3.1618e+12 514881
    ## - n_non_stop_words               1 2.2204e+08 3.1618e+12 514881
    ## - n_non_stop_unique_tokens       1 2.2474e+08 3.1618e+12 514881
    ## <none>                                        3.1616e+12 514881
    ## - num_videos                     1 2.3702e+08 3.1619e+12 514882
    ## - self_reference_max_shares      1 2.3979e+08 3.1619e+12 514882
    ## - kw_avg_max                     1 3.0560e+08 3.1619e+12 514882
    ## - min_positive_polarity          1 3.1317e+08 3.1619e+12 514882
    ## - kw_max_min                     1 3.1567e+08 3.1619e+12 514882
    ## - self_reference_min_shares      1 3.3633e+08 3.1620e+12 514882
    ## - LDA_03                         1 3.5066e+08 3.1620e+12 514883
    ## - kw_avg_min                     1 3.7370e+08 3.1620e+12 514883
    ## - weekday_is_thursday            1 4.2742e+08 3.1620e+12 514883
    ## - data_channel_is_world          1 7.6334e+08 3.1624e+12 514886
    ## - abs_title_sentiment_polarity   1 7.9297e+08 3.1624e+12 514886
    ## - abs_title_subjectivity         1 8.3540e+08 3.1625e+12 514887
    ## - n_tokens_title                 1 8.5054e+08 3.1625e+12 514887
    ## - num_self_hrefs                 1 9.3084e+08 3.1626e+12 514888
    ## - LDA_02                         1 9.4925e+08 3.1626e+12 514888
    ## - LDA_01                         1 9.7565e+08 3.1626e+12 514888
    ## - data_channel_is_socmed         1 9.9337e+08 3.1626e+12 514888
    ## - data_channel_is_tech           1 1.0006e+09 3.1626e+12 514888
    ## - global_subjectivity            1 1.0196e+09 3.1626e+12 514888
    ## - data_channel_is_lifestyle      1 1.4597e+09 3.1631e+12 514892
    ## - data_channel_is_bus            1 1.8910e+09 3.1635e+12 514896
    ## - kw_min_avg                     1 1.9354e+09 3.1636e+12 514896
    ## - num_hrefs                      1 3.0360e+09 3.1647e+12 514906
    ## - data_channel_is_entertainment  1 3.3369e+09 3.1650e+12 514909
    ## - kw_max_avg                     1 5.6332e+09 3.1673e+12 514929
    ## - kw_avg_avg                     1 1.1616e+10 3.1732e+12 514981
    ## 
    ## Step:  AIC=514879.5
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
    ##     min_positive_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - rate_positive_words            1 7.0417e+06 3.1616e+12 514878
    ## - avg_positive_polarity          1 1.9517e+07 3.1616e+12 514878
    ## - min_negative_polarity          1 3.1484e+07 3.1617e+12 514878
    ## - n_tokens_content               1 4.2933e+07 3.1617e+12 514878
    ## - title_sentiment_polarity       1 4.4745e+07 3.1617e+12 514878
    ## - self_reference_avg_sharess     1 4.8858e+07 3.1617e+12 514878
    ## - max_negative_polarity          1 5.3379e+07 3.1617e+12 514878
    ## - kw_min_max                     1 8.9364e+07 3.1617e+12 514878
    ## - kw_min_min                     1 9.5271e+07 3.1617e+12 514878
    ## - kw_max_max                     1 1.1346e+08 3.1617e+12 514879
    ## - global_rate_positive_words     1 1.3189e+08 3.1618e+12 514879
    ## - num_imgs                       1 1.7240e+08 3.1618e+12 514879
    ## - LDA_04                         1 2.1390e+08 3.1618e+12 514879
    ## <none>                                        3.1616e+12 514880
    ## - self_reference_max_shares      1 2.3896e+08 3.1619e+12 514880
    ## - num_videos                     1 2.3927e+08 3.1619e+12 514880
    ## - n_non_stop_words               1 2.3965e+08 3.1619e+12 514880
    ## - n_non_stop_unique_tokens       1 2.4218e+08 3.1619e+12 514880
    ## - kw_avg_max                     1 3.0572e+08 3.1619e+12 514880
    ## - min_positive_polarity          1 3.0676e+08 3.1619e+12 514880
    ## - kw_max_min                     1 3.1375e+08 3.1619e+12 514880
    ## - self_reference_min_shares      1 3.3611e+08 3.1620e+12 514880
    ## - LDA_03                         1 3.4938e+08 3.1620e+12 514881
    ## - kw_avg_min                     1 3.7160e+08 3.1620e+12 514881
    ## - weekday_is_thursday            1 4.2762e+08 3.1621e+12 514881
    ## - average_token_length           1 5.7481e+08 3.1622e+12 514883
    ## - data_channel_is_world          1 7.6153e+08 3.1624e+12 514884
    ## - abs_title_sentiment_polarity   1 7.8996e+08 3.1624e+12 514884
    ## - abs_title_subjectivity         1 8.3402e+08 3.1625e+12 514885
    ## - n_tokens_title                 1 8.7325e+08 3.1625e+12 514885
    ## - num_self_hrefs                 1 9.2197e+08 3.1626e+12 514886
    ## - LDA_02                         1 9.5274e+08 3.1626e+12 514886
    ## - LDA_01                         1 9.7312e+08 3.1626e+12 514886
    ## - data_channel_is_socmed         1 9.8526e+08 3.1626e+12 514886
    ## - data_channel_is_tech           1 9.9288e+08 3.1626e+12 514886
    ## - global_subjectivity            1 1.1033e+09 3.1627e+12 514887
    ## - data_channel_is_lifestyle      1 1.4528e+09 3.1631e+12 514890
    ## - data_channel_is_bus            1 1.8831e+09 3.1635e+12 514894
    ## - kw_min_avg                     1 1.9297e+09 3.1636e+12 514894
    ## - num_hrefs                      1 3.0914e+09 3.1647e+12 514905
    ## - data_channel_is_entertainment  1 3.3309e+09 3.1650e+12 514907
    ## - kw_max_avg                     1 5.6239e+09 3.1673e+12 514927
    ## - kw_avg_avg                     1 1.1608e+10 3.1732e+12 514979
    ## 
    ## Step:  AIC=514877.6
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
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - avg_positive_polarity          1 1.7804e+07 3.1617e+12 514876
    ## - min_negative_polarity          1 2.4487e+07 3.1617e+12 514876
    ## - n_tokens_content               1 4.7491e+07 3.1617e+12 514876
    ## - self_reference_avg_sharess     1 4.8959e+07 3.1617e+12 514876
    ## - title_sentiment_polarity       1 4.9710e+07 3.1617e+12 514876
    ## - max_negative_polarity          1 6.4379e+07 3.1617e+12 514876
    ## - kw_min_max                     1 8.9120e+07 3.1617e+12 514876
    ## - kw_min_min                     1 9.5512e+07 3.1617e+12 514876
    ## - kw_max_max                     1 1.1330e+08 3.1617e+12 514877
    ## - global_rate_positive_words     1 1.3695e+08 3.1618e+12 514877
    ## - num_imgs                       1 1.6987e+08 3.1618e+12 514877
    ## - LDA_04                         1 2.1349e+08 3.1618e+12 514877
    ## <none>                                        3.1616e+12 514878
    ## - num_videos                     1 2.3513e+08 3.1619e+12 514878
    ## - n_non_stop_words               1 2.3730e+08 3.1619e+12 514878
    ## - self_reference_max_shares      1 2.3921e+08 3.1619e+12 514878
    ## - n_non_stop_unique_tokens       1 2.3978e+08 3.1619e+12 514878
    ## - kw_avg_max                     1 3.0487e+08 3.1619e+12 514878
    ## - kw_max_min                     1 3.1378e+08 3.1619e+12 514878
    ## - min_positive_polarity          1 3.1702e+08 3.1620e+12 514878
    ## - self_reference_min_shares      1 3.3664e+08 3.1620e+12 514879
    ## - LDA_03                         1 3.5169e+08 3.1620e+12 514879
    ## - kw_avg_min                     1 3.7143e+08 3.1620e+12 514879
    ## - weekday_is_thursday            1 4.2774e+08 3.1621e+12 514879
    ## - average_token_length           1 6.4205e+08 3.1623e+12 514881
    ## - data_channel_is_world          1 7.5831e+08 3.1624e+12 514882
    ## - abs_title_sentiment_polarity   1 7.8379e+08 3.1624e+12 514882
    ## - abs_title_subjectivity         1 8.4507e+08 3.1625e+12 514883
    ## - n_tokens_title                 1 8.7885e+08 3.1625e+12 514883
    ## - num_self_hrefs                 1 9.1847e+08 3.1626e+12 514884
    ## - LDA_02                         1 9.6146e+08 3.1626e+12 514884
    ## - LDA_01                         1 9.7830e+08 3.1626e+12 514884
    ## - data_channel_is_socmed         1 9.7975e+08 3.1626e+12 514884
    ## - data_channel_is_tech           1 9.8698e+08 3.1626e+12 514884
    ## - global_subjectivity            1 1.1415e+09 3.1628e+12 514886
    ## - data_channel_is_lifestyle      1 1.4476e+09 3.1631e+12 514888
    ## - data_channel_is_bus            1 1.8772e+09 3.1635e+12 514892
    ## - kw_min_avg                     1 1.9315e+09 3.1636e+12 514893
    ## - num_hrefs                      1 3.1005e+09 3.1647e+12 514903
    ## - data_channel_is_entertainment  1 3.3247e+09 3.1650e+12 514905
    ## - kw_max_avg                     1 5.6182e+09 3.1673e+12 514925
    ## - kw_avg_avg                     1 1.1601e+10 3.1732e+12 514977
    ## 
    ## Step:  AIC=514875.7
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
    ##     abs_title_sentiment_polarity + weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - min_negative_polarity          1 2.3300e+07 3.1617e+12 514874
    ## - n_tokens_content               1 4.1922e+07 3.1617e+12 514874
    ## - title_sentiment_polarity       1 4.6451e+07 3.1617e+12 514874
    ## - self_reference_avg_sharess     1 4.9297e+07 3.1617e+12 514874
    ## - max_negative_polarity          1 6.8213e+07 3.1617e+12 514874
    ## - kw_min_max                     1 8.9629e+07 3.1617e+12 514875
    ## - kw_min_min                     1 9.6076e+07 3.1618e+12 514875
    ## - kw_max_max                     1 1.1269e+08 3.1618e+12 514875
    ## - global_rate_positive_words     1 1.5451e+08 3.1618e+12 514875
    ## - num_imgs                       1 1.6714e+08 3.1618e+12 514875
    ## - LDA_04                         1 2.1075e+08 3.1619e+12 514876
    ## <none>                                        3.1617e+12 514876
    ## - num_videos                     1 2.2920e+08 3.1619e+12 514876
    ## - n_non_stop_words               1 2.3528e+08 3.1619e+12 514876
    ## - n_non_stop_unique_tokens       1 2.3779e+08 3.1619e+12 514876
    ## - self_reference_max_shares      1 2.3962e+08 3.1619e+12 514876
    ## - kw_avg_max                     1 3.0255e+08 3.1620e+12 514876
    ## - kw_max_min                     1 3.1485e+08 3.1620e+12 514876
    ## - self_reference_min_shares      1 3.3760e+08 3.1620e+12 514877
    ## - LDA_03                         1 3.5182e+08 3.1620e+12 514877
    ## - kw_avg_min                     1 3.7157e+08 3.1620e+12 514877
    ## - weekday_is_thursday            1 4.2610e+08 3.1621e+12 514877
    ## - min_positive_polarity          1 4.7942e+08 3.1621e+12 514878
    ## - average_token_length           1 7.2902e+08 3.1624e+12 514880
    ## - data_channel_is_world          1 7.5518e+08 3.1624e+12 514880
    ## - abs_title_sentiment_polarity   1 7.7112e+08 3.1624e+12 514880
    ## - abs_title_subjectivity         1 8.3105e+08 3.1625e+12 514881
    ## - n_tokens_title                 1 8.7614e+08 3.1625e+12 514881
    ## - num_self_hrefs                 1 9.1576e+08 3.1626e+12 514882
    ## - LDA_02                         1 9.5547e+08 3.1626e+12 514882
    ## - LDA_01                         1 9.7508e+08 3.1626e+12 514882
    ## - data_channel_is_socmed         1 9.7862e+08 3.1626e+12 514882
    ## - data_channel_is_tech           1 9.8716e+08 3.1626e+12 514882
    ## - global_subjectivity            1 1.1603e+09 3.1628e+12 514884
    ## - data_channel_is_lifestyle      1 1.4573e+09 3.1631e+12 514887
    ## - data_channel_is_bus            1 1.8785e+09 3.1635e+12 514890
    ## - kw_min_avg                     1 1.9329e+09 3.1636e+12 514891
    ## - num_hrefs                      1 3.0869e+09 3.1647e+12 514901
    ## - data_channel_is_entertainment  1 3.3380e+09 3.1650e+12 514903
    ## - kw_max_avg                     1 5.6220e+09 3.1673e+12 514923
    ## - kw_avg_avg                     1 1.1598e+10 3.1733e+12 514975
    ## 
    ## Step:  AIC=514873.9
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
    ##     weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - title_sentiment_polarity       1 3.9138e+07 3.1617e+12 514872
    ## - self_reference_avg_sharess     1 4.9561e+07 3.1617e+12 514872
    ## - n_tokens_content               1 7.7281e+07 3.1618e+12 514873
    ## - max_negative_polarity          1 8.0689e+07 3.1618e+12 514873
    ## - kw_min_max                     1 8.9077e+07 3.1618e+12 514873
    ## - kw_min_min                     1 9.5825e+07 3.1618e+12 514873
    ## - kw_max_max                     1 1.1186e+08 3.1618e+12 514873
    ## - num_imgs                       1 1.6232e+08 3.1618e+12 514873
    ## - global_rate_positive_words     1 1.6616e+08 3.1618e+12 514873
    ## - LDA_04                         1 2.1379e+08 3.1619e+12 514874
    ## <none>                                        3.1617e+12 514874
    ## - num_videos                     1 2.3855e+08 3.1619e+12 514874
    ## - n_non_stop_words               1 2.3882e+08 3.1619e+12 514874
    ## - self_reference_max_shares      1 2.4097e+08 3.1619e+12 514874
    ## - n_non_stop_unique_tokens       1 2.4132e+08 3.1619e+12 514874
    ## - kw_avg_max                     1 3.0204e+08 3.1620e+12 514875
    ## - kw_max_min                     1 3.1434e+08 3.1620e+12 514875
    ## - self_reference_min_shares      1 3.3796e+08 3.1620e+12 514875
    ## - LDA_03                         1 3.5115e+08 3.1620e+12 514875
    ## - kw_avg_min                     1 3.7135e+08 3.1620e+12 514875
    ## - weekday_is_thursday            1 4.2692e+08 3.1621e+12 514876
    ## - min_positive_polarity          1 5.1138e+08 3.1622e+12 514876
    ## - average_token_length           1 7.1562e+08 3.1624e+12 514878
    ## - data_channel_is_world          1 7.5385e+08 3.1624e+12 514879
    ## - abs_title_sentiment_polarity   1 7.9672e+08 3.1625e+12 514879
    ## - abs_title_subjectivity         1 8.3251e+08 3.1625e+12 514879
    ## - n_tokens_title                 1 8.8134e+08 3.1626e+12 514880
    ## - num_self_hrefs                 1 9.2149e+08 3.1626e+12 514880
    ## - LDA_02                         1 9.5320e+08 3.1626e+12 514880
    ## - LDA_01                         1 9.7122e+08 3.1626e+12 514880
    ## - data_channel_is_socmed         1 9.8816e+08 3.1627e+12 514881
    ## - data_channel_is_tech           1 1.0027e+09 3.1627e+12 514881
    ## - global_subjectivity            1 1.3224e+09 3.1630e+12 514884
    ## - data_channel_is_lifestyle      1 1.4583e+09 3.1631e+12 514885
    ## - data_channel_is_bus            1 1.8987e+09 3.1636e+12 514889
    ## - kw_min_avg                     1 1.9349e+09 3.1636e+12 514889
    ## - num_hrefs                      1 3.0953e+09 3.1648e+12 514899
    ## - data_channel_is_entertainment  1 3.3360e+09 3.1650e+12 514901
    ## - kw_max_avg                     1 5.6233e+09 3.1673e+12 514921
    ## - kw_avg_avg                     1 1.1605e+10 3.1733e+12 514974
    ## 
    ## Step:  AIC=514872.3
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
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - self_reference_avg_sharess     1 5.0240e+07 3.1618e+12 514871
    ## - n_tokens_content               1 7.5826e+07 3.1618e+12 514871
    ## - max_negative_polarity          1 8.0852e+07 3.1618e+12 514871
    ## - kw_min_max                     1 8.8478e+07 3.1618e+12 514871
    ## - kw_min_min                     1 9.6865e+07 3.1618e+12 514871
    ## - kw_max_max                     1 1.1199e+08 3.1618e+12 514871
    ## - global_rate_positive_words     1 1.5189e+08 3.1619e+12 514872
    ## - num_imgs                       1 1.6577e+08 3.1619e+12 514872
    ## - LDA_04                         1 2.1607e+08 3.1619e+12 514872
    ## <none>                                        3.1617e+12 514872
    ## - n_non_stop_words               1 2.3379e+08 3.1620e+12 514872
    ## - n_non_stop_unique_tokens       1 2.3626e+08 3.1620e+12 514872
    ## - num_videos                     1 2.3999e+08 3.1620e+12 514872
    ## - self_reference_max_shares      1 2.4228e+08 3.1620e+12 514872
    ## - kw_avg_max                     1 3.0273e+08 3.1620e+12 514873
    ## - kw_max_min                     1 3.1482e+08 3.1620e+12 514873
    ## - self_reference_min_shares      1 3.3996e+08 3.1621e+12 514873
    ## - LDA_03                         1 3.5337e+08 3.1621e+12 514873
    ## - kw_avg_min                     1 3.7218e+08 3.1621e+12 514874
    ## - weekday_is_thursday            1 4.2533e+08 3.1621e+12 514874
    ## - min_positive_polarity          1 5.0177e+08 3.1622e+12 514875
    ## - average_token_length           1 7.2178e+08 3.1624e+12 514877
    ## - data_channel_is_world          1 7.5625e+08 3.1625e+12 514877
    ## - abs_title_subjectivity         1 8.0859e+08 3.1625e+12 514877
    ## - n_tokens_title                 1 8.7781e+08 3.1626e+12 514878
    ## - num_self_hrefs                 1 9.1980e+08 3.1626e+12 514878
    ## - LDA_02                         1 9.5252e+08 3.1627e+12 514879
    ## - LDA_01                         1 9.7861e+08 3.1627e+12 514879
    ## - data_channel_is_socmed         1 9.8188e+08 3.1627e+12 514879
    ## - data_channel_is_tech           1 9.9395e+08 3.1627e+12 514879
    ## - abs_title_sentiment_polarity   1 1.0352e+09 3.1628e+12 514879
    ## - global_subjectivity            1 1.3066e+09 3.1630e+12 514882
    ## - data_channel_is_lifestyle      1 1.4462e+09 3.1632e+12 514883
    ## - data_channel_is_bus            1 1.8917e+09 3.1636e+12 514887
    ## - kw_min_avg                     1 1.9345e+09 3.1637e+12 514887
    ## - num_hrefs                      1 3.1043e+09 3.1648e+12 514898
    ## - data_channel_is_entertainment  1 3.3333e+09 3.1650e+12 514900
    ## - kw_max_avg                     1 5.6255e+09 3.1673e+12 514920
    ## - kw_avg_avg                     1 1.1611e+10 3.1733e+12 514972
    ## 
    ## Step:  AIC=514870.7
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
    ##     weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_tokens_content               1 7.6339e+07 3.1618e+12 514869
    ## - max_negative_polarity          1 8.0401e+07 3.1618e+12 514869
    ## - kw_min_max                     1 8.6851e+07 3.1619e+12 514869
    ## - kw_min_min                     1 9.6498e+07 3.1619e+12 514870
    ## - kw_max_max                     1 1.1261e+08 3.1619e+12 514870
    ## - global_rate_positive_words     1 1.5011e+08 3.1619e+12 514870
    ## - num_imgs                       1 1.6371e+08 3.1619e+12 514870
    ## - LDA_04                         1 2.1644e+08 3.1620e+12 514871
    ## <none>                                        3.1618e+12 514871
    ## - n_non_stop_words               1 2.3607e+08 3.1620e+12 514871
    ## - n_non_stop_unique_tokens       1 2.3851e+08 3.1620e+12 514871
    ## - num_videos                     1 2.4677e+08 3.1620e+12 514871
    ## - kw_avg_max                     1 3.0498e+08 3.1621e+12 514871
    ## - kw_max_min                     1 3.2045e+08 3.1621e+12 514872
    ## - LDA_03                         1 3.5312e+08 3.1621e+12 514872
    ## - kw_avg_min                     1 3.7668e+08 3.1621e+12 514872
    ## - weekday_is_thursday            1 4.2363e+08 3.1622e+12 514872
    ## - min_positive_polarity          1 4.9945e+08 3.1623e+12 514873
    ## - self_reference_max_shares      1 5.2675e+08 3.1623e+12 514873
    ## - average_token_length           1 7.3054e+08 3.1625e+12 514875
    ## - data_channel_is_world          1 7.5173e+08 3.1625e+12 514875
    ## - self_reference_min_shares      1 7.7281e+08 3.1625e+12 514875
    ## - abs_title_subjectivity         1 8.0902e+08 3.1626e+12 514876
    ## - n_tokens_title                 1 8.7188e+08 3.1626e+12 514876
    ## - num_self_hrefs                 1 8.7819e+08 3.1626e+12 514876
    ## - LDA_02                         1 9.5168e+08 3.1627e+12 514877
    ## - LDA_01                         1 9.7852e+08 3.1627e+12 514877
    ## - data_channel_is_socmed         1 9.8071e+08 3.1627e+12 514877
    ## - data_channel_is_tech           1 9.9566e+08 3.1628e+12 514877
    ## - abs_title_sentiment_polarity   1 1.0362e+09 3.1628e+12 514878
    ## - global_subjectivity            1 1.2978e+09 3.1631e+12 514880
    ## - data_channel_is_lifestyle      1 1.4418e+09 3.1632e+12 514881
    ## - data_channel_is_bus            1 1.8850e+09 3.1637e+12 514885
    ## - kw_min_avg                     1 1.9445e+09 3.1637e+12 514886
    ## - num_hrefs                      1 3.1315e+09 3.1649e+12 514896
    ## - data_channel_is_entertainment  1 3.3291e+09 3.1651e+12 514898
    ## - kw_max_avg                     1 5.6463e+09 3.1674e+12 514918
    ## - kw_avg_avg                     1 1.1620e+10 3.1734e+12 514971
    ## 
    ## Step:  AIC=514869.4
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + min_positive_polarity + max_negative_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - max_negative_polarity          1 5.7714e+07 3.1619e+12 514868
    ## - kw_min_max                     1 8.4604e+07 3.1619e+12 514868
    ## - kw_min_min                     1 9.7732e+07 3.1619e+12 514868
    ## - kw_max_max                     1 1.0834e+08 3.1620e+12 514868
    ## - global_rate_positive_words     1 1.3064e+08 3.1620e+12 514869
    ## - n_non_stop_words               1 1.6629e+08 3.1620e+12 514869
    ## - n_non_stop_unique_tokens       1 1.6854e+08 3.1620e+12 514869
    ## - num_imgs                       1 1.9369e+08 3.1620e+12 514869
    ## - LDA_04                         1 2.2083e+08 3.1621e+12 514869
    ## <none>                                        3.1618e+12 514869
    ## - num_videos                     1 2.9185e+08 3.1621e+12 514870
    ## - kw_avg_max                     1 3.0473e+08 3.1621e+12 514870
    ## - kw_max_min                     1 3.2025e+08 3.1622e+12 514870
    ## - kw_avg_min                     1 3.7491e+08 3.1622e+12 514871
    ## - LDA_03                         1 3.7752e+08 3.1622e+12 514871
    ## - weekday_is_thursday            1 4.2456e+08 3.1623e+12 514871
    ## - self_reference_max_shares      1 5.2442e+08 3.1624e+12 514872
    ## - min_positive_polarity          1 5.6834e+08 3.1624e+12 514872
    ## - average_token_length           1 7.0217e+08 3.1625e+12 514874
    ## - data_channel_is_world          1 7.2903e+08 3.1626e+12 514874
    ## - self_reference_min_shares      1 7.7143e+08 3.1626e+12 514874
    ## - abs_title_subjectivity         1 8.2263e+08 3.1627e+12 514875
    ## - num_self_hrefs                 1 8.3538e+08 3.1627e+12 514875
    ## - n_tokens_title                 1 8.9058e+08 3.1627e+12 514875
    ## - LDA_02                         1 9.4416e+08 3.1628e+12 514876
    ## - data_channel_is_socmed         1 9.6185e+08 3.1628e+12 514876
    ## - data_channel_is_tech           1 9.7482e+08 3.1628e+12 514876
    ## - LDA_01                         1 9.9011e+08 3.1628e+12 514876
    ## - abs_title_sentiment_polarity   1 1.0352e+09 3.1629e+12 514876
    ## - global_subjectivity            1 1.3811e+09 3.1632e+12 514880
    ## - data_channel_is_lifestyle      1 1.4047e+09 3.1632e+12 514880
    ## - data_channel_is_bus            1 1.8528e+09 3.1637e+12 514884
    ## - kw_min_avg                     1 1.9400e+09 3.1638e+12 514884
    ## - data_channel_is_entertainment  1 3.2572e+09 3.1651e+12 514896
    ## - num_hrefs                      1 3.4705e+09 3.1653e+12 514898
    ## - kw_max_avg                     1 5.6227e+09 3.1675e+12 514917
    ## - kw_avg_avg                     1 1.1571e+10 3.1734e+12 514969
    ## 
    ## Step:  AIC=514867.9
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - kw_min_max                     1 8.6458e+07 3.1620e+12 514867
    ## - kw_min_min                     1 9.8782e+07 3.1620e+12 514867
    ## - kw_max_max                     1 1.0953e+08 3.1620e+12 514867
    ## - global_rate_positive_words     1 1.3542e+08 3.1620e+12 514867
    ## - n_non_stop_words               1 1.9004e+08 3.1621e+12 514868
    ## - num_imgs                       1 1.9210e+08 3.1621e+12 514868
    ## - n_non_stop_unique_tokens       1 1.9246e+08 3.1621e+12 514868
    ## - LDA_04                         1 2.2272e+08 3.1621e+12 514868
    ## <none>                                        3.1619e+12 514868
    ## - num_videos                     1 2.7727e+08 3.1622e+12 514868
    ## - kw_avg_max                     1 2.9992e+08 3.1622e+12 514869
    ## - kw_max_min                     1 3.1943e+08 3.1622e+12 514869
    ## - LDA_03                         1 3.7234e+08 3.1623e+12 514869
    ## - kw_avg_min                     1 3.7480e+08 3.1623e+12 514869
    ## - weekday_is_thursday            1 4.2402e+08 3.1623e+12 514870
    ## - self_reference_max_shares      1 5.2460e+08 3.1624e+12 514870
    ## - min_positive_polarity          1 5.4824e+08 3.1624e+12 514871
    ## - average_token_length           1 6.6797e+08 3.1626e+12 514872
    ## - data_channel_is_world          1 7.3845e+08 3.1626e+12 514872
    ## - self_reference_min_shares      1 7.8103e+08 3.1627e+12 514873
    ## - abs_title_subjectivity         1 8.2255e+08 3.1627e+12 514873
    ## - num_self_hrefs                 1 8.3908e+08 3.1627e+12 514873
    ## - n_tokens_title                 1 8.9302e+08 3.1628e+12 514874
    ## - LDA_02                         1 9.5057e+08 3.1629e+12 514874
    ## - data_channel_is_socmed         1 9.6351e+08 3.1629e+12 514874
    ## - data_channel_is_tech           1 9.8414e+08 3.1629e+12 514875
    ## - LDA_01                         1 9.8682e+08 3.1629e+12 514875
    ## - abs_title_sentiment_polarity   1 1.0379e+09 3.1629e+12 514875
    ## - data_channel_is_lifestyle      1 1.4194e+09 3.1633e+12 514878
    ## - global_subjectivity            1 1.4491e+09 3.1633e+12 514879
    ## - data_channel_is_bus            1 1.8668e+09 3.1638e+12 514882
    ## - kw_min_avg                     1 1.9433e+09 3.1638e+12 514883
    ## - data_channel_is_entertainment  1 3.2753e+09 3.1652e+12 514895
    ## - num_hrefs                      1 3.4321e+09 3.1653e+12 514896
    ## - kw_max_avg                     1 5.6271e+09 3.1675e+12 514915
    ## - kw_avg_avg                     1 1.1588e+10 3.1735e+12 514967
    ## 
    ## Step:  AIC=514866.7
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + LDA_01 + 
    ##     LDA_02 + LDA_03 + LDA_04 + global_subjectivity + global_rate_positive_words + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - kw_max_max                     1 8.4542e+07 3.1621e+12 514865
    ## - kw_min_min                     1 9.5628e+07 3.1621e+12 514865
    ## - global_rate_positive_words     1 1.3963e+08 3.1621e+12 514866
    ## - num_imgs                       1 1.8887e+08 3.1622e+12 514866
    ## - n_non_stop_words               1 1.9392e+08 3.1622e+12 514866
    ## - n_non_stop_unique_tokens       1 1.9638e+08 3.1622e+12 514866
    ## <none>                                        3.1620e+12 514867
    ## - LDA_04                         1 2.2981e+08 3.1622e+12 514867
    ## - num_videos                     1 2.8204e+08 3.1623e+12 514867
    ## - kw_max_min                     1 3.2824e+08 3.1623e+12 514868
    ## - LDA_03                         1 3.6877e+08 3.1624e+12 514868
    ## - kw_avg_min                     1 3.8199e+08 3.1624e+12 514868
    ## - weekday_is_thursday            1 4.2304e+08 3.1624e+12 514868
    ## - self_reference_max_shares      1 5.3144e+08 3.1625e+12 514869
    ## - kw_avg_max                     1 5.3509e+08 3.1625e+12 514869
    ## - min_positive_polarity          1 5.5362e+08 3.1625e+12 514870
    ## - average_token_length           1 6.8011e+08 3.1627e+12 514871
    ## - data_channel_is_world          1 7.7589e+08 3.1628e+12 514871
    ## - self_reference_min_shares      1 7.7727e+08 3.1628e+12 514871
    ## - num_self_hrefs                 1 8.2610e+08 3.1628e+12 514872
    ## - abs_title_subjectivity         1 8.2735e+08 3.1628e+12 514872
    ## - n_tokens_title                 1 9.0414e+08 3.1629e+12 514873
    ## - LDA_02                         1 9.4522e+08 3.1629e+12 514873
    ## - LDA_01                         1 1.0006e+09 3.1630e+12 514873
    ## - data_channel_is_tech           1 1.0117e+09 3.1630e+12 514874
    ## - abs_title_sentiment_polarity   1 1.0383e+09 3.1630e+12 514874
    ## - data_channel_is_socmed         1 1.0417e+09 3.1630e+12 514874
    ## - global_subjectivity            1 1.4544e+09 3.1634e+12 514877
    ## - data_channel_is_lifestyle      1 1.4703e+09 3.1635e+12 514878
    ## - data_channel_is_bus            1 1.8773e+09 3.1639e+12 514881
    ## - kw_min_avg                     1 2.1874e+09 3.1642e+12 514884
    ## - data_channel_is_entertainment  1 3.3934e+09 3.1654e+12 514894
    ## - num_hrefs                      1 3.4207e+09 3.1654e+12 514895
    ## - kw_max_avg                     1 5.7261e+09 3.1677e+12 514915
    ## - kw_avg_avg                     1 1.1769e+10 3.1738e+12 514968
    ## 
    ## Step:  AIC=514865.4
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + global_rate_positive_words + min_positive_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - global_rate_positive_words     1 1.3782e+08 3.1622e+12 514865
    ## - num_imgs                       1 1.8408e+08 3.1623e+12 514865
    ## - n_non_stop_words               1 2.0052e+08 3.1623e+12 514865
    ## - n_non_stop_unique_tokens       1 2.0304e+08 3.1623e+12 514865
    ## <none>                                        3.1621e+12 514865
    ## - LDA_04                         1 2.3638e+08 3.1623e+12 514865
    ## - num_videos                     1 2.8918e+08 3.1624e+12 514866
    ## - kw_max_min                     1 3.1093e+08 3.1624e+12 514866
    ## - LDA_03                         1 3.5073e+08 3.1624e+12 514866
    ## - kw_avg_min                     1 3.6224e+08 3.1624e+12 514867
    ## - weekday_is_thursday            1 4.2420e+08 3.1625e+12 514867
    ## - self_reference_max_shares      1 5.4032e+08 3.1626e+12 514868
    ## - min_positive_polarity          1 5.5338e+08 3.1626e+12 514868
    ## - kw_min_min                     1 6.6311e+08 3.1627e+12 514869
    ## - average_token_length           1 6.8913e+08 3.1628e+12 514869
    ## - kw_avg_max                     1 7.1815e+08 3.1628e+12 514870
    ## - self_reference_min_shares      1 7.7010e+08 3.1628e+12 514870
    ## - num_self_hrefs                 1 8.1741e+08 3.1629e+12 514871
    ## - data_channel_is_world          1 8.2964e+08 3.1629e+12 514871
    ## - abs_title_subjectivity         1 8.3346e+08 3.1629e+12 514871
    ## - n_tokens_title                 1 8.9120e+08 3.1630e+12 514871
    ## - LDA_02                         1 9.4437e+08 3.1630e+12 514872
    ## - LDA_01                         1 1.0010e+09 3.1631e+12 514872
    ## - abs_title_sentiment_polarity   1 1.0439e+09 3.1631e+12 514873
    ## - data_channel_is_tech           1 1.0445e+09 3.1631e+12 514873
    ## - data_channel_is_socmed         1 1.0809e+09 3.1632e+12 514873
    ## - global_subjectivity            1 1.4512e+09 3.1635e+12 514876
    ## - data_channel_is_lifestyle      1 1.5016e+09 3.1636e+12 514877
    ## - data_channel_is_bus            1 1.8895e+09 3.1640e+12 514880
    ## - kw_min_avg                     1 2.1141e+09 3.1642e+12 514882
    ## - num_hrefs                      1 3.3855e+09 3.1655e+12 514893
    ## - data_channel_is_entertainment  1 3.5624e+09 3.1656e+12 514895
    ## - kw_max_avg                     1 5.6453e+09 3.1677e+12 514913
    ## - kw_avg_avg                     1 1.1715e+10 3.1738e+12 514966
    ## 
    ## Step:  AIC=514864.6
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_non_stop_words               1 1.7355e+08 3.1624e+12 514864
    ## - n_non_stop_unique_tokens       1 1.7594e+08 3.1624e+12 514864
    ## - num_imgs                       1 1.9742e+08 3.1624e+12 514864
    ## - LDA_04                         1 2.2483e+08 3.1624e+12 514865
    ## <none>                                        3.1622e+12 514865
    ## - num_videos                     1 2.7003e+08 3.1625e+12 514865
    ## - kw_max_min                     1 3.1453e+08 3.1625e+12 514865
    ## - LDA_03                         1 3.2828e+08 3.1625e+12 514865
    ## - kw_avg_min                     1 3.6580e+08 3.1626e+12 514866
    ## - weekday_is_thursday            1 4.2532e+08 3.1626e+12 514866
    ## - min_positive_polarity          1 4.4213e+08 3.1627e+12 514866
    ## - self_reference_max_shares      1 5.4672e+08 3.1628e+12 514867
    ## - kw_min_min                     1 6.5424e+08 3.1629e+12 514868
    ## - kw_avg_max                     1 6.8216e+08 3.1629e+12 514869
    ## - average_token_length           1 7.7720e+08 3.1630e+12 514869
    ## - self_reference_min_shares      1 7.8078e+08 3.1630e+12 514869
    ## - data_channel_is_world          1 7.8719e+08 3.1630e+12 514870
    ## - num_self_hrefs                 1 8.6934e+08 3.1631e+12 514870
    ## - LDA_02                         1 9.0412e+08 3.1631e+12 514871
    ## - n_tokens_title                 1 9.1945e+08 3.1631e+12 514871
    ## - abs_title_subjectivity         1 9.4177e+08 3.1632e+12 514871
    ## - LDA_01                         1 9.8716e+08 3.1632e+12 514871
    ## - abs_title_sentiment_polarity   1 1.0222e+09 3.1632e+12 514872
    ## - data_channel_is_tech           1 1.0437e+09 3.1633e+12 514872
    ## - data_channel_is_socmed         1 1.0964e+09 3.1633e+12 514872
    ## - global_subjectivity            1 1.3166e+09 3.1635e+12 514874
    ## - data_channel_is_lifestyle      1 1.5045e+09 3.1637e+12 514876
    ## - data_channel_is_bus            1 1.8780e+09 3.1641e+12 514879
    ## - kw_min_avg                     1 2.1300e+09 3.1643e+12 514881
    ## - num_hrefs                      1 3.5012e+09 3.1657e+12 514893
    ## - data_channel_is_entertainment  1 3.5340e+09 3.1657e+12 514894
    ## - kw_max_avg                     1 5.6505e+09 3.1679e+12 514912
    ## - kw_avg_avg                     1 1.1714e+10 3.1739e+12 514965
    ## 
    ## Step:  AIC=514864.1
    ## shares ~ n_tokens_title + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_non_stop_unique_tokens       1 2.0162e+07 3.1624e+12 514862
    ## - num_imgs                       1 8.9196e+07 3.1625e+12 514863
    ## - LDA_04                         1 2.2661e+08 3.1626e+12 514864
    ## <none>                                        3.1624e+12 514864
    ## - num_videos                     1 2.4858e+08 3.1626e+12 514864
    ## - LDA_03                         1 3.0693e+08 3.1627e+12 514865
    ## - kw_max_min                     1 3.1380e+08 3.1627e+12 514865
    ## - min_positive_polarity          1 3.6089e+08 3.1627e+12 514865
    ## - kw_avg_min                     1 3.6564e+08 3.1627e+12 514865
    ## - weekday_is_thursday            1 4.2978e+08 3.1628e+12 514866
    ## - self_reference_max_shares      1 5.5974e+08 3.1629e+12 514867
    ## - kw_avg_max                     1 6.6109e+08 3.1630e+12 514868
    ## - kw_min_min                     1 6.9149e+08 3.1631e+12 514868
    ## - average_token_length           1 7.1205e+08 3.1631e+12 514868
    ## - self_reference_min_shares      1 7.7717e+08 3.1632e+12 514869
    ## - data_channel_is_world          1 8.0726e+08 3.1632e+12 514869
    ## - num_self_hrefs                 1 8.8064e+08 3.1633e+12 514870
    ## - n_tokens_title                 1 8.9636e+08 3.1633e+12 514870
    ## - LDA_02                         1 9.0879e+08 3.1633e+12 514870
    ## - abs_title_subjectivity         1 9.3069e+08 3.1633e+12 514870
    ## - LDA_01                         1 9.4393e+08 3.1633e+12 514870
    ## - abs_title_sentiment_polarity   1 1.0015e+09 3.1634e+12 514871
    ## - data_channel_is_tech           1 1.0647e+09 3.1634e+12 514871
    ## - data_channel_is_socmed         1 1.1088e+09 3.1635e+12 514872
    ## - global_subjectivity            1 1.3680e+09 3.1638e+12 514874
    ## - data_channel_is_lifestyle      1 1.5088e+09 3.1639e+12 514875
    ## - data_channel_is_bus            1 1.8969e+09 3.1643e+12 514879
    ## - kw_min_avg                     1 2.1473e+09 3.1645e+12 514881
    ## - num_hrefs                      1 3.3277e+09 3.1657e+12 514891
    ## - data_channel_is_entertainment  1 3.6189e+09 3.1660e+12 514894
    ## - kw_max_avg                     1 5.6560e+09 3.1680e+12 514912
    ## - kw_avg_avg                     1 1.1729e+10 3.1741e+12 514965
    ## 
    ## Step:  AIC=514862.3
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - num_imgs                       1 9.1632e+07 3.1625e+12 514861
    ## <none>                                        3.1624e+12 514862
    ## - LDA_04                         1 2.3041e+08 3.1626e+12 514862
    ## - num_videos                     1 2.4915e+08 3.1627e+12 514862
    ## - LDA_03                         1 3.1352e+08 3.1627e+12 514863
    ## - kw_max_min                     1 3.1385e+08 3.1627e+12 514863
    ## - min_positive_polarity          1 3.6016e+08 3.1628e+12 514863
    ## - kw_avg_min                     1 3.6564e+08 3.1628e+12 514864
    ## - weekday_is_thursday            1 4.3042e+08 3.1628e+12 514864
    ## - self_reference_max_shares      1 5.5946e+08 3.1630e+12 514865
    ## - kw_avg_max                     1 6.5974e+08 3.1631e+12 514866
    ## - kw_min_min                     1 6.9233e+08 3.1631e+12 514866
    ## - average_token_length           1 7.0414e+08 3.1631e+12 514866
    ## - self_reference_min_shares      1 7.7741e+08 3.1632e+12 514867
    ## - data_channel_is_world          1 8.1042e+08 3.1632e+12 514867
    ## - num_self_hrefs                 1 8.7904e+08 3.1633e+12 514868
    ## - n_tokens_title                 1 8.9466e+08 3.1633e+12 514868
    ## - LDA_02                         1 9.1792e+08 3.1633e+12 514868
    ## - abs_title_subjectivity         1 9.2725e+08 3.1633e+12 514868
    ## - LDA_01                         1 9.5628e+08 3.1634e+12 514869
    ## - abs_title_sentiment_polarity   1 9.9955e+08 3.1634e+12 514869
    ## - data_channel_is_tech           1 1.0692e+09 3.1635e+12 514870
    ## - data_channel_is_socmed         1 1.1157e+09 3.1635e+12 514870
    ## - global_subjectivity            1 1.3606e+09 3.1638e+12 514872
    ## - data_channel_is_lifestyle      1 1.5137e+09 3.1639e+12 514874
    ## - data_channel_is_bus            1 1.9119e+09 3.1643e+12 514877
    ## - kw_min_avg                     1 2.1452e+09 3.1645e+12 514879
    ## - num_hrefs                      1 3.3191e+09 3.1657e+12 514889
    ## - data_channel_is_entertainment  1 3.6154e+09 3.1660e+12 514892
    ## - kw_max_avg                     1 5.6546e+09 3.1681e+12 514910
    ## - kw_avg_avg                     1 1.1726e+10 3.1741e+12 514963
    ## 
    ## Step:  AIC=514861.1
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + num_videos + 
    ##     average_token_length + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - num_videos                     1 2.0491e+08 3.1627e+12 514861
    ## - LDA_04                         1 2.1795e+08 3.1627e+12 514861
    ## <none>                                        3.1625e+12 514861
    ## - LDA_03                         1 2.8296e+08 3.1628e+12 514862
    ## - kw_max_min                     1 3.1952e+08 3.1628e+12 514862
    ## - min_positive_polarity          1 3.7380e+08 3.1629e+12 514862
    ## - kw_avg_min                     1 3.7526e+08 3.1629e+12 514862
    ## - weekday_is_thursday            1 4.2715e+08 3.1629e+12 514863
    ## - self_reference_max_shares      1 5.5864e+08 3.1631e+12 514864
    ## - kw_min_min                     1 6.7616e+08 3.1632e+12 514865
    ## - kw_avg_max                     1 6.8896e+08 3.1632e+12 514865
    ## - average_token_length           1 6.9400e+08 3.1632e+12 514865
    ## - self_reference_min_shares      1 7.8111e+08 3.1633e+12 514866
    ## - num_self_hrefs                 1 8.1826e+08 3.1633e+12 514866
    ## - data_channel_is_world          1 8.3650e+08 3.1633e+12 514866
    ## - n_tokens_title                 1 8.9090e+08 3.1634e+12 514867
    ## - LDA_02                         1 8.9275e+08 3.1634e+12 514867
    ## - LDA_01                         1 9.2758e+08 3.1634e+12 514867
    ## - abs_title_subjectivity         1 9.2778e+08 3.1634e+12 514867
    ## - abs_title_sentiment_polarity   1 1.0178e+09 3.1635e+12 514868
    ## - data_channel_is_tech           1 1.0833e+09 3.1636e+12 514869
    ## - data_channel_is_socmed         1 1.1392e+09 3.1636e+12 514869
    ## - global_subjectivity            1 1.3520e+09 3.1638e+12 514871
    ## - data_channel_is_lifestyle      1 1.5280e+09 3.1640e+12 514873
    ## - data_channel_is_bus            1 1.9337e+09 3.1644e+12 514876
    ## - kw_min_avg                     1 2.1440e+09 3.1646e+12 514878
    ## - data_channel_is_entertainment  1 3.6113e+09 3.1661e+12 514891
    ## - num_hrefs                      1 3.8326e+09 3.1663e+12 514893
    ## - kw_max_avg                     1 5.7097e+09 3.1682e+12 514909
    ## - kw_avg_avg                     1 1.1865e+10 3.1744e+12 514963
    ## 
    ## Step:  AIC=514860.9
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - LDA_04                         1 2.1685e+08 3.1629e+12 514861
    ## <none>                                        3.1627e+12 514861
    ## - LDA_03                         1 2.4505e+08 3.1629e+12 514861
    ## - kw_max_min                     1 3.1946e+08 3.1630e+12 514862
    ## - kw_avg_min                     1 3.7131e+08 3.1631e+12 514862
    ## - min_positive_polarity          1 3.9427e+08 3.1631e+12 514862
    ## - weekday_is_thursday            1 4.2788e+08 3.1631e+12 514863
    ## - self_reference_max_shares      1 6.0586e+08 3.1633e+12 514864
    ## - kw_avg_max                     1 6.0768e+08 3.1633e+12 514864
    ## - average_token_length           1 6.9298e+08 3.1634e+12 514865
    ## - kw_min_min                     1 7.4776e+08 3.1634e+12 514865
    ## - self_reference_min_shares      1 7.5320e+08 3.1635e+12 514866
    ## - num_self_hrefs                 1 7.8463e+08 3.1635e+12 514866
    ## - data_channel_is_world          1 8.3274e+08 3.1635e+12 514866
    ## - LDA_02                         1 8.8996e+08 3.1636e+12 514867
    ## - n_tokens_title                 1 9.0873e+08 3.1636e+12 514867
    ## - abs_title_subjectivity         1 9.2240e+08 3.1636e+12 514867
    ## - LDA_01                         1 9.3523e+08 3.1636e+12 514867
    ## - abs_title_sentiment_polarity   1 1.0280e+09 3.1637e+12 514868
    ## - data_channel_is_tech           1 1.0871e+09 3.1638e+12 514868
    ## - data_channel_is_socmed         1 1.1339e+09 3.1638e+12 514869
    ## - global_subjectivity            1 1.4082e+09 3.1641e+12 514871
    ## - data_channel_is_lifestyle      1 1.5282e+09 3.1642e+12 514872
    ## - data_channel_is_bus            1 1.9483e+09 3.1646e+12 514876
    ## - kw_min_avg                     1 2.1595e+09 3.1649e+12 514878
    ## - data_channel_is_entertainment  1 3.5002e+09 3.1662e+12 514890
    ## - num_hrefs                      1 3.9404e+09 3.1666e+12 514893
    ## - kw_max_avg                     1 5.6805e+09 3.1684e+12 514909
    ## - kw_avg_avg                     1 1.1784e+10 3.1745e+12 514962
    ## 
    ## Step:  AIC=514860.8
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - LDA_03                         1 9.0317e+07 3.1630e+12 514860
    ## <none>                                        3.1629e+12 514861
    ## - kw_max_min                     1 3.1376e+08 3.1632e+12 514862
    ## - kw_avg_min                     1 3.6308e+08 3.1633e+12 514862
    ## - weekday_is_thursday            1 4.2646e+08 3.1633e+12 514863
    ## - min_positive_polarity          1 4.3563e+08 3.1634e+12 514863
    ## - kw_avg_max                     1 5.7059e+08 3.1635e+12 514864
    ## - self_reference_max_shares      1 6.0270e+08 3.1635e+12 514864
    ## - average_token_length           1 6.6753e+08 3.1636e+12 514865
    ## - LDA_02                         1 6.7667e+08 3.1636e+12 514865
    ## - LDA_01                         1 7.1839e+08 3.1636e+12 514865
    ## - self_reference_min_shares      1 7.5417e+08 3.1637e+12 514865
    ## - kw_min_min                     1 7.6976e+08 3.1637e+12 514866
    ## - num_self_hrefs                 1 8.1469e+08 3.1637e+12 514866
    ## - n_tokens_title                 1 8.8116e+08 3.1638e+12 514867
    ## - data_channel_is_world          1 8.8510e+08 3.1638e+12 514867
    ## - abs_title_subjectivity         1 9.2736e+08 3.1638e+12 514867
    ## - data_channel_is_socmed         1 9.9697e+08 3.1639e+12 514868
    ## - abs_title_sentiment_polarity   1 1.0368e+09 3.1640e+12 514868
    ## - global_subjectivity            1 1.3913e+09 3.1643e+12 514871
    ## - data_channel_is_tech           1 1.5817e+09 3.1645e+12 514873
    ## - data_channel_is_bus            1 1.7337e+09 3.1647e+12 514874
    ## - data_channel_is_lifestyle      1 1.7532e+09 3.1647e+12 514874
    ## - kw_min_avg                     1 2.2115e+09 3.1651e+12 514878
    ## - data_channel_is_entertainment  1 3.4608e+09 3.1664e+12 514889
    ## - num_hrefs                      1 3.9975e+09 3.1669e+12 514894
    ## - kw_max_avg                     1 5.7018e+09 3.1686e+12 514909
    ## - kw_avg_avg                     1 1.1864e+10 3.1748e+12 514963
    ## 
    ## Step:  AIC=514859.6
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + global_subjectivity + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_thursday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## <none>                                        3.1630e+12 514860
    ## - kw_max_min                     1 3.0583e+08 3.1633e+12 514860
    ## - kw_avg_min                     1 3.5171e+08 3.1634e+12 514861
    ## - weekday_is_thursday            1 4.2518e+08 3.1634e+12 514861
    ## - min_positive_polarity          1 4.6680e+08 3.1635e+12 514862
    ## - LDA_02                         1 5.8750e+08 3.1636e+12 514863
    ## - self_reference_max_shares      1 6.0248e+08 3.1636e+12 514863
    ## - kw_avg_max                     1 6.1365e+08 3.1636e+12 514863
    ## - average_token_length           1 6.3693e+08 3.1636e+12 514863
    ## - kw_min_min                     1 7.2597e+08 3.1637e+12 514864
    ## - self_reference_min_shares      1 7.5702e+08 3.1638e+12 514864
    ## - LDA_01                         1 7.6249e+08 3.1638e+12 514864
    ## - num_self_hrefs                 1 7.9962e+08 3.1638e+12 514865
    ## - data_channel_is_world          1 8.1111e+08 3.1638e+12 514865
    ## - n_tokens_title                 1 8.7146e+08 3.1639e+12 514865
    ## - abs_title_subjectivity         1 9.1110e+08 3.1639e+12 514866
    ## - data_channel_is_socmed         1 9.6534e+08 3.1640e+12 514866
    ## - abs_title_sentiment_polarity   1 1.0134e+09 3.1640e+12 514866
    ## - global_subjectivity            1 1.3645e+09 3.1644e+12 514870
    ## - data_channel_is_lifestyle      1 1.9953e+09 3.1650e+12 514875
    ## - kw_min_avg                     1 2.1482e+09 3.1652e+12 514876
    ## - data_channel_is_tech           1 2.1658e+09 3.1652e+12 514877
    ## - data_channel_is_bus            1 2.4365e+09 3.1654e+12 514879
    ## - data_channel_is_entertainment  1 3.3988e+09 3.1664e+12 514887
    ## - num_hrefs                      1 3.9440e+09 3.1670e+12 514892
    ## - kw_max_avg                     1 5.6203e+09 3.1686e+12 514907
    ## - kw_avg_avg                     1 1.1797e+10 3.1748e+12 514961

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
    ##     weekday_is_thursday, data = train)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -24546  -2142  -1204   -129 836762 
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                    2.840e+02  7.030e+02   0.404 0.686206    
    ## n_tokens_title                 8.703e+01  3.149e+01   2.764 0.005718 ** 
    ## num_hrefs                      3.814e+01  6.486e+00   5.879 4.16e-09 ***
    ## num_self_hrefs                -4.969e+01  1.877e+01  -2.647 0.008117 ** 
    ## average_token_length          -2.481e+02  1.050e+02  -2.363 0.018147 *  
    ## data_channel_is_lifestyle     -1.447e+03  3.461e+02  -4.182 2.90e-05 ***
    ## data_channel_is_entertainment -1.487e+03  2.724e+02  -5.458 4.86e-08 ***
    ## data_channel_is_bus           -1.279e+03  2.767e+02  -4.621 3.83e-06 ***
    ## data_channel_is_socmed        -9.927e+02  3.413e+02  -2.909 0.003631 ** 
    ## data_channel_is_tech          -1.195e+03  2.742e+02  -4.357 1.32e-05 ***
    ## data_channel_is_world         -9.778e+02  3.667e+02  -2.666 0.007674 ** 
    ## kw_min_min                     2.895e+00  1.148e+00   2.522 0.011658 *  
    ## kw_max_min                     8.203e-02  5.010e-02   1.637 0.101593    
    ## kw_avg_min                    -5.366e-01  3.056e-01  -1.756 0.079143 .  
    ## kw_avg_max                    -1.712e-03  7.383e-04  -2.319 0.020394 *  
    ## kw_min_avg                    -3.413e-01  7.865e-02  -4.339 1.44e-05 ***
    ## kw_max_avg                    -1.907e-01  2.717e-02  -7.019 2.29e-12 ***
    ## kw_avg_avg                     1.542e+00  1.517e-01  10.169  < 2e-16 ***
    ## self_reference_min_shares      9.854e-03  3.826e-03   2.576 0.010004 *  
    ## self_reference_max_shares      4.206e-03  1.830e-03   2.298 0.021572 *  
    ## LDA_01                        -9.753e+02  3.773e+02  -2.585 0.009738 ** 
    ## LDA_02                        -9.883e+02  4.355e+02  -2.269 0.023264 *  
    ## global_subjectivity            2.586e+03  7.478e+02   3.458 0.000544 ***
    ## min_positive_polarity         -1.961e+03  9.694e+02  -2.023 0.043112 *  
    ## abs_title_subjectivity         1.061e+03  3.756e+02   2.826 0.004719 ** 
    ## abs_title_sentiment_polarity   9.324e+02  3.129e+02   2.980 0.002882 ** 
    ## weekday_is_thursday           -3.197e+02  1.656e+02  -1.930 0.053563 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10680 on 27723 degrees of freedom
    ## Multiple R-squared:  0.0238, Adjusted R-squared:  0.02289 
    ## F-statistic:    26 on 26 and 27723 DF,  p-value: < 2.2e-16

Calculate the predicted mean square error on the train set:

``` r
train.pred <- predict(lm.step, train)
mean((train.pred - train$shares)^2)
```

    ## [1] 113982234

### On test set

Calculate the predicted mean square error on the test set:

``` r
test.pred <- predict(lm.step, test)
mean((test.pred - test$shares)^2)
```

    ## [1] 175204418

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
