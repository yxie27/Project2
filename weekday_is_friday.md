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

![](weekday_is_friday_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
hist(log(train$shares))
```

![](weekday_is_friday_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

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
    ## weekday_is_friday             Min.   :0.0000     1st Qu.:0.0000     Median :0.0000     Mean   :0.1435    
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
    ## weekday_is_friday             3rd Qu.:0.0000     Max.   :1.0000

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
    ##           Mean of squared residuals: 117183208
    ##                     % Var explained: -0.36

``` r
#variable importance measures
importance(rf)
```

    ##                                  %IncMSE IncNodePurity
    ## n_tokens_title                 0.5443122   40460826440
    ## n_tokens_content               3.4013595   46108779663
    ## n_unique_tokens                2.9201362   42790846921
    ## n_non_stop_words               8.1679312   34711463990
    ## n_non_stop_unique_tokens       8.0381118   59041608629
    ## num_hrefs                      4.8607273   63518549651
    ## num_self_hrefs                 3.2028930   41133684231
    ## num_imgs                       3.3246410   39595559140
    ## num_videos                     3.6570462   48832227310
    ## average_token_length           4.2945304   57231765339
    ## num_keywords                   6.5823787   18832710106
    ## data_channel_is_lifestyle      2.3032596    3953511061
    ## data_channel_is_entertainment  4.6907971    3778914989
    ## data_channel_is_bus            3.1185236   21346113039
    ## data_channel_is_socmed         5.1744413    2806032852
    ## data_channel_is_tech           6.8486322    1641863741
    ## data_channel_is_world          5.1893500    2849616567
    ## kw_min_min                     2.9872268   16970395062
    ## kw_max_min                     3.5759264   83585242503
    ## kw_avg_min                     5.1631204  122243928978
    ## kw_min_max                     4.0622971   49541960274
    ## kw_max_max                     4.0370673   56002958093
    ## kw_avg_max                     5.2983852  123838060171
    ## kw_min_avg                     4.2296614   66414926817
    ## kw_max_avg                     4.4991086  213351108881
    ## kw_avg_avg                     3.3512625  184874378278
    ## self_reference_min_shares      6.7008014   58279696411
    ## self_reference_max_shares      6.3570578   63353390507
    ## self_reference_avg_sharess     4.3749556  119294762553
    ## LDA_00                         3.8940435   94835053711
    ## LDA_01                         7.6560929   47216220381
    ## LDA_02                         4.8449573   80380485639
    ## LDA_03                         4.1670989  100685654187
    ## LDA_04                         5.4005064   83016351544
    ## global_subjectivity            5.4479990   82044417317
    ## global_sentiment_polarity      4.1792144   51416194598
    ## global_rate_positive_words     3.2621049   69752551988
    ## global_rate_negative_words     2.6067755   46468155518
    ## rate_positive_words            3.0509865   29908697486
    ## rate_negative_words            2.6083523   35720193493
    ## avg_positive_polarity          4.6072121   51646542565
    ## min_positive_polarity          0.5134083   15563161903
    ## max_positive_polarity          8.7629572   10562051485
    ## avg_negative_polarity          4.0711855   66098604441
    ## min_negative_polarity          3.0052790   34516895765
    ## max_negative_polarity          3.5056925   34036152729
    ## title_subjectivity             2.5110363   57610516498
    ## title_sentiment_polarity       3.8491167   71389573103
    ## abs_title_subjectivity         4.2915970   15243237397
    ## abs_title_sentiment_polarity   1.4034625   31500803104
    ## weekday_is_friday             -1.2477644    5119187485

``` r
#draw dotplot of variable importance as measured by Random Forest
varImpPlot(rf)
```

![](weekday_is_friday_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Calculate the predicted mean square error on the train set:

``` r
train.pred <- predict(rf, train)
mean((train.pred - train$shares)^2)
```

    ## [1] 25592277

### On test set

``` r
rf.test <- predict(rf, newdata = test)
mean((test$shares-rf.test)^2)
```

    ## [1] 177273854

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

    ## Start:  AIC=514900.6
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
    ##     weekday_is_friday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - global_sentiment_polarity      1 3.1150e+05 3.1620e+12 514899
    ## - title_subjectivity             1 5.8214e+05 3.1620e+12 514899
    ## - global_rate_negative_words     1 2.1296e+06 3.1620e+12 514899
    ## - n_unique_tokens                1 2.1943e+06 3.1620e+12 514899
    ## - LDA_00                         1 6.2484e+06 3.1620e+12 514899
    ## - LDA_04                         1 6.2540e+06 3.1620e+12 514899
    ## - LDA_03                         1 6.2562e+06 3.1620e+12 514899
    ## - LDA_02                         1 6.2612e+06 3.1620e+12 514899
    ## - LDA_01                         1 6.2622e+06 3.1620e+12 514899
    ## - num_keywords                   1 6.3046e+06 3.1620e+12 514899
    ## - avg_negative_polarity          1 6.5003e+06 3.1620e+12 514899
    ## - max_positive_polarity          1 6.9980e+06 3.1620e+12 514899
    ## - rate_positive_words            1 9.1720e+06 3.1620e+12 514899
    ## - rate_negative_words            1 1.0145e+07 3.1620e+12 514899
    ## - n_non_stop_words               1 1.3973e+07 3.1620e+12 514899
    ## - n_tokens_content               1 1.4501e+07 3.1620e+12 514899
    ## - avg_positive_polarity          1 2.1379e+07 3.1620e+12 514899
    ## - min_negative_polarity          1 2.4412e+07 3.1620e+12 514899
    ## - weekday_is_friday              1 3.3207e+07 3.1620e+12 514899
    ## - title_sentiment_polarity       1 3.6005e+07 3.1620e+12 514899
    ## - self_reference_avg_sharess     1 4.6367e+07 3.1620e+12 514899
    ## - max_negative_polarity          1 4.8595e+07 3.1620e+12 514899
    ## - global_rate_positive_words     1 5.4092e+07 3.1620e+12 514899
    ## - n_non_stop_unique_tokens       1 8.0027e+07 3.1621e+12 514899
    ## - kw_min_min                     1 9.0082e+07 3.1621e+12 514899
    ## - kw_min_max                     1 9.1088e+07 3.1621e+12 514899
    ## - kw_max_max                     1 1.0617e+08 3.1621e+12 514900
    ## - average_token_length           1 1.6090e+08 3.1621e+12 514900
    ## - num_imgs                       1 1.7579e+08 3.1622e+12 514900
    ## <none>                                        3.1620e+12 514901
    ## - self_reference_max_shares      1 2.3535e+08 3.1622e+12 514901
    ## - min_positive_polarity          1 2.4260e+08 3.1622e+12 514901
    ## - num_videos                     1 2.4449e+08 3.1622e+12 514901
    ## - kw_avg_max                     1 2.9998e+08 3.1623e+12 514901
    ## - kw_max_min                     1 3.1885e+08 3.1623e+12 514901
    ## - self_reference_min_shares      1 3.2902e+08 3.1623e+12 514901
    ## - kw_avg_min                     1 3.7897e+08 3.1624e+12 514902
    ## - abs_title_sentiment_polarity   1 4.9638e+08 3.1625e+12 514903
    ## - abs_title_subjectivity         1 7.3602e+08 3.1627e+12 514905
    ## - data_channel_is_world          1 7.7394e+08 3.1628e+12 514905
    ## - n_tokens_title                 1 8.6838e+08 3.1628e+12 514906
    ## - num_self_hrefs                 1 9.0257e+08 3.1629e+12 514907
    ## - global_subjectivity            1 9.8819e+08 3.1630e+12 514907
    ## - data_channel_is_socmed         1 1.0042e+09 3.1630e+12 514907
    ## - data_channel_is_tech           1 1.0068e+09 3.1630e+12 514907
    ## - data_channel_is_lifestyle      1 1.4600e+09 3.1634e+12 514911
    ## - kw_min_avg                     1 1.8741e+09 3.1639e+12 514915
    ## - data_channel_is_bus            1 1.8943e+09 3.1639e+12 514915
    ## - num_hrefs                      1 2.9862e+09 3.1650e+12 514925
    ## - data_channel_is_entertainment  1 3.3137e+09 3.1653e+12 514928
    ## - kw_max_avg                     1 5.6358e+09 3.1676e+12 514948
    ## - kw_avg_avg                     1 1.1445e+10 3.1734e+12 514999
    ## 
    ## Step:  AIC=514898.6
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
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_friday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - title_subjectivity             1 6.0192e+05 3.1620e+12 514897
    ## - n_unique_tokens                1 2.1976e+06 3.1620e+12 514897
    ## - global_rate_negative_words     1 2.9882e+06 3.1620e+12 514897
    ## - LDA_00                         1 6.2606e+06 3.1620e+12 514897
    ## - LDA_04                         1 6.2662e+06 3.1620e+12 514897
    ## - LDA_03                         1 6.2684e+06 3.1620e+12 514897
    ## - LDA_02                         1 6.2734e+06 3.1620e+12 514897
    ## - LDA_01                         1 6.2744e+06 3.1620e+12 514897
    ## - num_keywords                   1 6.2858e+06 3.1620e+12 514897
    ## - max_positive_polarity          1 7.0348e+06 3.1620e+12 514897
    ## - avg_negative_polarity          1 8.4119e+06 3.1620e+12 514897
    ## - rate_positive_words            1 9.1841e+06 3.1620e+12 514897
    ## - rate_negative_words            1 1.0000e+07 3.1620e+12 514897
    ## - n_non_stop_words               1 1.3975e+07 3.1620e+12 514897
    ## - n_tokens_content               1 1.4361e+07 3.1620e+12 514897
    ## - min_negative_polarity          1 2.4562e+07 3.1620e+12 514897
    ## - avg_positive_polarity          1 2.8435e+07 3.1620e+12 514897
    ## - weekday_is_friday              1 3.3248e+07 3.1620e+12 514897
    ## - title_sentiment_polarity       1 3.6835e+07 3.1620e+12 514897
    ## - self_reference_avg_sharess     1 4.6346e+07 3.1620e+12 514897
    ## - max_negative_polarity          1 5.1241e+07 3.1620e+12 514897
    ## - global_rate_positive_words     1 6.1229e+07 3.1620e+12 514897
    ## - n_non_stop_unique_tokens       1 7.9859e+07 3.1621e+12 514897
    ## - kw_min_min                     1 9.0027e+07 3.1621e+12 514897
    ## - kw_min_max                     1 9.1220e+07 3.1621e+12 514897
    ## - kw_max_max                     1 1.0622e+08 3.1621e+12 514898
    ## - average_token_length           1 1.6115e+08 3.1621e+12 514898
    ## - num_imgs                       1 1.7589e+08 3.1622e+12 514898
    ## <none>                                        3.1620e+12 514899
    ## - self_reference_max_shares      1 2.3526e+08 3.1622e+12 514899
    ## - num_videos                     1 2.4432e+08 3.1622e+12 514899
    ## - min_positive_polarity          1 2.4904e+08 3.1622e+12 514899
    ## - kw_avg_max                     1 2.9987e+08 3.1623e+12 514899
    ## - kw_max_min                     1 3.1870e+08 3.1623e+12 514899
    ## - self_reference_min_shares      1 3.2902e+08 3.1623e+12 514899
    ## - kw_avg_min                     1 3.7880e+08 3.1624e+12 514900
    ## - abs_title_sentiment_polarity   1 4.9643e+08 3.1625e+12 514901
    ## - abs_title_subjectivity         1 7.3709e+08 3.1627e+12 514903
    ## - data_channel_is_world          1 7.7364e+08 3.1628e+12 514903
    ## - n_tokens_title                 1 8.6843e+08 3.1628e+12 514904
    ## - num_self_hrefs                 1 9.0261e+08 3.1629e+12 514905
    ## - data_channel_is_socmed         1 1.0041e+09 3.1630e+12 514905
    ## - data_channel_is_tech           1 1.0066e+09 3.1630e+12 514905
    ## - global_subjectivity            1 1.0315e+09 3.1630e+12 514906
    ## - data_channel_is_lifestyle      1 1.4597e+09 3.1634e+12 514909
    ## - kw_min_avg                     1 1.8738e+09 3.1639e+12 514913
    ## - data_channel_is_bus            1 1.8943e+09 3.1639e+12 514913
    ## - num_hrefs                      1 2.9950e+09 3.1650e+12 514923
    ## - data_channel_is_entertainment  1 3.3136e+09 3.1653e+12 514926
    ## - kw_max_avg                     1 5.6364e+09 3.1676e+12 514946
    ## - kw_avg_avg                     1 1.1447e+10 3.1734e+12 514997
    ## 
    ## Step:  AIC=514896.6
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
    ##     weekday_is_friday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_unique_tokens                1 2.2145e+06 3.1620e+12 514895
    ## - global_rate_negative_words     1 3.0012e+06 3.1620e+12 514895
    ## - LDA_00                         1 6.2250e+06 3.1620e+12 514895
    ## - LDA_04                         1 6.2306e+06 3.1620e+12 514895
    ## - LDA_03                         1 6.2328e+06 3.1620e+12 514895
    ## - LDA_02                         1 6.2379e+06 3.1620e+12 514895
    ## - LDA_01                         1 6.2388e+06 3.1620e+12 514895
    ## - num_keywords                   1 6.2515e+06 3.1620e+12 514895
    ## - max_positive_polarity          1 7.0090e+06 3.1620e+12 514895
    ## - avg_negative_polarity          1 8.3922e+06 3.1620e+12 514895
    ## - rate_positive_words            1 9.1728e+06 3.1620e+12 514895
    ## - rate_negative_words            1 9.9772e+06 3.1620e+12 514895
    ## - n_non_stop_words               1 1.3915e+07 3.1620e+12 514895
    ## - n_tokens_content               1 1.4268e+07 3.1620e+12 514895
    ## - min_negative_polarity          1 2.4628e+07 3.1620e+12 514895
    ## - avg_positive_polarity          1 2.8127e+07 3.1620e+12 514895
    ## - weekday_is_friday              1 3.3239e+07 3.1620e+12 514895
    ## - title_sentiment_polarity       1 3.8460e+07 3.1620e+12 514895
    ## - self_reference_avg_sharess     1 4.6368e+07 3.1620e+12 514895
    ## - max_negative_polarity          1 5.1241e+07 3.1620e+12 514895
    ## - global_rate_positive_words     1 6.1494e+07 3.1620e+12 514895
    ## - n_non_stop_unique_tokens       1 7.9886e+07 3.1621e+12 514895
    ## - kw_min_min                     1 9.0139e+07 3.1621e+12 514895
    ## - kw_min_max                     1 9.1281e+07 3.1621e+12 514895
    ## - kw_max_max                     1 1.0605e+08 3.1621e+12 514896
    ## - average_token_length           1 1.6118e+08 3.1621e+12 514896
    ## - num_imgs                       1 1.7571e+08 3.1622e+12 514896
    ## <none>                                        3.1620e+12 514897
    ## - self_reference_max_shares      1 2.3526e+08 3.1622e+12 514897
    ## - num_videos                     1 2.4422e+08 3.1622e+12 514897
    ## - min_positive_polarity          1 2.4944e+08 3.1622e+12 514897
    ## - kw_avg_max                     1 3.0002e+08 3.1623e+12 514897
    ## - kw_max_min                     1 3.1889e+08 3.1623e+12 514897
    ## - self_reference_min_shares      1 3.2917e+08 3.1623e+12 514897
    ## - kw_avg_min                     1 3.7912e+08 3.1624e+12 514898
    ## - data_channel_is_world          1 7.7360e+08 3.1628e+12 514901
    ## - abs_title_sentiment_polarity   1 8.0621e+08 3.1628e+12 514902
    ## - abs_title_subjectivity         1 8.3161e+08 3.1628e+12 514902
    ## - n_tokens_title                 1 8.6785e+08 3.1628e+12 514902
    ## - num_self_hrefs                 1 9.0277e+08 3.1629e+12 514903
    ## - data_channel_is_socmed         1 1.0037e+09 3.1630e+12 514903
    ## - data_channel_is_tech           1 1.0067e+09 3.1630e+12 514903
    ## - global_subjectivity            1 1.0390e+09 3.1630e+12 514904
    ## - data_channel_is_lifestyle      1 1.4596e+09 3.1634e+12 514907
    ## - kw_min_avg                     1 1.8733e+09 3.1639e+12 514911
    ## - data_channel_is_bus            1 1.8940e+09 3.1639e+12 514911
    ## - num_hrefs                      1 2.9948e+09 3.1650e+12 514921
    ## - data_channel_is_entertainment  1 3.3133e+09 3.1653e+12 514924
    ## - kw_max_avg                     1 5.6359e+09 3.1676e+12 514944
    ## - kw_avg_avg                     1 1.1446e+10 3.1734e+12 514995
    ## 
    ## Step:  AIC=514894.6
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
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_friday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - global_rate_negative_words     1 3.1527e+06 3.1620e+12 514893
    ## - LDA_00                         1 5.6692e+06 3.1620e+12 514893
    ## - LDA_04                         1 5.6745e+06 3.1620e+12 514893
    ## - LDA_03                         1 5.6767e+06 3.1620e+12 514893
    ## - LDA_02                         1 5.6815e+06 3.1620e+12 514893
    ## - LDA_01                         1 5.6824e+06 3.1620e+12 514893
    ## - num_keywords                   1 6.3475e+06 3.1620e+12 514893
    ## - avg_negative_polarity          1 8.3904e+06 3.1620e+12 514893
    ## - max_positive_polarity          1 8.4056e+06 3.1620e+12 514893
    ## - rate_positive_words            1 9.3416e+06 3.1620e+12 514893
    ## - rate_negative_words            1 1.0155e+07 3.1620e+12 514893
    ## - n_non_stop_words               1 1.3797e+07 3.1620e+12 514893
    ## - n_tokens_content               1 2.5063e+07 3.1620e+12 514893
    ## - min_negative_polarity          1 2.6792e+07 3.1620e+12 514893
    ## - avg_positive_polarity          1 2.8240e+07 3.1620e+12 514893
    ## - weekday_is_friday              1 3.3035e+07 3.1620e+12 514893
    ## - title_sentiment_polarity       1 3.8825e+07 3.1620e+12 514893
    ## - self_reference_avg_sharess     1 4.6669e+07 3.1620e+12 514893
    ## - max_negative_polarity          1 4.9506e+07 3.1620e+12 514893
    ## - global_rate_positive_words     1 6.3440e+07 3.1620e+12 514893
    ## - kw_min_min                     1 9.0102e+07 3.1621e+12 514893
    ## - kw_min_max                     1 9.0992e+07 3.1621e+12 514893
    ## - kw_max_max                     1 1.0532e+08 3.1621e+12 514894
    ## - num_imgs                       1 1.7460e+08 3.1622e+12 514894
    ## - average_token_length           1 1.8611e+08 3.1622e+12 514894
    ## <none>                                        3.1620e+12 514895
    ## - self_reference_max_shares      1 2.3604e+08 3.1622e+12 514895
    ## - n_non_stop_unique_tokens       1 2.3876e+08 3.1622e+12 514895
    ## - num_videos                     1 2.4205e+08 3.1622e+12 514895
    ## - min_positive_polarity          1 2.6544e+08 3.1622e+12 514895
    ## - kw_avg_max                     1 3.0201e+08 3.1623e+12 514895
    ## - kw_max_min                     1 3.1881e+08 3.1623e+12 514895
    ## - self_reference_min_shares      1 3.2981e+08 3.1623e+12 514896
    ## - kw_avg_min                     1 3.7884e+08 3.1624e+12 514896
    ## - data_channel_is_world          1 7.7244e+08 3.1628e+12 514899
    ## - abs_title_sentiment_polarity   1 8.0465e+08 3.1628e+12 514900
    ## - abs_title_subjectivity         1 8.3367e+08 3.1628e+12 514900
    ## - n_tokens_title                 1 8.6782e+08 3.1629e+12 514900
    ## - num_self_hrefs                 1 9.0383e+08 3.1629e+12 514901
    ## - data_channel_is_socmed         1 1.0015e+09 3.1630e+12 514901
    ## - data_channel_is_tech           1 1.0050e+09 3.1630e+12 514901
    ## - global_subjectivity            1 1.0379e+09 3.1630e+12 514902
    ## - data_channel_is_lifestyle      1 1.4602e+09 3.1634e+12 514905
    ## - kw_min_avg                     1 1.8769e+09 3.1639e+12 514909
    ## - data_channel_is_bus            1 1.8936e+09 3.1639e+12 514909
    ## - num_hrefs                      1 3.0089e+09 3.1650e+12 514919
    ## - data_channel_is_entertainment  1 3.3146e+09 3.1653e+12 514922
    ## - kw_max_avg                     1 5.6402e+09 3.1676e+12 514942
    ## - kw_avg_avg                     1 1.1450e+10 3.1734e+12 514993
    ## 
    ## Step:  AIC=514892.6
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
    ##     weekday_is_friday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - LDA_00                         1 5.7010e+06 3.1620e+12 514891
    ## - LDA_04                         1 5.7063e+06 3.1620e+12 514891
    ## - LDA_03                         1 5.7085e+06 3.1620e+12 514891
    ## - LDA_02                         1 5.7133e+06 3.1620e+12 514891
    ## - LDA_01                         1 5.7142e+06 3.1620e+12 514891
    ## - num_keywords                   1 6.4276e+06 3.1620e+12 514891
    ## - max_positive_polarity          1 8.0434e+06 3.1620e+12 514891
    ## - avg_negative_polarity          1 8.2776e+06 3.1620e+12 514891
    ## - rate_negative_words            1 8.9455e+06 3.1620e+12 514891
    ## - rate_positive_words            1 9.8260e+06 3.1620e+12 514891
    ## - n_non_stop_words               1 1.3828e+07 3.1620e+12 514891
    ## - n_tokens_content               1 2.5569e+07 3.1620e+12 514891
    ## - min_negative_polarity          1 2.6336e+07 3.1620e+12 514891
    ## - avg_positive_polarity          1 2.8458e+07 3.1620e+12 514891
    ## - weekday_is_friday              1 3.3150e+07 3.1620e+12 514891
    ## - title_sentiment_polarity       1 4.0583e+07 3.1620e+12 514891
    ## - self_reference_avg_sharess     1 4.6788e+07 3.1620e+12 514891
    ## - max_negative_polarity          1 5.0866e+07 3.1620e+12 514891
    ## - kw_min_min                     1 8.9562e+07 3.1621e+12 514891
    ## - kw_min_max                     1 9.1058e+07 3.1621e+12 514891
    ## - kw_max_max                     1 1.0582e+08 3.1621e+12 514892
    ## - global_rate_positive_words     1 1.3947e+08 3.1621e+12 514892
    ## - num_imgs                       1 1.7336e+08 3.1622e+12 514892
    ## - average_token_length           1 1.8446e+08 3.1622e+12 514892
    ## <none>                                        3.1620e+12 514893
    ## - self_reference_max_shares      1 2.3653e+08 3.1622e+12 514893
    ## - n_non_stop_unique_tokens       1 2.3797e+08 3.1622e+12 514893
    ## - num_videos                     1 2.3893e+08 3.1622e+12 514893
    ## - min_positive_polarity          1 2.6287e+08 3.1622e+12 514893
    ## - kw_avg_max                     1 3.0241e+08 3.1623e+12 514893
    ## - kw_max_min                     1 3.1882e+08 3.1623e+12 514893
    ## - self_reference_min_shares      1 3.2967e+08 3.1623e+12 514894
    ## - kw_avg_min                     1 3.7926e+08 3.1624e+12 514894
    ## - data_channel_is_world          1 7.6944e+08 3.1628e+12 514897
    ## - abs_title_sentiment_polarity   1 8.0294e+08 3.1628e+12 514898
    ## - abs_title_subjectivity         1 8.3877e+08 3.1628e+12 514898
    ## - n_tokens_title                 1 8.6991e+08 3.1629e+12 514898
    ## - num_self_hrefs                 1 9.0339e+08 3.1629e+12 514899
    ## - data_channel_is_socmed         1 9.9913e+08 3.1630e+12 514899
    ## - data_channel_is_tech           1 1.0025e+09 3.1630e+12 514899
    ## - global_subjectivity            1 1.0368e+09 3.1630e+12 514900
    ## - data_channel_is_lifestyle      1 1.4574e+09 3.1634e+12 514903
    ## - kw_min_avg                     1 1.8766e+09 3.1639e+12 514907
    ## - data_channel_is_bus            1 1.8905e+09 3.1639e+12 514907
    ## - num_hrefs                      1 3.0367e+09 3.1650e+12 514917
    ## - data_channel_is_entertainment  1 3.3126e+09 3.1653e+12 514920
    ## - kw_max_avg                     1 5.6383e+09 3.1676e+12 514940
    ## - kw_avg_avg                     1 1.1449e+10 3.1734e+12 514991
    ## 
    ## Step:  AIC=514890.7
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
    ##     weekday_is_friday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - num_keywords                   1 6.2794e+06 3.1620e+12 514889
    ## - max_positive_polarity          1 7.8454e+06 3.1620e+12 514889
    ## - avg_negative_polarity          1 8.2250e+06 3.1620e+12 514889
    ## - rate_negative_words            1 8.4823e+06 3.1620e+12 514889
    ## - rate_positive_words            1 1.2481e+07 3.1620e+12 514889
    ## - n_tokens_content               1 2.4605e+07 3.1620e+12 514889
    ## - min_negative_polarity          1 2.6294e+07 3.1620e+12 514889
    ## - avg_positive_polarity          1 2.8067e+07 3.1620e+12 514889
    ## - weekday_is_friday              1 3.3054e+07 3.1620e+12 514889
    ## - title_sentiment_polarity       1 4.0458e+07 3.1620e+12 514889
    ## - self_reference_avg_sharess     1 4.7047e+07 3.1620e+12 514889
    ## - max_negative_polarity          1 5.0910e+07 3.1620e+12 514889
    ## - kw_min_min                     1 8.9659e+07 3.1621e+12 514889
    ## - kw_min_max                     1 9.0823e+07 3.1621e+12 514889
    ## - kw_max_max                     1 1.0586e+08 3.1621e+12 514890
    ## - global_rate_positive_words     1 1.3935e+08 3.1621e+12 514890
    ## - num_imgs                       1 1.7394e+08 3.1622e+12 514890
    ## - average_token_length           1 2.0389e+08 3.1622e+12 514890
    ## - LDA_04                         1 2.1310e+08 3.1622e+12 514891
    ## <none>                                        3.1620e+12 514891
    ## - n_non_stop_words               1 2.3248e+08 3.1622e+12 514891
    ## - n_non_stop_unique_tokens       1 2.3529e+08 3.1622e+12 514891
    ## - self_reference_max_shares      1 2.3693e+08 3.1622e+12 514891
    ## - num_videos                     1 2.3873e+08 3.1622e+12 514891
    ## - min_positive_polarity          1 2.6329e+08 3.1623e+12 514891
    ## - kw_avg_max                     1 3.0199e+08 3.1623e+12 514891
    ## - kw_max_min                     1 3.1928e+08 3.1623e+12 514891
    ## - self_reference_min_shares      1 3.3029e+08 3.1623e+12 514892
    ## - LDA_03                         1 3.4320e+08 3.1623e+12 514892
    ## - kw_avg_min                     1 3.7977e+08 3.1624e+12 514892
    ## - data_channel_is_world          1 7.6900e+08 3.1628e+12 514895
    ## - abs_title_sentiment_polarity   1 8.0371e+08 3.1628e+12 514896
    ## - abs_title_subjectivity         1 8.3811e+08 3.1628e+12 514896
    ## - n_tokens_title                 1 8.6977e+08 3.1629e+12 514896
    ## - num_self_hrefs                 1 9.0660e+08 3.1629e+12 514897
    ## - LDA_02                         1 9.4091e+08 3.1629e+12 514897
    ## - LDA_01                         1 9.6881e+08 3.1630e+12 514897
    ## - data_channel_is_socmed         1 1.0030e+09 3.1630e+12 514897
    ## - data_channel_is_tech           1 1.0047e+09 3.1630e+12 514898
    ## - global_subjectivity            1 1.0315e+09 3.1630e+12 514898
    ## - data_channel_is_lifestyle      1 1.4599e+09 3.1635e+12 514902
    ## - kw_min_avg                     1 1.8798e+09 3.1639e+12 514905
    ## - data_channel_is_bus            1 1.8942e+09 3.1639e+12 514905
    ## - num_hrefs                      1 3.0540e+09 3.1650e+12 514915
    ## - data_channel_is_entertainment  1 3.3169e+09 3.1653e+12 514918
    ## - kw_max_avg                     1 5.6421e+09 3.1676e+12 514938
    ## - kw_avg_avg                     1 1.1457e+10 3.1734e+12 514989
    ## 
    ## Step:  AIC=514888.8
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
    ##     weekday_is_friday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - max_positive_polarity          1 7.7085e+06 3.1620e+12 514887
    ## - avg_negative_polarity          1 8.2350e+06 3.1620e+12 514887
    ## - rate_negative_words            1 9.5422e+06 3.1620e+12 514887
    ## - rate_positive_words            1 1.3612e+07 3.1620e+12 514887
    ## - n_tokens_content               1 2.4291e+07 3.1620e+12 514887
    ## - min_negative_polarity          1 2.6401e+07 3.1620e+12 514887
    ## - avg_positive_polarity          1 2.8470e+07 3.1620e+12 514887
    ## - weekday_is_friday              1 3.3007e+07 3.1620e+12 514887
    ## - title_sentiment_polarity       1 4.0237e+07 3.1620e+12 514887
    ## - self_reference_avg_sharess     1 4.7606e+07 3.1620e+12 514887
    ## - max_negative_polarity          1 5.0810e+07 3.1620e+12 514887
    ## - kw_min_max                     1 8.9106e+07 3.1621e+12 514888
    ## - kw_min_min                     1 8.9915e+07 3.1621e+12 514888
    ## - kw_max_max                     1 1.1632e+08 3.1621e+12 514888
    ## - global_rate_positive_words     1 1.4163e+08 3.1621e+12 514888
    ## - num_imgs                       1 1.7443e+08 3.1622e+12 514888
    ## - average_token_length           1 2.0616e+08 3.1622e+12 514889
    ## - LDA_04                         1 2.1610e+08 3.1622e+12 514889
    ## <none>                                        3.1620e+12 514889
    ## - n_non_stop_words               1 2.3366e+08 3.1622e+12 514889
    ## - n_non_stop_unique_tokens       1 2.3648e+08 3.1622e+12 514889
    ## - num_videos                     1 2.3716e+08 3.1622e+12 514889
    ## - self_reference_max_shares      1 2.3785e+08 3.1622e+12 514889
    ## - min_positive_polarity          1 2.6271e+08 3.1623e+12 514889
    ## - kw_avg_max                     1 3.1175e+08 3.1623e+12 514889
    ## - kw_max_min                     1 3.1415e+08 3.1623e+12 514890
    ## - self_reference_min_shares      1 3.3198e+08 3.1623e+12 514890
    ## - LDA_03                         1 3.4551e+08 3.1623e+12 514890
    ## - kw_avg_min                     1 3.7382e+08 3.1624e+12 514890
    ## - data_channel_is_world          1 7.6572e+08 3.1628e+12 514893
    ## - abs_title_sentiment_polarity   1 8.0265e+08 3.1628e+12 514894
    ## - abs_title_subjectivity         1 8.3787e+08 3.1628e+12 514894
    ## - n_tokens_title                 1 8.6541e+08 3.1629e+12 514894
    ## - num_self_hrefs                 1 9.1957e+08 3.1629e+12 514895
    ## - LDA_02                         1 9.3955e+08 3.1629e+12 514895
    ## - LDA_01                         1 9.6498e+08 3.1630e+12 514895
    ## - data_channel_is_socmed         1 9.9828e+08 3.1630e+12 514896
    ## - data_channel_is_tech           1 1.0012e+09 3.1630e+12 514896
    ## - global_subjectivity            1 1.0300e+09 3.1630e+12 514896
    ## - data_channel_is_lifestyle      1 1.4573e+09 3.1635e+12 514900
    ## - data_channel_is_bus            1 1.8882e+09 3.1639e+12 514903
    ## - kw_min_avg                     1 1.9256e+09 3.1639e+12 514904
    ## - num_hrefs                      1 3.0481e+09 3.1650e+12 514913
    ## - data_channel_is_entertainment  1 3.3257e+09 3.1653e+12 514916
    ## - kw_max_avg                     1 5.6418e+09 3.1676e+12 514936
    ## - kw_avg_avg                     1 1.1641e+10 3.1736e+12 514989
    ## 
    ## Step:  AIC=514886.8
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
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_friday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - avg_negative_polarity          1 7.9905e+06 3.1620e+12 514885
    ## - rate_negative_words            1 1.0921e+07 3.1620e+12 514885
    ## - rate_positive_words            1 1.6189e+07 3.1620e+12 514885
    ## - avg_positive_polarity          1 2.1327e+07 3.1620e+12 514885
    ## - min_negative_polarity          1 2.8155e+07 3.1620e+12 514885
    ## - n_tokens_content               1 2.9606e+07 3.1620e+12 514885
    ## - weekday_is_friday              1 3.3129e+07 3.1620e+12 514885
    ## - title_sentiment_polarity       1 3.9989e+07 3.1620e+12 514885
    ## - self_reference_avg_sharess     1 4.7837e+07 3.1621e+12 514885
    ## - max_negative_polarity          1 4.8891e+07 3.1621e+12 514885
    ## - kw_min_max                     1 8.9124e+07 3.1621e+12 514886
    ## - kw_min_min                     1 9.0024e+07 3.1621e+12 514886
    ## - kw_max_max                     1 1.1538e+08 3.1621e+12 514886
    ## - global_rate_positive_words     1 1.3456e+08 3.1621e+12 514886
    ## - num_imgs                       1 1.7435e+08 3.1622e+12 514886
    ## - average_token_length           1 2.1059e+08 3.1622e+12 514887
    ## - LDA_04                         1 2.1607e+08 3.1622e+12 514887
    ## <none>                                        3.1620e+12 514887
    ## - n_non_stop_words               1 2.2854e+08 3.1622e+12 514887
    ## - n_non_stop_unique_tokens       1 2.3133e+08 3.1622e+12 514887
    ## - self_reference_max_shares      1 2.3820e+08 3.1622e+12 514887
    ## - num_videos                     1 2.4030e+08 3.1622e+12 514887
    ## - kw_avg_max                     1 3.1203e+08 3.1623e+12 514888
    ## - kw_max_min                     1 3.1333e+08 3.1623e+12 514888
    ## - min_positive_polarity          1 3.2127e+08 3.1623e+12 514888
    ## - self_reference_min_shares      1 3.3226e+08 3.1623e+12 514888
    ## - LDA_03                         1 3.4592e+08 3.1624e+12 514888
    ## - kw_avg_min                     1 3.7233e+08 3.1624e+12 514888
    ## - data_channel_is_world          1 7.6730e+08 3.1628e+12 514892
    ## - abs_title_sentiment_polarity   1 8.0151e+08 3.1628e+12 514892
    ## - abs_title_subjectivity         1 8.3643e+08 3.1628e+12 514892
    ## - n_tokens_title                 1 8.6814e+08 3.1629e+12 514892
    ## - num_self_hrefs                 1 9.2123e+08 3.1629e+12 514893
    ## - LDA_02                         1 9.4148e+08 3.1629e+12 514893
    ## - LDA_01                         1 9.6677e+08 3.1630e+12 514893
    ## - data_channel_is_tech           1 1.0042e+09 3.1630e+12 514894
    ## - data_channel_is_socmed         1 1.0051e+09 3.1630e+12 514894
    ## - global_subjectivity            1 1.0231e+09 3.1630e+12 514894
    ## - data_channel_is_lifestyle      1 1.4599e+09 3.1635e+12 514898
    ## - data_channel_is_bus            1 1.8939e+09 3.1639e+12 514901
    ## - kw_min_avg                     1 1.9277e+09 3.1639e+12 514902
    ## - num_hrefs                      1 3.0585e+09 3.1651e+12 514912
    ## - data_channel_is_entertainment  1 3.3254e+09 3.1653e+12 514914
    ## - kw_max_avg                     1 5.6429e+09 3.1676e+12 514934
    ## - kw_avg_avg                     1 1.1639e+10 3.1736e+12 514987
    ## 
    ## Step:  AIC=514884.9
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
    ##     abs_title_sentiment_polarity + weekday_is_friday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - rate_negative_words            1 1.0226e+07 3.1620e+12 514883
    ## - rate_positive_words            1 1.6055e+07 3.1620e+12 514883
    ## - avg_positive_polarity          1 2.1135e+07 3.1620e+12 514883
    ## - min_negative_polarity          1 2.4615e+07 3.1620e+12 514883
    ## - weekday_is_friday              1 3.2966e+07 3.1620e+12 514883
    ## - n_tokens_content               1 3.5703e+07 3.1620e+12 514883
    ## - title_sentiment_polarity       1 4.3956e+07 3.1621e+12 514883
    ## - self_reference_avg_sharess     1 4.8152e+07 3.1621e+12 514883
    ## - max_negative_polarity          1 5.4720e+07 3.1621e+12 514883
    ## - kw_min_max                     1 8.8599e+07 3.1621e+12 514884
    ## - kw_min_min                     1 9.0297e+07 3.1621e+12 514884
    ## - kw_max_max                     1 1.1448e+08 3.1621e+12 514884
    ## - global_rate_positive_words     1 1.3412e+08 3.1621e+12 514884
    ## - num_imgs                       1 1.7245e+08 3.1622e+12 514884
    ## - average_token_length           1 2.0973e+08 3.1622e+12 514885
    ## - LDA_04                         1 2.1511e+08 3.1622e+12 514885
    ## <none>                                        3.1620e+12 514885
    ## - n_non_stop_words               1 2.2946e+08 3.1622e+12 514885
    ## - n_non_stop_unique_tokens       1 2.3225e+08 3.1622e+12 514885
    ## - num_videos                     1 2.3621e+08 3.1623e+12 514885
    ## - self_reference_max_shares      1 2.3869e+08 3.1623e+12 514885
    ## - kw_max_min                     1 3.1282e+08 3.1623e+12 514886
    ## - kw_avg_max                     1 3.1434e+08 3.1623e+12 514886
    ## - min_positive_polarity          1 3.1730e+08 3.1623e+12 514886
    ## - self_reference_min_shares      1 3.3313e+08 3.1623e+12 514886
    ## - LDA_03                         1 3.4621e+08 3.1624e+12 514886
    ## - kw_avg_min                     1 3.7149e+08 3.1624e+12 514886
    ## - data_channel_is_world          1 7.6521e+08 3.1628e+12 514890
    ## - abs_title_sentiment_polarity   1 7.9408e+08 3.1628e+12 514890
    ## - abs_title_subjectivity         1 8.3473e+08 3.1628e+12 514890
    ## - n_tokens_title                 1 8.6794e+08 3.1629e+12 514891
    ## - num_self_hrefs                 1 9.2349e+08 3.1629e+12 514891
    ## - LDA_02                         1 9.3856e+08 3.1630e+12 514891
    ## - LDA_01                         1 9.6866e+08 3.1630e+12 514891
    ## - data_channel_is_tech           1 1.0014e+09 3.1630e+12 514892
    ## - data_channel_is_socmed         1 1.0023e+09 3.1630e+12 514892
    ## - global_subjectivity            1 1.0219e+09 3.1630e+12 514892
    ## - data_channel_is_lifestyle      1 1.4574e+09 3.1635e+12 514896
    ## - data_channel_is_bus            1 1.8899e+09 3.1639e+12 514899
    ## - kw_min_avg                     1 1.9266e+09 3.1639e+12 514900
    ## - num_hrefs                      1 3.0569e+09 3.1651e+12 514910
    ## - data_channel_is_entertainment  1 3.3363e+09 3.1654e+12 514912
    ## - kw_max_avg                     1 5.6477e+09 3.1677e+12 514932
    ## - kw_avg_avg                     1 1.1643e+10 3.1737e+12 514985
    ## 
    ## Step:  AIC=514883
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
    ##     weekday_is_friday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - rate_positive_words            1 6.8706e+06 3.1620e+12 514881
    ## - avg_positive_polarity          1 1.8196e+07 3.1620e+12 514881
    ## - min_negative_polarity          1 3.1702e+07 3.1621e+12 514881
    ## - weekday_is_friday              1 3.2539e+07 3.1621e+12 514881
    ## - title_sentiment_polarity       1 4.2876e+07 3.1621e+12 514881
    ## - n_tokens_content               1 4.3403e+07 3.1621e+12 514881
    ## - self_reference_avg_sharess     1 4.7693e+07 3.1621e+12 514881
    ## - max_negative_polarity          1 5.3622e+07 3.1621e+12 514881
    ## - kw_min_max                     1 8.8799e+07 3.1621e+12 514882
    ## - kw_min_min                     1 9.1573e+07 3.1621e+12 514882
    ## - kw_max_max                     1 1.1385e+08 3.1621e+12 514882
    ## - global_rate_positive_words     1 1.3469e+08 3.1622e+12 514882
    ## - num_imgs                       1 1.6999e+08 3.1622e+12 514882
    ## - LDA_04                         1 2.1115e+08 3.1622e+12 514883
    ## <none>                                        3.1620e+12 514883
    ## - self_reference_max_shares      1 2.3782e+08 3.1623e+12 514883
    ## - num_videos                     1 2.3853e+08 3.1623e+12 514883
    ## - n_non_stop_words               1 2.4784e+08 3.1623e+12 514883
    ## - n_non_stop_unique_tokens       1 2.5045e+08 3.1623e+12 514883
    ## - min_positive_polarity          1 3.1060e+08 3.1623e+12 514884
    ## - kw_max_min                     1 3.1085e+08 3.1623e+12 514884
    ## - kw_avg_max                     1 3.1448e+08 3.1623e+12 514884
    ## - self_reference_min_shares      1 3.3289e+08 3.1624e+12 514884
    ## - LDA_03                         1 3.4490e+08 3.1624e+12 514884
    ## - kw_avg_min                     1 3.6933e+08 3.1624e+12 514884
    ## - average_token_length           1 5.7767e+08 3.1626e+12 514886
    ## - data_channel_is_world          1 7.6333e+08 3.1628e+12 514888
    ## - abs_title_sentiment_polarity   1 7.9096e+08 3.1628e+12 514888
    ## - abs_title_subjectivity         1 8.3328e+08 3.1629e+12 514888
    ## - n_tokens_title                 1 8.9149e+08 3.1629e+12 514889
    ## - num_self_hrefs                 1 9.1417e+08 3.1629e+12 514889
    ## - LDA_02                         1 9.4217e+08 3.1630e+12 514889
    ## - LDA_01                         1 9.6605e+08 3.1630e+12 514889
    ## - data_channel_is_tech           1 9.9330e+08 3.1630e+12 514890
    ## - data_channel_is_socmed         1 9.9376e+08 3.1630e+12 514890
    ## - global_subjectivity            1 1.1069e+09 3.1631e+12 514891
    ## - data_channel_is_lifestyle      1 1.4501e+09 3.1635e+12 514894
    ## - data_channel_is_bus            1 1.8817e+09 3.1639e+12 514897
    ## - kw_min_avg                     1 1.9207e+09 3.1639e+12 514898
    ## - num_hrefs                      1 3.1107e+09 3.1651e+12 514908
    ## - data_channel_is_entertainment  1 3.3293e+09 3.1654e+12 514910
    ## - kw_max_avg                     1 5.6379e+09 3.1677e+12 514930
    ## - kw_avg_avg                     1 1.1634e+10 3.1737e+12 514983
    ## 
    ## Step:  AIC=514881
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
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_friday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - avg_positive_polarity          1 1.6561e+07 3.1620e+12 514879
    ## - min_negative_polarity          1 2.4859e+07 3.1621e+12 514879
    ## - weekday_is_friday              1 3.2832e+07 3.1621e+12 514879
    ## - title_sentiment_polarity       1 4.7670e+07 3.1621e+12 514879
    ## - self_reference_avg_sharess     1 4.7793e+07 3.1621e+12 514879
    ## - n_tokens_content               1 4.7942e+07 3.1621e+12 514879
    ## - max_negative_polarity          1 6.4546e+07 3.1621e+12 514880
    ## - kw_min_max                     1 8.8560e+07 3.1621e+12 514880
    ## - kw_min_min                     1 9.1809e+07 3.1621e+12 514880
    ## - kw_max_max                     1 1.1369e+08 3.1621e+12 514880
    ## - global_rate_positive_words     1 1.4066e+08 3.1622e+12 514880
    ## - num_imgs                       1 1.6751e+08 3.1622e+12 514881
    ## - LDA_04                         1 2.1074e+08 3.1622e+12 514881
    ## <none>                                        3.1620e+12 514881
    ## - num_videos                     1 2.3446e+08 3.1623e+12 514881
    ## - self_reference_max_shares      1 2.3808e+08 3.1623e+12 514881
    ## - n_non_stop_words               1 2.4550e+08 3.1623e+12 514881
    ## - n_non_stop_unique_tokens       1 2.4806e+08 3.1623e+12 514881
    ## - kw_max_min                     1 3.1087e+08 3.1623e+12 514882
    ## - kw_avg_max                     1 3.1362e+08 3.1623e+12 514882
    ## - min_positive_polarity          1 3.2083e+08 3.1624e+12 514882
    ## - self_reference_min_shares      1 3.3342e+08 3.1624e+12 514882
    ## - LDA_03                         1 3.4716e+08 3.1624e+12 514882
    ## - kw_avg_min                     1 3.6915e+08 3.1624e+12 514882
    ## - average_token_length           1 6.4621e+08 3.1627e+12 514885
    ## - data_channel_is_world          1 7.6016e+08 3.1628e+12 514886
    ## - abs_title_sentiment_polarity   1 7.8490e+08 3.1628e+12 514886
    ## - abs_title_subjectivity         1 8.4423e+08 3.1629e+12 514886
    ## - n_tokens_title                 1 8.9710e+08 3.1629e+12 514887
    ## - num_self_hrefs                 1 9.1075e+08 3.1629e+12 514887
    ## - LDA_02                         1 9.5071e+08 3.1630e+12 514887
    ## - LDA_01                         1 9.7114e+08 3.1630e+12 514888
    ## - data_channel_is_tech           1 9.8750e+08 3.1630e+12 514888
    ## - data_channel_is_socmed         1 9.8833e+08 3.1630e+12 514888
    ## - global_subjectivity            1 1.1450e+09 3.1632e+12 514889
    ## - data_channel_is_lifestyle      1 1.4450e+09 3.1635e+12 514892
    ## - data_channel_is_bus            1 1.8759e+09 3.1639e+12 514895
    ## - kw_min_avg                     1 1.9225e+09 3.1640e+12 514896
    ## - num_hrefs                      1 3.1197e+09 3.1652e+12 514906
    ## - data_channel_is_entertainment  1 3.3233e+09 3.1654e+12 514908
    ## - kw_max_avg                     1 5.6323e+09 3.1677e+12 514928
    ## - kw_avg_avg                     1 1.1627e+10 3.1737e+12 514981
    ## 
    ## Step:  AIC=514879.2
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
    ##     abs_title_sentiment_polarity + weekday_is_friday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - min_negative_polarity          1 2.3707e+07 3.1621e+12 514877
    ## - weekday_is_friday              1 3.2428e+07 3.1621e+12 514877
    ## - n_tokens_content               1 4.2557e+07 3.1621e+12 514878
    ## - title_sentiment_polarity       1 4.4598e+07 3.1621e+12 514878
    ## - self_reference_avg_sharess     1 4.8116e+07 3.1621e+12 514878
    ## - max_negative_polarity          1 6.8251e+07 3.1621e+12 514878
    ## - kw_min_max                     1 8.9049e+07 3.1621e+12 514878
    ## - kw_min_min                     1 9.2347e+07 3.1621e+12 514878
    ## - kw_max_max                     1 1.1311e+08 3.1622e+12 514878
    ## - global_rate_positive_words     1 1.5792e+08 3.1622e+12 514879
    ## - num_imgs                       1 1.6490e+08 3.1622e+12 514879
    ## - LDA_04                         1 2.0813e+08 3.1623e+12 514879
    ## <none>                                        3.1620e+12 514879
    ## - num_videos                     1 2.2877e+08 3.1623e+12 514879
    ## - self_reference_max_shares      1 2.3846e+08 3.1623e+12 514879
    ## - n_non_stop_words               1 2.4349e+08 3.1623e+12 514879
    ## - n_non_stop_unique_tokens       1 2.4608e+08 3.1623e+12 514879
    ## - kw_avg_max                     1 3.1134e+08 3.1624e+12 514880
    ## - kw_max_min                     1 3.1192e+08 3.1624e+12 514880
    ## - self_reference_min_shares      1 3.3435e+08 3.1624e+12 514880
    ## - LDA_03                         1 3.4730e+08 3.1624e+12 514880
    ## - kw_avg_min                     1 3.6930e+08 3.1624e+12 514880
    ## - min_positive_polarity          1 4.8140e+08 3.1625e+12 514881
    ## - average_token_length           1 7.3166e+08 3.1628e+12 514884
    ## - data_channel_is_world          1 7.5713e+08 3.1628e+12 514884
    ## - abs_title_sentiment_polarity   1 7.7281e+08 3.1628e+12 514884
    ## - abs_title_subjectivity         1 8.3090e+08 3.1629e+12 514884
    ## - n_tokens_title                 1 8.9444e+08 3.1629e+12 514885
    ## - num_self_hrefs                 1 9.0814e+08 3.1630e+12 514885
    ## - LDA_02                         1 9.4502e+08 3.1630e+12 514885
    ## - LDA_01                         1 9.6807e+08 3.1630e+12 514886
    ## - data_channel_is_socmed         1 9.8721e+08 3.1630e+12 514886
    ## - data_channel_is_tech           1 9.8767e+08 3.1630e+12 514886
    ## - global_subjectivity            1 1.1673e+09 3.1632e+12 514887
    ## - data_channel_is_lifestyle      1 1.4544e+09 3.1635e+12 514890
    ## - data_channel_is_bus            1 1.8771e+09 3.1639e+12 514894
    ## - kw_min_avg                     1 1.9239e+09 3.1640e+12 514894
    ## - num_hrefs                      1 3.1068e+09 3.1652e+12 514904
    ## - data_channel_is_entertainment  1 3.3361e+09 3.1654e+12 514906
    ## - kw_max_avg                     1 5.6359e+09 3.1677e+12 514927
    ## - kw_avg_avg                     1 1.1624e+10 3.1737e+12 514979
    ## 
    ## Step:  AIC=514877.4
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
    ##     weekday_is_friday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - weekday_is_friday              1 3.2848e+07 3.1621e+12 514876
    ## - title_sentiment_polarity       1 3.7357e+07 3.1621e+12 514876
    ## - self_reference_avg_sharess     1 4.8380e+07 3.1621e+12 514876
    ## - n_tokens_content               1 7.8495e+07 3.1621e+12 514876
    ## - max_negative_polarity          1 8.0836e+07 3.1622e+12 514876
    ## - kw_min_max                     1 8.8496e+07 3.1622e+12 514876
    ## - kw_min_min                     1 9.2098e+07 3.1622e+12 514876
    ## - kw_max_max                     1 1.1226e+08 3.1622e+12 514876
    ## - num_imgs                       1 1.6007e+08 3.1622e+12 514877
    ## - global_rate_positive_words     1 1.6982e+08 3.1622e+12 514877
    ## - LDA_04                         1 2.1116e+08 3.1623e+12 514877
    ## <none>                                        3.1621e+12 514877
    ## - num_videos                     1 2.3818e+08 3.1623e+12 514877
    ## - self_reference_max_shares      1 2.3983e+08 3.1623e+12 514877
    ## - n_non_stop_words               1 2.4716e+08 3.1623e+12 514878
    ## - n_non_stop_unique_tokens       1 2.4974e+08 3.1623e+12 514878
    ## - kw_avg_max                     1 3.1082e+08 3.1624e+12 514878
    ## - kw_max_min                     1 3.1139e+08 3.1624e+12 514878
    ## - self_reference_min_shares      1 3.3471e+08 3.1624e+12 514878
    ## - LDA_03                         1 3.4662e+08 3.1624e+12 514878
    ## - kw_avg_min                     1 3.6906e+08 3.1624e+12 514879
    ## - min_positive_polarity          1 5.1366e+08 3.1626e+12 514880
    ## - average_token_length           1 7.1810e+08 3.1628e+12 514882
    ## - data_channel_is_world          1 7.5580e+08 3.1628e+12 514882
    ## - abs_title_sentiment_polarity   1 7.9864e+08 3.1629e+12 514882
    ## - abs_title_subjectivity         1 8.3239e+08 3.1629e+12 514883
    ## - n_tokens_title                 1 8.9976e+08 3.1630e+12 514883
    ## - num_self_hrefs                 1 9.1392e+08 3.1630e+12 514883
    ## - LDA_02                         1 9.4270e+08 3.1630e+12 514884
    ## - LDA_01                         1 9.6419e+08 3.1630e+12 514884
    ## - data_channel_is_socmed         1 9.9689e+08 3.1631e+12 514884
    ## - data_channel_is_tech           1 1.0034e+09 3.1631e+12 514884
    ## - global_subjectivity            1 1.3309e+09 3.1634e+12 514887
    ## - data_channel_is_lifestyle      1 1.4554e+09 3.1635e+12 514888
    ## - data_channel_is_bus            1 1.8975e+09 3.1640e+12 514892
    ## - kw_min_avg                     1 1.9259e+09 3.1640e+12 514892
    ## - num_hrefs                      1 3.1153e+09 3.1652e+12 514903
    ## - data_channel_is_entertainment  1 3.3342e+09 3.1654e+12 514905
    ## - kw_max_avg                     1 5.6372e+09 3.1677e+12 514925
    ## - kw_avg_avg                     1 1.1632e+10 3.1737e+12 514977
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
    ##     global_rate_positive_words + min_positive_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - title_sentiment_polarity       1 3.7541e+07 3.1621e+12 514874
    ## - self_reference_avg_sharess     1 4.7885e+07 3.1622e+12 514874
    ## - n_tokens_content               1 7.8205e+07 3.1622e+12 514874
    ## - max_negative_polarity          1 8.0215e+07 3.1622e+12 514874
    ## - kw_min_max                     1 8.8057e+07 3.1622e+12 514874
    ## - kw_min_min                     1 9.1598e+07 3.1622e+12 514874
    ## - kw_max_max                     1 1.1306e+08 3.1622e+12 514875
    ## - num_imgs                       1 1.5993e+08 3.1623e+12 514875
    ## - global_rate_positive_words     1 1.6722e+08 3.1623e+12 514875
    ## - LDA_04                         1 2.1216e+08 3.1623e+12 514876
    ## <none>                                        3.1621e+12 514876
    ## - self_reference_max_shares      1 2.3845e+08 3.1623e+12 514876
    ## - num_videos                     1 2.3854e+08 3.1623e+12 514876
    ## - n_non_stop_words               1 2.4438e+08 3.1623e+12 514876
    ## - n_non_stop_unique_tokens       1 2.4694e+08 3.1624e+12 514876
    ## - kw_avg_max                     1 3.1086e+08 3.1624e+12 514876
    ## - kw_max_min                     1 3.1370e+08 3.1624e+12 514876
    ## - self_reference_min_shares      1 3.3400e+08 3.1624e+12 514877
    ## - LDA_03                         1 3.4781e+08 3.1625e+12 514877
    ## - kw_avg_min                     1 3.7124e+08 3.1625e+12 514877
    ## - min_positive_polarity          1 5.1486e+08 3.1626e+12 514878
    ## - average_token_length           1 7.1689e+08 3.1628e+12 514880
    ## - data_channel_is_world          1 7.5463e+08 3.1629e+12 514880
    ## - abs_title_sentiment_polarity   1 7.9783e+08 3.1629e+12 514881
    ## - abs_title_subjectivity         1 8.3037e+08 3.1629e+12 514881
    ## - n_tokens_title                 1 8.9971e+08 3.1630e+12 514882
    ## - num_self_hrefs                 1 9.0768e+08 3.1630e+12 514882
    ## - LDA_02                         1 9.4727e+08 3.1631e+12 514882
    ## - LDA_01                         1 9.6365e+08 3.1631e+12 514882
    ## - data_channel_is_socmed         1 9.9626e+08 3.1631e+12 514882
    ## - data_channel_is_tech           1 1.0014e+09 3.1631e+12 514882
    ## - global_subjectivity            1 1.3238e+09 3.1634e+12 514885
    ## - data_channel_is_lifestyle      1 1.4538e+09 3.1636e+12 514886
    ## - data_channel_is_bus            1 1.8968e+09 3.1640e+12 514890
    ## - kw_min_avg                     1 1.9231e+09 3.1640e+12 514891
    ## - num_hrefs                      1 3.1142e+09 3.1652e+12 514901
    ## - data_channel_is_entertainment  1 3.3301e+09 3.1654e+12 514903
    ## - kw_max_avg                     1 5.6338e+09 3.1677e+12 514923
    ## - kw_avg_avg                     1 1.1626e+10 3.1737e+12 514976
    ## 
    ## Step:  AIC=514874
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
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - self_reference_avg_sharess     1 4.8541e+07 3.1622e+12 514872
    ## - n_tokens_content               1 7.6770e+07 3.1622e+12 514873
    ## - max_negative_polarity          1 8.0374e+07 3.1622e+12 514873
    ## - kw_min_max                     1 8.7475e+07 3.1622e+12 514873
    ## - kw_min_min                     1 9.2602e+07 3.1622e+12 514873
    ## - kw_max_max                     1 1.1319e+08 3.1623e+12 514873
    ## - global_rate_positive_words     1 1.5323e+08 3.1623e+12 514873
    ## - num_imgs                       1 1.6329e+08 3.1623e+12 514873
    ## - LDA_04                         1 2.1440e+08 3.1624e+12 514874
    ## <none>                                        3.1621e+12 514874
    ## - n_non_stop_words               1 2.3939e+08 3.1624e+12 514874
    ## - self_reference_max_shares      1 2.3972e+08 3.1624e+12 514874
    ## - num_videos                     1 2.3995e+08 3.1624e+12 514874
    ## - n_non_stop_unique_tokens       1 2.4192e+08 3.1624e+12 514874
    ## - kw_avg_max                     1 3.1153e+08 3.1625e+12 514875
    ## - kw_max_min                     1 3.1417e+08 3.1625e+12 514875
    ## - self_reference_min_shares      1 3.3596e+08 3.1625e+12 514875
    ## - LDA_03                         1 3.4999e+08 3.1625e+12 514875
    ## - kw_avg_min                     1 3.7206e+08 3.1625e+12 514875
    ## - min_positive_polarity          1 5.0543e+08 3.1626e+12 514876
    ## - average_token_length           1 7.2293e+08 3.1629e+12 514878
    ## - data_channel_is_world          1 7.5698e+08 3.1629e+12 514879
    ## - abs_title_subjectivity         1 8.0708e+08 3.1629e+12 514879
    ## - n_tokens_title                 1 8.9618e+08 3.1630e+12 514880
    ## - num_self_hrefs                 1 9.0607e+08 3.1630e+12 514880
    ## - LDA_02                         1 9.4662e+08 3.1631e+12 514880
    ## - LDA_01                         1 9.7088e+08 3.1631e+12 514881
    ## - data_channel_is_socmed         1 9.9008e+08 3.1631e+12 514881
    ## - data_channel_is_tech           1 9.9280e+08 3.1631e+12 514881
    ## - abs_title_sentiment_polarity   1 1.0336e+09 3.1632e+12 514881
    ## - global_subjectivity            1 1.3084e+09 3.1635e+12 514883
    ## - data_channel_is_lifestyle      1 1.4420e+09 3.1636e+12 514885
    ## - data_channel_is_bus            1 1.8901e+09 3.1640e+12 514889
    ## - kw_min_avg                     1 1.9228e+09 3.1641e+12 514889
    ## - num_hrefs                      1 3.1231e+09 3.1653e+12 514899
    ## - data_channel_is_entertainment  1 3.3275e+09 3.1655e+12 514901
    ## - kw_max_avg                     1 5.6360e+09 3.1678e+12 514921
    ## - kw_avg_avg                     1 1.1631e+10 3.1738e+12 514974
    ## 
    ## Step:  AIC=514872.4
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + global_rate_positive_words + min_positive_polarity + 
    ##     max_negative_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_tokens_content               1 7.7275e+07 3.1623e+12 514871
    ## - max_negative_polarity          1 7.9933e+07 3.1623e+12 514871
    ## - kw_min_max                     1 8.5887e+07 3.1623e+12 514871
    ## - kw_min_min                     1 9.2258e+07 3.1623e+12 514871
    ## - kw_max_max                     1 1.1380e+08 3.1623e+12 514871
    ## - global_rate_positive_words     1 1.5146e+08 3.1623e+12 514872
    ## - num_imgs                       1 1.6128e+08 3.1624e+12 514872
    ## - LDA_04                         1 2.1476e+08 3.1624e+12 514872
    ## <none>                                        3.1622e+12 514872
    ## - n_non_stop_words               1 2.4164e+08 3.1624e+12 514873
    ## - n_non_stop_unique_tokens       1 2.4415e+08 3.1624e+12 514873
    ## - num_videos                     1 2.4661e+08 3.1624e+12 514873
    ## - kw_avg_max                     1 3.1376e+08 3.1625e+12 514873
    ## - kw_max_min                     1 3.1971e+08 3.1625e+12 514873
    ## - LDA_03                         1 3.4975e+08 3.1625e+12 514874
    ## - kw_avg_min                     1 3.7648e+08 3.1626e+12 514874
    ## - min_positive_polarity          1 5.0313e+08 3.1627e+12 514875
    ## - self_reference_max_shares      1 5.3012e+08 3.1627e+12 514875
    ## - average_token_length           1 7.3156e+08 3.1629e+12 514877
    ## - data_channel_is_world          1 7.5254e+08 3.1629e+12 514877
    ## - self_reference_min_shares      1 7.7280e+08 3.1630e+12 514877
    ## - abs_title_subjectivity         1 8.0749e+08 3.1630e+12 514878
    ## - num_self_hrefs                 1 8.6564e+08 3.1631e+12 514878
    ## - n_tokens_title                 1 8.9027e+08 3.1631e+12 514878
    ## - LDA_02                         1 9.4580e+08 3.1631e+12 514879
    ## - LDA_01                         1 9.7081e+08 3.1632e+12 514879
    ## - data_channel_is_socmed         1 9.8890e+08 3.1632e+12 514879
    ## - data_channel_is_tech           1 9.9449e+08 3.1632e+12 514879
    ## - abs_title_sentiment_polarity   1 1.0346e+09 3.1632e+12 514880
    ## - global_subjectivity            1 1.2998e+09 3.1635e+12 514882
    ## - data_channel_is_lifestyle      1 1.4376e+09 3.1636e+12 514883
    ## - data_channel_is_bus            1 1.8834e+09 3.1641e+12 514887
    ## - kw_min_avg                     1 1.9326e+09 3.1641e+12 514887
    ## - num_hrefs                      1 3.1499e+09 3.1653e+12 514898
    ## - data_channel_is_entertainment  1 3.3234e+09 3.1655e+12 514900
    ## - kw_max_avg                     1 5.6564e+09 3.1678e+12 514920
    ## - kw_avg_avg                     1 1.1639e+10 3.1738e+12 514972
    ## 
    ## Step:  AIC=514871.1
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + min_positive_polarity + max_negative_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - max_negative_polarity          1 5.7173e+07 3.1623e+12 514870
    ## - kw_min_max                     1 8.3638e+07 3.1624e+12 514870
    ## - kw_min_min                     1 9.3467e+07 3.1624e+12 514870
    ## - kw_max_max                     1 1.0948e+08 3.1624e+12 514870
    ## - global_rate_positive_words     1 1.3179e+08 3.1624e+12 514870
    ## - n_non_stop_words               1 1.7080e+08 3.1624e+12 514871
    ## - n_non_stop_unique_tokens       1 1.7312e+08 3.1624e+12 514871
    ## - num_imgs                       1 1.9120e+08 3.1625e+12 514871
    ## - LDA_04                         1 2.1916e+08 3.1625e+12 514871
    ## <none>                                        3.1623e+12 514871
    ## - num_videos                     1 2.9194e+08 3.1626e+12 514872
    ## - kw_avg_max                     1 3.1351e+08 3.1626e+12 514872
    ## - kw_max_min                     1 3.1951e+08 3.1626e+12 514872
    ## - LDA_03                         1 3.7416e+08 3.1626e+12 514872
    ## - kw_avg_min                     1 3.7471e+08 3.1626e+12 514872
    ## - self_reference_max_shares      1 5.2777e+08 3.1628e+12 514874
    ## - min_positive_polarity          1 5.7269e+08 3.1628e+12 514874
    ## - average_token_length           1 7.0298e+08 3.1630e+12 514875
    ## - data_channel_is_world          1 7.2967e+08 3.1630e+12 514876
    ## - self_reference_min_shares      1 7.7141e+08 3.1630e+12 514876
    ## - abs_title_subjectivity         1 8.2117e+08 3.1631e+12 514876
    ## - num_self_hrefs                 1 8.2276e+08 3.1631e+12 514876
    ## - n_tokens_title                 1 9.0930e+08 3.1632e+12 514877
    ## - LDA_02                         1 9.3826e+08 3.1632e+12 514877
    ## - data_channel_is_socmed         1 9.6986e+08 3.1632e+12 514878
    ## - data_channel_is_tech           1 9.7352e+08 3.1632e+12 514878
    ## - LDA_01                         1 9.8241e+08 3.1633e+12 514878
    ## - abs_title_sentiment_polarity   1 1.0336e+09 3.1633e+12 514878
    ## - global_subjectivity            1 1.3836e+09 3.1637e+12 514881
    ## - data_channel_is_lifestyle      1 1.4003e+09 3.1637e+12 514881
    ## - data_channel_is_bus            1 1.8510e+09 3.1641e+12 514885
    ## - kw_min_avg                     1 1.9281e+09 3.1642e+12 514886
    ## - data_channel_is_entertainment  1 3.2509e+09 3.1655e+12 514898
    ## - num_hrefs                      1 3.4916e+09 3.1658e+12 514900
    ## - kw_max_avg                     1 5.6326e+09 3.1679e+12 514918
    ## - kw_avg_avg                     1 1.1590e+10 3.1739e+12 514971
    ## 
    ## Step:  AIC=514869.6
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - kw_min_max                     1 8.5474e+07 3.1624e+12 514868
    ## - kw_min_min                     1 9.4492e+07 3.1624e+12 514868
    ## - kw_max_max                     1 1.1067e+08 3.1624e+12 514869
    ## - global_rate_positive_words     1 1.3657e+08 3.1625e+12 514869
    ## - num_imgs                       1 1.8964e+08 3.1625e+12 514869
    ## - n_non_stop_words               1 1.9478e+08 3.1625e+12 514869
    ## - n_non_stop_unique_tokens       1 1.9727e+08 3.1625e+12 514869
    ## - LDA_04                         1 2.2103e+08 3.1625e+12 514870
    ## <none>                                        3.1623e+12 514870
    ## - num_videos                     1 2.7743e+08 3.1626e+12 514870
    ## - kw_avg_max                     1 3.0865e+08 3.1626e+12 514870
    ## - kw_max_min                     1 3.1869e+08 3.1626e+12 514870
    ## - LDA_03                         1 3.6903e+08 3.1627e+12 514871
    ## - kw_avg_min                     1 3.7460e+08 3.1627e+12 514871
    ## - self_reference_max_shares      1 5.2794e+08 3.1629e+12 514872
    ## - min_positive_polarity          1 5.5261e+08 3.1629e+12 514872
    ## - average_token_length           1 6.6896e+08 3.1630e+12 514873
    ## - data_channel_is_world          1 7.3906e+08 3.1631e+12 514874
    ## - self_reference_min_shares      1 7.8097e+08 3.1631e+12 514874
    ## - abs_title_subjectivity         1 8.2109e+08 3.1631e+12 514875
    ## - num_self_hrefs                 1 8.2642e+08 3.1632e+12 514875
    ## - n_tokens_title                 1 9.1175e+08 3.1632e+12 514876
    ## - LDA_02                         1 9.4462e+08 3.1633e+12 514876
    ## - data_channel_is_socmed         1 9.7152e+08 3.1633e+12 514876
    ## - LDA_01                         1 9.7915e+08 3.1633e+12 514876
    ## - data_channel_is_tech           1 9.8279e+08 3.1633e+12 514876
    ## - abs_title_sentiment_polarity   1 1.0362e+09 3.1634e+12 514877
    ## - data_channel_is_lifestyle      1 1.4150e+09 3.1637e+12 514880
    ## - global_subjectivity            1 1.4514e+09 3.1638e+12 514880
    ## - data_channel_is_bus            1 1.8649e+09 3.1642e+12 514884
    ## - kw_min_avg                     1 1.9314e+09 3.1643e+12 514885
    ## - data_channel_is_entertainment  1 3.2689e+09 3.1656e+12 514896
    ## - num_hrefs                      1 3.4533e+09 3.1658e+12 514898
    ## - kw_max_avg                     1 5.6370e+09 3.1680e+12 514917
    ## - kw_avg_avg                     1 1.1607e+10 3.1739e+12 514969
    ## 
    ## Step:  AIC=514868.4
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + LDA_01 + 
    ##     LDA_02 + LDA_03 + LDA_04 + global_subjectivity + global_rate_positive_words + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - kw_max_max                     1 8.5701e+07 3.1625e+12 514867
    ## - kw_min_min                     1 9.1429e+07 3.1625e+12 514867
    ## - global_rate_positive_words     1 1.4077e+08 3.1626e+12 514868
    ## - num_imgs                       1 1.8644e+08 3.1626e+12 514868
    ## - n_non_stop_words               1 1.9868e+08 3.1626e+12 514868
    ## - n_non_stop_unique_tokens       1 2.0121e+08 3.1626e+12 514868
    ## <none>                                        3.1624e+12 514868
    ## - LDA_04                         1 2.2805e+08 3.1626e+12 514868
    ## - num_videos                     1 2.8217e+08 3.1627e+12 514869
    ## - kw_max_min                     1 3.2744e+08 3.1627e+12 514869
    ## - LDA_03                         1 3.6550e+08 3.1628e+12 514870
    ## - kw_avg_min                     1 3.8174e+08 3.1628e+12 514870
    ## - self_reference_max_shares      1 5.3476e+08 3.1629e+12 514871
    ## - kw_avg_max                     1 5.4674e+08 3.1630e+12 514871
    ## - min_positive_polarity          1 5.5799e+08 3.1630e+12 514871
    ## - average_token_length           1 6.8103e+08 3.1631e+12 514872
    ## - data_channel_is_world          1 7.7632e+08 3.1632e+12 514873
    ## - self_reference_min_shares      1 7.7723e+08 3.1632e+12 514873
    ## - num_self_hrefs                 1 8.1363e+08 3.1632e+12 514874
    ## - abs_title_subjectivity         1 8.2586e+08 3.1632e+12 514874
    ## - n_tokens_title                 1 9.2291e+08 3.1633e+12 514874
    ## - LDA_02                         1 9.3932e+08 3.1633e+12 514875
    ## - LDA_01                         1 9.9279e+08 3.1634e+12 514875
    ## - data_channel_is_tech           1 1.0102e+09 3.1634e+12 514875
    ## - abs_title_sentiment_polarity   1 1.0366e+09 3.1634e+12 514875
    ## - data_channel_is_socmed         1 1.0497e+09 3.1635e+12 514876
    ## - global_subjectivity            1 1.4566e+09 3.1639e+12 514879
    ## - data_channel_is_lifestyle      1 1.4655e+09 3.1639e+12 514879
    ## - data_channel_is_bus            1 1.8754e+09 3.1643e+12 514883
    ## - kw_min_avg                     1 2.1735e+09 3.1646e+12 514885
    ## - data_channel_is_entertainment  1 3.3863e+09 3.1658e+12 514896
    ## - num_hrefs                      1 3.4420e+09 3.1659e+12 514897
    ## - kw_max_avg                     1 5.7357e+09 3.1681e+12 514917
    ## - kw_avg_avg                     1 1.1788e+10 3.1742e+12 514970
    ## 
    ## Step:  AIC=514867.1
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + global_rate_positive_words + min_positive_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - global_rate_positive_words     1 1.3894e+08 3.1626e+12 514866
    ## - num_imgs                       1 1.8165e+08 3.1627e+12 514867
    ## - n_non_stop_words               1 2.0542e+08 3.1627e+12 514867
    ## - n_non_stop_unique_tokens       1 2.0800e+08 3.1627e+12 514867
    ## <none>                                        3.1625e+12 514867
    ## - LDA_04                         1 2.3464e+08 3.1627e+12 514867
    ## - num_videos                     1 2.8937e+08 3.1628e+12 514868
    ## - kw_max_min                     1 3.1003e+08 3.1628e+12 514868
    ## - LDA_03                         1 3.4740e+08 3.1628e+12 514868
    ## - kw_avg_min                     1 3.6185e+08 3.1629e+12 514868
    ## - self_reference_max_shares      1 5.4373e+08 3.1630e+12 514870
    ## - min_positive_polarity          1 5.5775e+08 3.1631e+12 514870
    ## - kw_min_min                     1 6.4981e+08 3.1631e+12 514871
    ## - average_token_length           1 6.9012e+08 3.1632e+12 514871
    ## - kw_avg_max                     1 7.3326e+08 3.1632e+12 514872
    ## - self_reference_min_shares      1 7.7001e+08 3.1633e+12 514872
    ## - num_self_hrefs                 1 8.0492e+08 3.1633e+12 514872
    ## - data_channel_is_world          1 8.3042e+08 3.1633e+12 514872
    ## - abs_title_subjectivity         1 8.3201e+08 3.1633e+12 514872
    ## - n_tokens_title                 1 9.0977e+08 3.1634e+12 514873
    ## - LDA_02                         1 9.3847e+08 3.1634e+12 514873
    ## - LDA_01                         1 9.9319e+08 3.1635e+12 514874
    ## - abs_title_sentiment_polarity   1 1.0423e+09 3.1635e+12 514874
    ## - data_channel_is_tech           1 1.0432e+09 3.1635e+12 514874
    ## - data_channel_is_socmed         1 1.0893e+09 3.1636e+12 514875
    ## - global_subjectivity            1 1.4534e+09 3.1639e+12 514878
    ## - data_channel_is_lifestyle      1 1.4969e+09 3.1640e+12 514878
    ## - data_channel_is_bus            1 1.8877e+09 3.1644e+12 514882
    ## - kw_min_avg                     1 2.0996e+09 3.1646e+12 514884
    ## - num_hrefs                      1 3.4064e+09 3.1659e+12 514895
    ## - data_channel_is_entertainment  1 3.5559e+09 3.1661e+12 514896
    ## - kw_max_avg                     1 5.6539e+09 3.1681e+12 514915
    ## - kw_avg_avg                     1 1.1732e+10 3.1742e+12 514968
    ## 
    ## Step:  AIC=514866.3
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_non_stop_words               1 1.7802e+08 3.1628e+12 514866
    ## - n_non_stop_unique_tokens       1 1.8048e+08 3.1628e+12 514866
    ## - num_imgs                       1 1.9495e+08 3.1628e+12 514866
    ## - LDA_04                         1 2.2309e+08 3.1629e+12 514866
    ## <none>                                        3.1626e+12 514866
    ## - num_videos                     1 2.7012e+08 3.1629e+12 514867
    ## - kw_max_min                     1 3.1364e+08 3.1629e+12 514867
    ## - LDA_03                         1 3.2496e+08 3.1630e+12 514867
    ## - kw_avg_min                     1 3.6542e+08 3.1630e+12 514868
    ## - min_positive_polarity          1 4.4560e+08 3.1631e+12 514868
    ## - self_reference_max_shares      1 5.5018e+08 3.1632e+12 514869
    ## - kw_min_min                     1 6.4098e+08 3.1633e+12 514870
    ## - kw_avg_max                     1 6.9678e+08 3.1633e+12 514870
    ## - average_token_length           1 7.7859e+08 3.1634e+12 514871
    ## - self_reference_min_shares      1 7.8073e+08 3.1634e+12 514871
    ## - data_channel_is_world          1 7.8776e+08 3.1634e+12 514871
    ## - num_self_hrefs                 1 8.5661e+08 3.1635e+12 514872
    ## - LDA_02                         1 8.9815e+08 3.1635e+12 514872
    ## - n_tokens_title                 1 9.3846e+08 3.1636e+12 514873
    ## - abs_title_subjectivity         1 9.4060e+08 3.1636e+12 514873
    ## - LDA_01                         1 9.7934e+08 3.1636e+12 514873
    ## - abs_title_sentiment_polarity   1 1.0206e+09 3.1637e+12 514873
    ## - data_channel_is_tech           1 1.0423e+09 3.1637e+12 514873
    ## - data_channel_is_socmed         1 1.1049e+09 3.1637e+12 514874
    ## - global_subjectivity            1 1.3175e+09 3.1640e+12 514876
    ## - data_channel_is_lifestyle      1 1.4998e+09 3.1641e+12 514877
    ## - data_channel_is_bus            1 1.8762e+09 3.1645e+12 514881
    ## - kw_min_avg                     1 2.1155e+09 3.1648e+12 514883
    ## - num_hrefs                      1 3.5229e+09 3.1662e+12 514895
    ## - data_channel_is_entertainment  1 3.5274e+09 3.1662e+12 514895
    ## - kw_max_avg                     1 5.6591e+09 3.1683e+12 514914
    ## - kw_avg_avg                     1 1.1731e+10 3.1744e+12 514967
    ## 
    ## Step:  AIC=514865.9
    ## shares ~ n_tokens_title + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_non_stop_unique_tokens       1 2.0800e+07 3.1628e+12 514864
    ## - num_imgs                       1 8.5940e+07 3.1629e+12 514865
    ## - LDA_04                         1 2.2487e+08 3.1630e+12 514866
    ## <none>                                        3.1628e+12 514866
    ## - num_videos                     1 2.4840e+08 3.1631e+12 514866
    ## - LDA_03                         1 3.0342e+08 3.1631e+12 514867
    ## - kw_max_min                     1 3.1289e+08 3.1631e+12 514867
    ## - min_positive_polarity          1 3.6293e+08 3.1632e+12 514867
    ## - kw_avg_min                     1 3.6527e+08 3.1632e+12 514867
    ## - self_reference_max_shares      1 5.6344e+08 3.1634e+12 514869
    ## - kw_avg_max                     1 6.7528e+08 3.1635e+12 514870
    ## - kw_min_min                     1 6.7821e+08 3.1635e+12 514870
    ## - average_token_length           1 7.1248e+08 3.1635e+12 514870
    ## - self_reference_min_shares      1 7.7707e+08 3.1636e+12 514871
    ## - data_channel_is_world          1 8.0809e+08 3.1636e+12 514871
    ## - num_self_hrefs                 1 8.6789e+08 3.1637e+12 514872
    ## - LDA_02                         1 9.0283e+08 3.1637e+12 514872
    ## - n_tokens_title                 1 9.1493e+08 3.1637e+12 514872
    ## - abs_title_subjectivity         1 9.2937e+08 3.1637e+12 514872
    ## - LDA_01                         1 9.3566e+08 3.1637e+12 514872
    ## - abs_title_sentiment_polarity   1 9.9951e+08 3.1638e+12 514873
    ## - data_channel_is_tech           1 1.0636e+09 3.1639e+12 514873
    ## - data_channel_is_socmed         1 1.1176e+09 3.1639e+12 514874
    ## - global_subjectivity            1 1.3695e+09 3.1642e+12 514876
    ## - data_channel_is_lifestyle      1 1.5042e+09 3.1643e+12 514877
    ## - data_channel_is_bus            1 1.8953e+09 3.1647e+12 514881
    ## - kw_min_avg                     1 2.1329e+09 3.1649e+12 514883
    ## - num_hrefs                      1 3.3449e+09 3.1662e+12 514893
    ## - data_channel_is_entertainment  1 3.6132e+09 3.1664e+12 514896
    ## - kw_max_avg                     1 5.6648e+09 3.1685e+12 514914
    ## - kw_avg_avg                     1 1.1747e+10 3.1746e+12 514967
    ## 
    ## Step:  AIC=514864.1
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - num_imgs                       1 8.8365e+07 3.1629e+12 514863
    ## <none>                                        3.1628e+12 514864
    ## - LDA_04                         1 2.2871e+08 3.1631e+12 514864
    ## - num_videos                     1 2.4898e+08 3.1631e+12 514864
    ## - LDA_03                         1 3.1007e+08 3.1631e+12 514865
    ## - kw_max_min                     1 3.1294e+08 3.1631e+12 514865
    ## - min_positive_polarity          1 3.6219e+08 3.1632e+12 514865
    ## - kw_avg_min                     1 3.6526e+08 3.1632e+12 514865
    ## - self_reference_max_shares      1 5.6315e+08 3.1634e+12 514867
    ## - kw_avg_max                     1 6.7389e+08 3.1635e+12 514868
    ## - kw_min_min                     1 6.7905e+08 3.1635e+12 514868
    ## - average_token_length           1 7.0442e+08 3.1635e+12 514868
    ## - self_reference_min_shares      1 7.7732e+08 3.1636e+12 514869
    ## - data_channel_is_world          1 8.1130e+08 3.1636e+12 514869
    ## - num_self_hrefs                 1 8.6627e+08 3.1637e+12 514870
    ## - LDA_02                         1 9.1205e+08 3.1637e+12 514870
    ## - n_tokens_title                 1 9.1320e+08 3.1637e+12 514870
    ## - abs_title_subjectivity         1 9.2588e+08 3.1638e+12 514870
    ## - LDA_01                         1 9.4812e+08 3.1638e+12 514870
    ## - abs_title_sentiment_polarity   1 9.9757e+08 3.1638e+12 514871
    ## - data_channel_is_tech           1 1.0682e+09 3.1639e+12 514871
    ## - data_channel_is_socmed         1 1.1246e+09 3.1640e+12 514872
    ## - global_subjectivity            1 1.3620e+09 3.1642e+12 514874
    ## - data_channel_is_lifestyle      1 1.5092e+09 3.1643e+12 514875
    ## - data_channel_is_bus            1 1.9104e+09 3.1647e+12 514879
    ## - kw_min_avg                     1 2.1308e+09 3.1650e+12 514881
    ## - num_hrefs                      1 3.3362e+09 3.1662e+12 514891
    ## - data_channel_is_entertainment  1 3.6096e+09 3.1664e+12 514894
    ## - kw_max_avg                     1 5.6633e+09 3.1685e+12 514912
    ## - kw_avg_avg                     1 1.1743e+10 3.1746e+12 514965
    ## 
    ## Step:  AIC=514862.8
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + num_videos + 
    ##     average_token_length + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - num_videos                     1 2.0563e+08 3.1631e+12 514863
    ## - LDA_04                         1 2.1653e+08 3.1631e+12 514863
    ## <none>                                        3.1629e+12 514863
    ## - LDA_03                         1 2.8026e+08 3.1632e+12 514863
    ## - kw_max_min                     1 3.1851e+08 3.1632e+12 514864
    ## - kw_avg_min                     1 3.7471e+08 3.1633e+12 514864
    ## - min_positive_polarity          1 3.7563e+08 3.1633e+12 514864
    ## - self_reference_max_shares      1 5.6233e+08 3.1635e+12 514866
    ## - kw_min_min                     1 6.6337e+08 3.1636e+12 514867
    ## - average_token_length           1 6.9447e+08 3.1636e+12 514867
    ## - kw_avg_max                     1 7.0289e+08 3.1636e+12 514867
    ## - self_reference_min_shares      1 7.8094e+08 3.1637e+12 514868
    ## - num_self_hrefs                 1 8.0720e+08 3.1637e+12 514868
    ## - data_channel_is_world          1 8.3695e+08 3.1638e+12 514868
    ## - LDA_02                         1 8.8746e+08 3.1638e+12 514869
    ## - n_tokens_title                 1 9.0940e+08 3.1638e+12 514869
    ## - LDA_01                         1 9.2010e+08 3.1638e+12 514869
    ## - abs_title_subjectivity         1 9.2641e+08 3.1638e+12 514869
    ## - abs_title_sentiment_polarity   1 1.0155e+09 3.1639e+12 514870
    ## - data_channel_is_tech           1 1.0820e+09 3.1640e+12 514870
    ## - data_channel_is_socmed         1 1.1478e+09 3.1641e+12 514871
    ## - global_subjectivity            1 1.3536e+09 3.1643e+12 514873
    ## - data_channel_is_lifestyle      1 1.5232e+09 3.1644e+12 514874
    ## - data_channel_is_bus            1 1.9318e+09 3.1649e+12 514878
    ## - kw_min_avg                     1 2.1296e+09 3.1651e+12 514880
    ## - data_channel_is_entertainment  1 3.6056e+09 3.1665e+12 514892
    ## - num_hrefs                      1 3.8459e+09 3.1668e+12 514895
    ## - kw_max_avg                     1 5.7176e+09 3.1686e+12 514911
    ## - kw_avg_avg                     1 1.1880e+10 3.1748e+12 514965
    ## 
    ## Step:  AIC=514862.7
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - LDA_04                         1 2.1543e+08 3.1633e+12 514863
    ## <none>                                        3.1631e+12 514863
    ## - LDA_03                         1 2.4246e+08 3.1634e+12 514863
    ## - kw_max_min                     1 3.1845e+08 3.1634e+12 514863
    ## - kw_avg_min                     1 3.7076e+08 3.1635e+12 514864
    ## - min_positive_polarity          1 3.9619e+08 3.1635e+12 514864
    ## - self_reference_max_shares      1 6.0979e+08 3.1637e+12 514866
    ## - kw_avg_max                     1 6.2072e+08 3.1637e+12 514866
    ## - average_token_length           1 6.9345e+08 3.1638e+12 514867
    ## - kw_min_min                     1 7.3437e+08 3.1639e+12 514867
    ## - self_reference_min_shares      1 7.5300e+08 3.1639e+12 514867
    ## - num_self_hrefs                 1 7.7373e+08 3.1639e+12 514867
    ## - data_channel_is_world          1 8.3318e+08 3.1640e+12 514868
    ## - LDA_02                         1 8.8467e+08 3.1640e+12 514868
    ## - abs_title_subjectivity         1 9.2102e+08 3.1640e+12 514869
    ## - n_tokens_title                 1 9.2746e+08 3.1641e+12 514869
    ## - LDA_01                         1 9.2774e+08 3.1641e+12 514869
    ## - abs_title_sentiment_polarity   1 1.0257e+09 3.1642e+12 514870
    ## - data_channel_is_tech           1 1.0858e+09 3.1642e+12 514870
    ## - data_channel_is_socmed         1 1.1425e+09 3.1643e+12 514871
    ## - global_subjectivity            1 1.4100e+09 3.1645e+12 514873
    ## - data_channel_is_lifestyle      1 1.5233e+09 3.1647e+12 514874
    ## - data_channel_is_bus            1 1.9465e+09 3.1651e+12 514878
    ## - kw_min_avg                     1 2.1451e+09 3.1653e+12 514879
    ## - data_channel_is_entertainment  1 3.4943e+09 3.1666e+12 514891
    ## - num_hrefs                      1 3.9541e+09 3.1671e+12 514895
    ## - kw_max_avg                     1 5.6883e+09 3.1688e+12 514911
    ## - kw_avg_avg                     1 1.1799e+10 3.1749e+12 514964
    ## 
    ## Step:  AIC=514862.5
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - LDA_03                         1 8.9031e+07 3.1634e+12 514861
    ## <none>                                        3.1633e+12 514863
    ## - kw_max_min                     1 3.1278e+08 3.1637e+12 514863
    ## - kw_avg_min                     1 3.6256e+08 3.1637e+12 514864
    ## - min_positive_polarity          1 4.3751e+08 3.1638e+12 514864
    ## - kw_avg_max                     1 5.8334e+08 3.1639e+12 514866
    ## - self_reference_max_shares      1 6.0663e+08 3.1639e+12 514866
    ## - average_token_length           1 6.6807e+08 3.1640e+12 514866
    ## - LDA_02                         1 6.7279e+08 3.1640e+12 514866
    ## - LDA_01                         1 7.1230e+08 3.1641e+12 514867
    ## - self_reference_min_shares      1 7.5396e+08 3.1641e+12 514867
    ## - kw_min_min                     1 7.5612e+08 3.1641e+12 514867
    ## - num_self_hrefs                 1 8.0350e+08 3.1641e+12 514868
    ## - data_channel_is_world          1 8.8539e+08 3.1642e+12 514868
    ## - n_tokens_title                 1 8.9968e+08 3.1642e+12 514868
    ## - abs_title_subjectivity         1 9.2596e+08 3.1643e+12 514869
    ## - data_channel_is_socmed         1 1.0057e+09 3.1643e+12 514869
    ## - abs_title_sentiment_polarity   1 1.0344e+09 3.1644e+12 514870
    ## - global_subjectivity            1 1.3931e+09 3.1647e+12 514873
    ## - data_channel_is_tech           1 1.5788e+09 3.1649e+12 514874
    ## - data_channel_is_bus            1 1.7334e+09 3.1651e+12 514876
    ## - data_channel_is_lifestyle      1 1.7474e+09 3.1651e+12 514876
    ## - kw_min_avg                     1 2.1968e+09 3.1655e+12 514880
    ## - data_channel_is_entertainment  1 3.4551e+09 3.1668e+12 514891
    ## - num_hrefs                      1 4.0111e+09 3.1674e+12 514896
    ## - kw_max_avg                     1 5.7096e+09 3.1691e+12 514911
    ## - kw_avg_avg                     1 1.1879e+10 3.1752e+12 514965
    ## 
    ## Step:  AIC=514861.3
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + global_subjectivity + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## <none>                                        3.1634e+12 514861
    ## - kw_max_min                     1 3.0492e+08 3.1637e+12 514862
    ## - kw_avg_min                     1 3.5128e+08 3.1638e+12 514862
    ## - min_positive_polarity          1 4.6853e+08 3.1639e+12 514863
    ## - LDA_02                         1 5.8481e+08 3.1640e+12 514864
    ## - self_reference_max_shares      1 6.0641e+08 3.1640e+12 514865
    ## - kw_avg_max                     1 6.2662e+08 3.1641e+12 514865
    ## - average_token_length           1 6.3771e+08 3.1641e+12 514865
    ## - kw_min_min                     1 7.1304e+08 3.1641e+12 514866
    ## - self_reference_min_shares      1 7.5679e+08 3.1642e+12 514866
    ## - LDA_01                         1 7.5742e+08 3.1642e+12 514866
    ## - num_self_hrefs                 1 7.8865e+08 3.1642e+12 514866
    ## - data_channel_is_world          1 8.1332e+08 3.1642e+12 514866
    ## - n_tokens_title                 1 8.8993e+08 3.1643e+12 514867
    ## - abs_title_subjectivity         1 9.0983e+08 3.1643e+12 514867
    ## - data_channel_is_socmed         1 9.7782e+08 3.1644e+12 514868
    ## - abs_title_sentiment_polarity   1 1.0113e+09 3.1644e+12 514868
    ## - global_subjectivity            1 1.3665e+09 3.1648e+12 514871
    ## - data_channel_is_lifestyle      1 1.9920e+09 3.1654e+12 514877
    ## - kw_min_avg                     1 2.1342e+09 3.1656e+12 514878
    ## - data_channel_is_tech           1 2.1673e+09 3.1656e+12 514878
    ## - data_channel_is_bus            1 2.4427e+09 3.1659e+12 514881
    ## - data_channel_is_entertainment  1 3.3937e+09 3.1668e+12 514889
    ## - num_hrefs                      1 3.9580e+09 3.1674e+12 514894
    ## - kw_max_avg                     1 5.6289e+09 3.1691e+12 514909
    ## - kw_avg_avg                     1 1.1814e+10 3.1752e+12 514963

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
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity, 
    ##     data = train)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -24502  -2134  -1206   -135 836819 
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                    2.156e+02  7.021e+02   0.307  0.75875    
    ## n_tokens_title                 8.794e+01  3.149e+01   2.793  0.00523 ** 
    ## num_hrefs                      3.820e+01  6.486e+00   5.890 3.91e-09 ***
    ## num_self_hrefs                -4.935e+01  1.877e+01  -2.629  0.00857 ** 
    ## average_token_length          -2.482e+02  1.050e+02  -2.364  0.01808 *  
    ## data_channel_is_lifestyle     -1.446e+03  3.461e+02  -4.178 2.95e-05 ***
    ## data_channel_is_entertainment -1.486e+03  2.725e+02  -5.454 4.98e-08 ***
    ## data_channel_is_bus           -1.280e+03  2.767e+02  -4.627 3.73e-06 ***
    ## data_channel_is_socmed        -9.990e+02  3.413e+02  -2.927  0.00342 ** 
    ## data_channel_is_tech          -1.195e+03  2.742e+02  -4.358 1.32e-05 ***
    ## data_channel_is_world         -9.792e+02  3.668e+02  -2.670  0.00759 ** 
    ## kw_min_min                     2.869e+00  1.148e+00   2.500  0.01243 *  
    ## kw_max_min                     8.191e-02  5.011e-02   1.635  0.10212    
    ## kw_avg_min                    -5.363e-01  3.056e-01  -1.755  0.07934 .  
    ## kw_avg_max                    -1.730e-03  7.383e-04  -2.343  0.01912 *  
    ## kw_min_avg                    -3.402e-01  7.865e-02  -4.325 1.53e-05 ***
    ## kw_max_avg                    -1.909e-01  2.717e-02  -7.024 2.21e-12 ***
    ## kw_avg_avg                     1.543e+00  1.517e-01  10.175  < 2e-16 ***
    ## self_reference_min_shares      9.853e-03  3.826e-03   2.575  0.01002 *  
    ## self_reference_max_shares      4.219e-03  1.830e-03   2.305  0.02116 *  
    ## LDA_01                        -9.721e+02  3.773e+02  -2.576  0.00999 ** 
    ## LDA_02                        -9.860e+02  4.355e+02  -2.264  0.02359 *  
    ## global_subjectivity            2.588e+03  7.478e+02   3.461  0.00054 ***
    ## min_positive_polarity         -1.964e+03  9.694e+02  -2.026  0.04274 *  
    ## abs_title_subjectivity         1.061e+03  3.756e+02   2.824  0.00475 ** 
    ## abs_title_sentiment_polarity   9.314e+02  3.129e+02   2.977  0.00291 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10680 on 27724 degrees of freedom
    ## Multiple R-squared:  0.02367,    Adjusted R-squared:  0.02279 
    ## F-statistic: 26.89 on 25 and 27724 DF,  p-value: < 2.2e-16

Calculate the predicted mean square error on the train set:

``` r
train.pred <- predict(lm.step, train)
mean((train.pred - train$shares)^2)
```

    ## [1] 113997556

### On test set

Calculate the predicted mean square error on the test set:

``` r
test.pred <- predict(lm.step, test)
mean((test.pred - test$shares)^2)
```

    ## [1] 175191776

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
