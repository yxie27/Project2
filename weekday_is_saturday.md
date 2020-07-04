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

![](weekday_is_saturday_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
hist(log(train$shares))
```

![](weekday_is_saturday_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

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
    ## weekday_is_saturday           Min.   :0.00000    1st Qu.:0.00000    Median :0.00000    Mean   :0.06123   
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
    ## weekday_is_saturday           3rd Qu.:0.00000    Max.   :1.00000

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
    ##           Mean of squared residuals: 117010297
    ##                     % Var explained: -0.21

``` r
#variable importance measures
importance(rf)
```

    ##                                   %IncMSE IncNodePurity
    ## n_tokens_title                 2.39868924   43480578095
    ## n_tokens_content               4.17458156   43879619329
    ## n_unique_tokens                2.98666319   37047208349
    ## n_non_stop_words               5.87858639   42059322096
    ## n_non_stop_unique_tokens       5.18634164   52828056663
    ## num_hrefs                      4.20951267   68310015157
    ## num_self_hrefs                 3.33067032   43256352533
    ## num_imgs                       0.80772714   37113754106
    ## num_videos                     4.98048679   39643369100
    ## average_token_length           4.03867237   56390223622
    ## num_keywords                   3.18110140   13154272539
    ## data_channel_is_lifestyle      2.09009845    5743853855
    ## data_channel_is_entertainment  4.67886910    3188354192
    ## data_channel_is_bus            2.47833757   17762627792
    ## data_channel_is_socmed         4.15972152    2284446218
    ## data_channel_is_tech          10.15540685    1612042677
    ## data_channel_is_world          2.69819614    2925097765
    ## kw_min_min                     3.94280611   17265806162
    ## kw_max_min                     2.45620882   75902863985
    ## kw_avg_min                     3.62906600   96066434835
    ## kw_min_max                     5.41356556   50022547229
    ## kw_max_max                     4.01888954   44533972269
    ## kw_avg_max                     4.67811368  125433688772
    ## kw_min_avg                     4.55385952   45608837475
    ## kw_max_avg                     5.75373822  176794495983
    ## kw_avg_avg                     4.92168936  189706829400
    ## self_reference_min_shares      7.90710624   58250905215
    ## self_reference_max_shares      5.90387600   78311997457
    ## self_reference_avg_sharess     4.89564256  160787244827
    ## LDA_00                         4.34614874  108516366398
    ## LDA_01                         8.93515732   56853566099
    ## LDA_02                         2.16931963   69965270533
    ## LDA_03                         4.45493085   95581864668
    ## LDA_04                         4.03440657   86071224174
    ## global_subjectivity            4.62427058   99399187197
    ## global_sentiment_polarity      6.99196065   47344320050
    ## global_rate_positive_words     3.43663563   59818358351
    ## global_rate_negative_words     4.12603247   44437520313
    ## rate_positive_words            7.81427010   30187837600
    ## rate_negative_words           10.82238275   32601502840
    ## avg_positive_polarity          2.37776944   49876613848
    ## min_positive_polarity          0.08613331   20155547754
    ## max_positive_polarity          2.88447727   15264707558
    ## avg_negative_polarity          3.52244951   57707915803
    ## min_negative_polarity          3.26584968   25386099130
    ## max_negative_polarity          2.74516642   29341986483
    ## title_subjectivity             3.41337135   65239782673
    ## title_sentiment_polarity       3.58471925   66812428695
    ## abs_title_subjectivity         6.00243334   16002493638
    ## abs_title_sentiment_polarity   2.20753173   32748956374
    ## weekday_is_saturday            3.05510819   32987959622

``` r
#draw dotplot of variable importance as measured by Random Forest
varImpPlot(rf)
```

![](weekday_is_saturday_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Calculate the predicted mean square error on the train set:

``` r
train.pred <- predict(rf, train)
mean((train.pred - train$shares)^2)
```

    ## [1] 27260429

### On test set

``` r
rf.test <- predict(rf, newdata = test)
mean((test$shares-rf.test)^2)
```

    ## [1] 177620024

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

    ## Start:  AIC=514895.2
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
    ##     weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - global_sentiment_polarity      1 2.0994e+05 3.1614e+12 514893
    ## - title_subjectivity             1 2.9878e+05 3.1614e+12 514893
    ## - n_unique_tokens                1 1.4971e+06 3.1614e+12 514893
    ## - global_rate_negative_words     1 2.0853e+06 3.1614e+12 514893
    ## - LDA_00                         1 5.6067e+06 3.1614e+12 514893
    ## - LDA_04                         1 5.6119e+06 3.1614e+12 514893
    ## - LDA_03                         1 5.6140e+06 3.1614e+12 514893
    ## - LDA_02                         1 5.6189e+06 3.1614e+12 514893
    ## - LDA_01                         1 5.6198e+06 3.1614e+12 514893
    ## - avg_negative_polarity          1 6.6907e+06 3.1614e+12 514893
    ## - max_positive_polarity          1 8.0578e+06 3.1614e+12 514893
    ## - rate_positive_words            1 8.9526e+06 3.1614e+12 514893
    ## - num_keywords                   1 9.5492e+06 3.1614e+12 514893
    ## - rate_negative_words            1 9.7406e+06 3.1614e+12 514893
    ## - n_non_stop_words               1 1.3192e+07 3.1614e+12 514893
    ## - n_tokens_content               1 1.5228e+07 3.1614e+12 514893
    ## - min_negative_polarity          1 2.3030e+07 3.1614e+12 514893
    ## - avg_positive_polarity          1 2.3246e+07 3.1614e+12 514893
    ## - title_sentiment_polarity       1 3.1902e+07 3.1614e+12 514893
    ## - self_reference_avg_sharess     1 4.4068e+07 3.1614e+12 514894
    ## - max_negative_polarity          1 4.8840e+07 3.1614e+12 514894
    ## - global_rate_positive_words     1 5.7433e+07 3.1614e+12 514894
    ## - n_non_stop_unique_tokens       1 7.6782e+07 3.1614e+12 514894
    ## - kw_min_min                     1 8.7457e+07 3.1615e+12 514894
    ## - kw_min_max                     1 9.6991e+07 3.1615e+12 514894
    ## - kw_max_max                     1 1.1376e+08 3.1615e+12 514894
    ## - average_token_length           1 1.6368e+08 3.1615e+12 514895
    ## - num_imgs                       1 1.7595e+08 3.1615e+12 514895
    ## <none>                                        3.1614e+12 514895
    ## - self_reference_max_shares      1 2.3312e+08 3.1616e+12 514895
    ## - min_positive_polarity          1 2.3731e+08 3.1616e+12 514895
    ## - num_videos                     1 2.5210e+08 3.1616e+12 514895
    ## - kw_avg_max                     1 2.8516e+08 3.1616e+12 514896
    ## - self_reference_min_shares      1 3.2334e+08 3.1617e+12 514896
    ## - kw_max_min                     1 3.2348e+08 3.1617e+12 514896
    ## - kw_avg_min                     1 3.7806e+08 3.1617e+12 514896
    ## - abs_title_sentiment_polarity   1 4.9729e+08 3.1619e+12 514898
    ## - weekday_is_saturday            1 6.4929e+08 3.1620e+12 514899
    ## - abs_title_subjectivity         1 7.5853e+08 3.1621e+12 514900
    ## - data_channel_is_world          1 7.7324e+08 3.1621e+12 514900
    ## - n_tokens_title                 1 8.7724e+08 3.1622e+12 514901
    ## - num_self_hrefs                 1 9.2317e+08 3.1623e+12 514901
    ## - global_subjectivity            1 9.7727e+08 3.1623e+12 514902
    ## - data_channel_is_socmed         1 1.0042e+09 3.1624e+12 514902
    ## - data_channel_is_tech           1 1.0239e+09 3.1624e+12 514902
    ## - data_channel_is_lifestyle      1 1.4774e+09 3.1628e+12 514906
    ## - data_channel_is_bus            1 1.8587e+09 3.1632e+12 514909
    ## - kw_min_avg                     1 1.9124e+09 3.1633e+12 514910
    ## - num_hrefs                      1 2.9127e+09 3.1643e+12 514919
    ## - data_channel_is_entertainment  1 3.2682e+09 3.1646e+12 514922
    ## - kw_max_avg                     1 5.6087e+09 3.1670e+12 514942
    ## - kw_avg_avg                     1 1.1358e+10 3.1727e+12 514993
    ## 
    ## Step:  AIC=514893.2
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
    ##     abs_title_sentiment_polarity + weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - title_subjectivity             1 3.1038e+05 3.1614e+12 514891
    ## - n_unique_tokens                1 1.4992e+06 3.1614e+12 514891
    ## - global_rate_negative_words     1 2.8184e+06 3.1614e+12 514891
    ## - LDA_00                         1 5.6161e+06 3.1614e+12 514891
    ## - LDA_04                         1 5.6212e+06 3.1614e+12 514891
    ## - LDA_03                         1 5.6234e+06 3.1614e+12 514891
    ## - LDA_02                         1 5.6283e+06 3.1614e+12 514891
    ## - LDA_01                         1 5.6292e+06 3.1614e+12 514891
    ## - max_positive_polarity          1 8.0907e+06 3.1614e+12 514891
    ## - avg_negative_polarity          1 8.4354e+06 3.1614e+12 514891
    ## - rate_positive_words            1 8.9624e+06 3.1614e+12 514891
    ## - num_keywords                   1 9.5306e+06 3.1614e+12 514891
    ## - rate_negative_words            1 9.6276e+06 3.1614e+12 514891
    ## - n_non_stop_words               1 1.3192e+07 3.1614e+12 514891
    ## - n_tokens_content               1 1.5115e+07 3.1614e+12 514891
    ## - min_negative_polarity          1 2.3152e+07 3.1614e+12 514891
    ## - avg_positive_polarity          1 3.1937e+07 3.1614e+12 514891
    ## - title_sentiment_polarity       1 3.2571e+07 3.1614e+12 514891
    ## - self_reference_avg_sharess     1 4.4051e+07 3.1614e+12 514892
    ## - max_negative_polarity          1 5.1258e+07 3.1614e+12 514892
    ## - global_rate_positive_words     1 6.5869e+07 3.1614e+12 514892
    ## - n_non_stop_unique_tokens       1 7.6651e+07 3.1614e+12 514892
    ## - kw_min_min                     1 8.7413e+07 3.1615e+12 514892
    ## - kw_min_max                     1 9.7106e+07 3.1615e+12 514892
    ## - kw_max_max                     1 1.1380e+08 3.1615e+12 514892
    ## - average_token_length           1 1.6390e+08 3.1615e+12 514893
    ## - num_imgs                       1 1.7603e+08 3.1615e+12 514893
    ## <none>                                        3.1614e+12 514893
    ## - self_reference_max_shares      1 2.3304e+08 3.1616e+12 514893
    ## - min_positive_polarity          1 2.4324e+08 3.1616e+12 514893
    ## - num_videos                     1 2.5197e+08 3.1616e+12 514893
    ## - kw_avg_max                     1 2.8507e+08 3.1616e+12 514894
    ## - self_reference_min_shares      1 3.2333e+08 3.1617e+12 514894
    ## - kw_max_min                     1 3.2336e+08 3.1617e+12 514894
    ## - kw_avg_min                     1 3.7793e+08 3.1617e+12 514895
    ## - abs_title_sentiment_polarity   1 4.9733e+08 3.1619e+12 514896
    ## - weekday_is_saturday            1 6.4943e+08 3.1620e+12 514897
    ## - abs_title_subjectivity         1 7.5949e+08 3.1621e+12 514898
    ## - data_channel_is_world          1 7.7303e+08 3.1621e+12 514898
    ## - n_tokens_title                 1 8.7728e+08 3.1622e+12 514899
    ## - num_self_hrefs                 1 9.2320e+08 3.1623e+12 514899
    ## - data_channel_is_socmed         1 1.0041e+09 3.1624e+12 514900
    ## - global_subjectivity            1 1.0188e+09 3.1624e+12 514900
    ## - data_channel_is_tech           1 1.0237e+09 3.1624e+12 514900
    ## - data_channel_is_lifestyle      1 1.4773e+09 3.1628e+12 514904
    ## - data_channel_is_bus            1 1.8587e+09 3.1632e+12 514907
    ## - kw_min_avg                     1 1.9122e+09 3.1633e+12 514908
    ## - num_hrefs                      1 2.9208e+09 3.1643e+12 514917
    ## - data_channel_is_entertainment  1 3.2681e+09 3.1646e+12 514920
    ## - kw_max_avg                     1 5.6095e+09 3.1670e+12 514940
    ## - kw_avg_avg                     1 1.1360e+10 3.1727e+12 514991
    ## 
    ## Step:  AIC=514891.2
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
    ##     weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_unique_tokens                1 1.5092e+06 3.1614e+12 514889
    ## - global_rate_negative_words     1 2.8275e+06 3.1614e+12 514889
    ## - LDA_00                         1 5.5919e+06 3.1614e+12 514889
    ## - LDA_04                         1 5.5971e+06 3.1614e+12 514889
    ## - LDA_03                         1 5.5992e+06 3.1614e+12 514889
    ## - LDA_02                         1 5.6042e+06 3.1614e+12 514889
    ## - LDA_01                         1 5.6050e+06 3.1614e+12 514889
    ## - max_positive_polarity          1 8.0710e+06 3.1614e+12 514889
    ## - avg_negative_polarity          1 8.4213e+06 3.1614e+12 514889
    ## - rate_positive_words            1 8.9544e+06 3.1614e+12 514889
    ## - num_keywords                   1 9.5013e+06 3.1614e+12 514889
    ## - rate_negative_words            1 9.6115e+06 3.1614e+12 514889
    ## - n_non_stop_words               1 1.3151e+07 3.1614e+12 514889
    ## - n_tokens_content               1 1.5048e+07 3.1614e+12 514889
    ## - min_negative_polarity          1 2.3198e+07 3.1614e+12 514889
    ## - avg_positive_polarity          1 3.1723e+07 3.1614e+12 514889
    ## - title_sentiment_polarity       1 3.3770e+07 3.1614e+12 514889
    ## - self_reference_avg_sharess     1 4.4066e+07 3.1614e+12 514890
    ## - max_negative_polarity          1 5.1258e+07 3.1614e+12 514890
    ## - global_rate_positive_words     1 6.6076e+07 3.1614e+12 514890
    ## - n_non_stop_unique_tokens       1 7.6671e+07 3.1614e+12 514890
    ## - kw_min_min                     1 8.7493e+07 3.1615e+12 514890
    ## - kw_min_max                     1 9.7153e+07 3.1615e+12 514890
    ## - kw_max_max                     1 1.1369e+08 3.1615e+12 514890
    ## - average_token_length           1 1.6392e+08 3.1615e+12 514891
    ## - num_imgs                       1 1.7591e+08 3.1615e+12 514891
    ## <none>                                        3.1614e+12 514891
    ## - self_reference_max_shares      1 2.3304e+08 3.1616e+12 514891
    ## - min_positive_polarity          1 2.4353e+08 3.1616e+12 514891
    ## - num_videos                     1 2.5190e+08 3.1616e+12 514891
    ## - kw_avg_max                     1 2.8517e+08 3.1616e+12 514892
    ## - self_reference_min_shares      1 3.2344e+08 3.1617e+12 514892
    ## - kw_max_min                     1 3.2350e+08 3.1617e+12 514892
    ## - kw_avg_min                     1 3.7816e+08 3.1617e+12 514893
    ## - weekday_is_saturday            1 6.4972e+08 3.1620e+12 514895
    ## - data_channel_is_world          1 7.7300e+08 3.1621e+12 514896
    ## - abs_title_sentiment_polarity   1 8.1807e+08 3.1622e+12 514896
    ## - abs_title_subjectivity         1 8.5252e+08 3.1622e+12 514897
    ## - n_tokens_title                 1 8.7698e+08 3.1622e+12 514897
    ## - num_self_hrefs                 1 9.2332e+08 3.1623e+12 514897
    ## - data_channel_is_socmed         1 1.0039e+09 3.1624e+12 514898
    ## - data_channel_is_tech           1 1.0238e+09 3.1624e+12 514898
    ## - global_subjectivity            1 1.0279e+09 3.1624e+12 514898
    ## - data_channel_is_lifestyle      1 1.4772e+09 3.1628e+12 514902
    ## - data_channel_is_bus            1 1.8585e+09 3.1632e+12 514905
    ## - kw_min_avg                     1 1.9119e+09 3.1633e+12 514906
    ## - num_hrefs                      1 2.9206e+09 3.1643e+12 514915
    ## - data_channel_is_entertainment  1 3.2679e+09 3.1646e+12 514918
    ## - kw_max_avg                     1 5.6092e+09 3.1670e+12 514938
    ## - kw_avg_avg                     1 1.1360e+10 3.1727e+12 514989
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
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + rate_negative_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - global_rate_negative_words     1 2.9485e+06 3.1614e+12 514887
    ## - LDA_00                         1 5.1605e+06 3.1614e+12 514887
    ## - LDA_04                         1 5.1655e+06 3.1614e+12 514887
    ## - LDA_03                         1 5.1676e+06 3.1614e+12 514887
    ## - LDA_02                         1 5.1723e+06 3.1614e+12 514887
    ## - LDA_01                         1 5.1731e+06 3.1614e+12 514887
    ## - avg_negative_polarity          1 8.4203e+06 3.1614e+12 514887
    ## - rate_positive_words            1 9.0929e+06 3.1614e+12 514887
    ## - max_positive_polarity          1 9.3395e+06 3.1614e+12 514887
    ## - num_keywords                   1 9.6009e+06 3.1614e+12 514887
    ## - rate_negative_words            1 9.7561e+06 3.1614e+12 514887
    ## - n_non_stop_words               1 1.3057e+07 3.1614e+12 514887
    ## - n_tokens_content               1 2.4878e+07 3.1614e+12 514887
    ## - min_negative_polarity          1 2.4975e+07 3.1614e+12 514887
    ## - avg_positive_polarity          1 3.1824e+07 3.1614e+12 514887
    ## - title_sentiment_polarity       1 3.4052e+07 3.1614e+12 514887
    ## - self_reference_avg_sharess     1 4.4310e+07 3.1614e+12 514888
    ## - max_negative_polarity          1 4.9935e+07 3.1614e+12 514888
    ## - global_rate_positive_words     1 6.7811e+07 3.1614e+12 514888
    ## - kw_min_min                     1 8.7463e+07 3.1615e+12 514888
    ## - kw_min_max                     1 9.6912e+07 3.1615e+12 514888
    ## - kw_max_max                     1 1.1307e+08 3.1615e+12 514888
    ## - num_imgs                       1 1.7613e+08 3.1615e+12 514889
    ## - average_token_length           1 1.8712e+08 3.1616e+12 514889
    ## <none>                                        3.1614e+12 514889
    ## - self_reference_max_shares      1 2.3369e+08 3.1616e+12 514889
    ## - n_non_stop_unique_tokens       1 2.4148e+08 3.1616e+12 514889
    ## - num_videos                     1 2.5062e+08 3.1616e+12 514889
    ## - min_positive_polarity          1 2.5778e+08 3.1616e+12 514889
    ## - kw_avg_max                     1 2.8681e+08 3.1617e+12 514890
    ## - kw_max_min                     1 3.2343e+08 3.1617e+12 514890
    ## - self_reference_min_shares      1 3.2398e+08 3.1617e+12 514890
    ## - kw_avg_min                     1 3.7793e+08 3.1617e+12 514891
    ## - weekday_is_saturday            1 6.5022e+08 3.1620e+12 514893
    ## - data_channel_is_world          1 7.7315e+08 3.1621e+12 514894
    ## - abs_title_sentiment_polarity   1 8.1686e+08 3.1622e+12 514894
    ## - abs_title_subjectivity         1 8.5433e+08 3.1622e+12 514895
    ## - n_tokens_title                 1 8.7696e+08 3.1622e+12 514895
    ## - num_self_hrefs                 1 9.2427e+08 3.1623e+12 514895
    ## - data_channel_is_socmed         1 1.0024e+09 3.1624e+12 514896
    ## - data_channel_is_tech           1 1.0233e+09 3.1624e+12 514896
    ## - global_subjectivity            1 1.0269e+09 3.1624e+12 514896
    ## - data_channel_is_lifestyle      1 1.4795e+09 3.1628e+12 514900
    ## - data_channel_is_bus            1 1.8595e+09 3.1632e+12 514904
    ## - kw_min_avg                     1 1.9151e+09 3.1633e+12 514904
    ## - num_hrefs                      1 2.9371e+09 3.1643e+12 514913
    ## - data_channel_is_entertainment  1 3.2709e+09 3.1646e+12 514916
    ## - kw_max_avg                     1 5.6130e+09 3.1670e+12 514936
    ## - kw_avg_avg                     1 1.1364e+10 3.1727e+12 514987
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
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - LDA_00                         1 5.1896e+06 3.1614e+12 514885
    ## - LDA_04                         1 5.1946e+06 3.1614e+12 514885
    ## - LDA_03                         1 5.1967e+06 3.1614e+12 514885
    ## - LDA_02                         1 5.2014e+06 3.1614e+12 514885
    ## - LDA_01                         1 5.2022e+06 3.1614e+12 514885
    ## - avg_negative_polarity          1 8.3109e+06 3.1614e+12 514885
    ## - rate_negative_words            1 8.6110e+06 3.1614e+12 514885
    ## - max_positive_polarity          1 8.9714e+06 3.1614e+12 514885
    ## - rate_positive_words            1 9.5550e+06 3.1614e+12 514885
    ## - num_keywords                   1 9.6969e+06 3.1614e+12 514885
    ## - n_non_stop_words               1 1.3086e+07 3.1614e+12 514885
    ## - min_negative_polarity          1 2.4550e+07 3.1614e+12 514885
    ## - n_tokens_content               1 2.5366e+07 3.1614e+12 514885
    ## - avg_positive_polarity          1 3.2050e+07 3.1614e+12 514886
    ## - title_sentiment_polarity       1 3.5638e+07 3.1614e+12 514886
    ## - self_reference_avg_sharess     1 4.4421e+07 3.1614e+12 514886
    ## - max_negative_polarity          1 5.1259e+07 3.1614e+12 514886
    ## - kw_min_min                     1 8.6948e+07 3.1615e+12 514886
    ## - kw_min_max                     1 9.6979e+07 3.1615e+12 514886
    ## - kw_max_max                     1 1.1358e+08 3.1615e+12 514886
    ## - global_rate_positive_words     1 1.4668e+08 3.1615e+12 514887
    ## - num_imgs                       1 1.7494e+08 3.1615e+12 514887
    ## - average_token_length           1 1.8554e+08 3.1616e+12 514887
    ## <none>                                        3.1614e+12 514887
    ## - self_reference_max_shares      1 2.3416e+08 3.1616e+12 514887
    ## - n_non_stop_unique_tokens       1 2.4070e+08 3.1616e+12 514887
    ## - num_videos                     1 2.4776e+08 3.1616e+12 514887
    ## - min_positive_polarity          1 2.5535e+08 3.1616e+12 514887
    ## - kw_avg_max                     1 2.8719e+08 3.1617e+12 514888
    ## - kw_max_min                     1 3.2345e+08 3.1617e+12 514888
    ## - self_reference_min_shares      1 3.2384e+08 3.1617e+12 514888
    ## - kw_avg_min                     1 3.7834e+08 3.1617e+12 514889
    ## - weekday_is_saturday            1 6.5054e+08 3.1620e+12 514891
    ## - data_channel_is_world          1 7.7030e+08 3.1621e+12 514892
    ## - abs_title_sentiment_polarity   1 8.1520e+08 3.1622e+12 514892
    ## - abs_title_subjectivity         1 8.5938e+08 3.1622e+12 514893
    ## - n_tokens_title                 1 8.7901e+08 3.1622e+12 514893
    ## - num_self_hrefs                 1 9.2383e+08 3.1623e+12 514893
    ## - data_channel_is_socmed         1 1.0001e+09 3.1624e+12 514894
    ## - data_channel_is_tech           1 1.0208e+09 3.1624e+12 514894
    ## - global_subjectivity            1 1.0263e+09 3.1624e+12 514894
    ## - data_channel_is_lifestyle      1 1.4769e+09 3.1628e+12 514898
    ## - data_channel_is_bus            1 1.8565e+09 3.1632e+12 514902
    ## - kw_min_avg                     1 1.9149e+09 3.1633e+12 514902
    ## - num_hrefs                      1 2.9638e+09 3.1643e+12 514911
    ## - data_channel_is_entertainment  1 3.2689e+09 3.1646e+12 514914
    ## - kw_max_avg                     1 5.6111e+09 3.1670e+12 514934
    ## - kw_avg_avg                     1 1.1363e+10 3.1727e+12 514985
    ## 
    ## Step:  AIC=514885.3
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
    ##     weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - avg_negative_polarity          1 8.2608e+06 3.1614e+12 514883
    ## - max_positive_polarity          1 8.7720e+06 3.1614e+12 514883
    ## - num_keywords                   1 9.5246e+06 3.1614e+12 514883
    ## - rate_negative_words            1 9.8092e+06 3.1614e+12 514883
    ## - rate_positive_words            1 1.4499e+07 3.1614e+12 514883
    ## - n_tokens_content               1 2.4451e+07 3.1614e+12 514883
    ## - min_negative_polarity          1 2.4511e+07 3.1614e+12 514883
    ## - avg_positive_polarity          1 3.1656e+07 3.1614e+12 514884
    ## - title_sentiment_polarity       1 3.5524e+07 3.1614e+12 514884
    ## - self_reference_avg_sharess     1 4.4662e+07 3.1614e+12 514884
    ## - max_negative_polarity          1 5.1303e+07 3.1614e+12 514884
    ## - kw_min_min                     1 8.7039e+07 3.1615e+12 514884
    ## - kw_min_max                     1 9.6750e+07 3.1615e+12 514884
    ## - kw_max_max                     1 1.1362e+08 3.1615e+12 514884
    ## - global_rate_positive_words     1 1.4657e+08 3.1615e+12 514885
    ## - num_imgs                       1 1.7550e+08 3.1615e+12 514885
    ## - LDA_04                         1 2.0218e+08 3.1616e+12 514885
    ## - average_token_length           1 2.0446e+08 3.1616e+12 514885
    ## <none>                                        3.1614e+12 514885
    ## - self_reference_max_shares      1 2.3454e+08 3.1616e+12 514885
    ## - n_non_stop_words               1 2.3529e+08 3.1616e+12 514885
    ## - n_non_stop_unique_tokens       1 2.3816e+08 3.1616e+12 514885
    ## - num_videos                     1 2.4757e+08 3.1616e+12 514885
    ## - min_positive_polarity          1 2.5574e+08 3.1616e+12 514886
    ## - kw_avg_max                     1 2.8679e+08 3.1617e+12 514886
    ## - kw_max_min                     1 3.2389e+08 3.1617e+12 514886
    ## - self_reference_min_shares      1 3.2443e+08 3.1617e+12 514886
    ## - LDA_03                         1 3.3552e+08 3.1617e+12 514886
    ## - kw_avg_min                     1 3.7882e+08 3.1618e+12 514887
    ## - weekday_is_saturday            1 6.5095e+08 3.1620e+12 514889
    ## - data_channel_is_world          1 7.6989e+08 3.1621e+12 514890
    ## - abs_title_sentiment_polarity   1 8.1595e+08 3.1622e+12 514890
    ## - abs_title_subjectivity         1 8.5875e+08 3.1622e+12 514891
    ## - n_tokens_title                 1 8.7888e+08 3.1623e+12 514891
    ## - num_self_hrefs                 1 9.2698e+08 3.1623e+12 514891
    ## - LDA_02                         1 9.4773e+08 3.1623e+12 514892
    ## - LDA_01                         1 9.6708e+08 3.1623e+12 514892
    ## - data_channel_is_socmed         1 1.0038e+09 3.1624e+12 514892
    ## - global_subjectivity            1 1.0213e+09 3.1624e+12 514892
    ## - data_channel_is_tech           1 1.0230e+09 3.1624e+12 514892
    ## - data_channel_is_lifestyle      1 1.4793e+09 3.1629e+12 514896
    ## - data_channel_is_bus            1 1.8601e+09 3.1632e+12 514900
    ## - kw_min_avg                     1 1.9179e+09 3.1633e+12 514900
    ## - num_hrefs                      1 2.9803e+09 3.1644e+12 514909
    ## - data_channel_is_entertainment  1 3.2731e+09 3.1646e+12 514912
    ## - kw_max_avg                     1 5.6148e+09 3.1670e+12 514933
    ## - kw_avg_avg                     1 1.1370e+10 3.1727e+12 514983
    ## 
    ## Step:  AIC=514883.3
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
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - max_positive_polarity          1 8.5105e+06 3.1614e+12 514881
    ## - rate_negative_words            1 9.1614e+06 3.1614e+12 514881
    ## - num_keywords                   1 9.5365e+06 3.1614e+12 514881
    ## - rate_positive_words            1 1.4408e+07 3.1614e+12 514881
    ## - min_negative_polarity          1 1.8696e+07 3.1614e+12 514882
    ## - n_tokens_content               1 3.0049e+07 3.1614e+12 514882
    ## - avg_positive_polarity          1 3.1162e+07 3.1614e+12 514882
    ## - title_sentiment_polarity       1 3.9299e+07 3.1614e+12 514882
    ## - self_reference_avg_sharess     1 4.4977e+07 3.1614e+12 514882
    ## - max_negative_polarity          1 5.7750e+07 3.1614e+12 514882
    ## - kw_min_min                     1 8.7315e+07 3.1615e+12 514882
    ## - kw_min_max                     1 9.6196e+07 3.1615e+12 514882
    ## - kw_max_max                     1 1.1270e+08 3.1615e+12 514882
    ## - global_rate_positive_words     1 1.4593e+08 3.1615e+12 514883
    ## - num_imgs                       1 1.7355e+08 3.1616e+12 514883
    ## - LDA_04                         1 2.0124e+08 3.1616e+12 514883
    ## - average_token_length           1 2.0366e+08 3.1616e+12 514883
    ## <none>                                        3.1614e+12 514883
    ## - self_reference_max_shares      1 2.3506e+08 3.1616e+12 514883
    ## - n_non_stop_words               1 2.3613e+08 3.1616e+12 514883
    ## - n_non_stop_unique_tokens       1 2.3899e+08 3.1616e+12 514883
    ## - num_videos                     1 2.4338e+08 3.1616e+12 514883
    ## - min_positive_polarity          1 2.5255e+08 3.1616e+12 514884
    ## - kw_avg_max                     1 2.8891e+08 3.1617e+12 514884
    ## - kw_max_min                     1 3.2335e+08 3.1617e+12 514884
    ## - self_reference_min_shares      1 3.2530e+08 3.1617e+12 514884
    ## - LDA_03                         1 3.3582e+08 3.1617e+12 514884
    ## - kw_avg_min                     1 3.7794e+08 3.1618e+12 514885
    ## - weekday_is_saturday            1 6.5076e+08 3.1620e+12 514887
    ## - data_channel_is_world          1 7.6779e+08 3.1622e+12 514888
    ## - abs_title_sentiment_polarity   1 8.0830e+08 3.1622e+12 514888
    ## - abs_title_subjectivity         1 8.5698e+08 3.1622e+12 514889
    ## - n_tokens_title                 1 8.7871e+08 3.1623e+12 514889
    ## - num_self_hrefs                 1 9.2931e+08 3.1623e+12 514889
    ## - LDA_02                         1 9.4477e+08 3.1623e+12 514890
    ## - LDA_01                         1 9.6903e+08 3.1624e+12 514890
    ## - data_channel_is_socmed         1 1.0010e+09 3.1624e+12 514890
    ## - global_subjectivity            1 1.0198e+09 3.1624e+12 514890
    ## - data_channel_is_tech           1 1.0201e+09 3.1624e+12 514890
    ## - data_channel_is_lifestyle      1 1.4767e+09 3.1629e+12 514894
    ## - data_channel_is_bus            1 1.8562e+09 3.1632e+12 514898
    ## - kw_min_avg                     1 1.9169e+09 3.1633e+12 514898
    ## - num_hrefs                      1 2.9788e+09 3.1644e+12 514907
    ## - data_channel_is_entertainment  1 3.2839e+09 3.1647e+12 514910
    ## - kw_max_avg                     1 5.6196e+09 3.1670e+12 514931
    ## - kw_avg_avg                     1 1.1373e+10 3.1728e+12 514981
    ## 
    ## Step:  AIC=514881.4
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
    ##     avg_positive_polarity + min_positive_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - num_keywords                   1 9.3584e+06 3.1614e+12 514880
    ## - rate_negative_words            1 1.0602e+07 3.1614e+12 514880
    ## - rate_positive_words            1 1.7203e+07 3.1614e+12 514880
    ## - min_negative_polarity          1 2.1869e+07 3.1614e+12 514880
    ## - avg_positive_polarity          1 2.3241e+07 3.1614e+12 514880
    ## - n_tokens_content               1 3.6264e+07 3.1614e+12 514880
    ## - title_sentiment_polarity       1 3.8987e+07 3.1614e+12 514880
    ## - self_reference_avg_sharess     1 4.5214e+07 3.1614e+12 514880
    ## - max_negative_polarity          1 5.5248e+07 3.1614e+12 514880
    ## - kw_min_min                     1 8.7427e+07 3.1615e+12 514880
    ## - kw_min_max                     1 9.6198e+07 3.1615e+12 514880
    ## - kw_max_max                     1 1.1183e+08 3.1615e+12 514880
    ## - global_rate_positive_words     1 1.3826e+08 3.1615e+12 514881
    ## - num_imgs                       1 1.7350e+08 3.1616e+12 514881
    ## - LDA_04                         1 2.0126e+08 3.1616e+12 514881
    ## - average_token_length           1 2.0830e+08 3.1616e+12 514881
    ## <none>                                        3.1614e+12 514881
    ## - n_non_stop_words               1 2.3066e+08 3.1616e+12 514881
    ## - n_non_stop_unique_tokens       1 2.3349e+08 3.1616e+12 514881
    ## - self_reference_max_shares      1 2.3542e+08 3.1616e+12 514881
    ## - num_videos                     1 2.4678e+08 3.1616e+12 514882
    ## - kw_avg_max                     1 2.8880e+08 3.1617e+12 514882
    ## - min_positive_polarity          1 3.1133e+08 3.1617e+12 514882
    ## - kw_max_min                     1 3.2241e+08 3.1617e+12 514882
    ## - self_reference_min_shares      1 3.2560e+08 3.1617e+12 514882
    ## - LDA_03                         1 3.3627e+08 3.1617e+12 514882
    ## - kw_avg_min                     1 3.7628e+08 3.1618e+12 514883
    ## - weekday_is_saturday            1 6.4997e+08 3.1620e+12 514885
    ## - data_channel_is_world          1 7.6944e+08 3.1622e+12 514886
    ## - abs_title_sentiment_polarity   1 8.0724e+08 3.1622e+12 514887
    ## - abs_title_subjectivity         1 8.5544e+08 3.1622e+12 514887
    ## - n_tokens_title                 1 8.8152e+08 3.1623e+12 514887
    ## - num_self_hrefs                 1 9.3111e+08 3.1623e+12 514888
    ## - LDA_02                         1 9.4684e+08 3.1623e+12 514888
    ## - LDA_01                         1 9.7082e+08 3.1624e+12 514888
    ## - data_channel_is_socmed         1 1.0079e+09 3.1624e+12 514888
    ## - global_subjectivity            1 1.0124e+09 3.1624e+12 514888
    ## - data_channel_is_tech           1 1.0233e+09 3.1624e+12 514888
    ## - data_channel_is_lifestyle      1 1.4794e+09 3.1629e+12 514892
    ## - data_channel_is_bus            1 1.8621e+09 3.1633e+12 514896
    ## - kw_min_avg                     1 1.9185e+09 3.1633e+12 514896
    ## - num_hrefs                      1 2.9894e+09 3.1644e+12 514906
    ## - data_channel_is_entertainment  1 3.2831e+09 3.1647e+12 514908
    ## - kw_max_avg                     1 5.6204e+09 3.1670e+12 514929
    ## - kw_avg_avg                     1 1.1371e+10 3.1728e+12 514979
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
    ##     global_rate_positive_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - rate_negative_words            1 1.2030e+07 3.1614e+12 514878
    ## - rate_positive_words            1 1.8794e+07 3.1614e+12 514878
    ## - min_negative_polarity          1 2.2011e+07 3.1614e+12 514878
    ## - avg_positive_polarity          1 2.4056e+07 3.1614e+12 514878
    ## - n_tokens_content               1 3.5733e+07 3.1614e+12 514878
    ## - title_sentiment_polarity       1 3.8741e+07 3.1614e+12 514878
    ## - self_reference_avg_sharess     1 4.5884e+07 3.1614e+12 514878
    ## - max_negative_polarity          1 5.5043e+07 3.1615e+12 514878
    ## - kw_min_min                     1 8.7741e+07 3.1615e+12 514878
    ## - kw_min_max                     1 9.4001e+07 3.1615e+12 514878
    ## - kw_max_max                     1 1.2449e+08 3.1615e+12 514879
    ## - global_rate_positive_words     1 1.4115e+08 3.1615e+12 514879
    ## - num_imgs                       1 1.7409e+08 3.1616e+12 514879
    ## - LDA_04                         1 2.0479e+08 3.1616e+12 514879
    ## - average_token_length           1 2.1101e+08 3.1616e+12 514879
    ## <none>                                        3.1614e+12 514880
    ## - n_non_stop_words               1 2.3215e+08 3.1616e+12 514880
    ## - n_non_stop_unique_tokens       1 2.3499e+08 3.1616e+12 514880
    ## - self_reference_max_shares      1 2.3653e+08 3.1616e+12 514880
    ## - num_videos                     1 2.4475e+08 3.1616e+12 514880
    ## - kw_avg_max                     1 2.9004e+08 3.1617e+12 514880
    ## - min_positive_polarity          1 3.1020e+08 3.1617e+12 514880
    ## - kw_max_min                     1 3.1568e+08 3.1617e+12 514880
    ## - self_reference_min_shares      1 3.2764e+08 3.1617e+12 514880
    ## - LDA_03                         1 3.3904e+08 3.1617e+12 514880
    ## - kw_avg_min                     1 3.6824e+08 3.1618e+12 514881
    ## - weekday_is_saturday            1 6.4671e+08 3.1620e+12 514883
    ## - data_channel_is_world          1 7.6528e+08 3.1622e+12 514884
    ## - abs_title_sentiment_polarity   1 8.0590e+08 3.1622e+12 514885
    ## - abs_title_subjectivity         1 8.5511e+08 3.1623e+12 514885
    ## - n_tokens_title                 1 8.7585e+08 3.1623e+12 514885
    ## - LDA_02                         1 9.4512e+08 3.1623e+12 514886
    ## - num_self_hrefs                 1 9.4627e+08 3.1623e+12 514886
    ## - LDA_01                         1 9.6594e+08 3.1624e+12 514886
    ## - data_channel_is_socmed         1 9.9903e+08 3.1624e+12 514886
    ## - global_subjectivity            1 1.0106e+09 3.1624e+12 514886
    ## - data_channel_is_tech           1 1.0187e+09 3.1624e+12 514886
    ## - data_channel_is_lifestyle      1 1.4761e+09 3.1629e+12 514890
    ## - data_channel_is_bus            1 1.8538e+09 3.1633e+12 514894
    ## - kw_min_avg                     1 1.9549e+09 3.1634e+12 514895
    ## - num_hrefs                      1 2.9815e+09 3.1644e+12 514904
    ## - data_channel_is_entertainment  1 3.2845e+09 3.1647e+12 514906
    ## - kw_max_avg                     1 5.6145e+09 3.1670e+12 514927
    ## - kw_avg_avg                     1 1.1534e+10 3.1729e+12 514979
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
    ##     global_rate_positive_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - rate_positive_words            1 7.9391e+06 3.1614e+12 514876
    ## - avg_positive_polarity          1 2.0651e+07 3.1614e+12 514876
    ## - min_negative_polarity          1 2.9225e+07 3.1614e+12 514876
    ## - title_sentiment_polarity       1 3.7646e+07 3.1614e+12 514876
    ## - n_tokens_content               1 4.4034e+07 3.1615e+12 514876
    ## - self_reference_avg_sharess     1 4.5404e+07 3.1615e+12 514876
    ## - max_negative_polarity          1 5.3850e+07 3.1615e+12 514876
    ## - kw_min_min                     1 8.9110e+07 3.1615e+12 514876
    ## - kw_min_max                     1 9.4216e+07 3.1615e+12 514876
    ## - kw_max_max                     1 1.2376e+08 3.1615e+12 514877
    ## - global_rate_positive_words     1 1.4179e+08 3.1616e+12 514877
    ## - num_imgs                       1 1.7140e+08 3.1616e+12 514877
    ## - LDA_04                         1 2.0056e+08 3.1616e+12 514877
    ## <none>                                        3.1614e+12 514878
    ## - self_reference_max_shares      1 2.3560e+08 3.1616e+12 514878
    ## - num_videos                     1 2.4728e+08 3.1617e+12 514878
    ## - n_non_stop_words               1 2.5188e+08 3.1617e+12 514878
    ## - n_non_stop_unique_tokens       1 2.5454e+08 3.1617e+12 514878
    ## - kw_avg_max                     1 2.9022e+08 3.1617e+12 514878
    ## - min_positive_polarity          1 3.0286e+08 3.1617e+12 514878
    ## - kw_max_min                     1 3.1351e+08 3.1617e+12 514878
    ## - self_reference_min_shares      1 3.2740e+08 3.1617e+12 514878
    ## - LDA_03                         1 3.3764e+08 3.1617e+12 514879
    ## - kw_avg_min                     1 3.6590e+08 3.1618e+12 514879
    ## - average_token_length           1 5.5864e+08 3.1620e+12 514881
    ## - weekday_is_saturday            1 6.4448e+08 3.1621e+12 514881
    ## - data_channel_is_world          1 7.6324e+08 3.1622e+12 514882
    ## - abs_title_sentiment_polarity   1 8.0246e+08 3.1622e+12 514883
    ## - abs_title_subjectivity         1 8.5350e+08 3.1623e+12 514883
    ## - n_tokens_title                 1 9.0094e+08 3.1623e+12 514884
    ## - num_self_hrefs                 1 9.3567e+08 3.1623e+12 514884
    ## - LDA_02                         1 9.4899e+08 3.1624e+12 514884
    ## - LDA_01                         1 9.6311e+08 3.1624e+12 514884
    ## - data_channel_is_socmed         1 9.8952e+08 3.1624e+12 514884
    ## - data_channel_is_tech           1 1.0097e+09 3.1624e+12 514884
    ## - global_subjectivity            1 1.0988e+09 3.1625e+12 514885
    ## - data_channel_is_lifestyle      1 1.4681e+09 3.1629e+12 514888
    ## - data_channel_is_bus            1 1.8447e+09 3.1633e+12 514892
    ## - kw_min_avg                     1 1.9484e+09 3.1634e+12 514893
    ## - num_hrefs                      1 3.0266e+09 3.1644e+12 514902
    ## - data_channel_is_entertainment  1 3.2746e+09 3.1647e+12 514904
    ## - kw_max_avg                     1 5.6034e+09 3.1670e+12 514925
    ## - kw_avg_avg                     1 1.1523e+10 3.1729e+12 514977
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
    ##     global_rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - avg_positive_polarity          1 1.8775e+07 3.1614e+12 514874
    ## - min_negative_polarity          1 2.1530e+07 3.1614e+12 514874
    ## - title_sentiment_polarity       1 4.2416e+07 3.1615e+12 514874
    ## - self_reference_avg_sharess     1 4.5507e+07 3.1615e+12 514874
    ## - n_tokens_content               1 4.8912e+07 3.1615e+12 514874
    ## - max_negative_polarity          1 6.5463e+07 3.1615e+12 514874
    ## - kw_min_min                     1 8.9358e+07 3.1615e+12 514874
    ## - kw_min_max                     1 9.3947e+07 3.1615e+12 514875
    ## - kw_max_max                     1 1.2358e+08 3.1615e+12 514875
    ## - global_rate_positive_words     1 1.4632e+08 3.1616e+12 514875
    ## - num_imgs                       1 1.6870e+08 3.1616e+12 514875
    ## - LDA_04                         1 2.0014e+08 3.1616e+12 514875
    ## <none>                                        3.1614e+12 514876
    ## - self_reference_max_shares      1 2.3587e+08 3.1617e+12 514876
    ## - num_videos                     1 2.4277e+08 3.1617e+12 514876
    ## - n_non_stop_words               1 2.4931e+08 3.1617e+12 514876
    ## - n_non_stop_unique_tokens       1 2.5192e+08 3.1617e+12 514876
    ## - kw_avg_max                     1 2.8935e+08 3.1617e+12 514876
    ## - kw_max_min                     1 3.1354e+08 3.1617e+12 514876
    ## - min_positive_polarity          1 3.1355e+08 3.1617e+12 514876
    ## - self_reference_min_shares      1 3.2796e+08 3.1617e+12 514877
    ## - LDA_03                         1 3.4005e+08 3.1618e+12 514877
    ## - kw_avg_min                     1 3.6572e+08 3.1618e+12 514877
    ## - average_token_length           1 6.1906e+08 3.1620e+12 514879
    ## - weekday_is_saturday            1 6.4371e+08 3.1621e+12 514879
    ## - data_channel_is_world          1 7.5979e+08 3.1622e+12 514880
    ## - abs_title_sentiment_polarity   1 7.9569e+08 3.1622e+12 514881
    ## - abs_title_subjectivity         1 8.6521e+08 3.1623e+12 514881
    ## - n_tokens_title                 1 9.0694e+08 3.1623e+12 514882
    ## - num_self_hrefs                 1 9.3188e+08 3.1624e+12 514882
    ## - LDA_02                         1 9.5811e+08 3.1624e+12 514882
    ## - LDA_01                         1 9.6852e+08 3.1624e+12 514882
    ## - data_channel_is_socmed         1 9.8354e+08 3.1624e+12 514882
    ## - data_channel_is_tech           1 1.0032e+09 3.1624e+12 514882
    ## - global_subjectivity            1 1.1382e+09 3.1626e+12 514884
    ## - data_channel_is_lifestyle      1 1.4624e+09 3.1629e+12 514887
    ## - data_channel_is_bus            1 1.8383e+09 3.1633e+12 514890
    ## - kw_min_avg                     1 1.9503e+09 3.1634e+12 514891
    ## - num_hrefs                      1 3.0361e+09 3.1645e+12 514900
    ## - data_channel_is_entertainment  1 3.2672e+09 3.1647e+12 514902
    ## - kw_max_avg                     1 5.5972e+09 3.1670e+12 514923
    ## - kw_avg_avg                     1 1.1515e+10 3.1729e+12 514975
    ## 
    ## Step:  AIC=514873.8
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
    ##     abs_title_sentiment_polarity + weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - min_negative_polarity          1 2.0393e+07 3.1615e+12 514872
    ## - title_sentiment_polarity       1 3.9328e+07 3.1615e+12 514872
    ## - n_tokens_content               1 4.3102e+07 3.1615e+12 514872
    ## - self_reference_avg_sharess     1 4.5850e+07 3.1615e+12 514872
    ## - max_negative_polarity          1 6.9428e+07 3.1615e+12 514872
    ## - kw_min_min                     1 8.9930e+07 3.1615e+12 514873
    ## - kw_min_max                     1 9.4472e+07 3.1615e+12 514873
    ## - kw_max_max                     1 1.2291e+08 3.1616e+12 514873
    ## - global_rate_positive_words     1 1.6495e+08 3.1616e+12 514873
    ## - num_imgs                       1 1.6591e+08 3.1616e+12 514873
    ## - LDA_04                         1 1.9744e+08 3.1616e+12 514874
    ## <none>                                        3.1614e+12 514874
    ## - self_reference_max_shares      1 2.3629e+08 3.1617e+12 514874
    ## - num_videos                     1 2.3657e+08 3.1617e+12 514874
    ## - n_non_stop_words               1 2.4716e+08 3.1617e+12 514874
    ## - n_non_stop_unique_tokens       1 2.4979e+08 3.1617e+12 514874
    ## - kw_avg_max                     1 2.8706e+08 3.1617e+12 514874
    ## - kw_max_min                     1 3.1464e+08 3.1618e+12 514875
    ## - self_reference_min_shares      1 3.2896e+08 3.1618e+12 514875
    ## - LDA_03                         1 3.4020e+08 3.1618e+12 514875
    ## - kw_avg_min                     1 3.6587e+08 3.1618e+12 514875
    ## - min_positive_polarity          1 4.7711e+08 3.1619e+12 514876
    ## - weekday_is_saturday            1 6.4109e+08 3.1621e+12 514877
    ## - average_token_length           1 7.0537e+08 3.1621e+12 514878
    ## - data_channel_is_world          1 7.5656e+08 3.1622e+12 514878
    ## - abs_title_sentiment_polarity   1 7.8250e+08 3.1622e+12 514879
    ## - abs_title_subjectivity         1 8.5054e+08 3.1623e+12 514879
    ## - n_tokens_title                 1 9.0406e+08 3.1623e+12 514880
    ## - num_self_hrefs                 1 9.2904e+08 3.1624e+12 514880
    ## - LDA_02                         1 9.5196e+08 3.1624e+12 514880
    ## - LDA_01                         1 9.6525e+08 3.1624e+12 514880
    ## - data_channel_is_socmed         1 9.8236e+08 3.1624e+12 514880
    ## - data_channel_is_tech           1 1.0034e+09 3.1624e+12 514881
    ## - global_subjectivity            1 1.1544e+09 3.1626e+12 514882
    ## - data_channel_is_lifestyle      1 1.4724e+09 3.1629e+12 514885
    ## - data_channel_is_bus            1 1.8397e+09 3.1633e+12 514888
    ## - kw_min_avg                     1 1.9517e+09 3.1634e+12 514889
    ## - num_hrefs                      1 3.0222e+09 3.1645e+12 514898
    ## - data_channel_is_entertainment  1 3.2808e+09 3.1647e+12 514901
    ## - kw_max_avg                     1 5.6012e+09 3.1670e+12 514921
    ## - kw_avg_avg                     1 1.1512e+10 3.1730e+12 514973
    ## 
    ## Step:  AIC=514872
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
    ##     weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - title_sentiment_polarity       1 3.3017e+07 3.1615e+12 514870
    ## - self_reference_avg_sharess     1 4.6081e+07 3.1615e+12 514870
    ## - n_tokens_content               1 7.6885e+07 3.1615e+12 514871
    ## - max_negative_polarity          1 8.1270e+07 3.1615e+12 514871
    ## - kw_min_min                     1 8.9694e+07 3.1615e+12 514871
    ## - kw_min_max                     1 9.3958e+07 3.1616e+12 514871
    ## - kw_max_max                     1 1.2212e+08 3.1616e+12 514871
    ## - num_imgs                       1 1.6143e+08 3.1616e+12 514871
    ## - global_rate_positive_words     1 1.7634e+08 3.1616e+12 514872
    ## - LDA_04                         1 2.0015e+08 3.1617e+12 514872
    ## <none>                                        3.1615e+12 514872
    ## - self_reference_max_shares      1 2.3754e+08 3.1617e+12 514872
    ## - num_videos                     1 2.4553e+08 3.1617e+12 514872
    ## - n_non_stop_words               1 2.5059e+08 3.1617e+12 514872
    ## - n_non_stop_unique_tokens       1 2.5322e+08 3.1617e+12 514872
    ## - kw_avg_max                     1 2.8653e+08 3.1617e+12 514873
    ## - kw_max_min                     1 3.1416e+08 3.1618e+12 514873
    ## - self_reference_min_shares      1 3.2927e+08 3.1618e+12 514873
    ## - LDA_03                         1 3.3956e+08 3.1618e+12 514873
    ## - kw_avg_min                     1 3.6565e+08 3.1618e+12 514873
    ## - min_positive_polarity          1 5.0731e+08 3.1620e+12 514874
    ## - weekday_is_saturday            1 6.4482e+08 3.1621e+12 514876
    ## - average_token_length           1 6.9312e+08 3.1622e+12 514876
    ## - data_channel_is_world          1 7.5532e+08 3.1622e+12 514877
    ## - abs_title_sentiment_polarity   1 8.0699e+08 3.1623e+12 514877
    ## - abs_title_subjectivity         1 8.5200e+08 3.1623e+12 514877
    ## - n_tokens_title                 1 9.0907e+08 3.1624e+12 514878
    ## - num_self_hrefs                 1 9.3454e+08 3.1624e+12 514878
    ## - LDA_02                         1 9.4984e+08 3.1624e+12 514878
    ## - LDA_01                         1 9.6166e+08 3.1624e+12 514878
    ## - data_channel_is_socmed         1 9.9135e+08 3.1625e+12 514879
    ## - data_channel_is_tech           1 1.0182e+09 3.1625e+12 514879
    ## - global_subjectivity            1 1.3101e+09 3.1628e+12 514882
    ## - data_channel_is_lifestyle      1 1.4734e+09 3.1629e+12 514883
    ## - data_channel_is_bus            1 1.8584e+09 3.1633e+12 514886
    ## - kw_min_avg                     1 1.9536e+09 3.1634e+12 514887
    ## - num_hrefs                      1 3.0298e+09 3.1645e+12 514897
    ## - data_channel_is_entertainment  1 3.2788e+09 3.1647e+12 514899
    ## - kw_max_avg                     1 5.6023e+09 3.1671e+12 514919
    ## - kw_avg_avg                     1 1.1519e+10 3.1730e+12 514971
    ## 
    ## Step:  AIC=514870.3
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
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - self_reference_avg_sharess     1 4.6679e+07 3.1615e+12 514869
    ## - n_tokens_content               1 7.5547e+07 3.1616e+12 514869
    ## - max_negative_polarity          1 8.1424e+07 3.1616e+12 514869
    ## - kw_min_min                     1 9.0620e+07 3.1616e+12 514869
    ## - kw_min_max                     1 9.3414e+07 3.1616e+12 514869
    ## - kw_max_max                     1 1.2228e+08 3.1616e+12 514869
    ## - global_rate_positive_words     1 1.6302e+08 3.1617e+12 514870
    ## - num_imgs                       1 1.6461e+08 3.1617e+12 514870
    ## - LDA_04                         1 2.0215e+08 3.1617e+12 514870
    ## <none>                                        3.1615e+12 514870
    ## - self_reference_max_shares      1 2.3873e+08 3.1617e+12 514870
    ## - n_non_stop_words               1 2.4589e+08 3.1617e+12 514870
    ## - num_videos                     1 2.4690e+08 3.1617e+12 514870
    ## - n_non_stop_unique_tokens       1 2.4848e+08 3.1617e+12 514870
    ## - kw_avg_max                     1 2.8706e+08 3.1618e+12 514871
    ## - kw_max_min                     1 3.1461e+08 3.1618e+12 514871
    ## - self_reference_min_shares      1 3.3108e+08 3.1618e+12 514871
    ## - LDA_03                         1 3.4155e+08 3.1618e+12 514871
    ## - kw_avg_min                     1 3.6639e+08 3.1619e+12 514872
    ## - min_positive_polarity          1 4.9854e+08 3.1620e+12 514873
    ## - weekday_is_saturday            1 6.4935e+08 3.1621e+12 514874
    ## - average_token_length           1 6.9862e+08 3.1622e+12 514874
    ## - data_channel_is_world          1 7.5753e+08 3.1622e+12 514875
    ## - abs_title_subjectivity         1 8.3033e+08 3.1623e+12 514876
    ## - n_tokens_title                 1 9.0578e+08 3.1624e+12 514876
    ## - num_self_hrefs                 1 9.3310e+08 3.1624e+12 514876
    ## - LDA_02                         1 9.4924e+08 3.1624e+12 514877
    ## - LDA_01                         1 9.6844e+08 3.1625e+12 514877
    ## - data_channel_is_socmed         1 9.8557e+08 3.1625e+12 514877
    ## - data_channel_is_tech           1 1.0102e+09 3.1625e+12 514877
    ## - abs_title_sentiment_polarity   1 1.0359e+09 3.1625e+12 514877
    ## - global_subjectivity            1 1.2958e+09 3.1628e+12 514880
    ## - data_channel_is_lifestyle      1 1.4623e+09 3.1630e+12 514881
    ## - data_channel_is_bus            1 1.8520e+09 3.1633e+12 514885
    ## - kw_min_avg                     1 1.9534e+09 3.1634e+12 514885
    ## - num_hrefs                      1 3.0376e+09 3.1645e+12 514895
    ## - data_channel_is_entertainment  1 3.2762e+09 3.1648e+12 514897
    ## - kw_max_avg                     1 5.6042e+09 3.1671e+12 514917
    ## - kw_avg_avg                     1 1.1523e+10 3.1730e+12 514969
    ## 
    ## Step:  AIC=514868.7
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
    ##     weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_tokens_content               1 7.6037e+07 3.1616e+12 514867
    ## - max_negative_polarity          1 8.0990e+07 3.1616e+12 514867
    ## - kw_min_min                     1 9.0283e+07 3.1616e+12 514868
    ## - kw_min_max                     1 9.1814e+07 3.1616e+12 514868
    ## - kw_max_max                     1 1.2291e+08 3.1617e+12 514868
    ## - global_rate_positive_words     1 1.6125e+08 3.1617e+12 514868
    ## - num_imgs                       1 1.6263e+08 3.1617e+12 514868
    ## - LDA_04                         1 2.0247e+08 3.1617e+12 514868
    ## <none>                                        3.1615e+12 514869
    ## - n_non_stop_words               1 2.4814e+08 3.1618e+12 514869
    ## - n_non_stop_unique_tokens       1 2.5071e+08 3.1618e+12 514869
    ## - num_videos                     1 2.5354e+08 3.1618e+12 514869
    ## - kw_avg_max                     1 2.8912e+08 3.1618e+12 514869
    ## - kw_max_min                     1 3.2004e+08 3.1619e+12 514870
    ## - LDA_03                         1 3.4131e+08 3.1619e+12 514870
    ## - kw_avg_min                     1 3.7069e+08 3.1619e+12 514870
    ## - min_positive_polarity          1 4.9630e+08 3.1620e+12 514871
    ## - self_reference_max_shares      1 5.4088e+08 3.1621e+12 514871
    ## - weekday_is_saturday            1 6.5121e+08 3.1622e+12 514872
    ## - average_token_length           1 7.0689e+08 3.1622e+12 514873
    ## - data_channel_is_world          1 7.5318e+08 3.1623e+12 514873
    ## - self_reference_min_shares      1 7.7128e+08 3.1623e+12 514873
    ## - abs_title_subjectivity         1 8.3078e+08 3.1624e+12 514874
    ## - num_self_hrefs                 1 8.9346e+08 3.1624e+12 514875
    ## - n_tokens_title                 1 8.9997e+08 3.1624e+12 514875
    ## - LDA_02                         1 9.4844e+08 3.1625e+12 514875
    ## - LDA_01                         1 9.6837e+08 3.1625e+12 514875
    ## - data_channel_is_socmed         1 9.8442e+08 3.1625e+12 514875
    ## - data_channel_is_tech           1 1.0119e+09 3.1626e+12 514876
    ## - abs_title_sentiment_polarity   1 1.0370e+09 3.1626e+12 514876
    ## - global_subjectivity            1 1.2874e+09 3.1628e+12 514878
    ## - data_channel_is_lifestyle      1 1.4581e+09 3.1630e+12 514880
    ## - data_channel_is_bus            1 1.8456e+09 3.1634e+12 514883
    ## - kw_min_avg                     1 1.9632e+09 3.1635e+12 514884
    ## - num_hrefs                      1 3.0635e+09 3.1646e+12 514894
    ## - data_channel_is_entertainment  1 3.2721e+09 3.1648e+12 514895
    ## - kw_max_avg                     1 5.6242e+09 3.1672e+12 514916
    ## - kw_avg_avg                     1 1.1532e+10 3.1731e+12 514968
    ## 
    ## Step:  AIC=514867.4
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + min_positive_polarity + max_negative_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - max_negative_polarity          1 5.8267e+07 3.1617e+12 514866
    ## - kw_min_max                     1 8.9513e+07 3.1617e+12 514866
    ## - kw_min_min                     1 9.1468e+07 3.1617e+12 514866
    ## - kw_max_max                     1 1.1846e+08 3.1617e+12 514866
    ## - global_rate_positive_words     1 1.4115e+08 3.1618e+12 514867
    ## - n_non_stop_words               1 1.7769e+08 3.1618e+12 514867
    ## - n_non_stop_unique_tokens       1 1.8009e+08 3.1618e+12 514867
    ## - num_imgs                       1 1.9245e+08 3.1618e+12 514867
    ## - LDA_04                         1 2.0670e+08 3.1618e+12 514867
    ## <none>                                        3.1616e+12 514867
    ## - kw_avg_max                     1 2.8886e+08 3.1619e+12 514868
    ## - num_videos                     1 2.9921e+08 3.1619e+12 514868
    ## - kw_max_min                     1 3.1984e+08 3.1619e+12 514868
    ## - LDA_03                         1 3.6521e+08 3.1620e+12 514869
    ## - kw_avg_min                     1 3.6894e+08 3.1620e+12 514869
    ## - self_reference_max_shares      1 5.3853e+08 3.1622e+12 514870
    ## - min_positive_polarity          1 5.6482e+08 3.1622e+12 514870
    ## - weekday_is_saturday            1 6.5245e+08 3.1623e+12 514871
    ## - average_token_length           1 6.7899e+08 3.1623e+12 514871
    ## - data_channel_is_world          1 7.3050e+08 3.1623e+12 514872
    ## - self_reference_min_shares      1 7.6989e+08 3.1624e+12 514872
    ## - abs_title_subjectivity         1 8.4457e+08 3.1625e+12 514873
    ## - num_self_hrefs                 1 8.5046e+08 3.1625e+12 514873
    ## - n_tokens_title                 1 9.1898e+08 3.1625e+12 514873
    ## - LDA_02                         1 9.4095e+08 3.1626e+12 514874
    ## - data_channel_is_socmed         1 9.6557e+08 3.1626e+12 514874
    ## - LDA_01                         1 9.7986e+08 3.1626e+12 514874
    ## - data_channel_is_tech           1 9.9098e+08 3.1626e+12 514874
    ## - abs_title_sentiment_polarity   1 1.0359e+09 3.1627e+12 514874
    ## - global_subjectivity            1 1.3701e+09 3.1630e+12 514877
    ## - data_channel_is_lifestyle      1 1.4209e+09 3.1630e+12 514878
    ## - data_channel_is_bus            1 1.8137e+09 3.1634e+12 514881
    ## - kw_min_avg                     1 1.9587e+09 3.1636e+12 514883
    ## - data_channel_is_entertainment  1 3.2008e+09 3.1648e+12 514893
    ## - num_hrefs                      1 3.3966e+09 3.1650e+12 514895
    ## - kw_max_avg                     1 5.6006e+09 3.1672e+12 514915
    ## - kw_avg_avg                     1 1.1483e+10 3.1731e+12 514966
    ## 
    ## Step:  AIC=514865.9
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_rate_positive_words + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - kw_min_max                     1 9.1425e+07 3.1618e+12 514865
    ## - kw_min_min                     1 9.2493e+07 3.1618e+12 514865
    ## - kw_max_max                     1 1.1971e+08 3.1618e+12 514865
    ## - global_rate_positive_words     1 1.4614e+08 3.1618e+12 514865
    ## - num_imgs                       1 1.9086e+08 3.1619e+12 514866
    ## - n_non_stop_words               1 2.0239e+08 3.1619e+12 514866
    ## - n_non_stop_unique_tokens       1 2.0496e+08 3.1619e+12 514866
    ## - LDA_04                         1 2.0855e+08 3.1619e+12 514866
    ## <none>                                        3.1617e+12 514866
    ## - kw_avg_max                     1 2.8417e+08 3.1620e+12 514866
    ## - num_videos                     1 2.8438e+08 3.1620e+12 514866
    ## - kw_max_min                     1 3.1902e+08 3.1620e+12 514867
    ## - LDA_03                         1 3.6010e+08 3.1620e+12 514867
    ## - kw_avg_min                     1 3.6883e+08 3.1620e+12 514867
    ## - self_reference_max_shares      1 5.3870e+08 3.1622e+12 514869
    ## - min_positive_polarity          1 5.4467e+08 3.1622e+12 514869
    ## - average_token_length           1 6.4506e+08 3.1623e+12 514870
    ## - weekday_is_saturday            1 6.5136e+08 3.1623e+12 514870
    ## - data_channel_is_world          1 7.3997e+08 3.1624e+12 514870
    ## - self_reference_min_shares      1 7.7953e+08 3.1625e+12 514871
    ## - abs_title_subjectivity         1 8.4447e+08 3.1625e+12 514871
    ## - num_self_hrefs                 1 8.5420e+08 3.1625e+12 514871
    ## - n_tokens_title                 1 9.2145e+08 3.1626e+12 514872
    ## - LDA_02                         1 9.4738e+08 3.1626e+12 514872
    ## - data_channel_is_socmed         1 9.6724e+08 3.1626e+12 514872
    ## - LDA_01                         1 9.7658e+08 3.1626e+12 514872
    ## - data_channel_is_tech           1 1.0004e+09 3.1627e+12 514873
    ## - abs_title_sentiment_polarity   1 1.0386e+09 3.1627e+12 514873
    ## - data_channel_is_lifestyle      1 1.4358e+09 3.1631e+12 514876
    ## - global_subjectivity            1 1.4381e+09 3.1631e+12 514877
    ## - data_channel_is_bus            1 1.8276e+09 3.1635e+12 514880
    ## - kw_min_avg                     1 1.9620e+09 3.1636e+12 514881
    ## - data_channel_is_entertainment  1 3.2188e+09 3.1649e+12 514892
    ## - num_hrefs                      1 3.3583e+09 3.1650e+12 514893
    ## - kw_max_avg                     1 5.6051e+09 3.1673e+12 514913
    ## - kw_avg_avg                     1 1.1500e+10 3.1732e+12 514965
    ## 
    ## Step:  AIC=514864.7
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + LDA_01 + 
    ##     LDA_02 + LDA_03 + LDA_04 + global_subjectivity + global_rate_positive_words + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - kw_min_min                     1 8.9368e+07 3.1619e+12 514863
    ## - kw_max_max                     1 9.2827e+07 3.1619e+12 514864
    ## - global_rate_positive_words     1 1.5058e+08 3.1619e+12 514864
    ## - num_imgs                       1 1.8754e+08 3.1620e+12 514864
    ## - n_non_stop_words               1 2.0646e+08 3.1620e+12 514865
    ## - n_non_stop_unique_tokens       1 2.0907e+08 3.1620e+12 514865
    ## - LDA_04                         1 2.1565e+08 3.1620e+12 514865
    ## <none>                                        3.1618e+12 514865
    ## - num_videos                     1 2.8931e+08 3.1621e+12 514865
    ## - kw_max_min                     1 3.2807e+08 3.1621e+12 514866
    ## - LDA_03                         1 3.5653e+08 3.1621e+12 514866
    ## - kw_avg_min                     1 3.7619e+08 3.1621e+12 514866
    ## - kw_avg_max                     1 5.1749e+08 3.1623e+12 514867
    ## - self_reference_max_shares      1 5.4577e+08 3.1623e+12 514867
    ## - min_positive_polarity          1 5.5022e+08 3.1623e+12 514868
    ## - weekday_is_saturday            1 6.4540e+08 3.1624e+12 514868
    ## - average_token_length           1 6.5742e+08 3.1624e+12 514868
    ## - self_reference_min_shares      1 7.7567e+08 3.1625e+12 514870
    ## - data_channel_is_world          1 7.7843e+08 3.1625e+12 514870
    ## - num_self_hrefs                 1 8.4062e+08 3.1626e+12 514870
    ## - abs_title_subjectivity         1 8.4936e+08 3.1626e+12 514870
    ## - n_tokens_title                 1 9.3299e+08 3.1627e+12 514871
    ## - LDA_02                         1 9.4188e+08 3.1627e+12 514871
    ## - LDA_01                         1 9.9066e+08 3.1628e+12 514871
    ## - data_channel_is_tech           1 1.0288e+09 3.1628e+12 514872
    ## - abs_title_sentiment_polarity   1 1.0390e+09 3.1628e+12 514872
    ## - data_channel_is_socmed         1 1.0475e+09 3.1628e+12 514872
    ## - global_subjectivity            1 1.4436e+09 3.1632e+12 514875
    ## - data_channel_is_lifestyle      1 1.4882e+09 3.1633e+12 514876
    ## - data_channel_is_bus            1 1.8385e+09 3.1636e+12 514879
    ## - kw_min_avg                     1 2.2121e+09 3.1640e+12 514882
    ## - data_channel_is_entertainment  1 3.3387e+09 3.1651e+12 514892
    ## - num_hrefs                      1 3.3471e+09 3.1651e+12 514892
    ## - kw_max_avg                     1 5.7064e+09 3.1675e+12 514913
    ## - kw_avg_avg                     1 1.1685e+10 3.1734e+12 514965
    ## 
    ## Step:  AIC=514863.5
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_avg_min + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + global_rate_positive_words + min_positive_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity + weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - global_rate_positive_words     1 1.4981e+08 3.1620e+12 514863
    ## - num_imgs                       1 1.8990e+08 3.1620e+12 514863
    ## - n_non_stop_words               1 2.1022e+08 3.1621e+12 514863
    ## - n_non_stop_unique_tokens       1 2.1286e+08 3.1621e+12 514863
    ## - LDA_04                         1 2.1516e+08 3.1621e+12 514863
    ## <none>                                        3.1619e+12 514863
    ## - num_videos                     1 3.0121e+08 3.1622e+12 514864
    ## - kw_max_min                     1 3.1062e+08 3.1622e+12 514864
    ## - kw_avg_min                     1 3.5508e+08 3.1622e+12 514865
    ## - LDA_03                         1 3.5648e+08 3.1622e+12 514865
    ## - self_reference_max_shares      1 5.4451e+08 3.1624e+12 514866
    ## - kw_avg_max                     1 5.4744e+08 3.1624e+12 514866
    ## - min_positive_polarity          1 5.5011e+08 3.1624e+12 514866
    ## - weekday_is_saturday            1 6.4747e+08 3.1625e+12 514867
    ## - average_token_length           1 6.5749e+08 3.1625e+12 514867
    ## - kw_max_max                     1 6.6720e+08 3.1625e+12 514867
    ## - self_reference_min_shares      1 7.7670e+08 3.1626e+12 514868
    ## - data_channel_is_world          1 7.8137e+08 3.1626e+12 514868
    ## - abs_title_subjectivity         1 8.4652e+08 3.1627e+12 514869
    ## - num_self_hrefs                 1 8.5674e+08 3.1627e+12 514869
    ## - n_tokens_title                 1 9.3026e+08 3.1628e+12 514870
    ## - LDA_02                         1 9.5073e+08 3.1628e+12 514870
    ## - LDA_01                         1 9.8699e+08 3.1628e+12 514870
    ## - data_channel_is_tech           1 1.0373e+09 3.1629e+12 514871
    ## - abs_title_sentiment_polarity   1 1.0394e+09 3.1629e+12 514871
    ## - data_channel_is_socmed         1 1.0487e+09 3.1629e+12 514871
    ## - global_subjectivity            1 1.4397e+09 3.1633e+12 514874
    ## - data_channel_is_lifestyle      1 1.4992e+09 3.1634e+12 514875
    ## - data_channel_is_bus            1 1.8404e+09 3.1637e+12 514878
    ## - kw_min_avg                     1 2.2086e+09 3.1641e+12 514881
    ## - num_hrefs                      1 3.3472e+09 3.1652e+12 514891
    ## - data_channel_is_entertainment  1 3.3788e+09 3.1652e+12 514891
    ## - kw_max_avg                     1 5.7232e+09 3.1676e+12 514912
    ## - kw_avg_avg                     1 1.1718e+10 3.1736e+12 514964
    ## 
    ## Step:  AIC=514862.8
    ## shares ~ n_tokens_title + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_avg_min + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_non_stop_words               1 1.8143e+08 3.1622e+12 514862
    ## - n_non_stop_unique_tokens       1 1.8394e+08 3.1622e+12 514862
    ## - LDA_04                         1 2.0383e+08 3.1622e+12 514863
    ## - num_imgs                       1 2.0393e+08 3.1622e+12 514863
    ## <none>                                        3.1620e+12 514863
    ## - num_videos                     1 2.8079e+08 3.1623e+12 514863
    ## - kw_max_min                     1 3.1424e+08 3.1623e+12 514864
    ## - LDA_03                         1 3.3277e+08 3.1623e+12 514864
    ## - kw_avg_min                     1 3.5866e+08 3.1624e+12 514864
    ## - min_positive_polarity          1 4.3280e+08 3.1624e+12 514865
    ## - kw_avg_max                     1 5.1683e+08 3.1625e+12 514865
    ## - self_reference_max_shares      1 5.5123e+08 3.1626e+12 514866
    ## - weekday_is_saturday            1 6.3765e+08 3.1626e+12 514866
    ## - kw_max_max                     1 6.5661e+08 3.1627e+12 514867
    ## - data_channel_is_world          1 7.3875e+08 3.1627e+12 514867
    ## - average_token_length           1 7.4699e+08 3.1628e+12 514867
    ## - self_reference_min_shares      1 7.8780e+08 3.1628e+12 514868
    ## - LDA_02                         1 9.0845e+08 3.1629e+12 514869
    ## - num_self_hrefs                 1 9.1163e+08 3.1629e+12 514869
    ## - abs_title_subjectivity         1 9.5970e+08 3.1630e+12 514869
    ## - n_tokens_title                 1 9.6010e+08 3.1630e+12 514869
    ## - LDA_01                         1 9.7269e+08 3.1630e+12 514869
    ## - abs_title_sentiment_polarity   1 1.0169e+09 3.1630e+12 514870
    ## - data_channel_is_tech           1 1.0366e+09 3.1630e+12 514870
    ## - data_channel_is_socmed         1 1.0651e+09 3.1631e+12 514870
    ## - global_subjectivity            1 1.2913e+09 3.1633e+12 514872
    ## - data_channel_is_lifestyle      1 1.5023e+09 3.1635e+12 514874
    ## - data_channel_is_bus            1 1.8290e+09 3.1638e+12 514877
    ## - kw_min_avg                     1 2.2238e+09 3.1642e+12 514880
    ## - data_channel_is_entertainment  1 3.3515e+09 3.1654e+12 514890
    ## - num_hrefs                      1 3.4665e+09 3.1655e+12 514891
    ## - kw_max_avg                     1 5.7273e+09 3.1677e+12 514911
    ## - kw_avg_avg                     1 1.1715e+10 3.1737e+12 514963
    ## 
    ## Step:  AIC=514862.4
    ## shares ~ n_tokens_title + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_avg_min + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - n_non_stop_unique_tokens       1 2.1229e+07 3.1622e+12 514861
    ## - num_imgs                       1 9.1583e+07 3.1623e+12 514861
    ## - LDA_04                         1 2.0537e+08 3.1624e+12 514862
    ## <none>                                        3.1622e+12 514862
    ## - num_videos                     1 2.5836e+08 3.1624e+12 514863
    ## - LDA_03                         1 3.1148e+08 3.1625e+12 514863
    ## - kw_max_min                     1 3.1381e+08 3.1625e+12 514863
    ## - min_positive_polarity          1 3.5040e+08 3.1625e+12 514863
    ## - kw_avg_min                     1 3.5887e+08 3.1625e+12 514864
    ## - kw_avg_max                     1 4.9527e+08 3.1627e+12 514865
    ## - self_reference_max_shares      1 5.6418e+08 3.1627e+12 514865
    ## - weekday_is_saturday            1 6.3136e+08 3.1628e+12 514866
    ## - average_token_length           1 6.8129e+08 3.1629e+12 514866
    ## - kw_max_max                     1 6.9801e+08 3.1629e+12 514867
    ## - data_channel_is_world          1 7.5674e+08 3.1629e+12 514867
    ## - self_reference_min_shares      1 7.8440e+08 3.1630e+12 514867
    ## - LDA_02                         1 9.1339e+08 3.1631e+12 514868
    ## - num_self_hrefs                 1 9.2393e+08 3.1631e+12 514868
    ## - LDA_01                         1 9.2864e+08 3.1631e+12 514869
    ## - n_tokens_title                 1 9.3648e+08 3.1631e+12 514869
    ## - abs_title_subjectivity         1 9.4784e+08 3.1631e+12 514869
    ## - abs_title_sentiment_polarity   1 9.9545e+08 3.1632e+12 514869
    ## - data_channel_is_tech           1 1.0568e+09 3.1632e+12 514870
    ## - data_channel_is_socmed         1 1.0762e+09 3.1633e+12 514870
    ## - global_subjectivity            1 1.3434e+09 3.1635e+12 514872
    ## - data_channel_is_lifestyle      1 1.5057e+09 3.1637e+12 514874
    ## - data_channel_is_bus            1 1.8478e+09 3.1640e+12 514877
    ## - kw_min_avg                     1 2.2457e+09 3.1644e+12 514880
    ## - num_hrefs                      1 3.2852e+09 3.1655e+12 514889
    ## - data_channel_is_entertainment  1 3.4306e+09 3.1656e+12 514890
    ## - kw_max_avg                     1 5.7387e+09 3.1679e+12 514911
    ## - kw_avg_avg                     1 1.1742e+10 3.1739e+12 514963
    ## 
    ## Step:  AIC=514860.6
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + data_channel_is_lifestyle + 
    ##     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + 
    ##     data_channel_is_tech + data_channel_is_world + kw_max_min + 
    ##     kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - num_imgs                       1 9.4118e+07 3.1623e+12 514859
    ## - LDA_04                         1 2.0906e+08 3.1624e+12 514860
    ## <none>                                        3.1622e+12 514861
    ## - num_videos                     1 2.5896e+08 3.1625e+12 514861
    ## - kw_max_min                     1 3.1386e+08 3.1625e+12 514861
    ## - LDA_03                         1 3.1829e+08 3.1625e+12 514861
    ## - min_positive_polarity          1 3.4967e+08 3.1626e+12 514862
    ## - kw_avg_min                     1 3.5888e+08 3.1626e+12 514862
    ## - kw_avg_max                     1 4.9404e+08 3.1627e+12 514863
    ## - self_reference_max_shares      1 5.6387e+08 3.1628e+12 514864
    ## - weekday_is_saturday            1 6.3087e+08 3.1628e+12 514864
    ## - average_token_length           1 6.7331e+08 3.1629e+12 514864
    ## - kw_max_max                     1 6.9893e+08 3.1629e+12 514865
    ## - data_channel_is_world          1 7.5981e+08 3.1630e+12 514865
    ## - self_reference_min_shares      1 7.8465e+08 3.1630e+12 514865
    ## - num_self_hrefs                 1 9.2224e+08 3.1631e+12 514867
    ## - LDA_02                         1 9.2276e+08 3.1631e+12 514867
    ## - n_tokens_title                 1 9.3471e+08 3.1631e+12 514867
    ## - LDA_01                         1 9.4116e+08 3.1631e+12 514867
    ## - abs_title_subjectivity         1 9.4426e+08 3.1632e+12 514867
    ## - abs_title_sentiment_polarity   1 9.9348e+08 3.1632e+12 514867
    ## - data_channel_is_tech           1 1.0614e+09 3.1633e+12 514868
    ## - data_channel_is_socmed         1 1.0831e+09 3.1633e+12 514868
    ## - global_subjectivity            1 1.3358e+09 3.1635e+12 514870
    ## - data_channel_is_lifestyle      1 1.5107e+09 3.1637e+12 514872
    ## - data_channel_is_bus            1 1.8628e+09 3.1641e+12 514875
    ## - kw_min_avg                     1 2.2436e+09 3.1645e+12 514878
    ## - num_hrefs                      1 3.2765e+09 3.1655e+12 514887
    ## - data_channel_is_entertainment  1 3.4269e+09 3.1656e+12 514889
    ## - kw_max_avg                     1 5.7373e+09 3.1679e+12 514909
    ## - kw_avg_avg                     1 1.1739e+10 3.1739e+12 514961
    ## 
    ## Step:  AIC=514859.4
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + num_videos + 
    ##     average_token_length + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_avg_min + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + LDA_04 + 
    ##     global_subjectivity + min_positive_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity + weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - LDA_04                         1 1.9723e+08 3.1625e+12 514859
    ## - num_videos                     1 2.1346e+08 3.1625e+12 514859
    ## <none>                                        3.1623e+12 514859
    ## - LDA_03                         1 2.8663e+08 3.1626e+12 514860
    ## - kw_max_min                     1 3.1888e+08 3.1626e+12 514860
    ## - min_positive_polarity          1 3.6322e+08 3.1627e+12 514861
    ## - kw_avg_min                     1 3.6767e+08 3.1627e+12 514861
    ## - kw_avg_max                     1 5.2316e+08 3.1628e+12 514862
    ## - self_reference_max_shares      1 5.6335e+08 3.1629e+12 514862
    ## - weekday_is_saturday            1 6.3174e+08 3.1629e+12 514863
    ## - average_token_length           1 6.6348e+08 3.1630e+12 514863
    ## - kw_max_max                     1 6.7640e+08 3.1630e+12 514863
    ## - data_channel_is_world          1 7.8711e+08 3.1631e+12 514864
    ## - self_reference_min_shares      1 7.8815e+08 3.1631e+12 514864
    ## - num_self_hrefs                 1 8.5922e+08 3.1632e+12 514865
    ## - LDA_02                         1 8.9721e+08 3.1632e+12 514865
    ## - LDA_01                         1 9.1226e+08 3.1632e+12 514865
    ## - n_tokens_title                 1 9.3025e+08 3.1632e+12 514866
    ## - abs_title_subjectivity         1 9.4501e+08 3.1632e+12 514866
    ## - abs_title_sentiment_polarity   1 1.0121e+09 3.1633e+12 514866
    ## - data_channel_is_tech           1 1.0768e+09 3.1634e+12 514867
    ## - data_channel_is_socmed         1 1.1079e+09 3.1634e+12 514867
    ## - global_subjectivity            1 1.3272e+09 3.1636e+12 514869
    ## - data_channel_is_lifestyle      1 1.5263e+09 3.1638e+12 514871
    ## - data_channel_is_bus            1 1.8851e+09 3.1642e+12 514874
    ## - kw_min_avg                     1 2.2386e+09 3.1645e+12 514877
    ## - data_channel_is_entertainment  1 3.4274e+09 3.1657e+12 514887
    ## - num_hrefs                      1 3.7826e+09 3.1661e+12 514891
    ## - kw_max_avg                     1 5.7874e+09 3.1681e+12 514908
    ## - kw_avg_avg                     1 1.1863e+10 3.1742e+12 514961
    ## 
    ## Step:  AIC=514859.1
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + num_videos + 
    ##     average_token_length + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_avg_min + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - LDA_03                         1 1.3032e+08 3.1626e+12 514858
    ## - num_videos                     1 2.1213e+08 3.1627e+12 514859
    ## <none>                                        3.1625e+12 514859
    ## - kw_max_min                     1 3.1419e+08 3.1628e+12 514860
    ## - kw_avg_min                     1 3.6065e+08 3.1629e+12 514860
    ## - min_positive_polarity          1 4.0101e+08 3.1629e+12 514861
    ## - kw_avg_max                     1 4.8727e+08 3.1630e+12 514861
    ## - self_reference_max_shares      1 5.6032e+08 3.1631e+12 514862
    ## - average_token_length           1 6.3930e+08 3.1631e+12 514863
    ## - weekday_is_saturday            1 6.4375e+08 3.1631e+12 514863
    ## - kw_max_max                     1 7.0443e+08 3.1632e+12 514863
    ## - LDA_02                         1 7.0767e+08 3.1632e+12 514863
    ## - LDA_01                         1 7.1533e+08 3.1632e+12 514863
    ## - self_reference_min_shares      1 7.8929e+08 3.1633e+12 514864
    ## - data_channel_is_world          1 8.3341e+08 3.1633e+12 514864
    ## - num_self_hrefs                 1 8.8971e+08 3.1634e+12 514865
    ## - n_tokens_title                 1 9.0433e+08 3.1634e+12 514865
    ## - abs_title_subjectivity         1 9.4984e+08 3.1634e+12 514865
    ## - data_channel_is_socmed         1 9.7926e+08 3.1635e+12 514866
    ## - abs_title_sentiment_polarity   1 1.0202e+09 3.1635e+12 514866
    ## - global_subjectivity            1 1.3115e+09 3.1638e+12 514869
    ## - data_channel_is_tech           1 1.5467e+09 3.1640e+12 514871
    ## - data_channel_is_bus            1 1.6916e+09 3.1642e+12 514872
    ## - data_channel_is_lifestyle      1 1.7406e+09 3.1642e+12 514872
    ## - kw_min_avg                     1 2.2940e+09 3.1648e+12 514877
    ## - data_channel_is_entertainment  1 3.3856e+09 3.1659e+12 514887
    ## - num_hrefs                      1 3.8376e+09 3.1663e+12 514891
    ## - kw_max_avg                     1 5.8129e+09 3.1683e+12 514908
    ## - kw_avg_avg                     1 1.1949e+10 3.1744e+12 514962
    ## 
    ## Step:  AIC=514858.3
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + num_videos + 
    ##     average_token_length + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_avg_min + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + global_subjectivity + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## - num_videos                     1 1.8332e+08 3.1628e+12 514858
    ## <none>                                        3.1626e+12 514858
    ## - kw_max_min                     1 3.0308e+08 3.1629e+12 514859
    ## - kw_avg_min                     1 3.4497e+08 3.1630e+12 514859
    ## - min_positive_polarity          1 4.3799e+08 3.1631e+12 514860
    ## - kw_avg_max                     1 5.3725e+08 3.1632e+12 514861
    ## - self_reference_max_shares      1 5.6437e+08 3.1632e+12 514861
    ## - LDA_02                         1 5.8588e+08 3.1632e+12 514861
    ## - average_token_length           1 6.0346e+08 3.1632e+12 514862
    ## - kw_max_max                     1 6.4457e+08 3.1633e+12 514862
    ## - weekday_is_saturday            1 6.4509e+08 3.1633e+12 514862
    ## - LDA_01                         1 6.6068e+08 3.1633e+12 514862
    ## - data_channel_is_world          1 7.0428e+08 3.1633e+12 514862
    ## - self_reference_min_shares      1 7.8974e+08 3.1634e+12 514863
    ## - num_self_hrefs                 1 8.6772e+08 3.1635e+12 514864
    ## - data_channel_is_socmed         1 8.7337e+08 3.1635e+12 514864
    ## - n_tokens_title                 1 8.9257e+08 3.1635e+12 514864
    ## - abs_title_subjectivity         1 9.3021e+08 3.1636e+12 514864
    ## - abs_title_sentiment_polarity   1 9.9334e+08 3.1636e+12 514865
    ## - global_subjectivity            1 1.2835e+09 3.1639e+12 514868
    ## - data_channel_is_lifestyle      1 1.8425e+09 3.1645e+12 514872
    ## - data_channel_is_tech           1 1.8937e+09 3.1645e+12 514873
    ## - data_channel_is_bus            1 2.1512e+09 3.1648e+12 514875
    ## - kw_min_avg                     1 2.2089e+09 3.1648e+12 514876
    ## - data_channel_is_entertainment  1 3.3104e+09 3.1659e+12 514885
    ## - num_hrefs                      1 3.7738e+09 3.1664e+12 514889
    ## - kw_max_avg                     1 5.7006e+09 3.1683e+12 514906
    ## - kw_avg_avg                     1 1.1837e+10 3.1745e+12 514960
    ## 
    ## Step:  AIC=514857.9
    ## shares ~ n_tokens_title + num_hrefs + num_self_hrefs + average_token_length + 
    ##     data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_avg_min + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + global_subjectivity + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_saturday
    ## 
    ##                                 Df  Sum of Sq        RSS    AIC
    ## <none>                                        3.1628e+12 514858
    ## - kw_max_min                     1 3.0417e+08 3.1631e+12 514859
    ## - kw_avg_min                     1 3.4296e+08 3.1632e+12 514859
    ## - min_positive_polarity          1 4.5483e+08 3.1633e+12 514860
    ## - kw_avg_max                     1 4.6417e+08 3.1633e+12 514860
    ## - average_token_length           1 6.0674e+08 3.1634e+12 514861
    ## - LDA_02                         1 6.0740e+08 3.1634e+12 514861
    ## - self_reference_max_shares      1 6.0849e+08 3.1634e+12 514861
    ## - weekday_is_saturday            1 6.3987e+08 3.1635e+12 514861
    ## - kw_max_max                     1 7.1884e+08 3.1635e+12 514862
    ## - data_channel_is_world          1 7.3264e+08 3.1635e+12 514862
    ## - LDA_01                         1 7.4074e+08 3.1636e+12 514862
    ## - self_reference_min_shares      1 7.6342e+08 3.1636e+12 514863
    ## - num_self_hrefs                 1 8.3796e+08 3.1636e+12 514863
    ## - n_tokens_title                 1 9.1150e+08 3.1637e+12 514864
    ## - data_channel_is_socmed         1 9.1411e+08 3.1637e+12 514864
    ## - abs_title_subjectivity         1 9.2684e+08 3.1637e+12 514864
    ## - abs_title_sentiment_polarity   1 1.0061e+09 3.1638e+12 514865
    ## - global_subjectivity            1 1.3396e+09 3.1642e+12 514868
    ## - data_channel_is_lifestyle      1 1.9277e+09 3.1647e+12 514873
    ## - data_channel_is_tech           1 2.0246e+09 3.1648e+12 514874
    ## - kw_min_avg                     1 2.2410e+09 3.1651e+12 514876
    ## - data_channel_is_bus            1 2.3241e+09 3.1651e+12 514876
    ## - data_channel_is_entertainment  1 3.2149e+09 3.1660e+12 514884
    ## - num_hrefs                      1 3.8903e+09 3.1667e+12 514890
    ## - kw_max_avg                     1 5.6967e+09 3.1685e+12 514906
    ## - kw_avg_avg                     1 1.1806e+10 3.1746e+12 514959

``` r
#summary the model
summary(lm.step)
```

    ## 
    ## Call:
    ## lm(formula = shares ~ n_tokens_title + num_hrefs + num_self_hrefs + 
    ##     average_token_length + data_channel_is_lifestyle + data_channel_is_entertainment + 
    ##     data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + 
    ##     data_channel_is_world + kw_max_min + kw_avg_min + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + global_subjectivity + 
    ##     min_positive_polarity + abs_title_subjectivity + abs_title_sentiment_polarity + 
    ##     weekday_is_saturday, data = train)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -25271  -2139  -1199   -135 836724 
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                    8.907e+02  6.704e+02   1.329 0.184003    
    ## n_tokens_title                 8.902e+01  3.149e+01   2.827 0.004708 ** 
    ## num_hrefs                      3.794e+01  6.497e+00   5.839 5.29e-09 ***
    ## num_self_hrefs                -5.086e+01  1.877e+01  -2.710 0.006729 ** 
    ## average_token_length          -2.422e+02  1.050e+02  -2.306 0.021110 *  
    ## data_channel_is_lifestyle     -1.431e+03  3.481e+02  -4.111 3.96e-05 ***
    ## data_channel_is_entertainment -1.456e+03  2.743e+02  -5.308 1.11e-07 ***
    ## data_channel_is_bus           -1.254e+03  2.777e+02  -4.513 6.40e-06 ***
    ## data_channel_is_socmed        -9.717e+02  3.433e+02  -2.831 0.004649 ** 
    ## data_channel_is_tech          -1.170e+03  2.778e+02  -4.213 2.53e-05 ***
    ## data_channel_is_world         -9.385e+02  3.703e+02  -2.534 0.011278 *  
    ## kw_max_min                     8.181e-02  5.010e-02   1.633 0.102518    
    ## kw_avg_min                    -5.297e-01  3.055e-01  -1.734 0.082961 .  
    ## kw_max_max                    -1.001e-03  3.986e-04  -2.510 0.012074 *  
    ## kw_avg_max                    -1.545e-03  7.661e-04  -2.017 0.043698 *  
    ## kw_min_avg                    -3.510e-01  7.920e-02  -4.432 9.37e-06 ***
    ## kw_max_avg                    -1.928e-01  2.728e-02  -7.066 1.63e-12 ***
    ## kw_avg_avg                     1.556e+00  1.529e-01  10.173  < 2e-16 ***
    ## self_reference_min_shares      9.897e-03  3.826e-03   2.587 0.009692 ** 
    ## self_reference_max_shares      4.228e-03  1.831e-03   2.309 0.020925 *  
    ## LDA_01                        -9.628e+02  3.779e+02  -2.548 0.010837 *  
    ## LDA_02                        -1.005e+03  4.355e+02  -2.307 0.021041 *  
    ## global_subjectivity            2.562e+03  7.478e+02   3.427 0.000612 ***
    ## min_positive_polarity         -1.936e+03  9.695e+02  -1.997 0.045869 *  
    ## abs_title_subjectivity         1.071e+03  3.757e+02   2.850 0.004371 ** 
    ## abs_title_sentiment_polarity   9.291e+02  3.129e+02   2.970 0.002984 ** 
    ## weekday_is_saturday            6.366e+02  2.688e+02   2.368 0.017878 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10680 on 27723 degrees of freedom
    ## Multiple R-squared:  0.02386,    Adjusted R-squared:  0.02295 
    ## F-statistic: 26.07 on 26 and 27723 DF,  p-value: < 2.2e-16

Calculate the predicted mean square error on the train set:

``` r
train.pred <- predict(lm.step, train)
mean((train.pred - train$shares)^2)
```

    ## [1] 113975195

### On test set

Calculate the predicted mean square error on the test set:

``` r
test.pred <- predict(lm.step, test)
mean((test.pred - test$shares)^2)
```

    ## [1] 175218996

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
