#load package
library(tidyverse)
library(haven)
library(knitr)
library(rmarkdown)

# Read data and remove the first two columns since they are non-predictive.
news<- read_csv("/Users/yilinxie/Desktop/ST558/Project/Project2/OnlineNewsPopularity.csv")

data.frame(output_file = "MondayAnalysis.md", params = list(weekday = "weekday_is_monday"))

#get unique weekdays
weekdays <- c("weekday_is_monday", "weekday_is_tuesday", "weekday_is_wednesday",
              "weekday_is_thursday", "weekday_is_friday", "weekday_is_saturday",
              "weekday_is_sunday")

#create filenames
output_file <- paste0(weekdays, ".md")

#create a list for each weekdays
params = lapply(weekdays, FUN = function(x){list(weekday = x)})

#put into a data frame 
reports <- tibble(output_file, params)
reports

#need to use x[[1]] to get at elements since tibble doesn't simplify
apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "README.Rmd", output_file = x[[1]], params = x[[2]])
      })
