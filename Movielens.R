# Install packages as required
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(Rcpp)) install.packages("Rcpp", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Download Movielens 10M dataset into a temp file named dl
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# The dataset consists of 3 files: ratings, movies and tags

# Extract ratings file which has 4 columns of userId, movieId, ratting and timestamp
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# Extract movies file which has 3 columns of movieId, title and genres
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

# Combine the 2 files into 1 dataset named movielens
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Split edx into training and test sets
set.seed(1, sample.kind="Rounding") 
index <- createDataPartition(y = edx$rating, times = 1, p = 0.2,
                             list = FALSE)
train_set <- edx[-index,]
test_set <- edx[index,]

# Make sure userId and movieId in test_set are also in train_set
test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Tabulate datasets and number of rows in each dataset
data.frame(Number_of_rows = c(nrow(train_set), nrow(test_set), nrow(validation)), row.names = c("train_set", "test_set", "validation"))

# List names of columns in train_set
colnames(train_set)

# Number of unique userIds in train_set
length(unique(train_set$userId))

# Plot b_u = average user rating
train_set %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black") +
  xlab("Ratings grouped by userId") + ylab("Number of ratings")

# Number of unique movieIds in train_set
length(unique(train_set$movieId))

# Plot b_i = average movie rating
train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating)) %>%
  ggplot(aes(b_i)) +
  geom_histogram(bins = 30, color = "black") +
  xlab("Ratings grouped by movieId") + ylab("Number of ratings")

# Number of unique genres in train_set
length(unique(train_set$genres))

# List first 6 entries of genres column in train_set
head(train_set$genres)

# Plot b_g = average genres rating
train_set %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating)) %>%
  ggplot(aes(b_g)) +
  geom_histogram(bins = 30, color = "black") +
  xlab("Ratings grouped by genres") + ylab("Number of ratings")

library(lubridate)
# Convert timestamp to week
train_set <- mutate(train_set, rating_date = as_datetime(timestamp))
train_set %>% mutate(rating_date = round_date(rating_date, unit = "week")) %>%
  # Plot ratings over time of rating
  group_by(rating_date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(rating_date, rating)) +
  geom_point() +
  geom_smooth(formula = y ~ x, method = "loess") +
  xlab("Date of rating rounded by week") + ylab("Rating")

library(stringr)
# Extract year of movie release from title, which were the 4 digits from the -5 to -2 positions of the titles
train_set <- train_set %>% mutate(release_year = as.Date(str_sub(title, -5, -2), format="%Y"))
# Plot ratings over year of release
train_set %>%
  group_by(release_year) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(release_year, rating)) +
  geom_point() +
  geom_smooth(formula = y ~ x, method = "loess") +
  xlab("Year of release of movie") + ylab("Rating")

# Define RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Assume all movies have same (average) rating
mu <- mean(train_set$rating)
mu

# Calculate RSME for just assuming average rating
rmse_naive <- RMSE(test_set$rating, mu)
rmse_naive

# Include movie effect
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Predict ratings and calculate RMSE with movie effect
predicted_ratings <- mu + test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
rmse_m <- RMSE(predicted_ratings, test_set$rating)
rmse_m

# Calculate regularised estimates of movie effect
# To do that, we pick a factor (lamda) that minimises RMSE
lambdas <- seq(0, 10, 0.25)
just_the_sum <- train_set %>%
  group_by(movieId) %>%
  summarize(s = sum(rating - mu), n_i = n())
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set %>%
    left_join(just_the_sum, by='movieId') %>%
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})
lambda <- lambdas[which.min(rmses)]

# Calculate b_i with lambda value
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())

# Predict ratings and calculate RMSE with movie effect and regularisation
predicted_ratings <- mu + test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
rmse_m <- RMSE(predicted_ratings, test_set$rating)
rmse_m

# Include user effect
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predict ratings and calculate RMSE with movie and user effects
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
rmse_m_u <- RMSE(predicted_ratings, test_set$rating)
rmse_m_u

# Include genre effect
genre_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

# Predict ratings and calculate RMSE with movie, user and genre effects
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)
rmse_m_u_g <- RMSE(predicted_ratings, test_set$rating)
rmse_m_u_g

# Extract year of movie release from title
library(stringr)
train_set <- train_set %>% mutate(release_year = as.Date(str_sub(title, -5, -2), format="%Y"))
test_set <- test_set %>% mutate(release_year = as.Date(str_sub(title, -5, -2), format="%Y"))

# Include release-year effect
release_year_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by="genres") %>%
  group_by(release_year) %>%
  summarize(b_r = mean(rating - mu - b_i - b_u - b_g))

# Predict ratings and calculate RMSE with movie, user, genre and release-year effects
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(release_year_avgs, by='release_year') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_r) %>%
  pull(pred)
rmse_m_u_g_r <- RMSE(predicted_ratings, test_set$rating)
rmse_m_u_g_r

#Tabulate RMSE results
rmse_results <- data.frame(method=c("Just avg", "+Movie", "+User", "+Genre", "+Release-year"), 
                           RMSE=c(rmse_naive, rmse_m, rmse_m_u, rmse_m_u_g, rmse_m_u_g_r))
rmse_results

# Load necessary libraries for Recosystem
library(recosystem)
library(Rcpp)

# Save the first 3 columns of train_set and test_set (userId, movieId and ratings) into text files
# Leave out the row names and column names
write.table(train_set[, 1:3],"train_subset.txt",sep="\t", row.names=FALSE, col.names=FALSE)
write.table(test_set[, 1:3],"test_subset.txt",sep="\t", row.names=FALSE, col.names=FALSE)

# Draw data from text files that contain userId, movieId and ratings
train_reco <- data_file("train_subset.txt", package = "recosystem")
test_reco <- data_file("test_subset.txt",  package = "recosystem")

# Set seed
set.seed(1, sample.kind="Rounding")

# Create a model object
r <- Reco()

# Select suitable tuning parameters using certain parameters in r$tune 
# Note that this might take about an hour or longer to run
opts <- r$tune(train_reco, opts = list(dim = c(20, 40), lrate = c(0.1, 0.2),
                                       costp_l1 = 0, costq_l1 = 0,
                                       nthread = 1, niter = 10))
opts

# Train model using training data train_reco
r$train(train_reco, opts = c(opts$min, nthread = 1, niter = 20))

# Predict ratings using test data test_reco
# Output goes to memory rather than a file
predicted_reco = r$predict(test_reco, out_memory())

# Calculate RMSE by comparing to test_set
rmse_reco <- RMSE(predicted_reco, test_set$rating)
rmse_reco

#Tabulate RMSE results
rmse_results <- data.frame(method=c("Just avg", "+Movie", "+User", "+Genre", "+Release-year", "Recosystem"), 
                           RMSE=c(rmse_naive, rmse_m, rmse_m_u, rmse_m_u_g, rmse_m_u_g_r, rmse_reco))
rmse_results

# Save the first 3 columns of validation set (userId, movieId and ratings) into text file
write.table(validation[, 1:3],"validation.txt",sep="\t", row.names=FALSE, col.names=FALSE)

# Draw data from validation.txt that contain userId, movieId and ratings
val_reco <- data_file("validation.txt", package = "recosystem")

# Predict ratings using data in validation set, sending results to memory rather than a file
predicted_val = r$predict(val_reco, out_memory())

# Calculate RMSE by comparing to validation set
rmse_val <- RMSE(predicted_val, validation$rating)
rmse_val