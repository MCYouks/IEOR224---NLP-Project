##### IEOR 242 Homework 4 Question 1 ####

setwd("/Users/Jillian/Desktop/UCB MEng/Spring 2018/INDENG 224/224-p/IEOR224---NLP-Project")

#Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre7')
#Sys.setenv(JAVA_HOME='C:\\Program Files (x86)\\Java\\jre7') 

#install.packages("rJava")
#install.packages('tm.plugin.webmining')
#install.packages("tm")
#install.packages("SnowballC")
#install.packages("wordcloud")
#install.packages('RTextTools')
library(rJava)
library(tm)
library(tm.plugin.webmining)
library(SnowballC)
library(wordcloud)
library(MASS)
library(caTools)
library(dplyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(stringr)
library(RTextTools)
library(ROCR)
library(ggplot2)
library(GGally)
library(caTools) # splits
library(rpart) # CART
library(rpart.plot) # CART plotting
library(gbm)
library(lda)
library(MASS)
library(tidyr)


# Function to compute accuracy of a classification model... you're welcome...
tableAccuracy <- function(test, pred) {
  t = table(test, pred)
  a = sum(diag(t))/length(test)
  return(a)
}

# Load the data set
sp500 <- read.csv("224 Proj Headlines.csv", stringsAsFactors = FALSE) 

# We want to predict Useful questions
# Lets create a new variable called "Useful" that converts the 
# "score" number to useful (or not useful)
# anything more than or equal to 1 is useful
#sp500$Label = as.factor(as.numeric(sp500$Label >= 1))
sp500$Label = as.numeric(sp500$Label)
sp500$Label[is.na(sp500$Label)] <- 0

# And remove the old "Avg" column - we won't use it anymore
#sp500$Score <- NULL

str(sp500)

# Before going any further, lets understand the rough distribution of
# useful questions in our data set
table(sp500$Label)
# So what is our baseline model? Predicts all weeks to face stock rise
301/(301+221)

###############
###############

#initial data cleaning
head(sp500[,3:ncol(sp500)])

sp500.body <- as.data.frame(sp500[,3:ncol(sp500)], stringsAsFactors = FALSE)
cols <- c('Top1','Top2','Top3','Top4','Top5','Top6','Top7','Top8','Top9','Top10','Top11','Top12','Top13','Top14','Top15','Top16','Top17','Top18','Top19','Top20','Top21','Top22','Top23','Top24','Top25')
sp500.body$Body <- apply( sp500[ , cols ] , 1 , paste , collapse = " " )

# remove HTML tags

for (i in 1:nrow(sp500.body)) {
  sp500.body$Body[i] <- extractHTMLStrip(sp500.body$Body[i])
}

# remove misc symbols
sp500.body$Body <- gsub("<script\\b[^<]*>[^<]*(?:<(?!/script>)[^<]*)*</script>", "", sp500.body$Body, perl=T)

# remove URL links, punctuation, numbers, newlines, and misc symbols
sp500.body$Body <- unlist(str_replace_all(sp500.body$Body,"[[:punct:]]|[[:digit:]]|http\\S+\\s*|\\n|<|>|=|_|-|#|\\$|\\|"," "))

# Step 1: Convert sp500.body to a "corpus.body"
# A vector source interprets each element of the vector as a document.
# corpus.body creates a collection of documents 
corpus.body = Corpus(VectorSource(sp500.body$Body))

corpus.body

# The sp500.body are now "documents"
corpus.body[[1]]
strwrap(corpus.body[[1]])

# Step 2: Change all the text to lower case.
# tm_map applies an operation to every document in our corpus.body
# Here, that operation is 'tolower', i.e., 'to lowercase'
corpus.body = tm_map(corpus.body, tolower)
# Lets check:
strwrap(corpus.body[[1]])

# Step 3: Remove all punctuation
corpus.body = tm_map(corpus.body, removePunctuation)
# Take a look:
strwrap(corpus.body[[1]])

# Use findFreqTerms to get a feeling for which words appear the most
# Words that appear at least 100 times:
frequencies = DocumentTermMatrix(corpus.body)
as.character(findFreqTerms(frequencies, lowfreq=100))
sp500body.freqterms <- as.character(findFreqTerms(frequencies, lowfreq=100))

# Step 4: Remove stop words
# First, take a look at tm's stopwords:
# "english" is all the stopwords in english
stopwords("english")[1:10]
length(stopwords("english"))
# Just remove stopwords:
# corpus.body = tm_map(corpus.body, removeWords, stopwords("english"))
# Remove stopwords, "ggplot2", "ggplot" - these are words common to all of our sp500.body
corpus.body = tm_map(corpus.body, removeWords, c("ggplot","ggplot2", sp500body.freqterms, stopwords("english")))
# Take a look:
strwrap(corpus.body[[1]])

# Step 5: Stem our document
# Recall, this means chopping off the ends of words that aren't maybe
# as necessary as the rest, like 'ing' and 'ed'
corpus.body = tm_map(corpus.body, stemDocument)
# Take a look:
strwrap(corpus.body[[1]])

# Seems we didn't catch all of the apples...
#corpus.body = tm_map(corpus.body, removeWords, c("appl"))

# Step 6: Create a word count matrix (rows are sp500.body, columns are words)
# We've finished our basic cleaning, so now we want to calculate frequencies
# of words across the sp500.body
frequencies = DocumentTermMatrix(corpus.body)
# We can get some summary information by looking at this structure
frequencies
# model is very sparse: a lot of words take value of 1

# Step 7: Account for sparsity
# We currently have way too many words, which will make it hard to train
# our models and may even lead to overfitting.

# Our solution to the possibility of overfitting is to only keep terms
# that appear in x% or more of the sp500.body. For example:
# 1% of the sp500.body or more 
sparse = removeSparseTerms(frequencies, 0.99)
sparse

# 0.5% of the sp500.body or more (= 6 or more)
#sparse = removeSparseTerms(frequencies, 0.995)
# How many did we keep?
#sparse

# Let's keep it at the 1%
#sparse = removeSparseTerms(frequencies, 0.99)
#sparse

# Step 8: Create data frame from the document-term matrix
sp500.bodyTM = as.data.frame(as.matrix(sparse))
# We have some variable names that start with a number, 
# which can cause R some problems. Let's fix this before going
# any further
colnames(sp500.bodyTM) = make.names(colnames(sp500.bodyTM))
# This isn't our original dataframe, so we need to bring that column
# with the dependent variable into this new one
#sp500.bodyTM$Negative = sp500.body$Negative
#sp500.bodyTM$Negative <- NULL

# Bonus: make a cool word cloud!
wordcloud(corpus.body, max.words = 100, random.order = FALSE, rot.per = .1,scale = c(1.5, 0.2), 
          colors = brewer.pal(8, "Dark2"))

####################################
# Part (a)(iv)
####################################

#colnames(sp500.titleTM)
colnames(sp500.bodyTM)

# This isn't our original dataframe, so we need to bring that column
# with the dependent variable into this new one
#sp500TM <- merge(sp500.titleTM,sp500.bodyTM,by=0)
sp500TM <- sp500.bodyTM
sp500TM$Label = sp500$Label
sp500TM$Row.names <- NULL

colnames(sp500TM) = make.names(colnames(sp500TM))

####################################
# Part (b)
####################################

# Split data into training and testing sets
set.seed(123)  # So we get the same results
spl = sample.split(sp500TM$Label, SplitRatio = 0.8)

sp500Train = sp500TM %>% filter(spl == TRUE)
sp500Test = sp500TM %>% filter(spl == FALSE)

# Side note: explain the pipe %>% 
# equivalent to filter(TweetsTM, spl==TRUE)

# Baseline accuracy
table(sp500Train$Label)

241/(241+177)

table(sp500Test$Label)

60/(60+44)

# Linear Discriminant Analysis
# Normally distributed features?
library(MASS)
lda.mod = lda(Label ~ ., data = sp500Train)

predict.lda = predict(lda.mod, newdata = sp500Test)$class
table(sp500Test$Label, predict.lda)
tableAccuracy(sp500Test$Label, predict.lda)


### Logistic Regression

sp500Log = glm(Label ~ ., data = sp500Train, family = "binomial")
# You may see a warning message - suspicious, but we will just ignore this
summary(sp500Log)

# Predictions on test set
PredictLog = predict(sp500Log, newdata = sp500Test, type = "response")
table(sp500Test$Label, PredictLog > 0.5)
tableAccuracy(sp500Test$Label, PredictLog > 0.5)
# Not as good as CART or RF

#TPR: 


rocr.log.pred <- prediction(PredictLog, sp500Test$Label)
logPerformance <- performance(rocr.log.pred, "tpr", "fpr")
plot(logPerformance, colorize = TRUE)
abline(0, 1)
as.numeric(performance(rocr.log.pred, "auc")@y.values)

# But what about training set?
PredictLogTrain = predict(sp500Log, type = "response")
table(sp500Train$Label, PredictLogTrain > 0.5)
tableAccuracy(sp500Train$Label, PredictLogTrain > 0.5)

### 
# Cross-validated CART model
set.seed(3421)
train.cart = train(Label ~ .,
                   data = sp500Train,
                   method = "rpart",
                   tuneGrid = data.frame(cp=seq(0, 0.4, 0.002)),
                   trControl = trainControl(method="cv", number=10))
train.cart
train.cart$results

ggplot(train.cart$results, aes(x = cp, y = Accuracy)) + geom_point(size = 2) + geom_line() + 
  ylab("CV Accuracy") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

mod.cart = train.cart$finalModel
prp(mod.cart)

predict.cart = predict(mod.cart, newdata = sp500Test, type = "class") # why no model.matrix? 
table(sp500Test$Label, predict.cart)
tableAccuracy(sp500Test$Label, predict.cart)


# Cross validated RF
# WARNING: this took me approx. 1 hour to run
set.seed(311)
train.rf = train(Label ~ .,
                 data = sp500Train,
                 method = "rf",
                 tuneGrid = data.frame(mtry = 1:25),
                 trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE))
train.rf
train.rf$results

ggplot(train.rf$results, aes(x = mtry, y = Accuracy)) + geom_point(size = 2) + geom_line() + 
  ylab("CV Accuracy") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

mod.rf = train.rf$finalModel
predict.rf = predict(mod.rf, newdata = sp500Test)
table(sp500Test$Useful, predict.rf)
tableAccuracy(sp500Test$Useful, predict.rf)

# Variable importance using dplyr and the pipe
# Step 1: turn mod.rf$importance into a data frame
# Step 2: create a new variable (column) called Words equal to the rownames of mod.rf$importance
# Step 3: arrange in descendending order according to variable importance measure
as.data.frame(mod.rf$importance) %>%
  mutate(Words = rownames(mod.rf$importance)) %>%
  arrange(desc(MeanDecreaseGini))

# Let's do stepwise regression
# WARNING: this took ~20 mins
# reduce the numbers of features used
sp500stepLog = step(sp500Log, direction = "backward")
summary(sp500stepLog)
length(sp500stepLog$coefficients)
# left with about 35 features
# may want to manually prune beyond this point...

PredictStepLog = predict(sp500stepLog, newdata = sp500Test, type = "response")
table(sp500Test$Useful, PredictStepLog > 0.5)
tableAccuracy(sp500Test$Useful, PredictStepLog > 0.5)




