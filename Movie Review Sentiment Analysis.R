# these library are required
library(tm)
library(SnowballC)
library(tidytext)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gutenbergr)
library(stringr)
library(text2vec)
library(keras)
library(Rtsne)
library(rJava)
library(RWeka)
library(maptpx)
library(textir)
library(qdap)
library(wordcloud)
library(dplyr)


setwd("E:/Kaggle case-study/Movie Review Sentiment Analysis")
# read data using data.table package
library(data.table)
require(data.table)
mtrain=fread("train.tsv")
View(mtrain)
head(mtrain)

# Print out the number of rows in tweets
nrow(mtrain)

clean_corpus <- function(corpus){
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, stopwords("en"))
  return(corpus)
}
# Apply your customized function to the tweet_corp: clean_corp
mtrain$Phrase = clean_corpus(mtrain$Phrase)
mtrain$Phrase[[1]][1]

# Isolate text from tweets: tweets_text
mtrain$text =mtrain$Phrase
str(mtrain$text)

library(tm)
mtrain_source = VectorSource(mtrain$Phrase)
mtrain_source
#nested list, or list of lists
# Make a volatile corpus: tweets_corpus
mtrain_corpus <- VCorpus(mtrain_source)
# Print out the tweets_corpus
mtrain_corpus
View(mtrain_corpus)
mtrain_corpus[[1]][1]



library(RTextTools)
library(e1071)
library(SparseM)

dtMatrix <- create_matrix(data["clean_corp"])

clean_corp= str_split(clean_corp," ")

#in Rtexttools we need to put dtm inside a container 
# Configure the training data
container <- create_container(dtMatrix, data$IsSunny, trainSize=1:11, virgin=FALSE)

# train a SVM Model
model <- train_model(container, "SVM", kernel="linear", cost=1)





























clean_corp= unlist(clean_corp)
clean_corp

####Applying preprocessing steps to a corpus
#Let's find the most frequent words in our tweets_text and see whether we should get rid of some
frequent_terms = freq_terms(clean_corp, 30)
plot(frequent_terms)

# ###### modification of data
# # convert to lower case
# mtrain$Phrase=tolower(mtrain$Phrase)
# # Remove punctuation: rm_punc
# mtrain$Phrase = removePunctuation(mtrain$Phrase)
# # Remove whitespace
# mtrain$Phrase=stripWhitespace(mtrain$Phrase)
# #mtrain$Phrase=data.frame(mtrain$Phrase)
# vPhrase = VectorSource(mtrain$Phrase)
# VCorpusphase=VCorpus(vPhrase)
# VCorpusphase[[1]][1]
# str(VCorpusphase[1])
# # Perform word stemming: stem_doc
# stem_doc <- stemDocument(mtrain$Phrase)
# # Print stem_doc
# stem_doc

# remove stop word
stwf=stopwords("english")
stwf
mtrain$Phrase=removeWords(mtrain$Phrase,stwf)

word.list=str_split(mtrain$Phrase[1],"\\s+")
mtrain$Phrase[1]=unlist(word.list)
mtrain[[mtrain$Phrase]][1]



# to see missing values
Num_NA= sapply(mtrain,function(y)length(which(is.na(y)==T)))
NA_Count= data.frame(Item=colnames(mtrain),Count=Num_NA)
NA_Count
 

#Encode bi_gram with atleast frquency
mPhrase=ngrams(mPhrase,"bi",3)
#document termfrequency
dmt=custom.dtm(mphase,"tf")

#### sentiment analysis########################


#calculate the polarity from qdap dictionary
pol=polarity(clean_text)
#word count in each row
wc=pol


#############################################
# naive bayes
mat= create_matrix(mtrain$Phrase, language="english", 
                   removeStopwords=FALSE, removeNumbers=TRUE, 
                   stemWords=FALSE, tm::weightTfIdf)

dtMatrix <- create_matrix(clean_corp)

mat = as.matrix(mat)

classifier = naiveBayes(mat[1:160,], as.factor(sentiment_all[1:160]))
predicted = predict(classifier, mat[161:180,]); predicted

table(sentiment_test, predicted)
recall_accuracy(sentiment_test, predicted)

# the other methods
mat= create_matrix(tweet_all, language="english", 
                   removeStopwords=FALSE, removeNumbers=TRUE, 
                   stemWords=FALSE, tm::weightTfIdf)

container = create_container(mat, as.numeric(sentiment_all),
                             trainSize=1:160, testSize=161:180,virgin=FALSE) #????????????removeSparseTerms

models = train_models(container, algorithms=c("MAXENT",
                                              "SVM",
                                              #"GLMNET", "BOOSTING", 
                                              "SLDA","BAGGING", 
                                              "RF", # "NNET", 
                                              "TREE" 
))

# test the model
results = classify_models(container, models)
table(as.numeric(as.numeric(sentiment_all[161:180])), results[,"FORESTS_LABEL"])
recall_accuracy(as.numeric(as.numeric(sentiment_all[161:180])), results[,"FORESTS_LABEL"])

# formal tests
analytics = create_analytics(container, results)
summary(analytics)

head(analytics@algorithm_summary)
head(analytics@label_summary)
head(analytics@document_summary)
analytics@ensemble_summary # Ensemble Agreement

# Cross Validation
N=3
cross_SVM = cross_validate(container,N,"SVM")
cross_GLMNET = cross_validate(container,N,"GLMNET")
cross_MAXENT = cross_validate(container,N,"MAXENT")
