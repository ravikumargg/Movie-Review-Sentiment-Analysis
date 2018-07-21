
# pickle_input <- function(object, filename, pickle = "pickle") {
#   builtins <- import_builtins()
#   pickle <- import(pickle)
#   handle <- builtins$open(filename, "wb")
#   on.exit(handle$close(), add = TRUE)
#   pickle$dump(object, handle, protocol = pickle$HIGHEST_PROTOCOL)
# }
# 
# pickle_output <- function(filename, pickle = "pickle") {
#   builtins <- import_builtins()
#   pickle <- import(pickle)
#   handle <- builtins$open(filename, "rb")
#   on.exit(handle$close(), add = TRUE)
#   pickle$load(handle)
# }

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
train=fread("train.tsv")
View(train)

# Print out the number of rows in tweets
nrow(train)

index=sample(1:156060,30000,replace = F)
index
mtrain=train[index,]

# Print out the number of rows in tweets
nrow(mtrain)

# ###### modification of data
# # convert to lower case
mtrain$Phrase=tolower(mtrain$Phrase)
# Remove punctuation: rm_punc
mtrain$Phrase = removePunctuation(mtrain$Phrase)
# Remove whitespace
mtrain$Phrase=stripWhitespace(mtrain$Phrase)

# remove stop word
stwf=stopwords("english")
mtrain$Phrase=removeWords(mtrain$Phrase,stwf)

#mtrain$Phrase=data.frame(mtrain$Phrase)
# vPhrase = VectorSource(mtrain$Phrase)
# VCorpusphase=VCorpus(vPhrase)
# VCorpusphase[[1]][1]
# str(VCorpusphase[1])
# Perform word stemming: stem_doc
#stem_doc <- stemDocument(mtrain$Phrase)
# Print stem_doc
#stem_doc

# word.list=str_split(mtrain$Phrase[1],"\\s+")
# mtrain$Phrase[1]=unlist(word.list)
# mtrain[[mtrain$Phrase]][1]


# to see missing values
Num_NA= sapply(mtrain,function(y)length(which(is.na(y)==T)))
NA_Count= data.frame(Item=colnames(mtrain),Count=Num_NA)
NA_Count

mtrain$Phrase[mtrain$Phrase==""]=NA

nrow(mtrain)


#Remove row which are NA
#mtrain=na.omit(mtrain)

library(RTextTools)
library(e1071)
library(SparseM)


dtMatrix_train = create_matrix(mtrain$Phrase)
View(dtMatrix_train)

#clean_corp= str_split(clean_corp," ")

#in Rtexttools we need to put dtm inside a container 
# Configure the training data
container= create_container(dtMatrix_train, mtrain$Sentiment, trainSize=1:30000, virgin=FALSE)

# train a SVM Model
model = train_model(container, "SVM", kernel="linear", cost=1)
summary(model) 

mtest$Sentiment=predict(model,newdata=mtest)

########### test data


mtest=fread("test.tsv")
View(mtest)

# Print out the number of rows in tweets
nrow(mtest)

# ###### modification of data
# # convert to lower case
mtest$Phrase=tolower(mtest$Phrase)
# Remove punctuation: rm_punc
mtest$Phrase = removePunctuation(mtest$Phrase)
# Remove whitespace
mtest$Phrase=stripWhitespace(mtest$Phrase)

# remove stop word
stwf=stopwords("english")
mtest$Phrase=removeWords(mtest$Phrase,stwf)

mtest$Phrase[mtest$Phrase==""]=NA
mtest$Sentiment=0

dtMatrix_test = create_matrix(mtest$Phrase)
View(dtMatrix_test)

mtrain = fread("../input/train.tsv")
mtest = fread("../input/test.tsv")
glimpse(mtrain)


