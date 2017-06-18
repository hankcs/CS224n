#!/bin/bash

DATASETS_DIR="utils/datasets"
mkdir -p $DATASETS_DIR
cd $DATASETS_DIR

# Get Stanford Sentiment Treebank
if hash wget 2>/dev/null; then
  wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
else
  curl -O http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
fi
unzip stanfordSentimentTreebank.zip
rm stanfordSentimentTreebank.zip

# Get 50D GloVe vectors
if hash wget 2>/dev/null; then
  wget http://nlp.stanford.edu/data/glove.6B.zip
else
  curl -O http://nlp.stanford.edu/data/glove.6B.zip
fi
unzip glove.6B.zip
rm glove.6B.100d.txt glove.6B.200d.txt glove.6B.300d.txt glove.6B.zip
