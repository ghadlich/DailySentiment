#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2021 Grant Hadlich
#
# Portions modified from https://tinyurl.com/nltksentiment Author: Shaumik Daityari
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE. 
import nltk
from nltk import classify, NaiveBayesClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('twitter_samples', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

import random, string, re

def is_happy_emoji(s):
    emojis = "😎😀😃😁😄😆🤣😂🙂🙃😉😊😇🥰😍🤩😘😗😚😙💋😺😸😹😻😼😽"
    count = 0
    for emoji in emojis:
        count += s.count(emoji)
        if count >= 1:
            return True
    return False

def is_angry_emoji(s):
    emojis = "😡😠🤬😈👿😤"
    count = 0
    for emoji in emojis:
        count += s.count(emoji)
        if count >= 1:
            return True
    return False

def is_sad_emoji(s):
    emojis = "😐😑😒😕😟🙁☹️🥺😦😧😨😰😥😢😭😱😖😣😞😓😩😫🖕😿"
    count = 0
    for emoji in emojis:
        count += s.count(emoji)
        if count >= 1:
            return True
    return False

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                    '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        if (token == ":)"):
            token = "happy_emoji"
        elif (token == ":D" or token == ":d"):
            token = "happy_emoji"
        elif (token == ":-)"):
            token = "happy_emoji"
        elif (is_happy_emoji(token)):
            token = "happy_emoji"
        elif (is_angry_emoji(token)):
            token = "sad_emoji"
        elif (token == ":(" or token == ":-("):
            token = "sad_emoji"
        elif (is_sad_emoji(token)):
            token = "sad_emoji"

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

class NaiveBayesModel(object):

    def __init__(self, seed=25):
        positive_tweets = twitter_samples.strings('positive_tweets.json')
        negative_tweets = twitter_samples.strings('negative_tweets.json')

        stop_words = stopwords.words('english')

        positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
        negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

        positive_cleaned_tokens_list = []
        negative_cleaned_tokens_list = []

        for tokens in positive_tweet_tokens:
            positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

        for tokens in negative_tweet_tokens:
            negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

        positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
        negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

        positive_dataset = [(tweet_dict, "Positive")
                            for tweet_dict in positive_tokens_for_model]

        negative_dataset = [(tweet_dict, "Negative")
                            for tweet_dict in negative_tokens_for_model]

        dataset = positive_dataset + negative_dataset

        random.seed(seed)

        random.shuffle(dataset)

        self.train_data = dataset[:7000]
        self.test_data = dataset[7000:]
        self.classifier = NaiveBayesClassifier.train(self.train_data)

    def name(self):
        return "NaiveBayes"

    def get_classifier(self):
        return self.classifier

    def get_accuracy(self):
        return classify.accuracy(self.classifier, self.test_data)

    def predict(self, input_tweet):
        custom_tokens = remove_noise(word_tokenize(input_tweet))

        return self.classifier.classify(dict([token, True] for token in custom_tokens)), 1.0

    def print_most_informative_features(self):
        self.classifier.show_most_informative_features(10)

    def create_text(self, data):
        positive = sum(data["Positive"].values())
        negative = abs(sum(data["Negative"].values()))
        total = positive + negative

        total_str = str(total)

        text = f"I analyzed the sentiment on the last {total_str} tweets from my home feed using a #NaiveBayes model from #NLTK. "
        if (positive>negative):
            percent = str(round(100*positive/total,1)) + "%"
            text += f"A majority ({percent}) were classified as positive."
        elif (positive == negative):
            text += f"There were an equal amount of positive and negative tweets."
        else:
            percent = str(round(100*negative/total,1)) + "%"
            text += f"A majority ({percent}) were classified as negative."

        text += "\n#Python #NLP #Classification #Sentiment #GrantBot"

        return text

if __name__ == "__main__":

    model = NaiveBayesModel(10)

    print("Model Accuracy: " + str(model.get_accuracy()))

    model.print_most_informative_features()

    text = "Sold! Enjoying with an ice cold @GuinnessIreland right now; Much love, Happy St. Patrick’s Day, and many thanks, folks!"

    print("Text: \"" + text + "\" is " + model.predict(text))
