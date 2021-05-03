#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2021 Grant Hadlich
#
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
from utils.utils import tweet
from utils.utils import get_tweets
from utils.utils import create_plot
from utils.utils import parse_tweets
from time import sleep
from datetime import datetime
from models.vader import VaderModel
from models.transformer import TransformerModel
from models.naive_bayes import NaiveBayesModel

if __name__ == "__main__":

    print("Loading Models")
    models = [TransformerModel(), VaderModel(), NaiveBayesModel()]
    print("Loaded Models: " + str([model.name() for model in models]))

    print("Pulling Statuses")
    statuses = get_tweets()

    print("Creating Initial Tweet")
    previous_id = tweet("I've been experimenting with #NLP #Sentiment #Classification with #Python. In the replies are a few models that analyze the sentiment of my home timeline feed on Twitter for the last 24 hours using the Twitter API.")

    for model in models:
        print("Running Model: " + model.name())
        data, _ = parse_tweets(model, statuses)

        filename = create_plot(data, model.name())
        text = model.create_text(data)
        print("Tweeting Data from Model: " + model.name())
        previous_id = tweet(text, filename, in_reply_to_status_id=previous_id)

    print("Completed Tweets")

    #while True:
    #    sleep(3600*12-10)
