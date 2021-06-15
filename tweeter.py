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
from twitterutils.twitterutils import tweet
from twitterutils.twitterutils import get_tweets
from utils.utils import create_plot
from utils.utils import parse_tweets
from time import sleep
from sentimentmodels.get_models import get_models
from datetime import datetime
import os

if __name__ == "__main__":
    now = datetime.now()
    filename = now.strftime("%Y_%m_%d_%H") + "_timeline.txt"

    print("Loading Models")
    models = get_models()
    print("Loaded Models: " + str([model.name() for model in models]))

    print("Pulling Statuses")
    dir = "./raw_tweets"

    os.makedirs(dir, exist_ok=True)

    path = os.path.join(dir, filename)
    statuses = get_tweets(output_file=path)

    print("Creating Initial Tweet")
    text = "How negative was my Twitter feed in the last few hours? "
    text += "In the replies are a few models that analyze the sentiment of my home timeline feed on Twitter for the last 24 hours using the Twitter API."
    text += "\nGitHub: https://github.com/ghadlich/DailySentiment"
    text += "\n#NLP #Python"
    previous_id = tweet(text, image_path="./data/logo.png")

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
