#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2021 Grant Hadlich
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
import os
import time
import math
import numpy as np
from datetime import datetime, timedelta
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

TWITTER_USER = os.environ.get("TWITTER_USER")

def roundup(x):
    if (x % 100 == 0):
        return x + 10

    return (int(math.ceil(x / 10.0))+1) * 10

def rounddown(x):
    if (-x % 100 == 0):
        return x - 50

    return (int(math.floor(-x / 10.0))+1) * -10

def create_plot(data, model_name=""):
    """ Creates a Histogram Plot of input Twitter Data """

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    current_hour = int(now.strftime("%H"))

    hour_labels = ["12:00AM", "1:00AM", "2:00AM", "3:00AM", "4:00AM", "5:00AM", "6:00AM", "7:00AM", "8:00AM", "9:00AM", "10:00AM", "11:00AM",
                     "12:00PM", "1:00PM", "2:00PM", "3:00PM", "4:00PM", "5:00PM", "6:00PM", "7:00PM", "8:00PM", "9:00PM", "10:00PM", "11:00PM",]

    xticks = []
    x_labels = []
    positive_values = []
    negative_values = []

    for i in range(24):
        j = (i+current_hour+1)%24

        if (data["Positive"][j] + -1*data["Negative"][j]) > 0:
            xticks.append(i)
            x_labels.append(hour_labels[j])
            positive_values.append(data["Positive"][j])
            negative_values.append(data["Negative"][j])

    fig, ax = plt.subplots(1, figsize=(16, 8))
    plt.bar(xticks, positive_values, color = '#337AE3', width =0.5)
    plt.bar(xticks, negative_values, color = '#DB4444', width =0.5)
    # x and y limits
    #plt.xlim(-0.5, 23.5)
    y_min = min(rounddown(min(data["Negative"].values())), -100)
    y_max = max(roundup(max(data["Positive"].values())), 100)
    plt.ylim(y_min, y_max)
    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    #grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.65)
    # x ticks
    plt.xticks(xticks, labels = x_labels, rotation=-45)
    plt.xlabel("Tweets in Hour (PST)")
    # title and legend
    legend_label = ['Positive Tweets', 'Negative Tweets']
    plt.legend(legend_label, ncol = 2, bbox_to_anchor=([1, 1.05, 0, 0]), frameon = False)
    if (model_name == ""):
        plt.title(f"{TWITTER_USER} Twitter Feed Sentiment - {date}\n", loc='left')
        filename = "./data/" + date + "_" + str(current_hour) + ".png"
    else:
        plt.title(f"{TWITTER_USER} Twitter Feed Sentiment - Model: {model_name} - {date}\n", loc='left')
        filename = "./data/" + date + "_" + str(current_hour) + "_" + model_name + ".png"

    fig.savefig(filename, dpi=256)

    return filename

def parse_tweets(model, statuses):
    data = dict()
    data["Positive"] = dict()
    data["Negative"] = dict()
    data["Neutral"] = dict()

    sorted_tweets = dict()
    sorted_tweets["Positive"] = []
    sorted_tweets["Negative"] = []
    sorted_tweets["Neutral"] = []

    for i in range (24):
        data["Positive"][i] = 0
        data["Negative"][i] = 0
        data["Neutral"][i] = 0

    now = datetime.now()

    for status in statuses:
        if (status.lang != "en"):
            continue

        d_0 = status.created_at - timedelta(hours=7, minutes=0)

        if (now-d_0 > timedelta(hours=23, minutes=0)):
            continue

        prediction, _ = model.predict(status.full_text)

        data[prediction][d_0.hour] += 1
        sorted_tweets[prediction].append(status)

    for i in range (24):
        data["Negative"][i] *= -1

    return data, sorted_tweets