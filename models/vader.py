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
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

class VaderModel(object):

    def __init__(self):
        self.classifier = SentimentIntensityAnalyzer()

    def name(self):
        return "Vader"

    def get_classifier(self):
        return self.classifier

    def predict(self, input_tweet):
        result = self.classifier.polarity_scores(input_tweet)

        if (abs(result['compound']) < 0.25):
            result['label'] = "Neutral"
        elif result['compound'] >= 0.25:
            result['label'] = "Positive"
        else:
            result['label'] = "Negative"

        return result['label'], result['compound']

    def create_text(self, data):
        positive = sum(data["Positive"].values())
        negative = abs(sum(data["Negative"].values()))
        neutral = abs(sum(data["Neutral"].values()))
        total = positive + negative + neutral
        pos_percent = str(round(100*positive/total,1)) + "%"
        neg_percent = str(round(100*negative/total,1)) + "%"
        neu_percent = str(round(100*neutral/total,1)) + "%"

        total_str = str(total)

        text = f"I analyzed the sentiment on the last {total_str} tweets from my home feed using a pretrained #VADER model from #NLTK. "
        if (positive>(negative+neutral)):
            text += f"A majority ({pos_percent}) were classified as positive with {neu_percent} neutral and {neg_percent} negative."
        elif (positive>negative and positive>neutral):
            text += f"A plurality ({pos_percent}) were classified as positive with {neu_percent} neutral and {neg_percent} negative."
        elif (negative>(positive+neutral)):
            text += f"A majority ({neg_percent}) were classified as negative with {neu_percent} neutral and {pos_percent} positive."
        elif (negative>positive and negative>neutral):
            text += f"A plurality ({neg_percent}) were classified as negative with {neu_percent} neutral and {pos_percent} positive."
        elif (neutral>(positive+negative)):
            text += f"A majority ({neu_percent}) were classified as neutral with {pos_percent} positive and {neg_percent} negative."
        elif (neutral>positive and neutral>negative):
            text += f"A plurality ({neu_percent}) were classified as neutral with {pos_percent} positive and {neg_percent} negative."
        else:
            text += f"There were an equal amount of positive, neutral, and negative tweets."

        text += "\n#Python #NLP #Classification #Sentiment #GrantBot"

        return text

if __name__ == "__main__":

    model = VaderModel()
    text = "Sold! Enjoying with an ice cold @GuinnessIreland right now; Much love, Happy St. Patrick’s Day, and many thanks, folks!"

    pred, score = model.predict(text)

    print("Text: \"" + text + "\" is " + pred + " with a score of " + str(score))
