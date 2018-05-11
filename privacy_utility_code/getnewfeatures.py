# coding=utf-8
import svc_emotions
import scipy
import numpy
from sklearn.pipeline import FeatureUnion

# list of emotions to get feature lists for
emotion_list = ['anger', 'disgust', 'fear', 'joy', 'love', 'sadness', 'surprise']

# load lists of tweets and corresponding labels into arrays
tweets = svc_emotions.file_to_array("tweets_full.txt")
labels = svc_emotions.file_to_array("labels_full.txt")