## naive bayes code from yifang -- very slight modifications only

import numpy
from sklearn.naive_bayes import MultinomialNB
import nltk
import re
import random
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import sys


from sklearn.ensemble import RandomForestClassifier

import pdb

def clean():
    with open("english_tweet_filtered.txt", "r") as infile:
        with open("english_tweet_filtered_cleaned.txt", "w") as outfile:
            text = ""
            line = ""
            for line in infile:
                line = line.strip()
                line = re.sub("[^\\x01-\\x7f]+", "", line)
                if re.match("[0-9]{6,}", line):
                    text = line
                elif re.match("^(Neutral|Positive|Negative|Noise)$", line):
                    label = line
                    outfile.write(label + "\t" + text + "\n")
                    outfile.flush()
                else:
                    text = ""
                    label = ""


def train(tweets, labels):
    stop = stopwords.words("english")

    text_features_dict = {}
    grams = set()
    for tweet in tweets:
        unigrams = re.split("[\\s]+", tweet)
        unigrams = [i for i in unigrams if i not in stop]
        for unigram in unigrams:
            grams.add(unigram)
            if unigram not in text_features_dict:
                text_features_dict[unigram] = True
        bigrams = nltk.bigrams(unigrams)
        for bigram in bigrams:
            grams.add(bigram)
            if unigram not in text_features_dict:
                text_features_dict[unigram] = True

    text_features = []
    for key, value in text_features_dict.items():
        text_features.append(key)

    features = []
    for tweet in tweets:
        unigrams = re.split("[\\s]+", tweet)
        unigrams = [i for i in unigrams if i not in stop]
        bigrams = nltk.bigrams(unigrams)
        unigramToCnt = nltk.FreqDist(unigrams)
        bigramToCnt = nltk.FreqDist(bigrams)
        feature = []
        for gram in grams:
            if gram in unigramToCnt:
                feature.append(unigramToCnt[gram])
            elif gram in bigramToCnt:
                feature.append(bigramToCnt[gram])
            else:
                feature.append(0)

        features.append(feature)

    map = {}
    for label in labels:
        if label not in map:
            map[label] = 0
        map[label] = map[label] + 1

    x = numpy.vstack(features)
    y = numpy.array(labels)
    # classifier = MultinomialNB(fit_prior=False)

    classifier = RandomForestClassifier()
    # classifier = MultinomialNB()
    classifier.fit(x,y)

    return classifier, grams, text_features

def trim(tweet):
    tweet = re.sub("http[^\\s]*", " ", tweet)
    tweet = re.sub("https[^\\s]*", " ", tweet)
    tweet = re.sub("www[^\\s]*", " ", tweet)
    return tweet

def test(tweets, labels, classifier, grams):
    stop = stopwords.words("english")
    features = []
    for tweet in tweets:
        unigrams = re.split("[\\s]+", tweet)
        unigrams = [i for i in unigrams if i not in stop]
        bigrams = nltk.bigrams(unigrams)
        unigramToCnt = nltk.FreqDist(unigrams)
        bigramToCnt = nltk.FreqDist(bigrams)
        feature = []
        for gram in grams:
            if gram in unigramToCnt:
                feature.append(unigramToCnt[gram])
            elif gram in bigramToCnt:
                feature.append(bigramToCnt[gram])
            else:
                feature.append(0)

        features.append(feature)

    predict = classifier.predict(features)

    correct = 0
    incorrect = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(0, len(labels)):
        label = labels[i]
        if (predict[i] == label):
            correct = correct + 1
            if ((predict[i] == 'false') or (predict[i] == 0.0)):
                tn += 1
            else:
                tp += 1
        else:
            incorrect = incorrect + 1
            if ((predict[i] == 'false') or (predict[i] == 0.0)):
                fn += 1
            else:
                fp += 1
            # print predict[i] + "\t" + label + "\t" + tweets[i]
    return correct, incorrect, tp, fp, tn, fn, predict


if __name__ == "__main__":

    reload(sys)
    sys.setdefaultencoding("utf-8")

    #stemmer = SnowballStemmer("english")
    stemmer = LancasterStemmer()
    lemmatizer = WordNetLemmatizer()
    #tokens = re.split("[\\s]+", "I saw two guys yesterday.")
    #tokens = [stemmer.stem(token) for token in tokens]
    #tokens = [lemmatizer.lemmatize(token) for token in tokens]

    lines = []

    with open("labelledtweets.txt", "r") as infile:
        for line in infile:
            line = line.strip()
            line = line.lower()
            line = trim(line)
            lines.append(line.strip())

    emotion_list = ['joy', 'love', 'sadness', 'surprise', 'fear', 'anger', 'disgust']
    for emotion in emotion_list:
        tweets = []
        labels = []
        random.shuffle(lines)
        for line in lines:
            tokens = re.split("[\t]+", line)
            if len(tokens) > 1:
                label = tokens[0]
                tweet = tokens[1]
                subs = re.split("[\\s]+", tweet)
                subs = [stemmer.stem(sub) for sub in subs]
                #tweet = " ".join(subs)
                if (label == emotion):
                    tweets.append(tweet)
                    labels.append("true")
                else:
                    tweets.append(tweet)
                    labels.append("false")

        step = len(tweets) / 10
        totalCorrect = 0
        totalIncorrect = 0
        totalTP = 0
        totalFP = 0
        totalTN = 0
        totalFN = 0

        train(tweets, labels)

        for i in range(0, 10):
            testing = tweets[i * step : step + i * step]
            testingLabels = labels[i * step : step + i * step]

            training = tweets[0 : i * step] + tweets[step + i * step : len(tweets)]
            trainingLabels = labels[0 : i * step] + labels[step + i * step : len(tweets)]

            classifier, grams = train(training, trainingLabels)
            correct, incorrect, tp, fp, tn, fn = test(testing, testingLabels, classifier, grams)
            totalCorrect = totalCorrect + correct
            totalIncorrect = totalIncorrect + incorrect
            totalFN += fn
            totalTP += tp
            totalTN += tn
            totalFP += fp
            print "*CORRECT : " + str(correct)
            print "*INCORRECT : " + str(incorrect)

        precision = totalTP / (totalTP + float(totalFP))
        recall = totalTP / (totalTP + float(totalFN))
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)


        print emotion
        print "TOTAL CORRECT : " + str(totalCorrect)
        print "TOTAL INCORRECT : " + str(totalIncorrect)
        print "TOTAL TP : " + str(totalTP)
        print "TOTAL FP : " + str(totalFP)
        print "TOTAL TN : " + str(totalTN)
        print "TOTAL FN : " + str(totalFN)
        print "PRECISION : " + str(precision)
        print "RECALL : " + str(recall)
        print "F1 : " + str(f1)
