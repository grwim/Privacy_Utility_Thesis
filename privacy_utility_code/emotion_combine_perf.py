import svc_emotions
import scipy
import numpy
from sklearn.pipeline import FeatureUnion
from sklearn.cross_validation import train_test_split
from sklearn import svm
import numpy as np
import re
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from twokenize import tokenize

import naive_bayes
from sklearn.naive_bayes import MultinomialNB

import pdb
import tweet_processing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier



def balanced_train_test_split(tweets, binary_labels, test_size, random_state, balance_nbr, none_counter):

    X_train, X_test, y_train, y_test = train_test_split(tweets, binary_labels, test_size=test_split, random_state=rand_state)

    X_train = X_train.tolist()
    X_test = X_test.tolist()
    y_train = y_train.tolist()
    y_test = y_test.tolist()

    count_current_emotion_occurences = len([x for x in y_test if x == 1.0])

    if count_current_emotion_occurences > balance_nbr: # if too many of a given emotion in test set, then need to remove from test and put in train
        num_to_remove = count_current_emotion_occurences - balance_nbr

        # obtain list of indices of Trues
        true_index_list = [i for i, x in enumerate(y_test) if x == 1.0]

        # take tweets and their corresponding labels from test and put into train until only specified number of a given emotion in test set
        i = 0
        while num_to_remove > 0:

            # increment appropriate count

            tweet = X_test[true_index_list[i] - i] # get tweet contents of index to be removed
            label = y_test[true_index_list[i] - i]

            del X_test[true_index_list[i] - i]
            del y_test[true_index_list[i] - i]

            X_train.append(tweet)
            y_train.append(label)

            i += 1
            num_to_remove -= 1
            # print true_index_list[i]
            # print len([x for x in y_test if x == 1.0])
            # pdb.set_trace()

    elif count_current_emotion_occurences < balance_nbr: # otherwise do opposite
        num_to_add = balance_nbr - count_current_emotion_occurences

        true_index_list = [i for i, x in enumerate(y_train) if x == 1.0]

        # take tweets and their corresponding labels from train and put into test until specified number of a given emotion in train set
        i = 0
        while num_to_add > 0:
            tweet = X_train[true_index_list[i] - i] # get tweet contents of index to be removed
            label = y_train[true_index_list[i] - i]

            del X_train[true_index_list[i] - i]
            del y_train[true_index_list[i] - i]

            X_test.append(tweet)
            y_test.append(label)


            i += 1
            num_to_add  -= 1

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    return X_train, X_test, y_train, y_test, none_counter

reload(sys)
sys.setdefaultencoding('utf8')

emotion_list = ['anger', 'disgust', 'fear', 'joy', 'love', 'sadness', 'surprise']

tweets = tweet_processing.clean("tweets_full.txt")
labels = svc_emotions.file_to_array("labels_full.txt")

# set parameters for testing
test_split = 0.2    # train/test split ratio
rand_state = 111    # random seed for split
bal_nbr = 20
none_counter = 0
start_index_rich = 0
start_index_non = 0

for emotion in emotion_list:
    print emotion

    combine = True

    if combine:
        if emotion == 'anger': # anger will be the combined
            binary_labels = svc_emotions.to_binary_classification_merged(labels, 'anger', 'disgust')
        elif emotion == 'love':
            binary_labels = svc_emotions.to_binary_classification_merged(labels, 'love', 'joy')
    else:
        binary_labels = svc_emotions.to_binary_classification(labels, emotion)

    X_train_bal, X_test_bal, y_train_bal, y_test_bal, none_counter = balanced_train_test_split(tweets, binary_labels, test_split, rand_state, bal_nbr, none_counter)


    nb_classifier, nb_grams, nb_features = naive_bayes.train(X_train_bal, y_train_bal)

    ## to obtain all tweets used in training that are classified as belonging to a given emotion
    # for x in range(len(y_train_bal)):
    #     if (y_train_bal[x] == 1.0):
    #         print X_train_bal[x]

    # show most important features
    importances = nb_classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in nb_classifier.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    feature_importance = []
    i = 0
    for gram in nb_grams:
        pair = []
        pair.append(std[i])
        pair.append(gram)
        feature_importance.append(pair)
        i += 1

    nb_correct, nb_incorrect, nb_tp, nb_fp, nb_tn, nb_fn, nb_predictions = naive_bayes.test(X_test_bal, y_test_bal, nb_classifier, nb_grams)

    all_correct = 0
    all_wrong = 0
    svm_correct = 0
    lr_correct = 0
    nb_correct = 0
    svm_tp = 0
    svm_fn = 0
    svm_tn = 0
    svm_fp = 0
    lr_tp = 0
    lr_fn = 0
    lr_tn = 0
    lr_fp = 0
    total_positive = 0
    total_negative = 0
    total = 0

    print "     nb true pos: " + str(nb_tp)
    print "     nb false neg: " + str(nb_fn)
    print "     nb false pos: " + str(nb_fp)
    print "     nb true neg: " + str(nb_tn)
    print "\n"
    print "     NB precision: " +    str( precision_score(   y_test_bal, nb_predictions)  )
    print "     NB recall: " +          str( recall_score(         y_test_bal, nb_predictions) )






















































