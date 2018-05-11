# encoding=utf8
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
import pdb
import tweet_processing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# REMINDER: naive_bayes is currently random forest


def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

def get_twenty_twenty ( tweets, binary_labels, start_index_rich, start_index_non ):
    # get all tweets


    # get twenty emotions from the given emotion
    count_rich = 0
    count_non = 0

    final_tweets = []
    final_labels = []

    # populate
    i = 0
    while ( count_rich < 20) :
        if  bool (  re.search( r'emotion_rich', tweets[i]) ):
            tweets[i] = re.sub(r'emotion_rich', "", tweets[i]) # remove emotion_rich tag

            final_tweets.append( tweets[i] )
            final_labels.append ( binary_labels[i] )

            start_index_rich = i + 1
            count_rich = count_rich + 1
        i = i + 1

    i = 0
    while ( count_non < 20) :
        if  not bool (  re.search( r'emotion_rich', tweets[i]) ):

            final_tweets.append( tweets[i] )
            final_labels.append ( binary_labels[i] )

            start_index_non = i + 1
            count_non = count_non + 1
        i = i + 1

    return final_tweets, final_labels, start_index_rich, start_index_non


def balanced_train_test_split(tweets, binary_labels, test_size, random_state, balance_nbr, none_counter):

    X_train, X_test, y_train, y_test = train_test_split(tweets, binary_labels, test_size=test_split, random_state=rand_state)

    X_train = X_train.tolist()
    X_test = X_test.tolist()
    y_train = y_train.tolist()
    y_test = y_test.tolist()

    count_current_emotion_occurences = len([x for x in y_test if x == 1.0])

    # examine

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
        # i = 0
        # while num_to_add > 0:
        #     tweet = X_train[true_index_list[i] - i] # get tweet contents of index to be removed
        #     label = y_train[true_index_list[i] - i]

        #     del X_train[true_index_list[i] - i]
        #     del y_train[true_index_list[i] - i]

        #     X_test.append(tweet)
        #     y_test.append(label)


        #     i += 1
        #     num_to_add  -= 1

    # add 20 non relevant tweets, increment counter so don't have same none-tweets for next emotion
    # i = 0
    # while i < 20:
    #     X_test.append( none_tweets[i] )
    #     y_test.append(0.0)

    #     none_counter = none_counter + 1
    #     i = i + 1

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    return X_train, X_test, y_train, y_test, none_counter


# set to handle encoding issues
reload(sys)
sys.setdefaultencoding('utf8')










# ------------        load data for testing ---------------
emotion_list = ['anger', 'disgust', 'fear', 'joy', 'love', 'sadness', 'surprise']

# NO LOVE for Mohammed set
# emotion_list = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

# check about 'none' label --> how to deal with?


# tweets = svc_emotions.file_to_array("tweets_full.txt")

#  normal tweets to incorporate #
tweets_legacy= tweet_processing.clean("tweets_full.txt")
labels_legacy = svc_emotions.file_to_array("labels_full.txt")



# new annotated tweets #

# file = open("emotion_tweets_labelled_Saif_Mohammad.txt")
# array = np.array([])
# for line in file:
#     array = np.append(array, line.rstrip())
# file.close()

# tweets from Mohammed Set
# tweets_mohammad =  tweet_processing.clean_Mohammed_Format("emotion_tweets_labelled_Saif_Mohammad.txt")
# labels_mohammad = tweet_processing.get_Labels_Mohammed_Format("emotion_tweets_labelled_Saif_Mohammad.txt")

# tweets = np.concatenate((tweets_legacy, tweets_mohammad))
# labels = np.concatenate((labels_legacy, labels_mohammad))

tweets = tweets_legacy
labels = labels_legacy

# set parameters for testing
test_split = 0.2	# train/test split ratio
rand_state = 111	# random seed for split
bal_nbr = 20
none_counter = 0
start_index_rich = 0
start_index_non = 0


#file to output results
outfile_name = 'alg_compare_' + str(rand_state) + '.txt'
outfile = open(outfile_name, 'w')

for emotion in emotion_list:

    print emotion

    outfile.write("\n\nEMOTION: " + emotion + "\n")


    binary_labels = svc_emotions.to_binary_classification(labels, emotion)

    # love & joy
    # anger & disgust

    if emotion == 'anger': # anger will be the combined
        binary_labels = svc_emotions.to_binary_classification_merged(labels, 'anger', 'disgust')
    elif emotion == 'love':
        binary_labels = svc_emotions.to_binary_classification_merged(labels, 'love', 'joy')

    # get predictions for SVM model
    # NOTE: must have vocab files for both tokens and wordnet, in form "tokens_emotion" and "wordnet_emotion" respectively
    # 1. get feature-document mx for all tweets
    estimators = [('tokens', svc_emotions.get_token_vectorizer(vocab=True, emotion=emotion)), ('wordnet', svc_emotions.get_wordnet_vectorizer(vocab=True, emotion=emotion))]
    full_vectorizer = FeatureUnion(estimators)
    term_mx = full_vectorizer.fit_transform(tweets)
    # 2. split data into test and train

    X_train, X_test, y_train, y_test = train_test_split(term_mx, binary_labels, test_size=test_split, random_state=rand_state)
    X_train_tweets, X_test_tweets, y_train_tweets, y_test_tweets = train_test_split(tweets, binary_labels, test_size=test_split, random_state=rand_state)

    # 3. generate predictions from model
    clf = svm.LinearSVC().fit(X_train, y_train)
    svm_predictions = clf.predict(X_test)


# --- LogisticRegression ---

    lr = LogisticRegression().fit(X_train, y_train)
    lr_predictions = lr.predict(X_test)


# --- TERM MATCH ---
    # get predictions for term match model
    # emotions_file = open("emotions_synonyms_depoche.txt")
    emotions_file = open("emotions_synonyms.txt")
    lines = emotions_file.readlines()
    # build dictionary of emotions and synonmys from file

    emotions_list = dict()
    for line in lines:
    	line = line.rstrip("\n")
    	words = line.split(",")
    	emotion_curr = words[0]
    	synonyms = words[1:]
    	emotions_list[emotion_curr] = synonyms
    emotions_file.close()

    # X_train, X_test, y_train, y_test = train_test_split(tweets, binary_labels, test_size=test_split, random_state=rand_state)
    X_train_bal, X_test_bal, y_train_bal, y_test_bal, none_counter = balanced_train_test_split(tweets, binary_labels, test_split, rand_state, bal_nbr, none_counter)

    # term_predictions = np.array([])

    # print emotion
    # print len([x for x in y_test if x == 1.0])

    # # get prediction for each tweet based on whether it contains a synonym for that emotion
    # for tweet in X_test:
    # 	emotionFound = False
    # 	for synonym in emotions_list[emotion]:
    # 		if emotionFound is not True:
    # 			string_pat = "\\b" + synonym.lower() + "\\b"
    # 			synonym_pattern = re.compile(string_pat)
    # 			if synonym_pattern.search(unicode(tweet, errors='ignore').lower()) is not None:
    # 				emotionFound = True
    # 	term_predictions = np.append(term_predictions, emotionFound)

    # naive_bayes_predictions

    # pdb.set_trace()

    X_train_bal, y_train_bal, start_index_rich, start_index_non =  get_twenty_twenty ( tweets, binary_labels, start_index_rich, start_index_non )



    # array = np.array([])
    # for line in tweets:
    #             array = np.append(array,  re.sub(r'[^\w\'] ', " ",  line).split() )

    #  vectorizer = CountVectorizer(tokenizer=tokenize, analyzer='word', )

    # pdb.set_trace()

    nb_classifier, nb_grams, nb_features = naive_bayes.train(X_train_bal, y_train_bal)

    # randForest_classifier = RandomForestClassifier()
    # randForest_classifier.fit(X_train_bal, y_train_bal)

    # show_most_informative_features(vectorizer, nb_classifier, 20)

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

    # if emotion == 'fear':
    #     pdb.set_trace()


    # print relvent tweets
    # for line in X_train_bal:
    #     print line

    print

    # for item in sorted(feature_importance, reverse=True):
    #     if item[0] != 0.0:
    #         # feature_rep = 'feature: '
    #         # feature_rep = feature_rep + str(item[1])

    #         # importance_rep = 'importance: '
    #         # importance_rep = importance_rep + str(item[0])

    #         # print  repr(feature_rep).rjust(2), repr(importance_rep).rjust(20)

    #         print item[1], item[0]

    #         # print '{0:40} : {1:15}'.format(item[1], item[0])

    # print
    # print
    # print

    # for line in X_train_bal:
    #     print line

    # sort on importance

    # Print the feature ranking
    #print("Feature ranking:")

    # Plot the feature importances of the forest
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar( range(nb_grams[features].shape[1]), importances[indices],
    #    color="r", yerr=std[indices], align="center")
    # plt.xticks(range(nb_grams[features].shape[1]), indices)
    # plt.xlim([-1, nb_grams[features].shape[1]])
    # plt.show()

    nb_correct, nb_incorrect, nb_tp, nb_fp, nb_tn, nb_fn, nb_predictions = naive_bayes.test(X_test_bal, y_test_bal, nb_classifier, nb_grams)

    # if emotion == 'joy':
    #     for i in range(len(nb_predictions)):
    #         if nb_predictions[i]:
    #             print i
    #     pdb.set_trace()

    for tweet in X_test:
        emotionFound = False

    # get overall counts
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


    # print emotion
    # print
    for i in range(0, nb_predictions.shape[0]):

        # print out prediction for each model and true value
        #print(term_predictions[i])
        #print(svm_predictions[i])
        #outfile.write(str(term_predictions[i])  + "\n")
        #outfile.write("BREAK"  + "\n")
        #outfile.write(str(svm_predictions[i])  + "\n")
        # print(y_test[i])
        # print(X_test[i].encode('utf8'))
        # outfile.write(X_test[i].encode('utf8') + "\t" + str(term_predictions[i]) + "\t" + str(svm_predictions[i]) + "\t" + str(y_test[i]) + "\n")

        # get overlap in correctly labelled tweets (for positively labelled tweets only)
        if y_test[i] == 1.0:
        	total_positive += 1
        	if lr_predictions[i] == svm_predictions[i] and lr_predictions[i] and nb_predictions[i] == y_test[i]:
        		all_correct += 1
        	elif lr_predictions[i] == svm_predictions[i] and lr_predictions[i] and nb_predictions[i] != y_test[i]:
        		all_wrong += 1
        	elif lr_predictions[i] == y_test[i]:
        		lr_correct += 1
                        # print X_train_tweets[i]
                        # print
        	elif svm_predictions[i] == y_test[i]:
        		svm_correct += 1

                        # print  X_train_tweets[i]
                        # print
                        # pdb.set_trace()

        	elif nb_predictions[i] == y_test[i]:
        		nb_correct += 1
                        # print "nb only: ", X_train_bal[i]
        else:
        	total_negative += 1

        # get true/false negative and positive rates for both models
        # data should help show why models have different precision/recall
        if lr_predictions[i] == 0.0 and y_test[i] == 0.0:
        	lr_tn += 1
        if lr_predictions[i] == 1.0 and y_test[i] == 1.0:
        	lr_tp += 1
        if lr_predictions[i] == 0.0 and y_test[i] == 1.0:
        	lr_fn += 1
        if lr_predictions[i] == 1.0 and y_test[i] == 0.0:
        	lr_fp += 1

        if svm_predictions[i] == 0.0 and y_test[i] == 0.0:
        	svm_tn += 1
        if svm_predictions[i] == 1.0 and y_test[i] == 1.0:
        	svm_tp += 1
        if svm_predictions[i] == 0.0 and y_test[i] == 1.0:
        	svm_fn += 1
        if svm_predictions[i] == 1.0 and y_test[i] == 0.0:
        	svm_fp += 1

    # pdb.set_trace()
    total = total_negative + total_positive
    print emotion, ' total tweets: ', total

    # print "     total  tweets: " + str(total)
    print "     total_positive positive tweets: " + str(total_positive)
    print "     all right: " + str(all_correct)
    print "     all wrong: " + str(all_wrong)
    print "     only svm correct: " + str(svm_correct)
    print "     only lr correct: " + str(lr_correct)
    print "     only nb correct: " + str(nb_correct)
    print "\n"
    print "     svm true pos: " + str(svm_tp)
    print "     svm false neg: " + str(svm_fn)
    print "     svm false pos: " + str(svm_fp)
    print "     svm true neg: " + str(svm_tn)
    print "\n"
    print "     lr true pos: " + str(lr_tp)
    print "     lr false neg: " + str(lr_fn)
    print "     lr false pos: " + str(lr_fp)
    print "     lr true neg: " + str(lr_tn)
    print "\n"
    print "     nb true pos: " + str(nb_tp)
    print "     nb false neg: " + str(nb_fn)
    print "     nb false pos: " + str(nb_fp)
    print "     nb true neg: " + str(nb_tn)
    print "\n"
    print "     LR Precision: " +    str(precision_score(   y_test, lr_predictions))
    print "     LR Recall: " +         str(recall_score(         y_test, lr_predictions))
    print "     SVM Precision: " + str(precision_score(   y_test, svm_predictions))
    print "     SVM Recall: " +      str(recall_score(         y_test, svm_predictions))
    print "     NB precision: " +    str( precision_score(   y_test_bal, nb_predictions)  )
    print "     NB recall: " +          str( recall_score(         y_test_bal, nb_predictions) )

    # write all this data to file.. super fun stuff
    outfile.write("     total  tweets: " + str(total) + "\n")
    outfile.write("     total_positive positive tweets: " + str(total_positive) + "\n")
    outfile.write("     all right: " + str(all_correct) + "\n")
    outfile.write("     all wrong: " + str(all_wrong) + "\n")
    outfile.write("     only svm correct: " + str(svm_correct) + "\n")
    outfile.write("     only lr correct: " + str(lr_correct) + "\n")
    outfile.write("     only nb correct: " + str(nb_correct) + "\n")
    outfile.write("\n")

    outfile.write("     svm true pos: " + str(svm_tp) + "\n")
    outfile.write("     svm false neg: " + str(svm_fn) + "\n")
    outfile.write("     svm false pos: " + str(svm_fp) + "\n")
    outfile.write("     svm true neg: " + str(svm_tn) + "\n")
    outfile.write("\n")

    outfile.write("     lr true pos: " + str(lr_tp) + "\n")
    outfile.write("     lr false neg: " + str(lr_fn) + "\n")
    outfile.write("     lr false pos: " + str(lr_fp) + "\n")
    outfile.write("     lr true neg: " + str(lr_tn) + "\n")
    outfile.write("\n")

    outfile.write("     nb true pos: " + str(nb_tp) + "\n")
    outfile.write("     nb false neg: " + str(nb_fn) + "\n")
    outfile.write("     nb false pos: " + str(nb_fp) + "\n")
    outfile.write("     nb true neg: " + str(nb_tn) + "\n")
    outfile.write("\n")

    outfile.write("     LR Precision: " +    str(precision_score(   y_test, lr_predictions)) + "\n")
    outfile.write("     LR Recall: " +         str(recall_score(         y_test, lr_predictions)) + "\n")
    outfile.write("     SVM Precision: " + str(precision_score(   y_test, svm_predictions)) + "\n")
    outfile.write("     SVM Recall: " +      str(recall_score(         y_test, svm_predictions)) + "\n")
    outfile.write("     NB precision: " +    str( precision_score(   y_test_bal, nb_predictions)  )  + "\n")
    outfile.write("     NB recall: " +          str( recall_score(         y_test_bal, nb_predictions) )  + "\n")

	# pdb.set_trace()

	# print("Term Accuracy: " + str(accuracy_score(y_test, term_predictions)) + "\n")
	# print("Term Precision: " + str(precision_score(y_test, term_predictions)) + "\n")
	# print("Term Recall: " + str(recall_score(y_test, term_predictions)) + "\n")
	# print("Term F1: " + str(f1_score(y_test, term_predictions)) + "\n")

	# print("SVM Accuracy: " + str(accuracy_score(y_test, svm_predictions)) + "\n")
	# print("SVM Precision: " + str(precision_score(y_test, svm_predictions)) + "\n")
	# print("SVM Recall: " + str(recall_score(y_test, svm_predictions)) + "\n")
	# print("SVM F1: " + str(f1_score(y_test, svm_predictions)) + "\n")

outfile.close()



