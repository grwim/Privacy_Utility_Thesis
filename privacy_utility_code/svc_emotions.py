# coding=utf-8
from twokenize import tokenize
import numpy
import scipy
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn import svm
from scipy import sparse
import numpy as np
from nltk.corpus import wordnet as wn


# helper function convert text file to array (one line in each array entry)
def file_to_array(filename):
	file = open(filename)
	array = np.array([])
	for line in file:
		array = np.append(array, line.rstrip())
	file.close()
	return array

# helper function to convert label list from multilabel to binary for a given emotion
def to_binary_classification(classfications, emotion):
	binary_class = np.array([])
	for item in classfications:
		if item == emotion:
			binary_class = np.append(binary_class, True)
		else:
			binary_class = np.append(binary_class, False)
	return binary_class

# helper function to convert label list from multilabel to binary for a given emotion
def to_binary_classification_merged(classfications, emotion_1, emotion_2):
	binary_class = np.array([])
	for item in classfications:
		if (item == emotion_1) | (item == emotion_2):
			binary_class = np.append(binary_class, True)
		else:
			binary_class = np.append(binary_class, False)
	return binary_class

# returns a sklearn vectorizer for unigrams, bigrams, and trigrams
# params: vocab (false--get vectorizer for all features, or true--get vectorizer for predefined feature list)
#		  emotion (emotion to get vectorizer for, only needed if vocab=true)
def get_token_vectorizer(vocab=False, emotion=""):
	if vocab:
		filename = "featurelists/" + emotion + "_tokens"
		vocab = file_to_array(filename)
		vectorizer = CountVectorizer(ngram_range=(1,2), tokenizer=tokenize, analyzer='word', vocabulary=vocab)
	else:
		vectorizer = CountVectorizer(ngram_range=(1,2), tokenizer=tokenize, analyzer='word')
	return vectorizer

def get_token_vectorizer_merged(vocab=False, emotion_1="", emotion_2=""):
	if vocab:
		filename_1 = "featurelists/" + emotion_1 + "_tokens"
		vocab_1 = file_to_array(filename_1)

		filename_2 = "featurelists/" + emotion_2 + "_tokens"
		vocab_2 = file_to_array(filename_2)

		vocab = vocab_1 + vocab_2
		vectorizer = CountVectorizer(ngram_range=(1,2), tokenizer=tokenize, analyzer='word', vocabulary=vocab)
	else:
		vectorizer = CountVectorizer(ngram_range=(1,2), tokenizer=tokenize, analyzer='word')
	return vectorizer


# returns a sklearn vectorizer for wordnet synsets and hypernyms
# params: vocab (false--get vectorizer for all features, or true--get vectorizer for predefined feature list)
#		  emotion (emotion to get vectorizer for, only needed if vocab=true)
def get_wordnet_vectorizer(vocab=False, emotion=""):
	if vocab:
		filename = "featurelists/" + emotion + "_wordnet"
		vocab = file_to_array(filename)
		vectorizer = CountVectorizer(tokenizer=get_synsets_hypernyms, analyzer='word', vocabulary=vocab)
	else:
		vectorizer = CountVectorizer(tokenizer=get_synsets_hypernyms, analyzer='word')
	return vectorizer

# def find_emotion_svc(tweets_list, emotion_list):
# 	for emotion in emotion_list:
# 		term_mx = compute_feature_mx(get_features(emotion), tweets_list)
# 		clf = get_classfier(emotion)
# 		predictions = clf.predict(term_mx)

# def compute_feature_mx(features, tweets):
# 	vectorizer = CountVectorizer(ngram_range=(1,3), tokenizer=tokenize, analyzer='word', vocabulary=features)
# 	term_mx = vectorizer.fit_transform(tweets)
# 	return term_mx

# function to get wordnet synsets and recursive hypernyms from every word in tweet
def get_synsets_hypernyms(text):
	# get individual tokens
	tokens = tokenize(text)
	synsets = [] 	 # synset names
	synset_objs = [] # corresponding synset objects
	for token in tokens:
		# get all synsets for a token
		token_synsets = wn.synsets(token)
		for synset in token_synsets:
			if synset.name() not in synsets:
				# add synset name and object to list
				synsets.append("s." + synset.name())
				synset_objs.append(synset)
	# once all synsets are found, add all recursive hypernyms for each synset
	synsets += get_hypernyms_recursive(synset_objs)
	return synsets

# function to get all recursive hypernyms for a list of synsets
def get_hypernyms_recursive(synsets):
	all_hypernyms = []
	for synset in synsets:
		# get hypernyms for given synset
		hypernyms = []
		_recurse_hypernyms(synset, hypernyms)
		if hypernyms != []:
			all_hypernyms += hypernyms
	hypernym_strings = []
	# get list of all hypernym names
	for hypernym in all_hypernyms:
		hypernym_str = "h." + hypernym.name()
		if hypernym_str not in hypernym_strings:
			hypernym_strings.append(hypernym_str)
	return hypernym_strings

# recurse over all hypernyms for synset
def _recurse_hypernyms(synset, hypernyms):
	synset_hypernyms = synset.hypernyms()
	if synset_hypernyms:
		# add all hypernyms to list
		hypernyms += synset_hypernyms
		# recurse to get hypernyms of each hypernym on this level
		for hypernym in synset_hypernyms:
			_recurse_hypernyms(hypernym, hypernyms)

# function for greedy additive feature selection algorithm
# should produce reduced list of features
def greedy_feature_select(term_mx, labels, feature_names):

	clf = svm.LinearSVC()

	total_features = term_mx		# full feature-document mx
	current_features = np.array([]) # feature-doc mx for selected features
	current_feature_names = []		# selected feature names
	used_feature_idxs = []			# indices for features that have already been selected

	prev_score = 0			# f1 score from previous iteration of loop
	still_improving = True 	# boolean for whether to continue iterating
	while (still_improving):
		max_score = 0	# best f1 score so far
		bestFeature = 0 # index of feature that, when added, gives best f1 score so far
		# print("current feature list: " + str(current_feature_names))

		# loop over all features in total matrix
		for j in range(0, int(total_features.shape[1])):
			if j not in used_feature_idxs:
				# make test mx by adding col j of total matrix to mx of already selected features
				if current_features.size == 0:
					testFeatures = total_features.getcol(j)
				else:
					testFeatures = scipy.sparse.hstack([current_features,total_features.getcol(j)])

				# get k-fold cross val score for test matrix
				# if average better than existing max score, save this score and feature
				scores = cross_validation.cross_val_score(clf, testFeatures, labels, scoring='f1', cv=5)
				if scores.mean() > max_score and bestFeature not in used_feature_idxs:
					max_score = scores.mean()
					bestFeature = j

		# continue iterating as long as adding a feature improves the score
		if (max_score > prev_score):
			# add next best feature to feature list if it improves overall score
			if current_features.size == 0:
				current_features = total_features.getcol(bestFeature)
			else:
				current_features = scipy.sparse.hstack([current_features, total_features.getcol(bestFeature)])
			current_feature_names.append(feature_names[bestFeature])
			used_feature_idxs.append(bestFeature)
			prev_score = max_score
		else:
			still_improving = False

	# print accuracy, precision recall, f1 scores
	accuracy = cross_validation.cross_val_score(clf, current_features, labels, scoring='accuracy', cv=5)
	precision = cross_validation.cross_val_score(clf, current_features, labels, scoring='precision', cv=5)
	recall = cross_validation.cross_val_score(clf, current_features, labels, scoring='recall', cv=5)
	f1 = cross_validation.cross_val_score(clf, current_features, labels, scoring='f1', cv=5)

	print("Accuracy: " + str(accuracy) + "\n")
	print("Precision: " + str(precision) + "\n")
	print("Recall: " + str(recall) + "\n")
	print("F1: " + str(f1) + "\n")

	# return selected feature names for saving
	return current_feature_names
