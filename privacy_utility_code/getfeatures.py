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

# build SVM classifiers (one each for uni/bi/tri-grams and wordnet hypernyms/synsets)
# and combine into one vectorizer
estimators = [('tokens', svc_emotions.get_token_vectorizer()), ('wordnet', svc_emotions.get_wordnet_vectorizer())]
full_vectorizer = FeatureUnion(estimators)

print 'GOT HERE'

# get full feature-tweet matrix (all features, all tweets)
term_mx = full_vectorizer.fit_transform(tweets)
features = full_vectorizer.get_feature_names()

#file to output results
outfile_name = 'features_output.txt'
outfile = open(outfile_name, 'w')

for feature in features:
	outfile.write("\n" + feature.encode('utf-8')) # ADDED .encode('utf-8') ~KMR

# loop through all emotions to reduce the feature list for each
for emotion in emotion_list:
            print "here"
	filename = emotion + "_features.txt"
	outfile = open(filename, 'w')
	# call greedy feature select alg
	selected_feature_names = svc_emotions.greedy_feature_select(term_mx, svc_emotions.to_binary_classification(labels, emotion), features)
	# write all feature names to file to use later
	for feature in selected_feature_names:
		outfile.write(feature.encode('utf8') + '\n')
	outfile.close()
