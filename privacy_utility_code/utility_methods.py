import re
import math
import operator
# from apyori import apriori
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import pdb
from itertools import izip
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.graph_objs import *
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize
import time
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
# import pyfpgrowth
from pymining import itemmining

# from mlxtend.frequent_patterns import apriori
# from mlxtend.preprocessing import TransactionEncoder

from scipy.sparse import csr_matrix
import cPickle

import ast


def load_pickle(filename):
    with open(filename, 'rb') as infile:
        matrix = cPickle.load(infile)
    return matrix

def dot_product2(v1, v2):
    return sum(map(operator.mul, v1, v2))


def vector_cos5(v1, v2):
    prod = dot_product2(v1, v2)
    len1 = math.sqrt(dot_product2(v1, v1))
    len2 = math.sqrt(dot_product2(v2, v2))

    result = 0.0
    if (len1 * len2) != 0:
        result = prod / (len1 * len2)

    return result

def Binary_Transform_TrainingData(training_data): # takes in training data, (replaces non-zero values with 1, zero values with 0)


    for row in range(len(training_data)):
        for column in range(len(training_data[i])):
            if isinstance(training_data[row][column], basestring): # if a string, convert to int
                training_data[row][column] = int(training_data[row][column])
            if training_data[row][column] != 0:
                training_data[row][column] = 1

    return training_data

def Binary_Transform_sparse(sparse_matrix): # takes in vector, (replaces non-zero values with 1, zero values with 0)
    sparse_matrix_binary = sparse_matrix.sign()

    # vector = np.sign(vector) # convert to bool
    # vector = [1 if  x != 0 else 0 for x in vector]

    return sparse_matrix_binary

def Binary_Transform_Vector(vector): # takes in vector, (replaces non-zero values with 1, zero values with 0)
    # sparse_matrix_binary = sparse_matrix.sign()

    # vector = np.sign(vector) # convert to bool
    vector = [1 if  x != 0 else 0 for x in vector]

    return vector








def Handle_Uniqueness_Metrics_BOW(wordList_representation_fileName, projection_name, binary=True):
    """ we consider the whole vector (identifying) anyone who has the exact same vector, we put them in the same equivalency class (in terms of EXACT and BINARY)
    our goal is to see which of these has the highest level of privacy
    for some of these, k might be one
    > for each representation, evaluate how many handles are distinct (k =1), as well as how many are similar
    > simple dictionary approach where a unique handle vector is a key, and k is value """

    # use frozen set representations of vectors, and compare those.

    # if binary, only store unique words for each handle vector as frozen set

    num_handles_withAllZeroVector_full = 0
    num_handles_withAllZeroVector_high = 0
    num_handles_withAllZeroVector_med = 0
    num_handles_withAllZeroVector_low = 0

    E_Class_Sizes_list_full = []
    E_Class_Sizes_list_high = []
    E_Class_Sizes_list_med = []
    E_Class_Sizes_list_low = []

    handles_inFreqBucket_list_low = []
    handles_inFreqBucket_list_med = []
    handles_inFreqBucket_list_high = []

    infile_handles_inFreqBucket_low = open('low_handles_4000000.txt', 'r')
    for line in infile_handles_inFreqBucket_low:
        handle_list = line.split(',')
        for handle in handle_list:
            handles_inFreqBucket_list_low.append(handle)

    infile_handles_inFreqBucket_med = open('med_handles_4000000.txt', 'r')
    for line in infile_handles_inFreqBucket_med:
        handle_list = line.split(',')
        for handle in handle_list:
            handles_inFreqBucket_list_med.append(handle)

    infile_handles_inFreqBucket_high = open('high_handles_4000000.txt', 'r')
    for line in infile_handles_inFreqBucket_high:
        handle_list = line.split(',')
        for handle in handle_list:
            handles_inFreqBucket_list_high.append(handle)

    handleRepresentation_toCount = {}

    wordCount_representation_fileName = 'clean_tweets_4000000_BagOfWords_summed.txt'

    with open(wordList_representation_fileName) as file_words, open(wordCount_representation_fileName) as file_counts:
        line_count = 0

        for words_line, counts_line in izip(file_words, file_counts):

            words_line = words_line.strip('\n')
            words_line = words_line.strip('\r')
            words_list = words_line.split(',')

            handle = words_list[0]
            feature_words = words_list[1:]


            counts_line = counts_line.strip('\n')
            counts_line = counts_line.strip('\r')
            counts_list = counts_line.split(',')
            feature_counts = counts_list[1:]

            featureToCount_list = []

            if binary:
                for feature in feature_words:
                    feature = feature + '1'
                    featureToCount_list.append(feature)
            else:
                for feature, count in izip(feature_words, feature_counts):
                    feature = feature + str(count)
                    featureToCount_list.append(feature)

            feature_words_set = frozenset(featureToCount_list)

            if feature_words_set in handleRepresentation_toCount:
                handleRepresentation_toCount[feature_words_set].append(handle)
                # handles_with_nonUniqueVector.append(handle)
            else:
                handleRepresentation_toCount[feature_words_set] = []
                handleRepresentation_toCount[feature_words_set].append(handle)

    count_num_handles_EclassGreaterThan_1_full = 0
    count_num_handles_EclassGreaterThan_1_high = 0
    count_num_handles_EclassGreaterThan_1_med = 0
    count_num_handles_EclassGreaterThan_1_low = 0

    for vector_set, handle_list_curr in handleRepresentation_toCount.iteritems():

            # if handle_list_inOrder == handle_list_curr:

        allZeroVector = True

        for value in vector_set:
            if value != 0:
                allZeroVector = False
                break

        if allZeroVector:  # count number of handles with all zero vector for each bucket

            num_handles_withAllZeroVector_full = len(handle_list_curr)

            for handle in handle_list_curr:

                if handle in handles_inFreqBucket_list_high:
                    num_handles_withAllZeroVector_high = num_handles_withAllZeroVector_high + 1

                elif handle in handles_inFreqBucket_list_med:
                    num_handles_withAllZeroVector_med = num_handles_withAllZeroVector_med + 1

                elif handle in handles_inFreqBucket_list_low:
                    num_handles_withAllZeroVector_low = num_handles_withAllZeroVector_low + 1
        else:

            # ignore all zero vector
            E_Class_Sizes_list_full.append(len(handle_list_curr))

            if len(handle_list_curr) > 1:
                print 'len(handle_list_curr): ', len(handle_list_curr)
                print vector_set

            size_E_high = 0
            size_E_med = 0
            size_E_low = 0

            for handle in handle_list_curr:
                if handle in handles_inFreqBucket_list_high:
                    size_E_high = size_E_high + 1
                elif handle in handles_inFreqBucket_list_med:
                    size_E_med = size_E_med + 1
                elif handle in handles_inFreqBucket_list_low:
                    size_E_low = size_E_low + 1

            # only add to list of equivalency class sizes if at least one handle in the given bucket for that vector
            if size_E_high > 0:
                E_Class_Sizes_list_high.append(size_E_high)
            if size_E_med > 0:
                E_Class_Sizes_list_med.append(size_E_med)
            if size_E_low > 0:
                E_Class_Sizes_list_low.append(size_E_low)


            if len(handle_list_curr) >= 2: # if more than one handle in equivalency class, increase count of users with equivalency class greater than one
                count_num_handles_EclassGreaterThan_1_full = count_num_handles_EclassGreaterThan_1_full + len(handle_list_curr)

                # increment counts for buckets of handles
                for handle in handle_list_curr:

                    if handle in handles_inFreqBucket_list_high:
                        count_num_handles_EclassGreaterThan_1_high = count_num_handles_EclassGreaterThan_1_high + 1

                    elif handle in handles_inFreqBucket_list_med:
                        count_num_handles_EclassGreaterThan_1_med = count_num_handles_EclassGreaterThan_1_med + 1

                    elif handle in handles_inFreqBucket_list_low:
                        count_num_handles_EclassGreaterThan_1_low = count_num_handles_EclassGreaterThan_1_low + 1

                # vector
                # k
                # handles involved
# OUTFILE functionality
                # outfile_line = vectorString
                # outfile_line = outfile_line + ', k=' + str(len(handle_list_curr))
                # for handle in handle_list_curr:
                #     outfile_line = outfile_line + ', ' + handle

                # outfile_line = outfile_line + '\n'
                # outfile.write(outfile_line)

                # lineWrite_count = lineWrite_count + 1


    # for handle in handles_with_nonUniqueVector: # add handles_with_nonUniqueVector
    #     dummy_vector = [-1] * len_vector
    #     outfile_line = str(dummy_vector)
    #     outfile_line = outfile_line + ', k=0' + str(len(handle_list_curr))
    #     outfile_line = outfile_line + ', ' + handle
    #     outfile_line = outfile_line + '\n'
    #     outfile.write(outfile_line)

    # pdb.set_trace()
    # outfile.close()

    # print 'Finished Handle_Uniqueness_Metrics() for ', summed_representation_fileName, ' binary=', binary
    # print 'Wrote ', str(lineWrite_count), 'lines'

    privacy_gained_full = count_num_handles_EclassGreaterThan_1_full / float(5626 - num_handles_withAllZeroVector_full)
    privacy_gained_high = count_num_handles_EclassGreaterThan_1_high / float(len(handles_inFreqBucket_list_high) - num_handles_withAllZeroVector_high)
    privacy_gained_med = count_num_handles_EclassGreaterThan_1_med / float(len(handles_inFreqBucket_list_med)- num_handles_withAllZeroVector_med)
    privacy_gained_low = count_num_handles_EclassGreaterThan_1_low / float(len(handles_inFreqBucket_list_low) - num_handles_withAllZeroVector_low)

    print 'privacy_gained_full: ', privacy_gained_full
    print 'privacy_gained_high: ', privacy_gained_high
    print 'privacy_gained_med: ', privacy_gained_med
    print 'privacy_gained_low: ', privacy_gained_low
    print
    print 'num_handles_withAllZeroVector_full: ', num_handles_withAllZeroVector_full
    print 'num_handles_withAllZeroVector_high: ', num_handles_withAllZeroVector_high
    print 'num_handles_withAllZeroVector_med: ', num_handles_withAllZeroVector_med
    print 'num_handles_withAllZeroVector_low: ', num_handles_withAllZeroVector_low
    print


    trace_full = go.Box(
        y = E_Class_Sizes_list_full,
        name='All <br> <i>protected=<i> ' + str(count_num_handles_EclassGreaterThan_1_full) + '<br>  <i>all-zero=<i> ' + str(num_handles_withAllZeroVector_full),
        boxpoints='all',
        # jitter=0.5,
        whiskerwidth=0.2,
        fillcolor='rgb(93, 164, 214)',
        marker=dict(
            # size = 2,
            color='rgb(93, 164, 214)',
        ),
        line=dict(width=1),
        boxmean='sd'
    )

    # if num_features_ToKeep < 1000:
    #     projection_name = 'first ' + str(num_features_ToKeep) + ' features, ' + projection_name

    trace_high = go.Box(
        y = E_Class_Sizes_list_high,
        name='High <br> <i>protected=<i> ' + str(count_num_handles_EclassGreaterThan_1_high) + '<br>  <i>all-zero=<i> ' + str(num_handles_withAllZeroVector_high),
        boxpoints='all',
        # jitter=0.5,
        whiskerwidth=0.2,
        fillcolor='rgb(255, 144, 14)',
        marker=dict(
            # size = 2,
            color='rgb(255, 144, 14)',
        ),
        line=dict(width=1),
        boxmean='sd'
    )
    trace_med = go.Box(
        y = E_Class_Sizes_list_med,
        name='Moderate <br> <i>protected=<i> ' + str(count_num_handles_EclassGreaterThan_1_med) + '<br>  <i>all-zero=<i> ' + str(num_handles_withAllZeroVector_med),
        boxpoints='all',
        # jitter=0.5,
        whiskerwidth=0.2,
        fillcolor='rgb(44, 160, 101)',
        marker=dict(
            # size = 2,
            color='rgb(44, 160, 101)',
        ),
        line=dict(width=1),
        boxmean='sd'
    )
    trace_low = go.Box(
        y = E_Class_Sizes_list_low,
        name='Low <br> <i>protected=<i> ' + str(count_num_handles_EclassGreaterThan_1_low) + '<br>  <i>all-zero=<i> ' + str(num_handles_withAllZeroVector_low),
        boxpoints='all',
        # jitter=0.5,
        whiskerwidth=0.2,
        fillcolor='rgb(255, 65, 54)',
        marker=dict(
            # size = 2,
            color='rgb(255, 65, 54)',
        ),
        line=dict(width=1),
        boxmean='sd'
    )

    layout = go.Layout(
        annotations=Annotations([
            Annotation(
                x=0.5004254919715793,
                y=-0.12191064079952971,
                showarrow=False,
                text='',
                xref='paper',
                yref='paper'
            ),
            Annotation(
                x=-0.07944728761514841,
                y=0.4714285714285711,
                showarrow=False,
                text='Equivalency class sizes, excluding all zero vector',
                textangle=-90,
                xref='paper',
                yref='paper'
            )
        ]),
        autosize=True,
        title='Distribution of Handle Equivalency Class Sizes: ' + projection_name,
        yaxis=dict(
           zeroline=False,
           range=[0, 18]
        ),
                # # margin=dict(
        # #     l=40,
        # #     r=30,
        # #     b=80,
        # #     t=100,
        # # ),
        # paper_bgcolor='rgb(243, 243, 243)',
        # plot_bgcolor='rgb(243, 243, 243)',
        showlegend=False
    )

    traces = [trace_full, trace_high, trace_med, trace_low]
    # traces = [trace_high, trace_med, trace_low]


    # fig = go.Figure(data=traces, layout=layout)
    # py.iplot(fig, filename=projection_name + ' - Equivalancy Class Sizes - 4 Buckets')


# if bag of words, pass in ___summed_wordCounts.txt
def Handle_Uniqueness_Metrics(summed_representation_fileName, handles_fileName, projection_name, binary=True, bag_of_words=True, reduced_runtime=True,tweet_freq_type='', num_features_ToKeep=1000):
    """ we consider the whole vector (identifying) anyone who has the exact same vector, we put them in the same equivalency class (in terms of EXACT and BINARY)
    our goal is to see which of these has the highest level of privacy
    for some of these, k might be one
    > for each representation, evaluate how many handles are distinct (k =1), as well as how many are similar
    > simple dictionary approach where a unique handle vector is a key, and k is value """

    # keep track of number of handles with all zero vectors for each handle bucket
    # keep a list of each equivalency class size

    num_handles_withAllZeroVector_full = 0
    num_handles_withAllZeroVector_high = 0
    num_handles_withAllZeroVector_med = 0
    num_handles_withAllZeroVector_low = 0

    E_Class_Sizes_list_full = []
    E_Class_Sizes_list_high = []
    E_Class_Sizes_list_med = []
    E_Class_Sizes_list_low = []

    handles_inFreqBucket_list_low = []
    handles_inFreqBucket_list_med = []
    handles_inFreqBucket_list_high = []

    infile_handles_inFreqBucket_low = open('low_handles_4000000.txt', 'r')
    for line in infile_handles_inFreqBucket_low:
        handle_list = line.split(',')
        for handle in handle_list:
            handles_inFreqBucket_list_low.append(handle)

    infile_handles_inFreqBucket_med = open('med_handles_4000000.txt', 'r')
    for line in infile_handles_inFreqBucket_med:
        handle_list = line.split(',')
        for handle in handle_list:
            handles_inFreqBucket_list_med.append(handle)

    infile_handles_inFreqBucket_high = open('high_handles_4000000.txt', 'r')
    for line in infile_handles_inFreqBucket_high:
        handle_list = line.split(',')
        for handle in handle_list:
            handles_inFreqBucket_list_high.append(handle)


    # count of users with an equivalency class of at least size 2.
        # divide this by total number handles
        # possibly make graph too.

    # be able to drop a certain number of features

    all_handles = True # boolen for whether all handles will be evaluated
    handles_inFreqBucket_list = []

    if tweet_freq_type == 'low':
        all_handles = False
    elif tweet_freq_type == 'med':
        all_handles = False
    elif tweet_freq_type == 'high':
        all_handles = False

    if not all_handles:
        infile_handles_inFreqBucket = open(tweet_freq_type + '_handles_' + '4000000.txt', 'r')
        for line in infile_handles_inFreqBucket:
            handles_inFreqBucket_list = line.split(',')

    # vectorRepresentation --> [handles with same representation]
    handleRepresentation_toCount = {}
    unique_handle_count = 0

    # print summed_representation_fileName

    outfile_name = summed_representation_fileName[:-4]

    outfile_name = outfile_name + '_Uniqueness_Metrics_' + tweet_freq_type
    if binary:
        outfile_name = outfile_name + 'BINARY.csv'
    else:
        outfile_name = outfile_name + 'EXACT.csv'

    # print 'outfile_name: ', outfile_name
    outfile = open(outfile_name, 'w')

    if bag_of_words:

        # use sparse?

        # list_of_featureVector_sparseMatrices = []

        # X = []
        # handle_list = []
        # sparse_representation_fileName
        # if isBagOfWords(sparse_representation_fileName):
        #     X = load_pickle(sparse_representation_fileName)
        #     #convert training data to floats
        #     if binary: # convert tweets to binary
        #         X = feature_values_floatFormat = Binary_Transform_sparse(X)

        # # store list of sparse matrices
        # # iterate over sparse matrix and count num 0s

        # if (d-e).nnz != 0: # these two matrices are different
        #     print

        # if (d-d).nnz == 0: # These two matrices are the same
        #     print

        # from sparse matrics, need to get number of handles corresponding to each vector

        summed_representation_fileName = summed_representation_fileName # the file with all of the the counts for the words that occured for a handle
        wordList_representation_fileName = summed_representation_fileName[:-4]
        featureSpace_fileName = summed_representation_fileName[:-10]

        wordList_representation_fileName = wordList_representation_fileName + '_wordLists.txt'
        featureSpace_fileName = featureSpace_fileName + 'featureSpace.txt'

        # open up bag of words word file

        # load in all words, create feature space

        # to reduce the memory requirements of this process for BoW, only construct vectors with 750k feature space for each pairwise calculation

        uniqueWord_ToIndex_dict = {} # assign a 'feature number' to each word
        uniqueWord_ToFreq_dict = {} # keep track of count for each word
        handle_toWordList = {}
        handle_toWordCountList = {}

        feature_space_size = 0
        vectors_list = []
        words_filename = summed_representation_fileName[:-15]
        represenation_name = words_filename
        words_filename = words_filename + '.txt'

        handle_list = []
        nonZero_Features_list = []

        # aquire feature space
        featureSpace_fileName = summed_representation_fileName[:-10]
        featureSpace_fileName = featureSpace_fileName + 'featureSpace.txt'
        featureSpace_file = open(featureSpace_fileName, 'r')

        featureSpace_list = []
        for line in featureSpace_file:
            line = line.strip('\n')
            line_elements = line.split(',')
            for feature in line_elements:
                featureSpace_list.append(feature)

        for word in featureSpace_list:
            uniqueWord_ToIndex_dict[word] = feature_space_size
            feature_space_size = feature_space_size + 1
            # print word, ' ', feature_space_size

        # aquire feature space & populate handle --> wordList dict
        with open(wordList_representation_fileName) as file_words:
            line_count = 0
            for words_line in file_words:
                words_line = words_line.strip('\n')
                words_line = words_line.strip('\r')
                words_list = words_line.split(',')

                handle = words_list[0]
                feature_words = words_list[1:]

                if not all_handles: #
                    if handle not in handles_inFreqBucket_list:
                        continue # if current handle is not in specified freqBucket, ignore


                handle_list.append(handle)
                # nonZero_Features_list.append(feature_words)

                # wordTo_freqCount_vectorDict = {}


                # START HERE - use this code to aquire a handle to feature vector dict, that does not include hapax legomenon
            # (aquiring dict of feature words to count) for  removal of hapax legomonen
                for word in feature_words:
                    if word not in uniqueWord_ToIndex_dict:
                        uniqueWord_ToIndex_dict[word] = unqiue_word_count
                        unqiue_word_count = unqiue_word_count + 1
                        print word, ' ', unqiue_word_count

                    if word not in uniqueWord_ToFreq_dict:
                        uniqueWord_ToFreq_dict[word] = 1
                    else:
                        uniqueWord_ToFreq_dict[word] = uniqueWord_ToFreq_dict[word] + 1

                handle_toWordList[handle] = feature_words

                line_count = line_count + 1

        # populate handle --> wordCountList
        with open(summed_representation_fileName) as file_wordCounts:
            line_count = 0
            for wordCounts_line in file_wordCounts:
                wordCounts_line = wordCounts_line.strip('\n')
                wordCounts_line = wordCounts_line.strip('\r')
                wordCounts_list = wordCounts_line.split(',')

                handle = wordCounts_list[0]
                wordCounts_list = wordCounts_list[1:]

                if not all_handles: #
                    if handle not in handles_inFreqBucket_list:
                        continue # if current handle is not in specified freqBucket, ignore

                wordCounts_list = [int(x) for x in wordCounts_list]

                handle_toWordCountList[handle] = wordCounts_list

                line_count = line_count + 1

        # start here, implement direct string comparison, after construction vectors for two handles from aquire feature space
        # get lists of handles with same vector, sorted by by length


        uniqueWord_ToIndex_dict = {}
        unqiue_word_count = 0
        features_Meaningful =  [k for k,v in sorted(uniqueWord_ToFreq_dict.items(), key=operator.itemgetter(1), reverse = True ) if v >= 2]
        for word in features_Meaningful:
                uniqueWord_ToIndex_dict[word] = unqiue_word_count
                unqiue_word_count = unqiue_word_count + 1

        print 'Size compressed: ', len(features_Meaningful)
        print 'Num features droped: ', ( len(uniqueWord_ToFreq_dict) - len(features_Meaningful) )

        high_k_vectors = []

        handle_toVector_dict = {} # used if reduced_runtime=True to store in memory all handles as full vector representations, such that the conversion to full vector occurs once per handle, rather than for each comparison

        print 'Starting handle vector comparisons... '
        print 'tweet_freq_type:', tweet_freq_type
        if binary:
            print 'binary:', str(binary)

        handles_written_toFile_list = [] # avoid multiple writes of the same vector

        lineWrite_count = 0
        handle_1_count = 0
        for handle_1 in handle_list:
            start_time = time.time()
            k = -1 # keep track of # of handles that are same (according to vector reprsentation)
            handle_list_curr = []  # keep track of handles with same vector as handle_1 (including handle_1)

            handle_1_count = handle_1_count + 1
            # construction of full 750k feature space vector for handle_list_inOrder
            wordList_1 = handle_toWordList[handle_1]

            wordCounts_list_1 = [int(x) for x in handle_toWordCountList[handle_1] ]
            handle_vector__nonZeroFeatureSet_1 = []


            # use sets of (value, nonzero valued feature index)

            # need to aquire count for each word in wordList for a given handle...

            if len(wordList_1) !=  len(wordCounts_list_1) :
                print 'ERROR: len(wordList_1) != len(wordCounts_list_1) in Handle_Uniqueness_Metrics'
                return


            feature_index = 0
            feature_value = 0
            nonZeroFeature_Value_FeatureIndex_list_1 = []
            for word_num in range(len(wordList_1)):

                if wordList_1[word_num] in uniqueWord_ToIndex_dict:
                    feature_index = uniqueWord_ToIndex_dict[wordList_1[word_num]]
                    feature_value = wordCounts_list_1[word_num]

                    if feature_value != 0: # nonzero feature value
                        if binary:
                            feature_value = 1

                    element = str([feature_value, feature_index])

                    nonZeroFeature_Value_FeatureIndex_list_1.append(element)

                # print handle_vector_1

            handle_vector_1 = set(nonZeroFeature_Value_FeatureIndex_list_1)

            if reduced_runtime:
                handle_toVector_dict[handle_1] = handle_vector_1

            for handle_2 in handle_list:
                # print 'Comparing ', handle_1, '[handle ', handle_1_count, 'of', len(handle_list),'] with ', handle_2

                handle_vector_2 = []

                if reduced_runtime and handle_2 in handle_toVector_dict: # aquire vector representation of handle already in dict, such that the transformation into a full vector (computationally expensive) only has to occur on a per-handle basis
                    handle_vector_2 = handle_toVector_dict[handle_2]

                    # if binary:
                    #     handle_vector_2 = Binary_Transform_Vector(handle_vector_2)

                else:
                    wordList_2 = handle_toWordList[handle_2]
                    wordCounts_list_2 = [int(x) for x in handle_toWordCountList[handle_2] ]
                    handle_vector_2 = []

                    if len(wordList_2) != len(wordCounts_list_2):
                        print 'ERROR: len(wordList_2) != len(wordCounts_list_2) in Handle_Uniqueness_Metrics'
                        return

                    nonZeroFeature_Value_FeatureIndex_list_2 = []
                    for word_num in range(len(wordList_2)):
                        # handle_vector_2[ uniqueWord_ToIndex_dict[wordList_2[word_num]] ] = wordCounts_list_2[word_num] #  fill the vector, with each value going to the featue index corresponding to the value's word

                        feature_index = uniqueWord_ToIndex_dict[wordList_2[word_num]]
                        feature_value = wordCounts_list_2[word_num]

                        if feature_value != 0: # nonzero feature value
                            if binary:
                                feature_value = 1

                        element = str([feature_value, feature_index])

                        nonZeroFeature_Value_FeatureIndex_list_2.append(element)

                    # print handle_vector_1

                    handle_vector_2 = set(nonZeroFeature_Value_FeatureIndex_list_2)

                    handle_toVector_dict[handle_2] = handle_vector_2

                    #  = csr_matrix(handle_vector_2)

                    # if binary:
                    #     handle_vector_2 = Binary_Transform_Vector(handle_vector_2)
                # construction of full 750k feature space vector for

                # check if same or not

                # if same ...     how to not store full vector to keep track of k?
                # compare a handle with all other handles. if this handle is same as k other handles, output specifics to file
                    # need way of not putting out statistis for all handles corresponding to same k

                # count only a 'collision' on self as k=0, (k starts at -1 for each iteration over handle_vector_1 values)
                # if (handle_vector_1 - handle_vector_2).nnz == 0: # quick way of checking if two sparse matrices are the same
                if handle_vector_1 == handle_vector_2:
                    k = k + 1
                    handle_list_curr.append(handle_2)
                    # print 'k=',str(k)

            if k >= 1: # output vector and corresponding data only if collision between at least two handles
                # if handle_1 not in handles_written_toFile_list:

                if handle_vector_1 not in high_k_vectors:
                    # handles_written_toFile_list.append(handle_1)

                    high_k_vectors.append(handle_vector_1)
                    # vector
                    # k
                    # handles involved
                    # outfile_line = str(handle_vector_1)
                    output_line = ''

                    if len(handle_vector_1) == 0: # then no nonZero featur values, thus allZero vector
                        allZeroVector = [0]
                        outfile_line = str(allZeroVector)
                    else:
                        allZeroVector = [1]
                        outfile_line = str(allZeroVector)
                    # need to convert to

                    outfile_line = outfile_line + ', k=' + str(k)
                    for handle in handle_list_curr:
                        outfile_line = outfile_line + ', ' + handle

                    outfile_line = outfile_line + '\n'
                    outfile.write(outfile_line)

                    lineWrite_count = lineWrite_count + 1
                    if k >= 2:
                        print 'K GREATER THAN 1 (more than singular self-collision)'

            elapsed_time = time.time() - start_time
            print 'Calculated similarty for ', handle_1, ', handle ', handle_1_count, ' out of ', len(handle_list), ' took ', elapsed_time, ' seconds'

        outfile.close()


        print 'Finished Handle_Uniqueness_Metrics() for ', summed_representation_fileName, ' binary=', binary
        print 'Wrote ', str(lineWrite_count), 'lines'


    else:

        # keep x number of features for each vector  ist[:x]

        count_num_handles_EclassGreaterThan_1_full = 0
        count_num_handles_EclassGreaterThan_1_high = 0
        count_num_handles_EclassGreaterThan_1_med = 0
        count_num_handles_EclassGreaterThan_1_low = 0

        # keep count of number of handles with equivalency class of size >= 2

        # for every non 'self' collision, need to add handle/vector to outputfile ?
        # handles_with_nonUniqueVector = []
        # len_vector = 0
        lineWrite_count = 0
        # load handles and tweets in
        with open(summed_representation_fileName) as file_vectors, open(filename_handles) as file_handles:
            line_count = 0


            for line, handle in izip(file_vectors, file_handles):
                line_count = line_count + 1
                # print 'processing tweet ', line_count

                line = line.strip('\n')
                line = line.strip('\r')
                line_elements = line.split(',')

                handle = line_elements[0]
                vector_elements = line_elements[1:]

                vector_elements = [int(x) for x in vector_elements]

                vector_elements = vector_elements[:num_features_ToKeep]

                len_vector = len(vector_elements)

                if binary:
                    vector_elements = Binary_Transform_Vector(vector_elements)

                vector_stringForm = str(vector_elements)

                # if bag of words, whats the best (in terms of minimal size) that I could use that would also work for binary
                    # only words, sort in order? what about binary?
                    # do I have to use whole feature space?
                    # apex legomenon?

                    # if dont want to store 750k feature space reprsentation, have to parse in pure words format and convert into vector representation

                if vector_stringForm in handleRepresentation_toCount:
                    handleRepresentation_toCount[vector_stringForm].append(handle)
                    # handles_with_nonUniqueVector.append(handle)
                else:
                    handleRepresentation_toCount[vector_stringForm] = []
                    handleRepresentation_toCount[vector_stringForm].append(handle)

    # go through and perform analysis
    # how many representations with different #s of corresponding handles
    # avg

    # aquire handles with 1, 2, 3, other similiar etc.

        # get lists of handles with same vector, sorted by by length
        lists_by_length = sorted(handleRepresentation_toCount.values(), key=len, reverse=True)

        handlesAlreadyCounts_list_low = []

        # for handle_list_inOrder in lists_by_length:

        for vectorString, handle_list_curr in handleRepresentation_toCount.iteritems():

                # if handle_list_inOrder == handle_list_curr:

            vector = ast.literal_eval(vectorString)

            allZeroVector = True

            for value in vector:
                if value != 0:
                    allZeroVector = False
                    break

            if allZeroVector:  # count number of handles with all zero vector for each bucket

                num_handles_withAllZeroVector_full = len(handle_list_curr)

                for handle in handle_list_curr:

                    if handle in handles_inFreqBucket_list_high:
                        num_handles_withAllZeroVector_high = num_handles_withAllZeroVector_high + 1

                    elif handle in handles_inFreqBucket_list_med:
                        num_handles_withAllZeroVector_med = num_handles_withAllZeroVector_med + 1

                    elif handle in handles_inFreqBucket_list_low:
                        num_handles_withAllZeroVector_low = num_handles_withAllZeroVector_low + 1
            else:

                # ignore all zero vector
                E_Class_Sizes_list_full.append(len(handle_list_curr))

                size_E_high = 0
                size_E_med = 0
                size_E_low = 0

                for handle in handle_list_curr:
                    if handle in handles_inFreqBucket_list_high:
                        size_E_high = size_E_high + 1
                    elif handle in handles_inFreqBucket_list_med:
                        size_E_med = size_E_med + 1
                    elif handle in handles_inFreqBucket_list_low:
                        size_E_low = size_E_low + 1

                # only add to list of equivalency class sizes if at least one handle in the given bucket for that vector
                if size_E_high > 0:
                    E_Class_Sizes_list_high.append(size_E_high)
                if size_E_med > 0:
                    E_Class_Sizes_list_med.append(size_E_med)
                if size_E_low > 0:
                    E_Class_Sizes_list_low.append(size_E_low)


                if len(handle_list_curr) >= 2: # if more than one handle in equivalency class, increase count of users with equivalency class greater than one
                    count_num_handles_EclassGreaterThan_1_full = count_num_handles_EclassGreaterThan_1_full + len(handle_list_curr)

                    # increment counts for buckets of handles
                    for handle in handle_list_curr:

                        if handle in handles_inFreqBucket_list_high:
                            count_num_handles_EclassGreaterThan_1_high = count_num_handles_EclassGreaterThan_1_high + 1

                        elif handle in handles_inFreqBucket_list_med:
                            count_num_handles_EclassGreaterThan_1_med = count_num_handles_EclassGreaterThan_1_med + 1

                        elif handle in handles_inFreqBucket_list_low:
                            count_num_handles_EclassGreaterThan_1_low = count_num_handles_EclassGreaterThan_1_low + 1

                    # vector
                    # k
                    # handles involved
# OUTFILE functionality
                    # outfile_line = vectorString
                    # outfile_line = outfile_line + ', k=' + str(len(handle_list_curr))
                    # for handle in handle_list_curr:
                    #     outfile_line = outfile_line + ', ' + handle

                    # outfile_line = outfile_line + '\n'
                    # outfile.write(outfile_line)

                    # lineWrite_count = lineWrite_count + 1


        # for handle in handles_with_nonUniqueVector: # add handles_with_nonUniqueVector
        #     dummy_vector = [-1] * len_vector
        #     outfile_line = str(dummy_vector)
        #     outfile_line = outfile_line + ', k=0' + str(len(handle_list_curr))
        #     outfile_line = outfile_line + ', ' + handle
        #     outfile_line = outfile_line + '\n'
        #     outfile.write(outfile_line)


        # outfile.close()

        # print 'Finished Handle_Uniqueness_Metrics() for ', summed_representation_fileName, ' binary=', binary
        # print 'Wrote ', str(lineWrite_count), 'lines'

        if num_features_ToKeep < 1000:
            projection_name = 'first ' + str(num_features_ToKeep) + ' features, ' + projection_name

        privacy_gained_full = count_num_handles_EclassGreaterThan_1_full / float(5626 - num_handles_withAllZeroVector_full)
        privacy_gained_high = count_num_handles_EclassGreaterThan_1_high / float(len(handles_inFreqBucket_list_high) - num_handles_withAllZeroVector_high)
        privacy_gained_med = count_num_handles_EclassGreaterThan_1_med / float(len(handles_inFreqBucket_list_med)- num_handles_withAllZeroVector_med)
        privacy_gained_low = count_num_handles_EclassGreaterThan_1_low / float(len(handles_inFreqBucket_list_low) - num_handles_withAllZeroVector_low)

        print 'privacy_gained_full: ', privacy_gained_full
        print 'privacy_gained_high: ', privacy_gained_high
        print 'privacy_gained_med: ', privacy_gained_med
        print 'privacy_gained_low: ', privacy_gained_low
        print
        print 'num_handles_withAllZeroVector_full: ', num_handles_withAllZeroVector_full
        print 'num_handles_withAllZeroVector_high: ', num_handles_withAllZeroVector_high
        print 'num_handles_withAllZeroVector_med: ', num_handles_withAllZeroVector_med
        print 'num_handles_withAllZeroVector_low: ', num_handles_withAllZeroVector_low
        print

        trace_full = go.Box(
            y = E_Class_Sizes_list_full,
            name='All <br> <i>protected=<i> ' + str(count_num_handles_EclassGreaterThan_1_full) + '<br>  <i>all-zero=<i> ' + str(num_handles_withAllZeroVector_full),
            boxpoints='all',
            # jitter=0.1,
            whiskerwidth=0.2,
            fillcolor='rgb(93, 164, 214)',
            marker=dict(
                # size = 2,
                color='rgb(93, 164, 214)',
            ),
            line=dict(width=1),
            boxmean='sd'
        )

        trace_high = go.Box(
            y = E_Class_Sizes_list_high,
            name='High <br> <i>protected=<i> ' + str(count_num_handles_EclassGreaterThan_1_high) + '<br>  <i>all-zero=<i> ' + str(num_handles_withAllZeroVector_high),
            boxpoints='all',
            # jitter=0.1,
            whiskerwidth=0.2,
            fillcolor='rgb(255, 144, 14)',
            marker=dict(
                # size = 2,
                color='rgb(255, 144, 14)',
            ),
            line=dict(width=1),
            boxmean='sd'
        )
        trace_med = go.Box(
            y = E_Class_Sizes_list_med,
            name='Moderate <br> <i>protected=<i> ' + str(count_num_handles_EclassGreaterThan_1_med) + '<br>  <i>all-zero=<i> ' + str(num_handles_withAllZeroVector_med),
            boxpoints='all',
            # jitter=0.1,
            whiskerwidth=0.2,
            fillcolor='rgb(44, 160, 101)',
            marker=dict(
                # size = 2,
                color='rgb(44, 160, 101)',
            ),
            line=dict(width=1),
            boxmean='sd'
        )
        trace_low = go.Box(
            y = E_Class_Sizes_list_low,
            name='Low <br> <i>protected=<i> ' + str(count_num_handles_EclassGreaterThan_1_low) + '<br>  <i>all-zero=<i> ' + str(num_handles_withAllZeroVector_low),
            boxpoints='all',
            # jitter=0.1,
            whiskerwidth=0.2,
            fillcolor='rgb(255, 65, 54)',
            marker=dict(
                # size = 2,
                color='rgb(255, 65, 54)',
            ),
            line=dict(width=1),
            boxmean='sd'
        )


        layout = go.Layout(
            annotations=Annotations([
                Annotation(
                    x=0.5004254919715793,
                    y=-0.12191064079952971,
                    showarrow=False,
                    text='',
                    xref='paper',
                    yref='paper'
                ),
                Annotation(
                    x=-0.07944728761514841,
                    y=0.4714285714285711,
                    showarrow=False,
                    text='Equivalency class sizes, excluding all zero vector',
                    textangle=-90,
                    xref='paper',
                    yref='paper'
                )
            ]),
            autosize=True,
            title='Distribution of Handle Equivalency Class Sizes: ' + projection_name,
            yaxis=dict(
               zeroline=False,
               range=[0, 20]
            ),
                    # # margin=dict(
            # #     l=40,
            # #     r=30,
            # #     b=80,
            # #     t=100,
            # # ),
            # paper_bgcolor='rgb(243, 243, 243)',
            # plot_bgcolor='rgb(243, 243, 243)',
            showlegend=False
        )

        # traces = [trace_full, trace_high, trace_med, trace_low]
        traces = [trace_high, trace_med, trace_low]


        fig = go.Figure(data=traces, layout=layout)
        py.iplot(fig, filename=projection_name + ' - Equivalancy Class Sizes - 3 Buckets')

def cosine_BagOfWords_pairWise(v1,v2):

    # need to change to take entire training data


    # change to not consider 0 valued elements

    # go through both vectors, only keep features that aren't both valued as 0

    # pass in dictionary or tuple
        # tuple (word, freq_count)
        # dictionary:  word --> freq_cout
        # do this approach only for bag of words
        # use other method for other representations


    nonZero_values_v1_list = []
    nonZero_values_v2_list = []

    nonZero_indices_v1_list = []
    nonZero_indices_v2_list = []

    # aquire list on nonZero indices for v1, v2
    # aquire list of nonZero values for v1, v2

    # this is done assuming that both vectors are of the same length, which is not necessarily the case...


    # iterate over biger vector, and dont add to smaller vector outside of range
    max_vectorSize = max ( len(v1), len(v2) )

    for index in range ( max_vectorSize ):
        v1_value_i = v1[index]
        v2_value_i = v2[index]

        if (v1_value_i != 0.0) & ( index < len(v1) ):
            nonZero_values_v1_list.append(v1_value_i)
            nonZero_indices_v1_list.append(index)

        if (v2_value_i != 0.0) & ( index < len(v2) ):
            nonZero_values_v2_list.append(v2_value_i)
            nonZero_indices_v2_list.append(index)

    # creat 'union' comparison vectors

    comparison_vector_1 = []
    comparison_vector_2 = []

    # need to
        # two cases: [1] v1 is bigger & [2] v2 is bigger
    #


    if len(nonZero_indices_v1_list) < len(nonZero_indices_v2_list): # if v1 smaller, start with featurs in v1

        for index in range(len(nonZero_indices_v1_list)): # fill vectors with values corresponding to features observed in first vector
            # append value for this index to vector_1
            comparison_vector_1.append( nonZero_values_v1_list[index] )

            # if this feature index is present in nonZero_indices_v2_list, then add the value for this featue in v2 to vector2
            v1_feature_index_i = nonZero_indices_v1_list[index]

            if v1_feature_index_i in nonZero_indices_v2_list: # check that same feature is in v2
                if index >= len(nonZero_values_v2_list):
                    pdb.set_trace()

                comparison_vector_2.append( nonZero_values_v2_list[index] )
            # else, append zero at this index
            else:
                comparison_vector_2.append( 0.0 )


        for index in range(len(nonZero_indices_v2_list)):
        # then do the same thing (except don't add to both vectors unless featue index not in v1) for featue indices in v2 that aren't in v1
            v2_featue_index_i = nonZero_indices_v2_list[index]

            if v2_featue_index_i not in nonZero_indices_v1_list: # check that same feature is not in v1; if so, then this feature has not been incorporated ye
                comparison_vector_2.append( nonZero_values_v2_list[index] )
                comparison_vector_1.append( 0.0 ) # append zero for this feature value for handle 1

    else: # if v2 smaller, start with featuers in v2

        for index in range(len(nonZero_indices_v2_list)): # fill vectors with values corresponding to features observed in first vector
            # append value for this index to vector_1
            comparison_vector_2.append( nonZero_values_v2_list[index] )

            # if this feature index is present in nonZero_indices_v2_list, then add the value for this featue in v2 to vector2
            v2_feature_index_i = nonZero_indices_v2_list[index]

            if v2_feature_index_i in nonZero_indices_v1_list: # check that same feature is in v2
                if index >= len(nonZero_values_v1_list):
                    pdb.set_trace()

                comparison_vector_1.append( nonZero_values_v1_list[index] )
            # else, append zero at this index
            else:
                comparison_vector_1.append( 0.0 )


        for index in range(len(nonZero_indices_v1_list)):
        # then do the same thing (except don't add to both vectors unless featue index not in v1) for featue indices in v2 that aren't in v1
            v1_featue_index_i = nonZero_indices_v1_list[index]

            if v1_featue_index_i not in nonZero_indices_v2_list: # check that same feature is not in v1; if so, then this feature has not been incorporated ye
                comparison_vector_1.append( nonZero_values_v1_list[index] )
                comparison_vector_2.append( 0.0 ) # append zero for this feature value for handle 1

    # for word, value in v1.iteritems():
    #     comparison_vector_1.append(value)

    #     if word in v2: # then append the value for the same feate
    #         comparison_vector_2.append(v2[word])
    #     else: # add zo
    #         comparison_vector_2.append(0.0)

    #     comparison_featureSpace.append(word)

    # for word, value in v2.iteritems(): # add words to featue space that weren't in v1

    #     if word not in comparison_featureSpace: # then the word is not in v1
    #         comparison_featureSpace.append(word)

    #         comparison_vector_1.append(0.0)
    #         comparison_vector_2.append(value)

    # pdb.set_trace()
    #  "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"


    result = 1 - spatial.distance.cosine(comparison_vector_1, comparison_vector_2) # aquire cosine similarity
    print result
    if result != result:
        pdb.set_trace()

    # safe_sparse_dot(X_normalized, X_normalized, dense_output=dense_output)

    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(comparison_vector_1)):
        x = comparison_vector_1[i]; y = comparison_vector_2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y

    result = 0.0
    if math.sqrt(sumxx*sumyy) != 0:
        result = sumxy/math.sqrt(sumxx*sumyy)

    print result

    return result

def cosine_BagOfWords(X): # takes in np.arry

    # for each handle, compare with all other handles

    result = []

    print 'In custom cosine method'

    # What if just aquire new training data, and then pass that back?

    # do I want to normalize?
    # X_normalized = normalize(X)

    count = 0
    for handleVector_1 in X:
        result_forAHandle = []
        # count_sub = 0
        start_time = time.time()
        for handleVector_2 in X:
            result_forAHandle.append( cosine_BagOfWords_pairWise(handleVector_1, handleVector_2) )
            # print 'Calculated pair-wise similarty ', count_sub, ' out of ', len(X)
            # count_sub = count_sub + 1
        result.append(result_forAHandle)
        elapsed_time = time.time() - start_time
        print 'Calculated similarty for handle ', count, ' out of ', len(X), ' took ', elapsed_time, ' seconds'
        count = count + 1

    result = np.array(result_forAHandle)
    return result

def mycosine(x1, x2):
    # how could i modify this to only consider the non-zero features from both x1 and x2?

    # what if i just keep track of what word each value corresponds to?

    # need:
        # feature space index of each non-zero value
        # value of each non-zero value

    # if i'm passed in two vectors of values, how to I obtain the ...the

    # alright. convert each bag of words handle represenation into a vector of size = total # of words in corpus...
        # need to assertain how large that number is though...


    x1 = x1.reshape(1,-1)
    x2 = x2.reshape(1,-1)
    ans = 1 - cosine_similarity(x1, x2)
    return max(ans[0][0], 0)

# implement bag of words representation?


# what should this do exactly?
#
def BagOfWords_Handling ():
    # aquire feature space (all words present) ....
        # convert feature space into word --> index in feature space dict
        # or should I just do direct string comparisons on the words themselves?
        #no,



    # seems like BagOfWords_handling should be occuring on simliarity level, rather than constructing a vector with many zeros
        # maybe pass boolean to simliarity metrics, such that they behave differently

    # difference about bag of words: the words (features) are not in the same order
        # aquire feature space by building word --> freq dict, and sorting

    # idea: convert ALL summed representations into:
        #[1] the indices (in the feature space) of features whose value's arent zero, and [2] the values of said features
        # for each similarity metric, only compare between these two sets

    print

    # check first line? or check fileName of tweet representation itself?


def isBagOfWords( input_fileName ):
    if 'BagOfWords' in input_fileName.split('_'):
        return True
    else:
        return False



def Frequent_ItemSets ( input_fileName, tweet_representation_fileName, projection_input=False, emotion=True, sup_div = 100, num_features_ToKeep = 1000 ):

    # for my projections, need convert tweets:  remove words that aren't in the given projection's feature space
    # in either case, will still be using clean tweets, as main input, but in the case of a projection evaluation, need to load in projection file that has feature space in it

    # if projection_input: # need to aquire feature space of said projection

    # will evaluate according to:
    # [1] Proportion of sets absent in projection that were present in frequent item set of baseline
    # [2] proportion of sets that are now present in frequent item set of projection, but were not in the baseline
    # [3] the number that are present that should be present

    tweets = []
    feature_space = []

    if projection_input:
        with open(tweet_representation_fileName) as file_tweets:

                    # issue: first line in tweet file is header, but first line in handle file corresponds to second line in tweet file
            header_line = file_tweets.readline()
            header_line = header_line.strip('\n')
            header_line = header_line.strip('\r')
            header_elements = header_line.split(',')

            feature_space = header_elements

            if num_features_ToKeep != 1000:
                feature_space = feature_space[:num_features_ToKeep]
                # print 'new feature_space size: ', str(len(feature_space))
                # print feature_space

            size_featureSpace = len(feature_space)

            # print 'size_featureSpace: ', str(size_featureSpace)

    if emotion:
            feature_space = []
            with open('emotion_words.txt') as file_tweets:

                        # issue: first line in tweet file is header, but first line in handle file corresponds to second line in tweet file
                header_line = file_tweets.readline()
                header_line = header_line.strip('\n')
                header_line = header_line.strip('\r')
                header_elements = header_line.split(',')

                feature_space = header_elements

                size_featureSpace = len(feature_space)

                # print 'size_featureSpace: ', str(size_featureSpace)

    num_tweets_withFeatures_inFeatureSpace = 0

    input_file = open(input_fileName)
    line_count = 0
    for line in input_file:

        # print 'processing line', line_count
        line_count = line_count + 1
        line = line.strip('\n')
        word_tokens = re.sub(r'[^\w\'] ', " ",  line).split() # convert into list of words

        if projection_input:
            word_tokens = [x for x in word_tokens if x in feature_space]

            if 0 != len(word_tokens):
                num_tweets_withFeatures_inFeatureSpace = num_tweets_withFeatures_inFeatureSpace + 1
            # remove words not in a feature space

        tweets.append(word_tokens)

    min_support = len(tweets) /  sup_div # 1% support

    if projection_input:
        min_support = num_tweets_withFeatures_inFeatureSpace / 500

    # print tweets[0]
    print 'Performing Frequent_ItemSets() with min_support=', str(min_support / float(4000000)) ,' for ', input_fileName

    # patterns = pyfpgrowth.find_frequent_patterns(tweets, min_support)

    # rules = pyfpgrowth.generate_association_rules(patterns, 0.7)

    relim_input = itemmining.get_relim_input(tweets)
    report = itemmining.relim(relim_input, min_support=min_support)
    # print report.items()
    print 'num sets: ', len(report.items())
    # print 'num_tweets_withFeatures_inFeatureSpace: ', str(num_tweets_withFeatures_inFeatureSpace)

    # te = TransactionEncoder()
    # te_ary = te.fit(dataset).transform(dataset)
    # oht_ary = te.fit(tweets).transform(tweets, sparse=True)
    # sparse_df = pd.SparseDataFrame(te_ary, columns=te.columns_, default_fill_value=False)

    # te = TransactionEncoder()
    # te_ary = te.fit(tweets).transform(tweets)
    # df = pd.DataFrame(te_ary, columns=te.columns_)
    # results = apriori(df, min_support=0.6, use_colnames=True)

    # te = TransactionEncoder()
    # oht_ary = te.fit(tweets).transform(tweets, sparse=True)
    # sparse_df = pd.SparseDataFrame(oht_ary, columns=te.columns_, default_fill_value=False)
    # pdb.set_trace()
    # results = apriori(sparse_df, min_support=min_support, use_colnames=True)

    return report

    # outfile_name = input_fileName[:-4] # remove '.txt'
    # outfile_name = outfile_name + '_Frequent_Itemsets.txt'
    # outfile = open(outfile_name, 'w')


# what should the parameters be of the apiori method? min_support? min_confidence? min_lift? min_length?
def Apriori_AssociationRules ( input_fileName, input_is_CSV = False ):

    # load in actual tweet data, make sure is working right
    # For sparse matrix representations, would we need convert each count for a given word to the word itself?


    # if input file not .csv format (i.e. original tweets), then

    # records can be a list of lists

    # load tweets from file
    tweets = []
    input_file = open(input_fileName)
    line_count = 0
    for line in input_file:
        print 'processing line', line_count
        line_count = line_count + 1
        word_tokens = re.sub(r'[^\w\'] ', " ",  line).split() # convert into list of words
        tweets.append(word_tokens)


    # dataset = pd.read_csv('apriori_data.csv', header = None)
    # records = [] for i in range(0, 101):
    # records.append([str(dataset.values[i,j]) for j in range(0, 10)])


    # train Apriori model
    rules = apriori(tweets, min_support = 0.001, min_confidence = 0.1, min_lift = 3, min_length = 1)

    # view rules
    results = list(rules)

    for rule in results:
        print rule.items, ', support: ', rule.support
        print rule.ordered_statistics
        print



# one hot encoder for bag of words model? Not needed if bag of words


# each of these handle documents would be represented as: the summed counts for the occurrence of each feature in the feature space, across all tweets corresponding to the handle


def Anomaly_Detection_LOF_2 ( sparse_representation_fileName, summed_representation_fileName, param_value_list, n_neighbors = 20, param_testing = False, binary=False ):
    print 'in 2'
    X = []
    handle_list = []

    if isBagOfWords(sparse_representation_fileName):
        X = load_pickle(sparse_representation_fileName)
        #convert training data to floats
        if binary: # convert tweets to binary
            X = feature_values_floatFormat = Binary_Transform_sparse(X)


        with open(summed_representation_fileName) as file:

            line_count = 0
            for line in file:
                # print 'Processing line ', line_count
                line_count = line_count + 1

                line = line.strip('\n')
                line = line.strip('\r')
                line_elements = line.split(',')

                handle = line_elements[0]
                feature_values = line_elements[1:]

                handle_list.append(handle)

    if param_testing:

        qualityMetric_list = []
        handles_toLabels_dict_list = []

        parameter_value_list = []
        sil_score_list = []
        percentageMaxPossilbeEntropy_list = []

        # X = preprocessing.normalize(X) # makes euclidean metric function as cosine similarity
        # X_norm = X
        # num_clusters_list = [2, 3, 4, 5, 6, 8, 10, 20, 50]
        # num_clusters_list = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        n_neighbors_list = param_value_list
        with open('cluster_parameterAnalysis.csv', 'a') as outfile:  # change to 'a' to append

            for n_neighbors in n_neighbors_list:

                print 'Calculating LOF..'
                lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, metric='euclidean')
                labels_ = lof_model.fit_predict(X)
                labels = labels_
                ranking_list = lof_model.negative_outlier_factor_.tolist()
                print 'Finished calculating LOF'

                print 'Calculating node quality metric...'
                quality_metric = metrics.silhouette_score(X, labels, metric='euclidean')
                print quality_metric, ', n_neighbors: ', str( n_neighbors )

                qualityMetric_list.append(quality_metric)
                handles_toLabels_dict = {}
                for index in range(len(labels)):
                    handles_toLabels_dict[handle_list[index]]  = labels[index]
                handles_toLabels_dict_list.append(handles_toLabels_dict)

                # pdb.set_trace()

                # largestCluster_size = 0
                # for cluster_id in range(cluster_num):
                #     cluster_size = len([k for k,v in sorted(handles_toLabels_dict.items(), key=operator.itemgetter(1), reverse = True ) if v == cluster_id])
                #     if cluster_size > largestCluster_size:
                #         largestCluster_size = cluster_size
                # print 'Size of largest cluster:', str(largestCluster_size)


                # BEGIN parameter results collection #
                    # for easy copy and pasting
                        # parameter_value_list = []
                        # sil_score_list = []
                        # percentageMaxPossilbeEntropy_list = []

                        # return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list

                        # param_value_list
                    # end for easy copy and pasting

                # size_ofEachCluster_dict = {}
                # for label in labels:
                #     if label in size_ofEachCluster_dict:
                #         size_ofEachCluster_dict[label] = size_ofEachCluster_dict[label] + 1
                #     else:
                #         size_ofEachCluster_dict[label] = 1

                # size_ofEachCluster_list = size_ofEachCluster_dict.values()
                # entropy = stats.entropy(size_ofEachCluster_list)
                # size_ofEachCluster_EVEN_list = [1 for x in range(len(size_ofEachCluster_list))] # create assumption of completely eeven distribution of labels across clusters
                # maxPossible_entropy = stats.entropy(size_ofEachCluster_EVEN_list)

                #   percentageMaxPossilbeEntropy = entropy of actual distribution / maximum possible entropy
                # percentageMaxPossilbeEntropy = entropy / float(maxPossible_entropy)

                # qualityMetric_list.append(quality_metric)

                # parameter_value_list.append(n_neighbors)
                # sil_score_list.append(quality_metric)
                # percentageMaxPossilbeEntropy_list.append(percentageMaxPossilbeEntropy)
                # # END parameter results collection #

                # outfile_line = str(quality_metric) + ', LOF, ' + 'n_neighbors=' + str(n_neighbors) + 'binary=' + str(binary) + ', ' + 'largest_clusterSize=' + str(largestCluster_size) + ', ' + summed_representation_fileName + '\n'
                # outfile.write(outfile_line)

        print 'Aquired labels and quality metrics to evaluate: qualityMetric_list, labels_list, num_clusters_list' + 'binary=' + str(binary)
        print 'handles_toLabels_dict_list'

        # return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list
    else:

        print 'Calculating LOF..'
        lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, metric='euclidean')
        labels_ = lof_model.fit_predict(X)
        labels = labels_
        print 'Finished calculating LOF'

        # print 'Calculating node quality metric...'
        quality_metric = metrics.silhouette_score(X, labels, metric='euclidean')
        # print quality_metric, ', n_neighbors: ', str( n_neighbors )

        # need to get list of handles...

        ranking_list = lof_model.negative_outlier_factor_.tolist()

        # qualityMetric_list.append(quality_metric)
        # handles_toLabels_dict = {}
        # for index in range(len(labels)):
        #     handles_toLabels_dict[handle_list[index]]  = labels[index]
        # handles_toLabels_dict_list.append(handles_toLabels_dict)

        # largestCluster_size = 0
        # for cluster_id in range(cluster_num):
        #     cluster_size = len([k for k,v in sorted(handles_toLabels_dict.items(), key=operator.itemgetter(1), reverse = True ) if v == cluster_id])
        #     if cluster_size > largestCluster_size:
        #         largestCluster_size = cluster_size
        # print 'Size of largest cluster:', str(largestCluster_size)

        return ranking_list, handle_list

        # print 'ERROR: expected param_testing = True'

def Anomaly_Detection_LOF ( summed_representation_fileName, parameter_value_list, n_neighbors=20, param_testing=False, binary=False, num_features_ToKeep = 10 ):
    print 'in 1'
    # TO DO:
# sort output anomalies, and then sort
# start with value of 10 for
# 5, 10, 20, 50
# see what handles are output as anomalies, and then look at their actual tweets or summed vector representation
    handle_list = []
    # DECIDE:
    # standardize features? in between 0 and 1

    # contamination assumption (incidence of anomolies)? default is .1

    # list of lists, sparse representation of feature occurrences across all tweets for training model
    training_data = []

    # load in summed representation file
    with open(summed_representation_fileName) as file:

        line_count = 0
        for line in file:
            line_count = line_count + 1

            line = line.strip('\n')
            line = line.strip('\r')
            line_elements = line.split(',')

            handle = line_elements[0]
            feature_values = line_elements[1:]

            feature_values = feature_values[:num_features_ToKeep]

            handle_list.append(handle)

            # print 'num_features kept: ', str(len(feature_values))

            #convert training data to floats
            feature_values_floatFormat = [ int(x) for x in feature_values]
            if ( not isBagOfWords(summed_representation_fileName) ) and binary: # convert tweets to binary
                feature_values = Binary_Transform_Vector(feature_values_floatFormat)

            training_data.append( feature_values )

    # if isBagOfWords(summed_representation_fileName):
    #     uniqueWord_ToIndex_dict = {} # assign a 'feature number' to each word
    #     uniqueWord_ToFreq_dict = {} # keep track of count for each word

    #     unqiue_word_count = 0
    #     vectors_list = []
    #     words_filename = summed_representation_fileName[:-4]
    #     represenation_name = words_filename
    #     words_filename = words_filename + '_wordLists.txt'

    #     handle_list = []
    #     nonZero_Features_list = []

    #     with open(words_filename) as file_words:
    #         line_count = 0
    #         for words_line in file_words:
    #             words_line = words_line.strip('\n')
    #             words_line = words_line.strip('\r')
    #             words_list = words_line.split(',')

    #             handle = words_list[0]
    #             feature_words = words_list[1:]

    #             handle_list.append(handle)
    #             nonZero_Features_list.append(feature_words)

    #             # wordTo_freqCount_vectorDict = {}

    #             for word in feature_words:
    #                 if word not in uniqueWord_ToIndex_dict:
    #                     uniqueWord_ToIndex_dict[word] = unqiue_word_count
    #                     unqiue_word_count = unqiue_word_count + 1
    #                     # print word, ' ', unqiue_word_count

    #                 if word not in uniqueWord_ToFreq_dict:
    #                     uniqueWord_ToFreq_dict[word] = 1
    #                 else:
    #                     uniqueWord_ToFreq_dict[word] = uniqueWord_ToFreq_dict[word] + 1


    #             line_count = line_count + 1

    #     uniqueWord_ToIndex_dict = {}
    #     unqiue_word_count = 0
    #     features_Meaningful =  [k for k,v in sorted(uniqueWord_ToFreq_dict.items(), key=operator.itemgetter(1), reverse = True ) if v >= 0]
    #     for word in features_Meaningful:
    #             uniqueWord_ToIndex_dict[word] = unqiue_word_count
    #             unqiue_word_count = unqiue_word_count + 1

    #     print 'Size compressed: ', len(features_Meaningful)
    #     print 'Num features droped: ', ( len(uniqueWord_ToFreq_dict) - len(features_Meaningful) )

    #     print 'Constructing feature space for Anomaly_Detection_LOF', summed_representation_fileName, ' ...'
    #     training_data_NEW = []
    #     count = 0
    #     for nonZero_Features, value_vector in izip (nonZero_Features_list, training_data):
    #         handle_vector = [0.0] * len(features_Meaningful)
    #         curr_index = 0
    #         for word, value in izip (nonZero_Features, value_vector):
    #             if word in uniqueWord_ToIndex_dict:
    #                 handle_vector[ uniqueWord_ToIndex_dict[word] ] = value #  fill the vector, with each value going to the featue index corresponding to the value's word
    #                 curr_index = curr_index + 1

    #         training_data_NEW.append(handle_vector)
    #         count = count + 1
    #     print 'Created vectors with full feature spaces, number: ', len(training_data)

    #     n_samples = len(training_data_NEW)
    #     # training_data_NEW = np.asarray(training_data_NEW)
    #     # training_data_NEW.reshape(n_samples, -1)
    #     training_data = training_data_NEW
    #     training_data_NEW = []

    # train the model
    # lof_model.train( training_data )

    if param_testing:
        n_neighbors_list = parameter_value_list

        sil_score_list = []
        percentageMaxPossilbeEntropy_list = []

        parameter_value_list = []
        sil_score_list = []
        percentageMaxPossilbeEntropy_list = []

        with open('cluster_parameterAnalysis.csv', 'a') as outfile:  # change to 'a' to append

            for n_neighbors in n_neighbors_list:

                print 'Calculating LOF..'
                lof_model = LocalOutlierFactor(n_neighbors=n_neighbors)
                labels_ = lof_model.fit_predict(training_data)
                labels = labels_
                ranking_list = lof_model.negative_outlier_factor_.tolist()
                print 'Finished calculating LOF'

                # print 'Calculating node quality metric...'
                # quality_metric = metrics.silhouette_score(training_data, labels, metric='euclidean')
                # print quality_metric, ', num clusters: ', str( n_neighbors )

                    # BEGIN parameter results collection #
                    # for easy copy and pasting
                        # parameter_value_list = []
                        # sil_score_list = []
                        # percentageMaxPossilbeEntropy_list = []

                        # return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list

                        # param_value_list
                    # end for easy copy and pasting
                pdb.set_trace()

                size_ofEachCluster_dict = {}
                for label in labels:
                    if label in size_ofEachCluster_dict:
                        size_ofEachCluster_dict[label] = size_ofEachCluster_dict[label] + 1
                    else:
                        size_ofEachCluster_dict[label] = 1

                size_ofEachCluster_list = size_ofEachCluster_dict.values()
                entropy = stats.entropy(size_ofEachCluster_list)
                size_ofEachCluster_EVEN_list = [1 for x in range(len(size_ofEachCluster_list))] # create assumption of completely eeven distribution of labels across clusters
                maxPossible_entropy = stats.entropy(size_ofEachCluster_EVEN_list)

                #   percentageMaxPossilbeEntropy = entropy of actual distribution / maximum possible entropy
                percentageMaxPossilbeEntropy = entropy / float(maxPossible_entropy)

                # qualityMetric_list.append(quality_metric)

                parameter_value_list.append(n_neighbors)
                sil_score_list.append(quality_metric)
                percentageMaxPossilbeEntropy_list.append(percentageMaxPossilbeEntropy)
                # END parameter results collection #

                outfile_line = str(quality_metric) + ', LOF, ' + 'n_neighbors=' + str(n_neighbors) + ', ' + summed_representation_fileName + '\n'
                outfile.write(outfile_line)

            outfile.close

        # return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list

    else:

        print 'Calculating LOF..'
        lof_model = LocalOutlierFactor(n_neighbors=n_neighbors)
        labels_ = lof_model.fit_predict(training_data)
        labels = labels_

        print 'Finished calculating LOF'
        ranking_list = lof_model.negative_outlier_factor_.tolist()

        # print 'Calculating node quality metric...'
        # quality_metric = metrics.silhouette_score(training_data, labels, metric='euclidean')
        # print quality_metric, ', n_neighbors: ', str( n_neighbors )

        return ranking_list, handle_list


def Clustering_K_Means_2 ( sparse_representation_fileName, summed_representation_fileName, param_value_list, num_clusters = 3, param_testing = False, binary=False ):

    X = []
    handle_list = []

    if isBagOfWords(sparse_representation_fileName):
        X = load_pickle(sparse_representation_fileName)

        #convert training data to floats
        if binary: # convert tweets to binary
            X = feature_values_floatFormat = Binary_Transform_sparse(X)

        with open(summed_representation_fileName) as file:

            line_count = 0
            for line in file:
                # print 'Processing line ', line_count
                line_count = line_count + 1

                line = line.strip('\n')
                line = line.strip('\r')
                line_elements = line.split(',')

                handle = line_elements[0]
                feature_values = line_elements[1:]

                handle_list.append(handle)
    else:
        print
        training_data = []
        handle_list = []

        line_count = 0
        # load in summed representation file
        print summed_representation_fileName
        with open(summed_representation_fileName) as file:


            for line in file:
                # print 'Processing line ', line_count
                line_count = line_count + 1

                line = line.strip('\n')
                line = line.strip('\r')
                line_elements = line.split(',')

                handle = line_elements[0]
                feature_values = line_elements[1:]

                handle_list.append(handle)

                #convert training data to floats
                feature_values_floatFormat = [ float(x) for x in feature_values]
                if ( not isBagOfWords(summed_representation_fileName) ) and binary: # convert tweets to binary
                    feature_values_floatFormat = Binary_Transform_Vector(feature_values_floatFormat)

                training_data.append( feature_values_floatFormat )

        X = np.asarray(training_data)

    if param_testing:
        qualityMetric_list = []
        handles_toLabels_dict_list = []

        sil_score_list = []
        percentageMaxPossilbeEntropy_list = []

        # X = preprocessing.normalize(X) # makes euclidean metric function as cosine similarity
        # X_norm = X
        # num_clusters_list = [2, 3, 4, 5, 6, 8, 10, 20, 50]
        # num_clusters_list = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        num_clusters_list = param_value_list

        with open('cluster_parameterAnalysis.csv', 'a') as outfile:  # change to 'a' to append

            for cluster_num in num_clusters_list:

                # print 'Calculating K_Means Clustering...'
                kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(X)
                labels = kmeans.labels_
                # print 'Finished calculating K_Means Clustering'

                # print 'Calculating node quality metric...'
                quality_metric = metrics.silhouette_score(X, labels, metric='euclidean')
                # print quality_metric, ', num clusters: ', str( cluster_num )

                qualityMetric_list.append(quality_metric)
                handles_toLabels_dict = {}
                for index in range(len(labels)):
                    handles_toLabels_dict[handle_list[index]]  = labels[index]
                handles_toLabels_dict_list.append(handles_toLabels_dict)

                largestCluster_size = 0
                for cluster_id in range(cluster_num):
                    cluster_size = len([k for k,v in sorted(handles_toLabels_dict.items(), key=operator.itemgetter(1), reverse = True ) if v == cluster_id])
                    if cluster_size > largestCluster_size:
                        largestCluster_size = cluster_size
                # print 'Size of largest cluster:', str(largestCluster_size)

                # BEGIN parameter results collection #
                    # for easy copy and pasting
                        # parameter_value_list = []
                        # sil_score_list = []
                        # percentageMaxPossilbeEntropy_list = []

                        # return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list

                        # param_value_list
                    # end for easy copy and pasting

                size_ofEachCluster_dict = {}
                for label in labels:
                    if label in size_ofEachCluster_dict:
                        size_ofEachCluster_dict[label] = size_ofEachCluster_dict[label] + 1
                    else:
                        size_ofEachCluster_dict[label] = 1

                size_ofEachCluster_list = size_ofEachCluster_dict.values()
                entropy = stats.entropy(size_ofEachCluster_list)
                size_ofEachCluster_EVEN_list = [1 for x in range(len(size_ofEachCluster_list))] # create assumption of completely eeven distribution of labels across clusters
                maxPossible_entropy = stats.entropy(size_ofEachCluster_EVEN_list)

                #   percentageMaxPossilbeEntropy = entropy of actual distribution / maximum possible entropy
                percentageMaxPossilbeEntropy = entropy / float(maxPossible_entropy)

                qualityMetric_list.append(quality_metric)

                # parameter_value_list.append(cluster_num)
                sil_score_list.append(quality_metric)
                percentageMaxPossilbeEntropy_list.append(percentageMaxPossilbeEntropy)
                # END parameter results collection #

                outfile_line = str(quality_metric) + ', K Means, ' + 'num_clusters=' + str(cluster_num) + 'binary=' + str(binary) + ', ' + 'largest_clusterSize=' + str(largestCluster_size) + ', ' + summed_representation_fileName + '\n'
                outfile.write(outfile_line)

        print 'Aquired labels and quality metrics to evaluate: qualityMetric_list, labels_list, num_clusters_list' + 'binary=' + str(binary)
        print 'handles_toLabels_dict_list'

        # return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list

    else:
        # X = preprocessing.normalize(X) # makes euclidean metric function as cosine similarity
        print 'Calculating K_Means Clustering...'
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
        labels = kmeans.labels_
        print 'Finished calculating K_Means Clustering'


        print 'Calculating node quality metric...'
        quality_metric = metrics.silhouette_score(X, labels, metric='euclidean')
        print quality_metric, ', num clusters: ', str( num_clusters )

        handle_toNodeLabel_dict = {}

        if len(handle_list) != len(labels):
            print 'ERROR: length of node labels for handles does not match number of handles \n'
            return

        # for x in range(len(labels)):
        #     handle = handle_list[x]
        #     label = labels[x]
        #     handle_toNodeLabel_dict[handle] = label

        # output_fileName = summed_representation_fileName[:-4] # remove '.txt'
        # output_fileName = output_fileName + '_Utility_Clustering_K_Means' + '_' + str( num_clusters ) + 'Clusters.txt'
        # outfile = open(output_fileName, 'w')

        # for handle, label in sorted(handle_toNodeLabel_dict.iteritems(), key=operator.itemgetter(1), reverse = True):
        #     output_line = handle + ', ' + str( label ) + '\n'
        #     outfile.write(output_line)

        # outfile.close()

        return labels, handle_list



def Clustering_K_Means ( summed_representation_fileName, param_value_list, num_clusters = 3, param_testing = False, binary=False, num_features_ToKeep = 1000 ):

    # output format: the members of the clusters (handles)
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    # need to experiement with different parameters, such as n_clusters?
    # lisa code for cluster quality?

    # list of lists, sparse representation of feature occurrences across all tweets for training model
    training_data = []
    handle_list = []

    line_count = 0
    # load in summed representation file
    print summed_representation_fileName
    with open(summed_representation_fileName) as file:

        for line in file:
            # print 'Processing line ', line_count
            line_count = line_count + 1

            line = line.strip('\n')
            line = line.strip('\r')
            line_elements = line.split(',')

            handle = line_elements[0]
            feature_values = line_elements[1:]

            handle_list.append(handle)

            #convert training data to floats
            feature_values_floatFormat = [ float(x) for x in feature_values]

            feature_values_floatFormat = feature_values_floatFormat[:num_features_ToKeep]

            if ( not isBagOfWords(summed_representation_fileName) ) and binary: # convert tweets to binary
                feature_values_floatFormat = Binary_Transform_Vector(feature_values_floatFormat)

            training_data.append( feature_values_floatFormat )

    print 'num_features kept: ', str(num_features_ToKeep)

    X = np.asarray(training_data)

    if isBagOfWords(summed_representation_fileName):

        uniqueWord_ToIndex_dict = {} # assign a 'feature number' to each word
        uniqueWord_ToFreq_dict = {} # keep track of count for each word

        unqiue_word_count = 0
        vectors_list = []
        words_filename = summed_representation_fileName[:-4]
        represenation_name = words_filename
        words_filename = words_filename + '_wordLists.txt'

        handle_list = []
        nonZero_Features_list = []

        with open(words_filename) as file_words:
            line_count = 0
            for words_line in file_words:
                words_line = words_line.strip('\n')
                words_line = words_line.strip('\r')
                words_list = words_line.split(',')

                handle = words_list[0]
                feature_words = words_list[1:]

                handle_list.append(handle)
                nonZero_Features_list.append(feature_words)

                # wordTo_freqCount_vectorDict = {}

                for word in feature_words:
                    if word not in uniqueWord_ToIndex_dict:
                        uniqueWord_ToIndex_dict[word] = unqiue_word_count
                        unqiue_word_count = unqiue_word_count + 1
                        # print word, ' ', unqiue_word_count

                    if word not in uniqueWord_ToFreq_dict:
                        uniqueWord_ToFreq_dict[word] = 1
                    else:
                        uniqueWord_ToFreq_dict[word] = uniqueWord_ToFreq_dict[word] + 1

                line_count = line_count + 1

        features_Meaningful =  [k for k,v in sorted(uniqueWord_ToFreq_dict.items(), key=operator.itemgetter(1), reverse = True ) if v >= 0]
        uniqueWord_ToIndex_dict = {}
        unqiue_word_count = 0
        for word in features_Meaningful:
                uniqueWord_ToIndex_dict[word] = unqiue_word_count
                unqiue_word_count = unqiue_word_count + 1

        # print 'Size compressed: ', len(features_Meaningful)
        # print 'Num features droped: ', ( len(uniqueWord_ToFreq_dict) - len(features_Meaningful) )

        print 'Constructing feature space for Clustering_K_Means', summed_representation_fileName, 'binary=', str(binary), ' ...'
        training_data_NEW = []
        count = 0
        for nonZero_Features, value_vector in izip (nonZero_Features_list, training_data):
            handle_vector = [0.0] * len(features_Meaningful)
            curr_index = 0
            for word, value in izip (nonZero_Features, value_vector):
                if word in uniqueWord_ToIndex_dict:
                    handle_vector[ uniqueWord_ToIndex_dict[word] ] = value #  fill the vector, with each value going to the featue index corresponding to the value's word
                    if binary:
                        handle_vector = Binary_Transform_Vector(handle_vector)

                    curr_index = curr_index + 1

            training_data_NEW.append(handle_vector)
            count = count + 1
        print 'Created vectors with full feature spaces, number: ', len(training_data)

        n_samples = len(training_data_NEW)
        training_data_NEW = np.asarray(training_data_NEW)
        training_data_NEW.reshape(n_samples, -1)
        X = training_data_NEW
        emptyList = []
        training_data_NEW = np.asarray(emptyList) # clear training data

    # parameters to change: num_clusters
    # append to consolidated parameters analysis file
    # write out for each permuation: silhouette_score, function, parameters involved, input_file

    # can qeury handle_list as well

    if param_testing:
        qualityMetric_list = []
        handles_toLabels_dict_list = []

        sil_score_list = []
        percentageMaxPossilbeEntropy_list = []

        parameter_value_list = []
        sil_score_list = []
        percentageMaxPossilbeEntropy_list = []


        # X = preprocessing.normalize(X) # makes euclidean metric function as cosine similarity
        # X_norm = X
        # num_clusters_list = [2, 3, 4, 5, 6, 8, 10, 20, 50]
        # num_clusters_list = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        num_clusters_list = param_value_list
        with open('cluster_parameterAnalysis.csv', 'a') as outfile:  # change to 'a' to append

            for cluster_num in num_clusters_list:

                print 'Calculating K_Means Clustering...'
                kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(X)
                labels = kmeans.labels_
                print 'Finished calculating K_Means Clustering'

                print 'Calculating node quality metric...'
                quality_metric = metrics.silhouette_score(X, labels, metric='euclidean')
                print quality_metric, ', num clusters: ', str( cluster_num )

                qualityMetric_list.append(quality_metric)
                handles_toLabels_dict = {}
                for index in range(len(labels)):
                    handles_toLabels_dict[handle_list[index]]  = labels[index]
                handles_toLabels_dict_list.append(handles_toLabels_dict)

                largestCluster_size = 0
                for cluster_id in range(cluster_num):
                    cluster_size = len([k for k,v in sorted(handles_toLabels_dict.items(), key=operator.itemgetter(1), reverse = True ) if v == cluster_id])
                    if cluster_size > largestCluster_size:
                        largestCluster_size = cluster_size
                # print 'Size of largest cluster:', str(largestCluster_size)


                # BEGIN parameter results collection #
                    # for easy copy and pasting
                        # parameter_value_list = []
                        # sil_score_list = []
                        # percentageMaxPossilbeEntropy_list = []

                        # return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list

                        # param_value_list
                    # end for easy copy and pasting

                size_ofEachCluster_dict = {}
                for label in labels:
                    if label in size_ofEachCluster_dict:
                        size_ofEachCluster_dict[label] = size_ofEachCluster_dict[label] + 1
                    else:
                        size_ofEachCluster_dict[label] = 1

                size_ofEachCluster_list = size_ofEachCluster_dict.values()
                entropy = stats.entropy(size_ofEachCluster_list)
                size_ofEachCluster_EVEN_list = [1 for x in range(len(size_ofEachCluster_list))] # create assumption of completely eeven distribution of labels across clusters
                maxPossible_entropy = stats.entropy(size_ofEachCluster_EVEN_list)

                #   percentageMaxPossilbeEntropy = entropy of actual distribution / maximum possible entropy
                percentageMaxPossilbeEntropy = entropy / float(maxPossible_entropy)

                qualityMetric_list.append(quality_metric)

                parameter_value_list.append(cluster_num)
                sil_score_list.append(quality_metric)
                percentageMaxPossilbeEntropy_list.append(percentageMaxPossilbeEntropy)
                # END parameter results collection #

                outfile_line = str(quality_metric) + ', K Means, ' + 'num_clusters=' + str(cluster_num) + 'binary=' + str(binary) + ', ' + 'largest_clusterSize=' + str(largestCluster_size) + ', ' + summed_representation_fileName + '\n'
                outfile.write(outfile_line)

        # print 'Aquired labels and quality metrics to evaluate: qualityMetric_list, labels_list, num_clusters_list' + 'binary=' + str(binary)
        # print 'handles_toLabels_dict_list'

        # return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list

        pdb.set_trace()

    else:
        # X = preprocessing.normalize(X) # makes euclidean metric function as cosine similarity
        print 'Calculating K_Means Clustering...'
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
        labels = kmeans.labels_
        print 'Finished calculating K_Means Clustering'


        print 'Calculating node quality metric...'
        quality_metric = metrics.silhouette_score(X, labels, metric='euclidean')
        print quality_metric, ', num clusters: ', str( num_clusters )

        handle_toNodeLabel_dict = {}

        if len(handle_list) != len(labels):
            print 'ERROR: length of node labels for handles does not match number of handles \n'
            return

        # for x in range(len(labels)):
        #     handle = handle_list[x]
        #     label = labels[x]
        #     handle_toNodeLabel_dict[handle] = label

        # output_fileName = summed_representation_fileName[:-4] # remove '.txt'
        # output_fileName = output_fileName + '_Utility_Clustering_K_Means' + '_' + str( num_clusters ) + 'Clusters.txt'
        # outfile = open(output_fileName, 'w')

        # for handle, label in sorted(handle_toNodeLabel_dict.iteritems(), key=operator.itemgetter(1), reverse = True):
        #     output_line = handle + ', ' + str( label ) + '\n'
        #     outfile.write(output_line)

        # outfile.close()

        return labels, handle_list


def Clustering_Hierarchical_2 (sparse_representation_fileName, summed_representation_fileName, parameter_value_list, num_clusters= 3, param_testing = False, binary=False ):

    X = []
    handle_list = []
    training_data = []
    num_features = 0

    if isBagOfWords(sparse_representation_fileName):
        X = load_pickle(sparse_representation_fileName)

                #convert training data to floats
        if binary: # convert tweets to binary
            X = feature_values_floatFormat = Binary_Transform_sparse(X)

        X = X.toarray() # sk learn hierarchical takes only dense numpy arrays

        with open(summed_representation_fileName) as file:

            line_count = 0
            for line in file:
                # print 'Processing line ', line_count
                line_count = line_count + 1

                line = line.strip('\n')
                line = line.strip('\r')
                line_elements = line.split(',')

                handle = line_elements[0]
                feature_values = line_elements[1:]

                handle_list.append(handle)
    else:

        represenation_name = ''

        # load in summed representation file
        print summed_representation_fileName
        with open(summed_representation_fileName) as file:

            line_count = 0
            for line in file:
                line_count = line_count + 1
                # print 'Processing line ', line_count
                line = line.strip('\n')
                line = line.strip('\r')
                line_elements = line.split(',')

                handle = line_elements[0]
                feature_values = line_elements[1:]

                handle_list.append(handle)

                feature_values_floatFormat = []
                for x in feature_values:
                    feature_values_floatFormat.append(x)
                    # print(x)

                feature_values_floatFormat = [ int(x) for x in feature_values]

                if ( not isBagOfWords(summed_representation_fileName) ) and binary: # convert tweets to binary
                    feature_values_floatFormat = Binary_Transform_Vector(feature_values_floatFormat)

                training_data.append( feature_values_floatFormat )

        # print 'Calculating Hierarchical Clustering...'
        # clusters = linkage(training_data, 'weighted', metric=mycosine)
        # print 'Finished calculating Hierarchical Clustering...'

        num_features = len(training_data[0])

        X = np.asarray(training_data)

    n_samples = len(training_data)
    # training_data = np.asarray(training_data)
    # training_data.reshape(n_samples, num_features)

    if param_testing:
        qualityMetric_list = []
        num_clusters_list = param_value_list
        handles_toLabels_dict_list = []

        sil_score_list = []
        percentageMaxPossilbeEntropy_list = []

        with open('cluster_parameterAnalysis.csv', 'a') as outfile:  # change to 'a' to append

            parameter_value_list = []
            sil_score_list = []
            percentageMaxPossilbeEntropy_list = []

            for cluster_num in num_clusters_list:

                # ward, euclidean good : )
                # try normalization

                print 'Calculating Hierarchical Clustering...'
                model = AgglomerativeClustering(n_clusters=cluster_num,linkage="ward", affinity='euclidean').fit(X)
                labels = model.labels_

                # Z = linkage(X)
                print 'Finished calculating Hierarchical Clustering...'

                pdb.set_trace()

                print 'Calculating node quality metric...'
                quality_metric = metrics.silhouette_score(X, labels, metric='euclidean')
                print quality_metric, ', num clusters: ', str( cluster_num )

                qualityMetric_list.append(quality_metric)

                handles_toLabels_dict = {}
                for index in range(len(labels)):
                    handles_toLabels_dict[handle_list[index]]  = labels[index]
                handles_toLabels_dict_list.append(handles_toLabels_dict)

                largestCluster_size = 0
                for cluster_id in range(cluster_num):
                    cluster_size = len([k for k,v in sorted(handles_toLabels_dict.items(), key=operator.itemgetter(1), reverse = True ) if v == cluster_id])
                    if cluster_size > largestCluster_size:
                        largestCluster_size = cluster_size
                print 'Size of largest cluster:', str(largestCluster_size)


                # BEGIN parameter results collection #
                    # for easy copy and pasting
                        # parameter_value_list = []
                        # sil_score_list = []
                        # percentageMaxPossilbeEntropy_list = []

                        # return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list

                        # param_value_list
                    # end for easy copy and pasting

                size_ofEachCluster_dict = {}
                for label in labels:
                    if label in size_ofEachCluster_dict:
                        size_ofEachCluster_dict[label] = size_ofEachCluster_dict[label] + 1
                    else:
                        size_ofEachCluster_dict[label] = 1

                size_ofEachCluster_list = size_ofEachCluster_dict.values()
                entropy = stats.entropy(size_ofEachCluster_list)
                size_ofEachCluster_EVEN_list = [1 for x in range(len(size_ofEachCluster_list))] # create assumption of completely eeven distribution of labels across clusters
                maxPossible_entropy = stats.entropy(size_ofEachCluster_EVEN_list)

                #   percentageMaxPossilbeEntropy = entropy of actual distribution / maximum possible entropy
                percentageMaxPossilbeEntropy = entropy / float(maxPossible_entropy)

                qualityMetric_list.append(quality_metric)

                parameter_value_list.append(cluster_num)
                sil_score_list.append(quality_metric)
                percentageMaxPossilbeEntropy_list.append(percentageMaxPossilbeEntropy)
                # END parameter results collection #

                outfile_line = str(quality_metric) + ', Hierarchical, ' + 'num_clusters=' + str(cluster_num) + 'binary=' + str(binary) + ', ' + 'largest_clusterSize=' + str(largestCluster_size) + ', '  + summed_representation_fileName + '\n'
                outfile.write(outfile_line)

        pdb.set_trace()


        print 'Aquired labels and quality metrics to evaluate: qualityMetric_list, labels_list, num_clusters_list' + 'binary=' + str(binary)
        print 'handles_toLabels_dict_list'

        return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list

    else:
        print 'Calculating Hierarchical Clustering...'
        # model = AgglomerativeClustering(n_clusters=num_clusters,linkage="average", affinity='cosine').fit(X)
        model = AgglomerativeClustering(n_clusters=num_clusters,linkage="ward", affinity='euclidean').fit(X)
        # model = AgglomerativeClustering(n_clusters=num_clusters,linkage='ward', affinity='euclidean').fit(X)

        labels = model.labels_
        print 'Finished calculating Hierarchical Clustering...'

        print 'Calculating node quality metric...'
        quality_metric = metrics.silhouette_score(X, labels, metric='euclidean')
        print quality_metric

        return labels, handle_list



def Clustering_Hierarchical (summed_representation_fileName, param_value_list, num_clusters= 3, param_testing = False, binary=False ):
    # output format: dendrogram
        # list of lists, sparse representation of feature occurrences across all tweets for training model
    training_data = []
    handle_list = []

    represenation_name = ''

    num_features = 0

    # load in summed representation file
    print summed_representation_fileName
    with open(summed_representation_fileName) as file:

        line_count = 0
        for line in file:
            line_count = line_count + 1
            # print 'Processing line ', line_count
            line = line.strip('\n')
            line = line.strip('\r')
            line_elements = line.split(',')

            handle = line_elements[0]
            feature_values = line_elements[1:]

            handle_list.append(handle)

            feature_values_floatFormat = []
            for x in feature_values:
                feature_values_floatFormat.append(x)
                # print(x)

            feature_values_floatFormat = [ int(x) for x in feature_values]

            if ( not isBagOfWords(summed_representation_fileName) ) and binary: # convert tweets to binary
                feature_values_floatFormat = Binary_Transform_Vector(feature_values_floatFormat)

            training_data.append( feature_values_floatFormat )

    # print 'Calculating Hierarchical Clustering...'
    # clusters = linkage(training_data, 'weighted', metric=mycosine)
    # print 'Finished calculating Hierarchical Clustering...'

    num_features = len(training_data[0])

    X = np.asarray(training_data)
    # X_Norm = preprocessing.normalize(X)

    if isBagOfWords(summed_representation_fileName):
        # then need to procure values from word file

        """ maybe use this:
                #Vectorizing
                X = CountVectorizer().fit_transform(docs)
                X = TfidfTransformer().fit_transform(X)
                #Clustering
                X = X.todense()
        """

        uniqueWord_ToIndex_dict = {} # assign a 'feature number' to each word
        uniqueWord_ToFreq_dict = {} # keep track of count for each word

        unqiue_word_count = 0
        vectors_list = []
        words_filename = summed_representation_fileName[:-4]
        represenation_name = words_filename
        words_filename = words_filename + '_wordLists.txt'

        handle_list = []
        nonZero_Features_list = []

        # improve by importing feature space directly from file?

        with open(words_filename) as file_words:
            line_count = 0
            for words_line in file_words:
                words_line = words_line.strip('\n')
                words_line = words_line.strip('\r')
                words_list = words_line.split(',')

                handle = words_list[0]
                feature_words = words_list[1:]

                handle_list.append(handle)
                nonZero_Features_list.append(feature_words)

                # wordTo_freqCount_vectorDict = {}

                for word in feature_words:
                    if word not in uniqueWord_ToIndex_dict:
                        uniqueWord_ToIndex_dict[word] = unqiue_word_count
                        unqiue_word_count = unqiue_word_count + 1
                        # print word, ' ', unqiue_word_count

                    if word not in uniqueWord_ToFreq_dict:
                        uniqueWord_ToFreq_dict[word] = 1
                    else:
                        uniqueWord_ToFreq_dict[word] = uniqueWord_ToFreq_dict[word] + 1


                line_count = line_count + 1


        # max_nonZero_featureCount = 0
        # for nonZero_Features in nonZero_Features_list:
        #     if len(nonZero_Features) > max_nonZero_featureCount:
        #         max_nonZero_featureCount = len(nonZero_Features)

        # m = min(d.values())
        # check if should use 1 or 1.0
            # or do thresh hold thing
        # what are implications for .... --> don't add word to feature space if less
        # sort unqiue_word_count list reverse, and then update the values of keys that have values above X... ?
        print 'Size uncompressed: ', len(uniqueWord_ToFreq_dict)
        features_Meaningful =  [k for k,v in sorted(uniqueWord_ToFreq_dict.items(), key=operator.itemgetter(1), reverse = True ) if v >= 0] # only gets me the values that are below
        # make another dict of word --> freq, then sort and create dictionary that is sorted
        # no, just aquire list of words, from sorted dictionary

        uniqueWord_ToIndex_dict = {}
        unqiue_word_count = 0
        for word in features_Meaningful:
                uniqueWord_ToIndex_dict[word] = unqiue_word_count
                unqiue_word_count = unqiue_word_count + 1

        print 'Size compressed: ', len(features_Meaningful)
        print 'Num features droped: ', ( len(uniqueWord_ToFreq_dict) - len(features_Meaningful) )

        print 'Constructing feature space for ', 'Hierarchical and', summed_representation_fileName, 'binary=', str(binary), ' ...'
        training_data_NEW = []
        count = 0
        for nonZero_Features, value_vector in izip (nonZero_Features_list, training_data):
            handle_vector = [0.0] * len(features_Meaningful)

            # for i in range(len(value_vector)):
            #     handle_vector[i][0] = value_vector[i]

            curr_index = 0
            for word, value in izip (nonZero_Features, value_vector):

                if word in uniqueWord_ToIndex_dict:
                    handle_vector[ uniqueWord_ToIndex_dict[word] ] = value #  fill the vector, with each value going to the featue index corresponding to the value's word
                    if binary:
                        handle_vector = Binary_Transform_Vector(handle_vector)
                    curr_index = curr_index + 1

            # word_compressed_feature_space_index = handle_vector[ unique_word_dict[word] ]
            # handle_vector[word_compressed_feature_space_index] = value

            training_data_NEW.append(handle_vector)

            count = count + 1
        print 'Created vectors with full feature spaces, number: ', len(training_data)

        # yeah, no this isnt going to work, because i'm using indices that words have for the entire feature space..
        # just fill up, in order, from the begining of this compressed space, the feature values, and then later aquire the words they correspond to
            # to get the corresponding words, read in from file? OR what if each element of the vector space is a (word, feature_value) tuple?

        # now that all vectors have same shape, won't get errors within packages
        # BUT still need to have both values, and corresponding words in distance calculation


        # for i in range(len(feature_words)):
        #     wordTo_freqCount_vectorDict[feature_words[i]] = training_data[line_count][i]

        # pdb.set_trace()
        # vectors_list.append(wordTo_freqCount_vectorDict)

        test_vector_1 = np.asarray(training_data_NEW[0])
        test_vector_2 = np.asarray(training_data_NEW[1])

        n_samples = len(training_data_NEW)
        training_data_NEW = np.asarray(training_data_NEW)
        training_data_NEW.reshape(n_samples, -1)

        # training_data_NEW = np.asarray(training_data_NEW)

        # pdb.set_trace()

        # test_vector_1 = test_vector_1.reshape(1, -1)
        # test_vector_2 = test_vector_2.reshape(1, -1)


        # if this method is quicker, then should incorporate
        # result_dot = safe_sparse_dot(X_normalized, X_normalized, dense_output=True)


        # result_custom = cosine_BagOfWords(training_data_NEW)
        # pdb.set_trace()

        # this one seems to be much faster --> why?
        # training_data_NEW = np.asarray(training_data_NEW)
        # result = cosine_similarity(training_data_NEW)

        # print result
        # pdb.set_trace()

        # and then make list of dictionaries: word --> freq_count}
        # and then make training data = to the above list of dictionaries
        # and then use custom affinity metric

        if param_testing:
            qualityMetric_list = []
            num_clusters_list = param_value_list
            handles_toLabels_dict_list = []

            parameter_value_list = []
            sil_score_list = []
            percentageMaxPossilbeEntropy_list = []

            with open('cluster_parameterAnalysis.csv', 'a') as outfile:  # change to 'a' to append

                for cluster_num in num_clusters_list:

                    print 'Calculating Hierarchical Clustering...'
                    model = AgglomerativeClustering(n_clusters=cluster_num,linkage="ward", affinity='euclidean').fit(training_data)
                    labels = model.labels_
                    print 'Finished calculating Hierarchical Clustering...'

                    print 'Calculating node quality metric...'
                    quality_metric = metrics.silhouette_score(training_data, labels, metric='euclidean')
                    print quality_metric, ', num clusters: ', str( cluster_num )

                    qualityMetric_list.append(quality_metric)

                    handles_toLabels_dict = {}
                    for index in range(len(labels)):
                        handles_toLabels_dict[handle_list[index]]  = labels[index]
                    handles_toLabels_dict_list.append(handles_toLabels_dict)

                    largestCluster_size = 0
                    for cluster_id in range(cluster_num):
                        cluster_size = len([k for k,v in sorted(handles_toLabels_dict.items(), key=operator.itemgetter(1), reverse = True ) if v == cluster_id])
                        if cluster_size > largestCluster_size:
                            largestCluster_size = cluster_size
                    print 'Size of largest cluster:', str(largestCluster_size)


                    # BEGIN parameter results collection #
                        # for easy copy and pasting
                            # parameter_value_list = []
                            # sil_score_list = []
                            # percentageMaxPossilbeEntropy_list = []

                            # return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list
                        # end for easy copy and pasting

                    size_ofEachCluster_dict = {}
                    for label in labels:
                        if label in size_ofEachCluster_dict:
                            size_ofEachCluster_dict[label] = size_ofEachCluster_dict[label] + 1
                        else:
                            size_ofEachCluster_dict[label] = 1

                    size_ofEachCluster_list = size_ofEachCluster_dict.values()
                    entropy = stats.entropy(size_ofEachCluster_list)
                    size_ofEachCluster_EVEN_list = [1 for x in range(len(size_ofEachCluster_list))] # create assumption of completely eeven distribution of labels across clusters
                    maxPossible_entropy = stats.entropy(size_ofEachCluster_EVEN_list)

                    #   percentageMaxPossilbeEntropy = entropy of actual distribution / maximum possible entropy
                    percentageMaxPossilbeEntropy = entropy / float(maxPossible_entropy)

                    qualityMetric_list.append(quality_metric)

                    parameter_value_list.append(cluster_num)
                    sil_score_list.append(quality_metric)
                    percentageMaxPossilbeEntropy_list.append(percentageMaxPossilbeEntropy)
                    # END parameter results collection #

                    outfile_line = str(quality_metric) + ', Hierarchical, ' + 'num_clusters=' + str(cluster_num) + 'binary=' + str(binary) + ', ' + 'largest_clusterSize=' + str(largestCluster_size) + ', '  + summed_representation_fileName + '\n'
                    outfile.write(outfile_line)

            print 'Aquired labels and quality metrics to evaluate: qualityMetric_list, labels_list, num_clusters_list' + 'binary=' + str(binary)
            print 'handles_toLabels_dict_list'

            return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list

        else:

            print 'Calculating Hierarchical Clustering...'
            # model = linkage(training_data_NEW, method='average', metric='cosine')

            # model = AgglomerativeClustering(n_clusters=num_clusters,linkage="complete", affinity='cosine').fit(training_data_NEW)
            model = AgglomerativeClustering(n_clusters=num_clusters,linkage='ward', affinity='euclidean').fit(training_data_NEW)

            labels = model.labels_
            print 'Finished calculating Hierarchical Clustering...'
            # with warnings.catch_warnings(): #ignore depreciation warnings caused from float indexing into np.array
                # warnings.simplefilter("ignore")

            # print 'Calculating node quality metric...'
            # pairwise_distances = cosine_similarity(training_data_NEW)
            # for handle_index in range(len(pairwise_distances)): # replace 'nan' with 0.0
            #     features = pairwise_distances[int(handle_index)]
            #     for feature_index in pairwise_distances[int(handle_index)]:
            #         if np.isnan( pairwise_distances[int(handle_index)][int(feature_index)] ):
            #             pairwise_distances[int(handle_index)][int(feature_index)] = 0.0

            print 'Calculating node quality metric...'
            quality_metric = metrics.silhouette_score(training_data, labels, metric='euclidean')
            print quality_metric

        # quality_metric = metrics.silhouette_score(pairwise_distances, labels, metric='precomputed')
        # quality_metric, coph_dists = cophenet(Z, pdist(training_data_NEW))

        # fig = ff.create_dendogram(X, orientation='left', labels=handles )
        # fig['layout'].update({'width':800, 'height':800})
        # py.iplot(fig, filename= 'Clustering_Hierarchical' +  'dendrogram_with_labels')

    else:
        n_samples = len(training_data)
        training_data = np.asarray(training_data)
        training_data.reshape(n_samples, num_features)

        if param_testing:
            qualityMetric_list = []
            num_clusters_list = param_value_list
            handles_toLabels_dict_list = []
            sil_score_list = []
            percentageMaxPossilbeEntropy_list = []

            with open('cluster_parameterAnalysis.csv', 'a') as outfile:  # change to 'a' to append

                for cluster_num in num_clusters_list:

                    # ward, euclidean good : )
                    # try normalization

                    print 'Calculating Hierarchical Clustering...'
                    model = AgglomerativeClustering(n_clusters=cluster_num,linkage="ward", affinity='euclidean').fit(training_data)
                    labels = model.labels_
                    print 'Finished calculating Hierarchical Clustering...'

                    print 'Calculating node quality metric...'
                    quality_metric = metrics.silhouette_score(training_data, labels, metric='euclidean')
                    print quality_metric, ', num clusters: ', str( cluster_num )

                    qualityMetric_list.append(quality_metric)

                    handles_toLabels_dict = {}
                    for index in range(len(labels)):
                        handles_toLabels_dict[handle_list[index]]  = labels[index]
                    handles_toLabels_dict_list.append(handles_toLabels_dict)

                    largestCluster_size = 0
                    for cluster_id in range(cluster_num):
                        cluster_size = len([k for k,v in sorted(handles_toLabels_dict.items(), key=operator.itemgetter(1), reverse = True ) if v == cluster_id])
                        if cluster_size > largestCluster_size:
                            largestCluster_size = cluster_size
                    print 'Size of largest cluster:', str(largestCluster_size)


                    # begin parameter results collection #
                        # for easy copy and pasting
                            # parameter_value_list = []
                            # sil_score_list = []
                            # percentageMaxPossilbeEntropy_list = []

                            # return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list
                        # end for easy copy and pasting

                    size_ofEachCluster_dict = {}
                    for label in labels:
                        if label in size_ofEachCluster_dict:
                            size_ofEachCluster_dict[label] = size_ofEachCluster_dict[label] + 1
                        else:
                            size_ofEachCluster_dict[label] = 1

                    size_ofEachCluster_list = size_ofEachCluster_dict.values()
                    entropy = stats.entropy(size_ofEachCluster_list)
                    size_ofEachCluster_EVEN_list = [1 for x in range(len(size_ofEachCluster_list))] # create assumption of completely eeven distribution of labels across clusters
                    maxPossible_entropy = stats.entropy(size_ofEachCluster_EVEN_list)

                    #   percentageMaxPossilbeEntropy = entropy of actual distribution / maximum possible entropy
                    percentageMaxPossilbeEntropy = entropy / float(maxPossible_entropy)

                    qualityMetric_list.append(quality_metric)

                    parameter_value_list.append(cluster_num)
                    sil_score_list.append(quality_metric)
                    percentageMaxPossilbeEntropy_list.append(percentageMaxPossilbeEntropy)
                    # end parameter results collection #

                    outfile_line = str(quality_metric) + ', Hierarchical, ' + 'num_clusters=' + str(cluster_num) + 'binary=' + str(binary) + ', ' + 'largest_clusterSize=' + str(largestCluster_size) + ', '  + summed_representation_fileName + '\n'
                    outfile.write(outfile_line)

            print 'Aquired labels and quality metrics to evaluate: qualityMetric_list, labels_list, num_clusters_list' + 'binary=' + str(binary)
            print 'handles_toLabels_dict_list'

            return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list

        else:
            print 'Calculating Hierarchical Clustering...'
            # model = AgglomerativeClustering(n_clusters=num_clusters,linkage="average", affinity='cosine').fit(X)
            model = AgglomerativeClustering(n_clusters=num_clusters,linkage="ward", affinity='euclidean').fit(training_data)
            # model = AgglomerativeClustering(n_clusters=num_clusters,linkage='ward', affinity='euclidean').fit(X)

            labels = model.labels_
            print 'Finished calculating Hierarchical Clustering...'

            print 'Calculating node quality metric...'
            quality_metric = metrics.silhouette_score(training_data, labels, metric='euclidean')
            print quality_metric

            # fig = ff.create_dendogram(X, orientation='left', labels=handles )
            # fig['layout'].update({'width':800, 'height':800})
            # represenation_name = represenation_name + '_dendrogram'
            # py.iplot(fig, filename=represenation_name)

            return labels, handle_list


    # c, coph_dists = cophenet(clusters, pdist(training_data))

    # print 'Closer to 1, the better the clustering preseves the original distances: ', str(c)
    # pdb.set_trace()

    # plt.figure(figsize = (10, 8))
    # plt.scatter(training_data[:,0], training_data[:,0])

def Clustering_DB_Scan_2 (sparse_representation_fileName, summed_representation_fileName, param_value_list, param_testing = False, binary=False):

    X = []
    handle_list = []
    training_data = []
    num_features = 0



    if isBagOfWords(sparse_representation_fileName):
        X = load_pickle(sparse_representation_fileName)

        with open(summed_representation_fileName) as file:

            line_count = 0
            for line in file:
                # print 'Processing line ', line_count
                line_count = line_count + 1

                line = line.strip('\n')
                line = line.strip('\r')
                line_elements = line.split(',')

                handle = line_elements[0]
                feature_values = line_elements[1:]

                handle_list.append(handle)
    else:

        represenation_name = ''

        # load in summed representation file
        print summed_representation_fileName
        with open(summed_representation_fileName) as file:

            line_count = 0
            for line in file:
                line_count = line_count + 1
                # print 'Processing line ', line_count
                line = line.strip('\n')
                line = line.strip('\r')
                line_elements = line.split(',')

                handle = line_elements[0]
                feature_values = line_elements[1:]

                handle_list.append(handle)

                feature_values_floatFormat = []
                for x in feature_values:
                    feature_values_floatFormat.append(x)
                    # print(x)

                feature_values_floatFormat = [ float(x) for x in feature_values]

                if ( not isBagOfWords(summed_representation_fileName) ) and binary: # convert tweets to binary
                    feature_values_floatFormat = Binary_Transform_Vector(feature_values_floatFormat)

                training_data.append( feature_values_floatFormat )

        # print 'Calculating Hierarchical Clustering...'
        # clusters = linkage(training_data, 'weighted', metric=mycosine)
        # print 'Finished calculating Hierarchical Clustering...'

        num_features = len(training_data[0])

        X = np.asarray(training_data)

    if param_testing:
        min_samples_list = param_value_list
        qualityMetric_list = []
        # min_samples_list = [3, 4, 5, 6, 8]
        handles_toLabels_dict_list = []

        eps_list = [0.6, 0.7, 0.8] # The maximum distance between two samples for them to be considered as in the same neighborhood.
        best_quality_metric_score = -1
        best_min_sample_num = 2
        best_eps = 0
        best_min_sample_num = 0

        parameter_value_list = []
        sil_score_list = []
        percentageMaxPossilbeEntropy_list = []

        with open('cluster_parameterAnalysis.csv', 'a') as outfile:  # change to 'a' to append

            for min_sample_num in min_samples_list:

                print 'Calculating DB Scan Clustering...'
                db = DBSCAN(eps=0.7, min_samples=min_sample_num, metric='euclidean').fit(X)
                labels = db.labels_
                print 'Finished calculating DB Scan Clustering...'

                print 'Calculating node quality metric...'
                quality_metric = metrics.silhouette_score(X, labels, metric='euclidean')
                print quality_metric, ', min_sample_num: ', str( min_sample_num ), ', eps:', str(0.7)
                if quality_metric > best_quality_metric_score:
                    best_quality_metric_score = quality_metric
                    best_min_sample_num = min_sample_num
                    best_eps = 0.7


                # begin parameter results collection #
                    # for easy copy and pasting
                        # parameter_value_list = []
                        # sil_score_list = []
                        # percentageMaxPossilbeEntropy_list = []

                        # return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list
                    # end for easy copy and pasting

                size_ofEachCluster_dict = {}
                for label in labels:
                    if label in size_ofEachCluster_dict:
                        size_ofEachCluster_dict[label] = size_ofEachCluster_dict[label] + 1
                    else:
                        size_ofEachCluster_dict[label] = 1

                size_ofEachCluster_list = size_ofEachCluster_dict.values()
                entropy = stats.entropy(size_ofEachCluster_list)
                size_ofEachCluster_EVEN_list = [1 for x in range(len(size_ofEachCluster_list))] # create assumption of completely eeven distribution of labels across clusters
                maxPossible_entropy = stats.entropy(size_ofEachCluster_EVEN_list)

                #   percentageMaxPossilbeEntropy = entropy of actual distribution / maximum possible entropy
                percentageMaxPossilbeEntropy = entropy / float(maxPossible_entropy)

                qualityMetric_list.append(quality_metric)

                parameter_value_list.append(min_sample_num)
                sil_score_list.append(quality_metric)
                percentageMaxPossilbeEntropy_list.append(percentageMaxPossilbeEntropy)
                # end parameter results collection #

                handles_toLabels_dict = {}
                for index in range(len(labels)):
                    handles_toLabels_dict[handle_list[index]]  = labels[index]
                handles_toLabels_dict_list.append(handles_toLabels_dict)

                largestCluster_size = 0
                unique_labels = []
                for label in labels:
                    if label not in unique_labels:
                        unique_labels.append(label)

                for label_num in unique_labels:
                    cluster_size = len([k for k,v in sorted(handles_toLabels_dict.items(), key=operator.itemgetter(1), reverse = True ) if v == label_num])
                    if cluster_size > largestCluster_size:
                        largestCluster_size = cluster_size
                print 'Size of largest cluster:', str(largestCluster_size)

                outfile_line = str(quality_metric) + ', DB Scan, ' + 'min_sample_num=' + str(min_sample_num) + ', ' + 'eps=' + str(0.7) + 'binary=' + str(binary) + ', ' + 'largest_clusterSize=' + str(largestCluster_size) + ', ' + summed_representation_fileName + '\n'
                outfile.write(outfile_line)




                # for eps_num in eps_list:
                #     print 'Calculating DB Scan Clustering...'
                #     db = DBSCAN(eps=eps_num, min_samples=min_sample_num, metric='euclidean').fit(training_data)
                #     labels = db.labels_
                #     print 'Finished calculating DB Scan Clustering...'

                #     print 'Calculating node quality metric...'
                #     quality_metric = metrics.silhouette_score(training_data, labels, metric='euclidean')
                #     print quality_metric, ', min_sample_num: ', str( min_sample_num ), ', eps:', str(eps_num)

                #     if quality_metric > best_quality_metric_score:
                #         best_quality_metric_score = quality_metric
                #         best_min_sample_num = min_sample_num
                #         best_eps = eps_num

                #     outfile_line = str(quality_metric) + ', DB Scan, ' + 'min_sample_num=' + str(min_sample_num) + ', ' + 'eps=' + str(eps_num) + ', ' + summed_representation_fileName + '\n'
                #     outfile.write(outfile_line)

                #     handles_toLabels_dict = {}
                #     for index in range(len(labels)):
                #         handles_toLabels_dict[handle_list[index]]  = labels[index]
                #     handles_toLabels_dict_list.append(handles_toLabels_dict)

            # print 'BEST:', best_quality_metric_score, ', min_sample_num: ', str( best_min_sample_num ), ', eps:', str(best_eps)
            outfile_line = 'BEST:'+ str(best_quality_metric_score) + ', DB Scan, ' + 'min_sample_num=' + str(best_min_sample_num) + ', ' + 'eps=' + str(best_eps) + ', ' + summed_representation_fileName + '\n'
            outfile.write(outfile_line)

        print 'Aquired labels and quality metrics to evaluate: qualityMetric_list, labels_list, num_clusters_list'
        print 'handles_toLabels_dict_list'

        pdb.set_trace()

        return parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list

    else:
        print 'Calculating DB Scan Clustering...'
        db = DBSCAN(eps=0.3, min_samples=5, metric='euclidean').fit(X)

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        print 'Finished calculating DB Scan Clustering...'

        print 'Calculating node quality metric...'
        quality_metric = metrics.silhouette_score(X, labels, metric='euclidean')
        print quality_metric

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print 'Number of clusters: ', n_clusters_

        pdb.set_trace()
        # heat map
        pairwise_distances = cosine_similarity(X)
        # a = np.random.random((16, 16))
        plt.imshow(pairwise_distances, cmap='hot', interpolation='nearest')
        plt.show()
        # end heat map

def Clustering_DB_Scan (summed_representation_fileName, param_testing = False, binary=False):
    # output format: heat plot
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN

    # list of lists, sparse representation of feature occurrences across all tweets for training model
    training_data = []
    handle_list = []
    featureSpace_size = 0

    # load in summed representation file
    print summed_representation_fileName
    with open(summed_representation_fileName) as file:

        line_count = 0
        for line in file:
            line_count = line_count + 1
            # print 'Processing line ', line_count
            line = line.strip('\n')
            line = line.strip('\r')
            line_elements = line.split(',')

            handle = line_elements[0]
            feature_values = line_elements[1:]

            handle_list.append(handle)

            feature_values_floatFormat = [ float(x) for x in feature_values]

            if ( not isBagOfWords(summed_representation_fileName) ) and binary: # convert tweets to binary
                feature_values_floatFormat = Binary_Transform_Vector(feature_values_floatFormat)

            training_data.append( feature_values_floatFormat )

    for column in training_data[0]:
        featureSpace_size = featureSpace_size + 1

    n_samples = len(training_data)

    if isBagOfWords(summed_representation_fileName):
        uniqueWord_ToIndex_dict = {} # assign a 'feature number' to each word
        uniqueWord_ToFreq_dict = {} # keep track of count for each word

        unqiue_word_count = 0
        vectors_list = []
        words_filename = summed_representation_fileName[:-4]
        represenation_name = words_filename
        words_filename = words_filename + '_wordLists.txt'

        nonZero_Features_list = []
        handle_list = []

        with open(words_filename) as file_words:
            line_count = 0
            for words_line in file_words:
                words_line = words_line.strip('\n')
                words_line = words_line.strip('\r')
                words_list = words_line.split(',')

                handle = words_list[0]
                feature_words = words_list[1:]

                handle_list.append(handle)
                nonZero_Features_list.append(feature_words)

                # wordTo_freqCount_vectorDict = {}

                for word in feature_words:
                    if word not in uniqueWord_ToIndex_dict:
                        uniqueWord_ToIndex_dict[word] = unqiue_word_count
                        unqiue_word_count = unqiue_word_count + 1
                        # print word, ' ', unqiue_word_count

                    if word not in uniqueWord_ToFreq_dict:
                        uniqueWord_ToFreq_dict[word] = 1
                    else:
                        uniqueWord_ToFreq_dict[word] = uniqueWord_ToFreq_dict[word] + 1


                line_count = line_count + 1

        uniqueWord_ToIndex_dict = {}
        unqiue_word_count = 0
        features_Meaningful =  [k for k,v in sorted(uniqueWord_ToFreq_dict.items(), key=operator.itemgetter(1), reverse = True ) if v >= 0]
        for word in features_Meaningful:
                uniqueWord_ToIndex_dict[word] = unqiue_word_count
                unqiue_word_count = unqiue_word_count + 1

        print 'Size compressed: ', len(features_Meaningful)
        print 'Num features droped: ', ( len(uniqueWord_ToFreq_dict) - len(features_Meaningful) )

        print 'Constructing feature space for ', 'Clustering_DB_Scan', ' and ', summed_representation_fileName, 'binary=', str(binary), ' ...'
        training_data_NEW = []
        count = 0
        for nonZero_Features, value_vector in izip (nonZero_Features_list, training_data):
            handle_vector = [0.0] * len(features_Meaningful)
            curr_index = 0
            for word, value in izip (nonZero_Features, value_vector):
                if word in uniqueWord_ToIndex_dict:
                    handle_vector[ uniqueWord_ToIndex_dict[word] ] = value #  fill the vector, with each value going to the featue index corresponding to the value's word
                    if binary:
                        handle_vector = Binary_Transform_Vector(handle_vector)

                    curr_index = curr_index + 1

            training_data_NEW.append(handle_vector)
            count = count + 1
        print 'Created vectors with full feature spaces, number: ', len(training_data)

        n_samples = len(training_data_NEW)
        # training_data_NEW = np.asarray(training_data_NEW)
        # training_data_NEW.reshape(n_samples, -1)
        training_data = training_data_NEW
        training_data_NEW = []


    # regardless of whether BoW, convert to np array and reshape appriopriately
    training_data = np.array(training_data)
    training_data.reshape(n_samples, featureSpace_size)

    # still try normalized data into euclidean = cos similarity
    if param_testing:
        min_samples_list = [2, 3, 4, 5, 6, 8, 10, 50, 100, 120, 140, 160, 180, 200, 300]
        qualityMetric_list = []
        # min_samples_list = [3, 4, 5, 6, 8]
        handles_toLabels_dict_list = []

        eps_list = [0.6, 0.7, 0.8] # The maximum distance between two samples for them to be considered as in the same neighborhood.
        best_quality_metric_score = -1
        best_min_sample_num = 2
        best_eps = 0
        best_min_sample_num = 0

        with open('cluster_parameterAnalysis.csv', 'a') as outfile:  # change to 'a' to append

            for min_sample_num in min_samples_list:

                print 'Calculating DB Scan Clustering...'
                db = DBSCAN(eps=0.7, min_samples=min_sample_num, metric='euclidean').fit(training_data)
                labels = db.labels_
                print 'Finished calculating DB Scan Clustering...'

                print 'Calculating node quality metric...'
                quality_metric = metrics.silhouette_score(training_data, labels, metric='euclidean')
                print quality_metric, ', min_sample_num: ', str( min_sample_num ), ', eps:', str(0.7)
                if quality_metric > best_quality_metric_score:
                    best_quality_metric_score = quality_metric
                    best_min_sample_num = min_sample_num
                    best_eps = 0.7

                qualityMetric_list.append(quality_metric)

                handles_toLabels_dict = {}
                for index in range(len(labels)):
                    handles_toLabels_dict[handle_list[index]]  = labels[index]
                handles_toLabels_dict_list.append(handles_toLabels_dict)

                largestCluster_size = 0
                unique_labels = []
                for label in labels:
                    if label not in unique_labels:
                        unique_labels.append(label)

                for label_num in unique_labels:
                    cluster_size = len([k for k,v in sorted(handles_toLabels_dict.items(), key=operator.itemgetter(1), reverse = True ) if v == label_num])
                    if cluster_size > largestCluster_size:
                        largestCluster_size = cluster_size
                print 'Size of largest cluster:', str(largestCluster_size)

                outfile_line = str(quality_metric) + ', DB Scan, ' + 'min_sample_num=' + str(min_sample_num) + ', ' + 'eps=' + str(0.7) + 'binary=' + str(binary) + ', ' + 'largest_clusterSize=' + str(largestCluster_size) + ', ' + summed_representation_fileName + '\n'
                outfile.write(outfile_line)

                # for eps_num in eps_list:
                #     print 'Calculating DB Scan Clustering...'
                #     db = DBSCAN(eps=eps_num, min_samples=min_sample_num, metric='euclidean').fit(training_data)
                #     labels = db.labels_
                #     print 'Finished calculating DB Scan Clustering...'

                #     print 'Calculating node quality metric...'
                #     quality_metric = metrics.silhouette_score(training_data, labels, metric='euclidean')
                #     print quality_metric, ', min_sample_num: ', str( min_sample_num ), ', eps:', str(eps_num)

                #     if quality_metric > best_quality_metric_score:
                #         best_quality_metric_score = quality_metric
                #         best_min_sample_num = min_sample_num
                #         best_eps = eps_num

                #     outfile_line = str(quality_metric) + ', DB Scan, ' + 'min_sample_num=' + str(min_sample_num) + ', ' + 'eps=' + str(eps_num) + ', ' + summed_representation_fileName + '\n'
                #     outfile.write(outfile_line)

                #     handles_toLabels_dict = {}
                #     for index in range(len(labels)):
                #         handles_toLabels_dict[handle_list[index]]  = labels[index]
                #     handles_toLabels_dict_list.append(handles_toLabels_dict)

            # print 'BEST:', best_quality_metric_score, ', min_sample_num: ', str( best_min_sample_num ), ', eps:', str(best_eps)
            outfile_line = 'BEST:'+ str(best_quality_metric_score) + ', DB Scan, ' + 'min_sample_num=' + str(best_min_sample_num) + ', ' + 'eps=' + str(best_eps) + ', ' + summed_representation_fileName + '\n'
            outfile.write(outfile_line)

        print 'Aquired labels and quality metrics to evaluate: qualityMetric_list, labels_list, num_clusters_list'
        print 'handles_toLabels_dict_list'

    else:
        print 'Calculating DB Scan Clustering...'
        db = DBSCAN(eps=0.3, min_samples=5, metric='euclidean').fit(training_data)

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        print 'Finished calculating DB Scan Clustering...'

        print 'Calculating node quality metric...'
        quality_metric = metrics.silhouette_score(training_data, labels, metric='euclidean')
        print quality_metric

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print 'Number of clusters: ', n_clusters_

        pdb.set_trace()
        # heat map
        pairwise_distances = cosine_similarity(training_data)
        # a = np.random.random((16, 16))
        plt.imshow(pairwise_distances, cmap='hot', interpolation='nearest')
        plt.show()
        # end heat map


# DO something like this to obtain optimal parameters for clustering methods
def Clustering_Parameter_Evaluation ( X ):
    figures = []
    num_features = X.shape[1]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig = tools.make_subplots(rows=1, cols=num_features,
                                  print_grid=False,
                                  subplot_titles=('The silhouette plot for the various clusters.',
                                                  'The visualization of the clustered data.'))

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        fig['layout']['xaxis1'].update(title='The silhouette coefficient values',
                                       range=[-0.1, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        fig['layout']['yaxis1'].update(title='Cluster label',
                                       showticklabels=False,
                                       range=[0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.

        clusterer = AgglomerativeClustering(n_clusters=num_clusters,linkage="average", affinity='cosine').fit(X)
        # clusterer = KMeans(n_clusters=n_clusters, random_state=10)

        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower = 10

        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            colors = cm.spectral(cluster_labels.astype(float) / n_clusters)

            filled_area = go.Scatter(y=np.arange(y_lower, y_upper),
                                     x=ith_cluster_silhouette_values,
                                     mode='lines',
                                     showlegend=False,
                                     line=dict(width=0.5,
                                              color=colors),
                                     fill='tozerox')
            fig.append_trace(filled_area, 1, 1)

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples


        # The vertical line for average silhouette score of all the values
        axis_line = go.Scatter(x=[silhouette_avg],
                               y=[0, len(X) + (n_clusters + 1) * 10],
                               showlegend=False,
                               mode='lines',
                               line=dict(color="red", dash='dash',
                                         width =1) )

        fig.append_trace(axis_line, 1, 1)

        # 2nd Plot showing the actual clusters formed
        colors = matplotlib.colors.colorConverter.to_rgb(cm.spectral(float(i) / n_clusters))
        colors = 'rgb'+str(colors)
        clusters = go.Scatter(x=X[:, 0],
                              y=X[:, 1],
                              showlegend=False,
                              mode='markers',
                              marker=dict(color=colors,
                                         size=4)
                             )
        fig.append_trace(clusters, 1, 2)

        # Labeling the clusters
        centers_ = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        centers = go.Scatter(x=centers_[:, 0],
                             y=centers_[:, 1],
                             showlegend=False,
                             mode='markers',
                             marker=dict(color='green', size=10,
                                         line=dict(color='black',
                                                                 width=1))
                            )

        fig.append_trace(centers, 1, 2)

        fig['layout']['xaxis2'].update(title='Feature space for the 1st feature',
                                       zeroline=False)
        fig['layout']['yaxis2'].update(title='Feature space for the 2nd feature',
                                      zeroline=False)


        fig['layout'].update(title="Silhouette analysis for KMeans clustering on sample data "
                             "with n_clusters = %d" % n_clusters)

        figures.append(fig)



# compute proportions of zeros across values in feature space for summed handle document
# aquire size of feature space
# for each handle, compute number 0 values for feature space, aquire incidence by dividing this number by size of feature space
# - also get number of tweet for each handle, such that can compare against the incidence of non-zero value
def Zero_Incidence_Metrics_Graph ( tweet_representation_fileName, filename_handles, summed_representation_fileName, graph_name=''):

    # do for each projection
        # maybe: have median and mean lines on graph with number below and

    # do check - make sure is summed file, not normal

    # handle --> [count for each of the feature in the feature space ]

    handle_toTweetCountDict = {}

    print 'Opening ', tweet_representation_fileName, '...'
    print 'Opening ', filename_handles, '...'
    print 'Opening ', summed_representation_fileName, '...'

    handles_inFreqBucket_list_low = []
    handles_inFreqBucket_list_med = []
    handles_inFreqBucket_list_high = []

    zeroValueProportion_list_full = []
    zeroValueProportion_list_high = []
    zeroValueProportion_list_med = []
    zeroValueProportion_list_low = []

    size_featureSpace = 0

    infile_handles_inFreqBucket_low = open('low_handles_4000000.txt', 'r')
    for line in infile_handles_inFreqBucket_low:
        handle_list = line.split(',')
        for handle in handle_list:
            handles_inFreqBucket_list_low.append(handle)

    infile_handles_inFreqBucket_med = open('med_handles_4000000.txt', 'r')
    for line in infile_handles_inFreqBucket_med:
        handle_list = line.split(',')
        for handle in handle_list:
            handles_inFreqBucket_list_med.append(handle)

    infile_handles_inFreqBucket_high = open('high_handles_4000000.txt', 'r')
    for line in infile_handles_inFreqBucket_high:
        handle_list = line.split(',')
        for handle in handle_list:
            handles_inFreqBucket_list_high.append(handle)

    # load handles and tweets in
    with open(tweet_representation_fileName) as file_tweets, open(filename_handles) as file_handles:
        line_count = 0

        # issue: first line in tweet file is header, but first line in handle file corresponds to second line in tweet file
        header_line = file_tweets.readline()
        header_line = header_line.strip('\n')
        header_line = header_line.strip('\r')
        header_elements = header_line.split(',')

        size_featureSpace = len(header_elements)

        for tweet, handle in izip(file_tweets, file_handles):

            print 'processing tweet ', line_count
            tweet = tweet.strip('\n')
            tweet = tweet.strip('\r')

            handle = handle.strip('\n')
            handle = handle.strip('\r')

            # populates handle to tweet space count dict
            if handle in handle_toTweetCountDict:
                handle_toTweetCountDict[handle] = handle_toTweetCountDict[handle] + 1
            else:
                handle_toTweetCountDict[handle] = 1

            line_count = line_count + 1
    print 'Aquisition of tweet counts per handle acquired'

    # if bag of words .... merely subtract the size of handle_vector from size of feature space to get # of zeros

    handle_toNullFeatureCountDict = {}
    if isBagOfWords(summed_representation_fileName):
        featureSpace_fileName = summed_representation_fileName[:-10]
        featureSpace_fileName = featureSpace_fileName + 'featureSpace.txt'
        featureSpace_file = open(featureSpace_fileName, 'r')

        featureSpace_list = []
        for line in featureSpace_file:
            line = line.strip('\n')
            line_elements = line.split(',')
            for feature in line_elements:
                featureSpace_list.append(feature)

        featureSpace_file.close()
        size_featureSpace = len(featureSpace_list)

        print 'size_featureSpace:', size_featureSpace

        with open (summed_representation_fileName) as file_summed_handle_vectors:
            for line in file_summed_handle_vectors:
                line = line.strip('\n')
                line = line.strip('\r')
                line_elements = line.split(',')
                num_zeroes = 0
                handle = line_elements[0]

                NonZero_features = line_elements[1:]
                num_NonZero_features = len( NonZero_features )
                num_zeroes = size_featureSpace - num_NonZero_features
                handle_toNullFeatureCountDict[handle] = num_zeroes

                zeroValueProportion_list_full.append(num_zeroes / float(size_featureSpace) )

                if handle in handles_inFreqBucket_list_high:
                    zeroValueProportion_list_high.append(num_zeroes / float(size_featureSpace) )
                elif handle in handles_inFreqBucket_list_med:
                    zeroValueProportion_list_med.append(num_zeroes / float(size_featureSpace) )
                elif handle in handles_inFreqBucket_list_low:
                    zeroValueProportion_list_low.append(num_zeroes / float(size_featureSpace) )
    else:
        print 'size_featureSpace:', size_featureSpace
        with open (summed_representation_fileName) as file_summed_handle_vectors:
            for line in file_summed_handle_vectors:
                line = line.strip('\n')
                line = line.strip('\r')
                line_elements = line.split(',')
                num_zeroes = 0
                handle = line_elements[0]

                feature_values = line_elements[1:]
                # size_featureSpace = len( feature_values )
                for value in feature_values:
                    if 0 == int(value):
                        num_zeroes = num_zeroes + 1
                handle_toNullFeatureCountDict[handle] = num_zeroes

                zeroValueProportion_list_full.append(num_zeroes / float(size_featureSpace) )

                if handle in handles_inFreqBucket_list_high:
                    zeroValueProportion_list_high.append(num_zeroes / float(size_featureSpace) )
                elif handle in handles_inFreqBucket_list_med:
                    zeroValueProportion_list_med.append(num_zeroes / float(size_featureSpace) )
                elif handle in handles_inFreqBucket_list_low:
                    zeroValueProportion_list_low.append(num_zeroes / float(size_featureSpace) )

    # calculate mean, median

    sorted_handle_NullCounts = sorted(handle_toNullFeatureCountDict.iteritems(), key=operator.itemgetter(1), reverse = True)

    # pdb.set_trace()
    num_handles = len ( sorted_handle_NullCounts )
    middle_handle_index = num_handles / 2
    median_nullCount = sorted_handle_NullCounts[middle_handle_index][1]

    nullCount_sum = 0
    for handle, count in sorted_handle_NullCounts:
        nullCount_sum = nullCount_sum + count

    mean_nullCount = nullCount_sum / float( num_handles )

    # ALSO CALCULUATE: proportion of handles below mean and median
    count_handlesBelowMedian = 0
    count_handlesBelowMean = 0

    output_fileName = tweet_representation_fileName[:-4] # remove '.txt'
    output_fileName = output_fileName + '_Zero_Incidence_Metrics.txt'
    outfile = open(output_fileName, 'w')

    output_line = ','.join(header_elements)
    output_line = output_line + '\n'
    outfile.write(output_line)

    for handle, nullFeatureCount in sorted(handle_toNullFeatureCountDict.iteritems(), key=operator.itemgetter(1)):
        output_line = handle + ', ' + 'zero_valued_feature_count: ' + str( nullFeatureCount ) + ', ' + ' tweet_count: ' + str( handle_toTweetCountDict[handle] ) + '\n'
        outfile.write(output_line)

        if nullFeatureCount < median_nullCount:
            count_handlesBelowMedian = count_handlesBelowMedian + 1
        if nullFeatureCount < mean_nullCount:
            count_handlesBelowMean = count_handlesBelowMean + 1

    output_line = 'mean zero_value_feature_count: ' + str ( mean_nullCount ) + ', median zero_value_feature_count: ' + str ( median_nullCount ) + '\n'
    outfile.write(output_line)
    output_line = 'proprtion of handles below mean: ' + str ( count_handlesBelowMean / float ( num_handles ) ) + ', proprtion of handles below median: ' + str ( count_handlesBelowMedian / float ( num_handles ) )
    outfile.write(output_line)
    outfile.close()

    x_data = ['Full Handle Set', 'High Tweet Frequency Handles',
          'Med Tweet Frequency Handles', 'Low Tweet Frequency Handles',]

    y_data = [zeroValueProportion_list_full, zeroValueProportion_list_high, zeroValueProportion_list_med, zeroValueProportion_list_low,]

    colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)']

    traces = []

    for xd, yd, cls in zip(x_data, y_data, colors):
        traces.append(go.Box(
            y=yd,
            name=xd,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=cls,
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
            boxmean='sd'
        ))

    layout = go.Layout(
        title='Zero-Valued Incidence Across Handle Vectors, ' + graph_name,
        yaxis=dict(
           zeroline=False
        ),
        # # margin=dict(
        # #     l=40,
        # #     r=30,
        # #     b=80,
        # #     t=100,
        # # ),
        # paper_bgcolor='rgb(243, 243, 243)',
        # plot_bgcolor='rgb(243, 243, 243)',
        showlegend=False
    )

    fig = go.Figure(data=traces, layout=layout)
    py.iplot(fig, filename=graph_name)

    # BE SURE TO GRAPH fration of 0s for vector handle

# compute proportions of zeros across values in feature space for summed handle document
# aquire size of feature space
# for each handle, compute number 0 values for feature space, aquire incidence by dividing this number by size of feature space
# - also get number of tweet for each handle, such that can compare against the incidence of non-zero value
def Zero_Incidence_Metrics ( tweet_representation_fileName, filename_handles, summed_representation_fileName, bag_of_words=False, tweet_freq_type='' ):

    # do for each projection
        # maybe: have median and mean lines on graph with number below and

    # do check - make sure is summed file, not normal

    # handle --> [count for each of the feature in the feature space ]
    handle_toTweetCountDict = {}

    print 'Opening ', tweet_representation_fileName, '...'
    print 'Opening ', filename_handles, '...'
    print 'Opening ', summed_representation_fileName, '...'




    all_handles = True # boolen for whether all handles will be evaluated
    handles_inFreqBucket_list = []

    if tweet_freq_type == 'low':
        all_handles = False
    elif tweet_freq_type == 'mid':
        all_handles = False
    elif tweet_freq_type == 'high':
        all_handles = False


    print 'all_handles:', str(all_handles)
    if not all_handles:
        infile_handles_inFreqBucket = open(tweet_freq_type + '_handles_' + '4000000.txt', 'r')
        for line in infile_handles_inFreqBucket:
            handles_inFreqBucket_list = line.split(',')

    # load handles and tweets in
    with open(tweet_representation_fileName) as file_tweets, open(filename_handles) as file_handles:
        line_count = 0

        # issue: first line in tweet file is header, but first line in handle file corresponds to second line in tweet file
        header_line = file_tweets.readline()
        header_line = header_line.strip('\n')
        header_line = header_line.strip('\r')
        header_elements = header_line.split(',')

        for tweet, handle in izip(file_tweets, file_handles):

            print 'processing tweet ', line_count
            tweet = tweet.strip('\n')
            tweet = tweet.strip('\r')
            tweet_elements = tweet.split(',')

            handle = handle.strip('\n')
            handle = handle.strip('\r')

            if not all_handles:
                if handle not in handles_inFreqBucket_list:
                    continue # if current handle is not in specified freqBucket, ignore

            # populates handle to feature space count dict
            if handle in handle_toTweetCountDict:
                handle_toTweetCountDict[handle] = handle_toTweetCountDict[handle] + 1
            else:
                handle_toTweetCountDict[handle] = 1

            line_count = line_count + 1
    print 'Aquisition of tweet counts per handle acquired'

    # if bag of words .... merely subtract the size of handle_vector from size of feature space to get # of zeros

    size_featureSpace = 0
    handle_toNullFeatureCountDict = {}
    if bag_of_words:
        featureSpace_fileName = summed_representation_fileName[:-10]
        featureSpace_fileName = featureSpace_fileName + 'featureSpace.txt'
        featureSpace_file = open(featureSpace_fileName, 'r')

        featureSpace_list = []
        for line in featureSpace_file:
            line = line.strip('\n')
            line_elements = line.split(',')
            for feature in line_elements:
                featureSpace_list.append(feature)

        featureSpace_file.close()
        size_featureSpace = len(featureSpace_list)

        with open (summed_representation_fileName) as file_summed_handle_vectors:
            for line in file_summed_handle_vectors:
                line = line.strip('\n')
                line = line.strip('\r')
                line_elements = line.split(',')
                num_zeroes = 0
                handle = line_elements[0]

                if not all_handles:
                    if handle not in handles_inFreqBucket_list:
                        continue # if current handle is not in specified freqBucket, ignore

                NonZero_features = line_elements[1:]
                num_NonZero_features = len( NonZero_features )
                num_zeroes = size_featureSpace - num_NonZero_features
                handle_toNullFeatureCountDict[handle] = num_zeroes
    else:
        with open (summed_representation_fileName) as file_summed_handle_vectors:
            for line in file_summed_handle_vectors:
                line = line.strip('\n')
                line = line.strip('\r')
                line_elements = line.split(',')
                num_zeroes = 0
                handle = line_elements[0]

                if not all_handles:
                    if handle not in handles_inFreqBucket_list:
                        continue # if current handle is not in specified freqBucket, ignore

                feature_values = line_elements[1:]
                size_featureSpace = len( feature_values )
                for value in feature_values:
                    if 0 == int(value):
                        num_zeroes = num_zeroes + 1
                handle_toNullFeatureCountDict[handle] = num_zeroes

    # calculate mean, median

    sorted_handle_NullCounts = sorted(handle_toNullFeatureCountDict.iteritems(), key=operator.itemgetter(1), reverse = True)

    # pdb.set_trace()
    num_handles = len ( sorted_handle_NullCounts )
    middle_handle_index = num_handles / 2
    median_nullCount = sorted_handle_NullCounts[middle_handle_index][1]

    nullCount_sum = 0
    for handle, count in sorted_handle_NullCounts:
        nullCount_sum = nullCount_sum + count

    mean_nullCount = nullCount_sum / float( num_handles )

    # ALSO CALCULUATE: proportion of handles below mean and median
    count_handlesBelowMedian = 0
    count_handlesBelowMean = 0

    output_fileName = tweet_representation_fileName[:-4] # remove '.txt'
    output_fileName = output_fileName + '_Zero_Incidence_Metrics.txt'
    outfile = open(output_fileName, 'w')

    output_line = ','.join(header_elements)
    output_line = output_line + '\n'
    outfile.write(output_line)

    for handle, nullFeatureCount in sorted(handle_toNullFeatureCountDict.iteritems(), key=operator.itemgetter(1)):
        output_line = handle + ', ' + 'zero_valued_feature_count: ' + str( nullFeatureCount ) + ', ' + ' tweet_count: ' + str( handle_toTweetCountDict[handle] ) + '\n'
        outfile.write(output_line)

        if nullFeatureCount < median_nullCount:
            count_handlesBelowMedian = count_handlesBelowMedian + 1
        if nullFeatureCount < mean_nullCount:
            count_handlesBelowMean = count_handlesBelowMean + 1

    output_line = 'mean zero_value_feature_count: ' + str ( mean_nullCount ) + ', median zero_value_feature_count: ' + str ( median_nullCount ) + '\n'
    outfile.write(output_line)
    output_line = 'proprtion of handles below mean: ' + str ( count_handlesBelowMean / float ( num_handles ) ) + ', proprtion of handles below median: ' + str ( count_handlesBelowMedian / float ( num_handles ) )
    outfile.write(output_line)
    outfile.close()


# calculates # non-zeros, zeros hits between two summed handle documents
def Binary_Similarity ( handleVector_1, handleVector_2 ):

    count_nonZero_hits = 0
    count_zero_hits = 0
    numFeatures = len(handleVector_1)

    if len(handleVector_1) != len(handleVector_2):
        print 'ERROR: difference in feature space size for two handle vectors passed into Binary_Similarity'

    for x in range ( len( handleVector_1 ) ):
        if float( handleVector_1[x] ) == 0:
            if float( handleVector_2[x] ) == 0:
                count_zero_hits = count_zero_hits + 1
        else:
            if float ( handleVector_2[x] ) != 0: # two non-zero values for same feature
                count_nonZero_hits = count_nonZero_hits + 1

    incidence_nonZero_hits = count_nonZero_hits / float( numFeatures )
    incidence_zero_hits = count_zero_hits / float( numFeatures )

    return incidence_nonZero_hits, incidence_zero_hits


def Exact_Similarity ( handleVector_1, handleVector_2 ):

    count_nonZero_hits = 0
    count_zero_hits = 0
    numFeatures = len(handleVector_1)

    if len(handleVector_1) != len(handleVector_2):
        print 'ERROR: difference in feature space size for two handle vectors passed into Exact_Similarity'

    for x in range ( len( handleVector_1 ) ):
        if float( handleVector_1[x] ) == 0:
            if float(  handleVector_2[x] ) == 0:
                count_zero_hits = count_zero_hits + 1
        else:
            if ( float( handleVector_2[x] ) == float( handleVector_1[x] ) ): # exact match
                count_nonZero_hits = count_nonZero_hits + 1

    incidence_nonZero_hits = count_nonZero_hits / float( numFeatures )
    incidence_zero_hits = count_zero_hits / float( numFeatures )

    return incidence_nonZero_hits, incidence_zero_hits




# calculates difference between two handle vectors
# some sort of calculation that will save time? on lisa paper
def Distortion_Metric (summed_wordlist_fileName_BASELINE, summed_representation_fileName_projection, tweets_projection, emot=False, num_features_ToKeep = 1000):

# the fraction of 1s i've lost in the projection from the baseline for each handle vector

    # inpute bag of words, and one other.

    # convert projections to binary.

    # for each handle:
        # for each feature:
            # x = count number of features non-zero in baseline projection
            # y = count number of features non-zero in non-baseline projection
    # num_nonZeroFeature_lost_handle_i = x - y

    proprtion_nonZero_features_dropped_perHandle_list = []

    feature_space = []
    # get feature space from tweets_projection

    if not emot:
        with open(tweets_projection) as file_tweets:

            # issue: first line in tweet file is header, but first line in handle file corresponds to second line in tweet file
            header_line = file_tweets.readline()
            header_line = header_line.strip('\n')
            header_line = header_line.strip('\r')
            header_elements = header_line.split(',')

            feature_space = header_elements

            size_featureSpace = len(feature_space)

            feature_space = feature_space[:num_features_ToKeep]

            print 'size_featureSpace: ', str(feature_space)
    else:
        with open('emotion_words.txt') as file_tweets:

            # issue: first line in tweet file is header, but first line in handle file corresponds to second line in tweet file
            header_line = file_tweets.readline()
            header_line = header_line.strip('\n')
            header_line = header_line.strip('\r')
            header_elements = header_line.split(',')

            feature_space = header_elements

            size_featureSpace = len(feature_space)

            print 'size_featureSpace: ', str(size_featureSpace)


    handle_toListOfNonZeroFeatures = {}
    line_count = 0
    # get nonZero features for each handle for this projection
    handle_toListOfNonZeroFeatures = {}
    with open(summed_representation_fileName_projection) as file_handleVectors:
        for line in file_handleVectors:
            line = line.strip('\n')
            line = line.strip('\r')
            line_elements = line.split(',')
            num_zeroes = 0
            handle = line_elements[0]
            feature_values = line_elements[1:]
            feature_values = [int(x) for x in feature_values]

            nonZero_features = []
            for feature_value, feature in izip(feature_values, feature_space):
                if feature_value != 0:
                    nonZero_features.append(feature)

            handle_toListOfNonZeroFeatures[handle] = nonZero_features

            line_count = line_count + 1

    # get nonZero features for each handle for BASELINE projection
    with open(summed_wordlist_fileName_BASELINE) as file_words:
        line_count = 0
        for words_line in file_words:
            words_line = words_line.strip('\n')
            words_line = words_line.strip('\r')
            words_list = words_line.split(',')

            handle = words_list[0]
            feature_words = words_list[1:]

            count_Dropped = 0
            count_nonZeroFeaturesBaseline = len(feature_words)

            # print 'count_nonZeroFeaturesBaseline: ', str(count_nonZeroFeaturesBaseline)
            nonZero_features_projection = handle_toListOfNonZeroFeatures[handle]

            for nonZero_feature_baseline in feature_words:

                if nonZero_feature_baseline not in nonZero_features_projection:
                    count_Dropped = count_Dropped + 1

            if count_nonZeroFeaturesBaseline == 0:
                fraction_nonZero_features_lost = count_Dropped / float(1)
                proprtion_nonZero_features_dropped_perHandle_list.append(fraction_nonZero_features_lost)
                # print str(fraction_nonZero_features_lost)
            else:
                fraction_nonZero_features_lost = count_Dropped / float(count_nonZeroFeaturesBaseline)
                proprtion_nonZero_features_dropped_perHandle_list.append(fraction_nonZero_features_lost)
                # print str(fraction_nonZero_features_lost)

    distortion = sum(proprtion_nonZero_features_dropped_perHandle_list) / len(proprtion_nonZero_features_dropped_perHandle_list)

    print 'distortion: ', str(distortion)
    pdb.set_trace()

    # get total number of non-zero features for each handle for a projection

    # get total number of non-zero features

    # PUT IN TABLE


    # may want to do distortion of result also

    # had a certain fraction of ones

    # count # of flipped bits in terms of non-zero valued,
    # count # of flipped bits in terms of previously zero valued

    # do exact

    count_nonZero_hits = 0
    count_zero_hits = 0
    numFeatures = len(handleVector_1)

    if len(handleVector_1) != len(handleVector_2):
        print 'ERROR: difference in feature space size for two handle vectors passed into Distortion'

    for x in range ( len( handleVector_1 ) ):
        if handleVector_1[x] == 0:
            print

        else:
            if handleVector_2[x] != 0: # two non-zero values for same feature
                print

    incidence_nonZero_hits = count_nonZero_hits / float( numFeatures )
    incidence_zero_hits = count_zero_hits / float( numFeatures )

    return incidence_nonZero_hits, incidence_zero_hits

    # need to ascertain number of features

    # need to account for header line

    # would need to compress




def median(l):
    srt = sorted(l)
    mid = len(l)//2
    if len(l) % 2: # f list length mod 2 has a remainder the list is an odd lenght
            return srt[mid]
    else:
        med = (srt[mid] + srt[mid-1]) / 2  # in a list [1,2,3,4] srt[mid]-> 2, srt[mid-1] -> 3
        return med

# do --> implement wrapper function for similiarity metrics
    # calculate mean, median, max, min, similarity for both exact and binary for each handle (involves computing pair-wise similarity for each handle against all other handles)
    # then run over night

# can easily expand for distortion metric capability
def Aggregate_Similarity( summed_representation_fileName):

    # dict { dict [mean, median, max, min, ] --> [float, float, float, float] } --> dict handle
    handle_toAggregateSimilarityMetrics = {}
    handle_toAggregateSimilarityMetrics['nonZero_incidence_BINARY_list'] = {}
    handle_toAggregateSimilarityMetrics['zero_incidence_BINARY_list'] = {}
    handle_toAggregateSimilarityMetrics['nonZero_incidence_EXACT_list'] = {}
    handle_toAggregateSimilarityMetrics['zero_incidence_EXACT_list'] = {}

    handle_toVectorRepresentation = {}

    size_featureSpace = 0

    line_count = 0
    # load in representation file, store each handle representation in list, aquired by comma delimmitaiton.
    with open (summed_representation_fileName) as file_summed_handle_vectors:
        for line in file_summed_handle_vectors:

            line = line.strip('\n')
            line = line.strip('\r')
            line_elements = line.split(',')
            num_zeroes = 0
            handle = line_elements[0]
            feature_values = line_elements[1:]
            size_featureSpace = len( feature_values )

            handle_toVectorRepresentation[handle] = feature_values
            # if line_count > 300:
                # break
            line_count = line_count + 1

    incidence_nonZero_hits = 0
    incidence_zero_hits = 0

    # for each handle x
        # for each handle y
            # compare x with y, using specified metric

    output_fileName = summed_representation_fileName[:-4] # remove '.txt'
    output_fileName = output_fileName + '_Aggregate_Similarity_Metrics.txt'
    outfile = open(output_fileName, 'w')

    value_mean_aggregate_list_nonZero_incidence_BINARY = []
    value_min_aggregate_lis_nonZero_incidence_BINARY = []
    value_max_aggregate_list_nonZero_incidence_BINARY  = []
    value_median_aggregate_list_nonZero_incidence_BINARY  = []

    value_mean_aggregate_list_Zero_incidence_BINARY = []
    value_min_aggregate_lis_Zero_incidence_BINARY  = []
    value_max_aggregate_list_Zero_incidence_BINARY  = []
    value_median_aggregate_list_Zero_incidence_BINARY  = []

    value_mean_aggregate_list_nonZero_incidence_EXACT = []
    value_min_aggregate_lis_nonZero_incidence_EXACT = []
    value_max_aggregate_list_nonZero_incidence_EXACT  = []
    value_median_aggregate_list_nonZero_incidence_EXACT  = []

    value_mean_aggregate_list_Zero_incidence_EXACT = []
    value_min_aggregate_lis_Zero_incidence_EXACT = []
    value_max_aggregate_list_Zero_incidence_EXACT  = []
    value_median_aggregate_list_Zero_incidence_EXACT  = []

    for handle_1, vectorSpace_1 in handle_toVectorRepresentation.iteritems():
        print 'Calculating metrics for ', handle_1
        nonZero_incidence_BINARY_list = []
        zero_incidence_BINARY_list = []
        nonZero_incidence_EXACT_list = []
        zero_incidence_EXACT_list = []
        for handle_2, vectorSpace_2 in handle_toVectorRepresentation.iteritems():

            incidence_nonZero_hits_BINARY, incidence_zero_hits_BINARY = Binary_Similarity(vectorSpace_1, vectorSpace_2)
            incidence_nonZero_hits_EXACT, incidence_zero_hits_EXACT = Exact_Similarity(vectorSpace_1, vectorSpace_2)

            if not (handle_1 == handle_2):
                nonZero_incidence_BINARY_list.append(incidence_nonZero_hits_BINARY)
                zero_incidence_BINARY_list.append(incidence_zero_hits_BINARY)
                nonZero_incidence_EXACT_list.append(incidence_nonZero_hits_EXACT)
                zero_incidence_EXACT_list.append(incidence_zero_hits_EXACT)

        outfile_line = handle_1 + 'feature space size: ' + str(size_featureSpace) + '\n'

        value_sum = sum ( nonZero_incidence_BINARY_list )
        value_min = max ( nonZero_incidence_BINARY_list )
        value_max = min ( nonZero_incidence_BINARY_list )
        value_median = median ( nonZero_incidence_BINARY_list )

        value_mean_aggregate_list_nonZero_incidence_BINARY.append(value_sum / float(len(handle_toVectorRepresentation)))
        value_min_aggregate_lis_nonZero_incidence_BINARY.append(value_min)
        value_max_aggregate_list_nonZero_incidence_BINARY.append(value_max)
        value_median_aggregate_list_nonZero_incidence_BINARY.append(value_median)

        outfile_line = outfile_line +  ', average nonZero_incidence_BINARY: ' + str( value_sum / float(len(handle_toVectorRepresentation)) ) +  ', min nonZero_incidence_BINARY: ' + str( value_min ) +  ', max nonZero_incidence_BINARY: ' + str( value_max ) +  ', median nonZero_incidence_BINARY: ' + str( value_median ) + '\n'


        value_sum = sum ( zero_incidence_BINARY_list )
        value_min = max ( zero_incidence_BINARY_list )
        value_max = min ( zero_incidence_BINARY_list )
        value_median = median ( zero_incidence_BINARY_list )

        value_mean_aggregate_list_Zero_incidence_BINARY.append(value_sum / float(len(handle_toVectorRepresentation)))
        value_min_aggregate_lis_Zero_incidence_BINARY.append(value_min)
        value_max_aggregate_list_Zero_incidence_BINARY.append(value_max)
        value_median_aggregate_list_Zero_incidence_BINARY.append(value_median)

        outfile_line = outfile_line +  ', average Zero_incidence_BINARY: ' + str( value_sum / float(len(handle_toVectorRepresentation)) ) +  ', min Zero_incidence_BINARY: ' + str( value_min ) +  ', max Zero_incidence_BINARY: ' + str( value_max ) +  ', median Zero_incidence_BINARY: ' + str( value_median ) + '\n'


        value_sum = sum ( nonZero_incidence_EXACT_list )
        value_min = max ( nonZero_incidence_EXACT_list )
        value_max = min ( nonZero_incidence_EXACT_list )
        value_median = median ( nonZero_incidence_EXACT_list )

        value_mean_aggregate_list_nonZero_incidence_EXACT.append(value_sum / float(len(handle_toVectorRepresentation)))
        value_min_aggregate_lis_nonZero_incidence_EXACT.append(value_min)
        value_max_aggregate_list_nonZero_incidence_EXACT.append(value_max)
        value_median_aggregate_list_nonZero_incidence_EXACT.append(value_median)

        outfile_line = outfile_line +  ', average nonZero_incidence_EXACT: ' + str( value_sum / float(len(handle_toVectorRepresentation)) ) +  ', min nonZero_incidence_EXACT: ' + str( value_min ) +  ', max nonZero_incidence_EXACT: ' + str( value_max ) +  ', median nonZero_incidence_EXACT: ' + str( value_median ) + '\n'

        value_sum = sum ( zero_incidence_EXACT_list )
        value_min = max ( zero_incidence_EXACT_list )
        value_max = min ( zero_incidence_EXACT_list )
        value_median = median ( zero_incidence_EXACT_list )

        value_mean_aggregate_list_Zero_incidence_EXACT.append(value_sum / float(len(handle_toVectorRepresentation)))
        value_min_aggregate_lis_Zero_incidence_EXACT.append(value_min)
        value_max_aggregate_list_Zero_incidence_EXACT.append(value_max)
        value_median_aggregate_list_Zero_incidence_EXACT.append(value_median)

        outfile_line = outfile_line +  ', average Zero_incidence_EXACT: ' + str( value_sum / float(len(handle_toVectorRepresentation)) ) +  ', min Zero_incidence_EXACT: ' + str( value_min ) +  ', max Zero_incidence_EXACT: ' + str( value_max ) +  ', median Zero_incidence_EXACT: ' + str( value_median ) + '\n' + '\n'

        outfile.write(outfile_line)

    outfile_line = 'Aggregate Statistics: ' + '\n'
    outfile_line = outfile_line + 'aggregate average nonZero_incidence_BINARY: ' + str( sum(value_mean_aggregate_list_nonZero_incidence_BINARY) / float(len(handle_toVectorRepresentation)) ) + ', aggregate min nonZero_incidence_BINARY: ' + str( sum(value_min_aggregate_lis_nonZero_incidence_BINARY) / float(len(handle_toVectorRepresentation)) ) + ', aggregate max nonZero_incidence_BINARY: ' + str( sum(value_max_aggregate_list_nonZero_incidence_BINARY) / float(len(handle_toVectorRepresentation)) ) + ', aggregate median nonZero_incidence_BINARY: ' + str( sum(value_median_aggregate_list_nonZero_incidence_BINARY) / float(len(handle_toVectorRepresentation)) ) + '\n'
    outfile_line = outfile_line + 'aggregate average Zero_incidence_BINARY: ' + str( sum(value_mean_aggregate_list_Zero_incidence_BINARY) / float(len(handle_toVectorRepresentation)) ) + ', aggregate min Zero_incidence_BINARY: ' + str( sum(value_min_aggregate_lis_Zero_incidence_BINARY) / float(len(handle_toVectorRepresentation)) ) + ', aggregate max Zero_incidence_BINARY: ' + str( sum(value_max_aggregate_list_Zero_incidence_BINARY) / float(len(handle_toVectorRepresentation)) ) + ', aggregate median Zero_incidence_BINARY: ' + str( sum(value_median_aggregate_list_Zero_incidence_BINARY) / float(len(handle_toVectorRepresentation)) ) + '\n'
    outfile_line = outfile_line + 'aggregate average nonZero_incidence_EXACT: ' + str( sum(value_mean_aggregate_list_nonZero_incidence_EXACT) / float(len(handle_toVectorRepresentation)) ) + ', aggregate min nonZero_incidence_EXACT: ' + str( sum(value_min_aggregate_lis_nonZero_incidence_EXACT) / float(len(handle_toVectorRepresentation)) ) + ', aggregate max nonZero_incidence_EXACT: ' + str( sum(value_max_aggregate_list_nonZero_incidence_EXACT) / float(len(handle_toVectorRepresentation)) ) + ', aggregate median nonZero_incidence_EXACT: ' + str( sum(value_median_aggregate_list_nonZero_incidence_EXACT) / float(len(handle_toVectorRepresentation)) ) + '\n'
    outfile_line = outfile_line + 'aggregate average Zero_incidence_EXACT: ' + str( sum(value_mean_aggregate_list_Zero_incidence_EXACT) / float(len(handle_toVectorRepresentation)) ) + ', aggregate min Zero_incidence_EXACT: ' + str( sum(value_min_aggregate_lis_Zero_incidence_EXACT) / float(len(handle_toVectorRepresentation)) ) + ', aggregate max Zero_incidence_EXACT: ' + str( sum(value_max_aggregate_list_Zero_incidence_EXACT) / float(len(handle_toVectorRepresentation)) ) + ', aggregate median Zero_incidence_EXACT: ' + str( sum(value_median_aggregate_list_Zero_incidence_EXACT) / float(len(handle_toVectorRepresentation)) ) + '\n'

    outfile.write(outfile_line)
    outfile.close()





# do --> implement wrapper function for distortion metrics

def isHashtag (uncleaned_tweets_filename, tf_idf_wordList_filename, num_tweets):
    tf_idf_wordList = []
    is_hashtag_wordList = []
    all_hashtags_wordList = []

    with open(tf_idf_wordList_filename) as input_file:
        for line in input_file:
            tf_idf_wordList = line.split(',')


    output_fileName = 'uncleaned_tweets'
    output_fileName = output_fileName + str ( num_tweets ) + '.txt'
    outfile = open(output_fileName, 'w')


    line_count = 0
    with open(uncleaned_tweets_filename) as input_file:
        for line in input_file:
            print 'processing line ', line_count, ' of ', num_tweets

            line = unicode(line,'utf-8')
            line = line.encode('unicode-escape')

            # remove unicode encodings of ascii characters
            line = line.decode('unicode_escape').encode('ascii','ignore')

            line = line.lower()

            outfile.write(line)

            line = line.strip('\n')
            line = line.strip('\r')

            # do encoding stuff, make lower case
            line_elements = line.split(' ')
            identified_hashtag_words = []

            # regex to identifiy hashtags, and then capture the word; build a list of such hashtags for a given tweet; then compare each element ...
            for element in line_elements:
                if bool (  re.search( r'\#\w+', element) ):
                    hashtag_word = element[1:] # remove the pound sign
                    print 'original: ', element
                    p = re.compile(r'(\w+)') #remove non letter characters
                    hashtag_word = p.search(hashtag_word).group(1)
                    print 'extracted word: ', hashtag_word
                    identified_hashtag_words.append(hashtag_word)
                    all_hashtags_wordList.append(hashtag_word)

            for element in identified_hashtag_words:
                if element in tf_idf_wordList and element not in is_hashtag_wordList: # is a hashtag, and has not been added to list already
                    is_hashtag_wordList.append(element)

            line_count = line_count + 1
            if line_count > num_tweets:
                break

    # aquire list of words in tf_idf_wordList but not in is_hashtag_wordList --> these are non-hashtag words
    not_hahstag_wordList = []
    for word in tf_idf_wordList:
        if word not in is_hashtag_wordList:
            not_hahstag_wordList.append(word)

    # now have two lists:
        # not_hahstag_wordList
        # is_hashtag_wordList

    # calculate proportion of tf-idf words that are hashtags
    num_hashtags_areTFIDF = len(is_hashtag_wordList)
    proportion_hashtag = num_hashtags_areTFIDF / float ( len(tf_idf_wordList) )
    print 'Proportion of ', str( len(tf_idf_wordList) ), ' tf-idf feature space words that are hashtags is: ', str( proportion_hashtag )
    pdb.set_trace()


def Graph_CollisionsOnCollisionVectors_Distributions_SingleProjection( projection_similarity_file_name, min_k=1, projection_name=''):

    vector_string_toK_dict = {}
    vector_string_toHandleList_dict = {}

    handles_inFreqBucket_list_low = []
    handles_inFreqBucket_list_med = []
    handles_inFreqBucket_list_high = []

    # vector_string_toK_dict_EmotionDict[vector_string] = k
    vector_string_toK_dict_full = {}
    vector_string_toK_dict_high = {}
    vector_string_toK_dict_med = {}
    vector_string_toK_dict_low = {}

    # vector_string_toHandleList_dict_EmotionDict[vector_string] = handle_list
    vector_string_toHandleList_dict_full = {}
    vector_string_toHandleList_dict_high = {}
    vector_string_toHandleList_dict_med = {}
    vector_string_toHandleList_dict_low = {}

    size_featureSpace = 0

    infile_handles_inFreqBucket_low = open('low_handles_4000000.txt', 'r')
    for line in infile_handles_inFreqBucket_low:
        handle_list = line.split(',')
        for handle in handle_list:
            handles_inFreqBucket_list_low.append(handle)

    infile_handles_inFreqBucket_med = open('med_handles_4000000.txt', 'r')
    for line in infile_handles_inFreqBucket_med:
        handle_list = line.split(',')
        for handle in handle_list:
            handles_inFreqBucket_list_med.append(handle)

    infile_handles_inFreqBucket_high = open('high_handles_4000000.txt', 'r')
    for line in infile_handles_inFreqBucket_high:
        handle_list = line.split(',')
        for handle in handle_list:
            handles_inFreqBucket_list_high.append(handle)

    handles_with_nonUniqueVectors_num_full = 0
    handles_with_nonUniqueVectors_num_high = 0
    handles_with_nonUniqueVectors_num_med = 0
    handles_with_nonUniqueVectors_num_low = 0

    handles_with_allZeroVector_count_full = 0
    handles_with_allZeroVector_count_high = 0
    handles_with_allZeroVector_count_med = 0
    handles_with_allZeroVector_count_low = 0


    uniqueVector_toCorrespondingHandleCount_dict_full = {}
    uniqueVector_toCorrespondingHandleCount_dict_high = {}
    uniqueVector_toCorrespondingHandleCount_dict_med = {}
    uniqueVector_toCorrespondingHandleCount_dict_low = {}

    with open(projection_similarity_file_name) as infile:
        line_count = 0

        is_all_zero_vector = True # keep track that singular all zero vector for each file is not included in graph
        all_zero_vector_occured = False

        for line in infile:
            line_count = line_count + 1
            print 'Projection Similarity file ', projection_name, ': processing line ', line_count

            line = unicode(line,'utf-8')
            line = line.encode('unicode-escape')

            # line = line.lower()

            line = line.strip('\n')
            line = line.strip('\\n')
            line = line.strip('\r')

            # line_elements = line.split(' ')

            # use regex to extract components
            vector_string = re.findall('\[.+\]', line)[0]
            vector = ast.literal_eval(vector_string)

            if all_zero_vector_occured: # only have to check if vector is not all zero, if all zero vector has not occured yet
                is_all_zero_vector = False
            else:
                is_all_zero_vector = True # keep track that singular all zero vector for each file is not included in graph

                for value in vector:
                    if value != 0:
                        is_all_zero_vector = False

            # capture vector, capture non vector?

            k = int(re.findall('k=(\d+)', line)[0])

            capture_list = re.findall(', (\w{3,})', line) # list of handles corresponding to this vector

            handle_list = [x for x in capture_list if not x.isdigit()] # accounts for above capture picking up large numbers

            # DO THIS FOR OTHER GRAPH TOOOOOOO

            # what is being dropped off of handle list? collision number

            if not is_all_zero_vector:
                handles_with_nonUniqueVectors_num_full = handles_with_nonUniqueVectors_num_full  + k - 1 # subtract one to account for a vector being unique for every handle

                added_to_high = False
                added_to_med = False
                added_to_low = False


                for handle in handle_list:
                    if handle in handles_inFreqBucket_list_high:
                        handles_with_nonUniqueVectors_num_high = handles_with_nonUniqueVectors_num_high  + 1
                        added_to_high = True
                    elif handle in handles_inFreqBucket_list_med:
                        handles_with_nonUniqueVectors_num_med = handles_with_nonUniqueVectors_num_med  + 1
                        added_to_med = True
                    elif handle in handles_inFreqBucket_list_low:
                        handles_with_nonUniqueVectors_num_low = handles_with_nonUniqueVectors_num_low  + 1
                        added_to_low = True

                # if added_to_high:
                #     handles_with_nonUniqueVectors_num_high = handles_with_nonUniqueVectors_num_high - 1 # subtract one to account for a vector being unique for every handle
                # elif added_to_med:
                #     handles_with_nonUniqueVectors_num_med = handles_with_nonUniqueVectors_num_med - 1 # subtract one to account for a vector being unique for every handle
                # elif added_to_low:
                #     handles_with_nonUniqueVectors_num_low = handles_with_nonUniqueVectors_num_low - 1  # subtract one to account for a vector being unique for every handle

            else:
                handles_with_allZeroVector_count_full = k

                for handle in handle_list:
                    if handle in handles_inFreqBucket_list_high:
                        handles_with_allZeroVector_count_high = handles_with_allZeroVector_count_high  + 1
                    elif handle in handles_inFreqBucket_list_med:
                        handles_with_allZeroVector_count_med = handles_with_allZeroVector_count_med + 1
                    elif handle in handles_inFreqBucket_list_low:
                        handles_with_allZeroVector_count_low = handles_with_allZeroVector_count_low  + 1

            # exclude the all-zero vector and establish threshold for number of collisions required for a vector to be included in the graph
            if len(handle_list) >= min_k and not is_all_zero_vector:

                uniqueVector_toCorrespondingHandleCount_dict_full[vector_string] = k

                for handle in handle_list:
                    if handle in handles_inFreqBucket_list_high:
                        handles_with_nonUniqueVectors_num_high = handles_with_nonUniqueVectors_num_high  + 1
                        added_to_high = True
                        if vector_string in uniqueVector_toCorrespondingHandleCount_dict_high:
                            uniqueVector_toCorrespondingHandleCount_dict_high[vector_string] = uniqueVector_toCorrespondingHandleCount_dict_high[vector_string] + 1
                        else:
                            uniqueVector_toCorrespondingHandleCount_dict_high[vector_string] = 1

                    elif handle in handles_inFreqBucket_list_med:
                        handles_with_nonUniqueVectors_num_med = handles_with_nonUniqueVectors_num_med  + 1
                        added_to_med = True
                        if vector_string in uniqueVector_toCorrespondingHandleCount_dict_med:
                            uniqueVector_toCorrespondingHandleCount_dict_med[vector_string] = uniqueVector_toCorrespondingHandleCount_dict_med[vector_string] + 1
                        else:
                            uniqueVector_toCorrespondingHandleCount_dict_med[vector_string] = 1
                    elif handle in handles_inFreqBucket_list_low:
                        handles_with_nonUniqueVectors_num_low = handles_with_nonUniqueVectors_num_low  + 1
                        added_to_low = True
                        if vector_string in uniqueVector_toCorrespondingHandleCount_dict_low:
                            uniqueVector_toCorrespondingHandleCount_dict_low[vector_string] = uniqueVector_toCorrespondingHandleCount_dict_low[vector_string] + 1
                        else:
                            uniqueVector_toCorrespondingHandleCount_dict_low[vector_string] = 1

                vector_string_toK_dict_full[vector_string] = k
                vector_string_toHandleList_dict_full[vector_string] = handle_list

                for handle in handle_list:
                    if handle in handles_inFreqBucket_list_high:

                        if vector_string in vector_string_toK_dict_high:
                            vector_string_toK_dict_high[vector_string] = vector_string_toK_dict_high[vector_string] + 1
                        else:
                            vector_string_toK_dict_high[vector_string] = 1
                        if vector_string in vector_string_toHandleList_dict_high:
                            vector_string_toHandleList_dict_high[vector_string].append(handle)
                        else:
                            vector_string_toHandleList_dict_high[vector_string] = []
                            vector_string_toHandleList_dict_high[vector_string].append(handle)

                    elif handle in handles_inFreqBucket_list_med:

                        if vector_string in vector_string_toK_dict_med:
                            vector_string_toK_dict_med[vector_string] = vector_string_toK_dict_med[vector_string] + 1
                        else:
                            vector_string_toK_dict_med[vector_string] = 1
                        if vector_string in vector_string_toHandleList_dict_med:
                            vector_string_toHandleList_dict_med[vector_string].append(handle)
                        else:
                            vector_string_toHandleList_dict_med[vector_string] = []
                            vector_string_toHandleList_dict_med[vector_string].append(handle)

                    elif handle in handles_inFreqBucket_list_low:

                        if vector_string in vector_string_toK_dict_low:
                            vector_string_toK_dict_low[vector_string] = vector_string_toK_dict_low[vector_string] + 1
                        else:
                            vector_string_toK_dict_low[vector_string] = 1
                        if vector_string in vector_string_toHandleList_dict_low:
                            vector_string_toHandleList_dict_low[vector_string].append(handle)
                        else:
                            vector_string_toHandleList_dict_low[vector_string] = []
                            vector_string_toHandleList_dict_low[vector_string].append(handle)

    # k_value_list_full = []
    # k_value_list_high = []
    # k_value_list_med = []
    # k_value_list_low = []

    # for vector_string, k in sorted(vector_string_toK_dict_full.items(), key=operator.itemgetter(1), reverse = False):
    #     k_value_list_full.append(k)

    # for vector_string, k in sorted(vector_string_toK_dict_high.items(), key=operator.itemgetter(1), reverse = False):
    #     k_value_list_high.append(k)

    # for vector_string, k in sorted(vector_string_toK_dict_med.items(), key=operator.itemgetter(1), reverse = False):
    #     k_value_list_med.append(k)

    # for vector_string, k in sorted(vector_string_toK_dict_low.items(), key=operator.itemgetter(1), reverse = False):
    #     k_value_list_low.append(k)

# does every vector have one collision, itelsf?
# yes, this is the case. every vector has at least k=1, one 'colllision' on itself
    # in my graphs, k=1 should be turned into k=0, etc.

    num_high_handles = len(handles_inFreqBucket_list_high)
    num_med_handles = len(handles_inFreqBucket_list_med)
    num_low_handles = len(handles_inFreqBucket_list_low)

    total_handle_num = 5626 # this includes handles with all-zero vectors
    # print 'All Handles', str(handles_with_nonUniqueVectors_num_full + num_full)
    # print 'handles_with_nonUniqueVectors_num_full', str(handles_with_nonUniqueVectors_num_full)
    print 'num_high_handles', str(num_high_handles)
    print 'num_med_handles', str(num_med_handles)
    print 'num_low_handles', str(num_low_handles)

    # print 'handles_with_nonUniqueVectors_num_full:', handles_with_nonUniqueVectors_num_full
    # print 'handles_with_nonUniqueVectors_num_high:', handles_with_nonUniqueVectors_num_high
    # print 'handles_with_nonUniqueVectors_num_med:', handles_with_nonUniqueVectors_num_med
    # print 'handles_with_nonUniqueVectors_num_low:', handles_with_nonUniqueVectors_num_low

    # print 'handles_with_allZeroVector_count_full', handles_with_allZeroVector_count_full
    # print 'handles_with_allZeroVector_count_high', handles_with_allZeroVector_count_high
    # print 'handles_with_allZeroVector_count_med', handles_with_allZeroVector_count_med
    # print 'handles_with_allZeroVector_count_low', handles_with_allZeroVector_count_low

    # num_uniqueVectors_full = len(k_value_list_full) # non unique vector is a vector with at least one handle,
    # num_uniqueVectors_high = len(k_value_list_high)
    # num_uniqueVectors_med = len(k_value_list_med)
    # num_uniqueVectors_low = len(k_value_list_low)

    # print 'num_uniqueVectors_full', num_uniqueVectors_full
    # print 'num_uniqueVectors_high', num_uniqueVectors_high
    # print 'num_uniqueVectors_med', num_uniqueVectors_med
    # print 'num_uniqueVectors_low', num_uniqueVectors_low

    # handles_with_nonUniqueVectors_num_full_2 =  sum(k_value_list_full)

    # # num handles that correspond to all zero vector + num vectors that have more than one corresponding handle + num of collisions
    # total_full = handles_with_allZeroVector_count_full + num_uniqueVectors_full + handles_with_nonUniqueVectors_num_full
    # print 'handles_with_allZeroVector_count_full + num_uniqueVectors_full + handles_with_nonUniqueVectors_num_full:', str(total_full)

    # total_high = handles_with_allZeroVector_count_high + num_uniqueVectors_high + handles_with_nonUniqueVectors_num_high
    # total_med = handles_with_allZeroVector_count_med + num_uniqueVectors_med + handles_with_nonUniqueVectors_num_med
    # total_low = handles_with_allZeroVector_count_low + num_uniqueVectors_low + handles_with_nonUniqueVectors_num_low

    # print 'total_high', total_high
    # print 'total_med', total_med
    # print 'total_low', total_low

    # exclude handles with all-zero vector (already done)
    # add handles that do not have unique vectors (i.e. num_collisions) to output list
    # add handles that are unique to output list, because they are not currently there
        # this entails add n '0's to each k_value_list, where n is the number of

    # num_zeroColissionHandles_full = num_zeroColission_handles - num_high_handles - num_med_handles - num_low_handles
    # num_zeroColissionHandles_high = num_zeroColission_handles - num_high_handles - num_med_handles - num_low_handles
    # num_zeroColissionHandles_med = num_zeroColission_handles - num_high_handles - num_med_handles - num_low_handles
    # num_zeroColissionHandles_low = num_zeroColission_handles - num_high_handles - num_med_handles - num_low_handles


    # num_zeroCollision_handlesToAdd_high = num_high_handles - total_high
    # num_zeroCollision_handlesToAdd_med = num_med_handles - total_med
    # num_zeroCollision_handlesToAdd_low = num_low_handles - total_low

    # for i in range(handles_with_nonUniqueVectors_num_full):
    #     k_value_list_full.insert(0, 0)
    # for i in range(num_zeroCollision_handlesToAdd_high):
    #     k_value_list_high.insert(0, 0)
    # for i in range(num_zeroCollision_handlesToAdd_med):
    #     k_value_list_med.insert(0, 0)
    # for i in range(num_zeroCollision_handlesToAdd_low):
    #     k_value_list_low.insert(0, 0)

    # add handles with

    # WHY am i plotting handles across x axis? shouldn't it be unique vectors?
    # if unique vectors, then can use full set of unique vectors, sorted by k,
    # if I have unique handles as my x axis...

    k_value_list_full = []
    k_value_list_high = []
    k_value_list_med = []
    k_value_list_low = []

    for vector, collision_rate in uniqueVector_toCorrespondingHandleCount_dict_full.iteritems():
        k_value_list_full.append(collision_rate)

    for vector, collision_rate in uniqueVector_toCorrespondingHandleCount_dict_high.iteritems():
        k_value_list_high.append(collision_rate)

    for vector, collision_rate in uniqueVector_toCorrespondingHandleCount_dict_med.iteritems():
        k_value_list_med.append(collision_rate)

    for vector, collision_rate in uniqueVector_toCorrespondingHandleCount_dict_low.iteritems():
        k_value_list_low.append(collision_rate)



    x_data = ['Unique, non-all-zero collision vectors, <br> that represent at least one Full tweet_freq handle <br> <i>n=<i> ' + str(len(k_value_list_full)), 'Unique, non-all-zero collision vectors, <br> that represent at least one High tweet_freq handle <br> <i>n=<i> ' + str(len(k_value_list_high)),
      'Unique, non-all-zero collision vectors, <br> that represent at least one Med tweet_freq handle <br> <i>n=<i> ' + str(len(k_value_list_med)), 'Unique, non-all-zero collision vectors, <br> that represent at least one Low tweet_freq handle <br> <i>n=<i> ' + str(len(k_value_list_low)),]

    if len (k_value_list_high) == 0: # add a dummy zero to show zero vectors for this list
        k_value_list_low.append(0)

    if len (k_value_list_med) == 0: # add a dummy zero to show zero vectors for this list
        k_value_list_med.append(0)

    if len (k_value_list_low) == 0: # add a dummy zero to show zero vectors for this list
        k_value_list_med.append(0)


    y_data = [k_value_list_full, k_value_list_high, k_value_list_med, k_value_list_low,]

    colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)']

    traces = []

    for xd, yd, cls in zip(x_data, y_data, colors):
        traces.append(go.Box(
            y=yd,
            name=xd,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=cls,
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
        ))

    layout = go.Layout(
        annotations=Annotations([
            Annotation(
                x=0.5004254919715793,
                y=-0.16191064079952971,
                showarrow=False,
                text='',
                xref='paper',
                yref='paper'
            ),
            Annotation(
                x=-0.04944728761514841,
                y=0.4714285714285711,
                showarrow=False,
                text='Number hits',
                textangle=-90,
                xref='paper',
                yref='paper'
            )
        ]),
        autosize=True,
        margin=Margin(
            b=100
        ),
        title='Distribution of hit rates on non-zero ' + projection_name + ' vectors,' + '<br> hit rate being defined as the number of handles corresponding to the same vector projection <br> (overlap can occur between the vectors represented across these sets)',
        yaxis=dict(
           zeroline=False
        ),
        # # margin=dict(
        # #     l=40,
        # #     r=30,
        # #     b=80,
        # #     t=100,
        # # ),
        # paper_bgcolor='rgb(243, 243, 243)',
        # plot_bgcolor='rgb(243, 243, 243)',
        showlegend=False
    )

    fig = go.Figure(data=traces, layout=layout)
    py.iplot(fig, filename=projection_name + '- Collision Rates on Collision Vectors Distributions')



def Graph_HandleEquivalancyClassSize_Distributions_SingleProjection( projection_similarity_file_name, min_num_handles_same_vector=0, projection_name=''):

    vector_string_toNumHandlesSameVector_dict = {}
    vector_string_toHandleList_dict = {}

    handles_inFreqBucket_list_low = []
    handles_inFreqBucket_list_med = []
    handles_inFreqBucket_list_high = []

    # vector_string_toK_dict_EmotionDict[vector_string] = k
    vector_string_toNumHandlesSameVector_dict_full = {}
    vector_string_toNumHandlesSameVector_dict_high = {}
    vector_string_toNumHandlesSameVector_dict_med = {}
    vector_string_toNumHandlesSameVector_dict_low = {}

    # vector_string_toHandleList_dict_EmotionDict[vector_string] = handle_list
    vector_string_toHandleList_dict_full = {}
    vector_string_toHandleList_dict_high = {}
    vector_string_toHandleList_dict_med = {}
    vector_string_toHandleList_dict_low = {}

    size_featureSpace = 0

    infile_handles_inFreqBucket_low = open('low_handles_4000000.txt', 'r')
    for line in infile_handles_inFreqBucket_low:
        handle_list = line.split(',')
        for handle in handle_list:
            handles_inFreqBucket_list_low.append(handle)

    infile_handles_inFreqBucket_med = open('med_handles_4000000.txt', 'r')
    for line in infile_handles_inFreqBucket_med:
        handle_list = line.split(',')
        for handle in handle_list:
            handles_inFreqBucket_list_med.append(handle)

    infile_handles_inFreqBucket_high = open('high_handles_4000000.txt', 'r')
    for line in infile_handles_inFreqBucket_high:
        handle_list = line.split(',')
        for handle in handle_list:
            handles_inFreqBucket_list_high.append(handle)

    handles_with_nonUniqueVectors_num_full = 0
    handles_with_nonUniqueVectors_num_high = 0
    handles_with_nonUniqueVectors_num_med = 0
    handles_with_nonUniqueVectors_num_low = 0

    handles_with_allZeroVector_count_full = 0
    handles_with_allZeroVector_count_high = 0
    handles_with_allZeroVector_count_med = 0
    handles_with_allZeroVector_count_low = 0

    equivalenceClassSizes_list_full = []
    equivalenceClassSizes_list_high = []
    equivalenceClassSizes_list_med = []
    equivalenceClassSizes_list_low = []

    num_handles_withZero_vector_TOTAL = 0

    with open(projection_similarity_file_name) as infile:
        line_count = 0

        is_all_zero_vector = True # keep track that singular all zero vector for each file is not included in graph
        all_zero_vector_occured = False


        # is_all_zero_vector_BINARY = True # keep track that singular all zero vector for each file is not included in graph
        # all_zero_vector_occured_BINARY = False

        for line in infile:
            line_count = line_count + 1
            print 'Projection Similarity file ', projection_name, ': processing line ', line_count

            line = unicode(line,'utf-8')
            line = line.encode('unicode-escape')

            # line = line.lower()

            line = line.strip('\n')
            line = line.strip('\\n')
            line = line.strip('\r')

            # line_elements = line.split(' ')

            # use regex to extract components
            vector_string = re.findall('\[.+\]', line)[0]
            vector = ast.literal_eval(vector_string)

            # vector_BINARY= Binary_Transform_Vector(vector)

            # convert vector
                # store in seperate binary list

                # create _binary varients for all neccesary variables

            if all_zero_vector_occured: # only have to check if vector is not all zero, if all zero vector has not occured yet
                is_all_zero_vector = False
            else:
                is_all_zero_vector = True # keep track that singular all zero vector for each file is not included in graph

                for value in vector:
                    if value != 0:
                        is_all_zero_vector = False

            # if all_zero_vector_occured_BINARY: # only have to check if vector is not all zero, if all zero vector has not occured yet
            #     is_all_zero_vector_BINARY = False
            # else:
            #     is_all_zero_vector_BINARY = True # keep track that singular all zero vector for each file is not included in graph

            #     for value in vector_BINARY:
            #         if value != 0:
            #             is_all_zero_vector_BINARY = False

            num_handles_same_vector = int(re.findall('k=(\d+)', line)[0])

            # handle_list = re.findall('\w{2,}', line)[1:]

            capture_list = re.findall(', (\w{3,})', line) # list of handles corresponding to this vector

            handle_list = [x for x in capture_list if not x.isdigit()] # accounts for above capture picking up large numbers

            num_handles_same_vector = len(handle_list)

            if not is_all_zero_vector:

                handles_with_nonUniqueVectors_num_full = handles_with_nonUniqueVectors_num_full  + num_handles_same_vector - 1# subtract one to account for a vector being unique for every handle

                added_to_high = False
                added_to_med = False
                added_to_low = False

                for handle in handle_list:
                    if handle in handles_inFreqBucket_list_high:
                        handles_with_nonUniqueVectors_num_high = handles_with_nonUniqueVectors_num_high  + 1
                        added_to_high = True
                    elif handle in handles_inFreqBucket_list_med:
                        handles_with_nonUniqueVectors_num_med = handles_with_nonUniqueVectors_num_med  + 1
                        added_to_med = True
                    elif handle in handles_inFreqBucket_list_low:
                        handles_with_nonUniqueVectors_num_low = handles_with_nonUniqueVectors_num_low  + 1
                        added_to_low = True

                if added_to_high:
                    handles_with_nonUniqueVectors_num_high = handles_with_nonUniqueVectors_num_high - 1 # subtract one to account for a vector being unique for every handle
                elif added_to_med:
                    handles_with_nonUniqueVectors_num_med = handles_with_nonUniqueVectors_num_med - 1 # subtract one to account for a vector being unique for every handle
                elif added_to_low:
                    handles_with_nonUniqueVectors_num_low = handles_with_nonUniqueVectors_num_low - 1  # subtract one to account for a vector being unique for every handle

                # STORE SIZE OF EACH EQUIVALENCE CLASS
                equivalenceClassSizes_list_full.append(num_handles_same_vector)

                size_high = 0
                size_med = 0
                size_low = 0

                added_to_high = False
                added_to_med = False
                added_to_low = False
                for handle in handle_list:
                    if handle in handles_inFreqBucket_list_high:
                        size_high = size_high  + 1
                        added_to_high = True
                    elif handle in handles_inFreqBucket_list_med:
                        size_med = size_med  + 1
                        added_to_med = True
                    elif handle in handles_inFreqBucket_list_low:
                        size_low = size_low  + 1
                        added_to_low = True

                if added_to_high:
                    equivalenceClassSizes_list_high.append(size_high)
                if added_to_med:
                    equivalenceClassSizes_list_med.append(size_med)
                if added_to_low:
                    equivalenceClassSizes_list_low.append(size_low)

            else:
                handles_with_allZeroVector_count_full = num_handles_same_vector

                for handle in handle_list:
                    if handle in handles_inFreqBucket_list_high:
                        handles_with_allZeroVector_count_high = handles_with_allZeroVector_count_high  + 1
                    elif handle in handles_inFreqBucket_list_med:
                        handles_with_allZeroVector_count_med = handles_with_allZeroVector_count_med + 1
                    elif handle in handles_inFreqBucket_list_low:
                        handles_with_allZeroVector_count_low = handles_with_allZeroVector_count_low  + 1

            # exclude the all-zero vector and establish threshold for number of collisions required for a vector to be included in the graph
            if num_handles_same_vector >= min_num_handles_same_vector and not is_all_zero_vector:

                vector_string_toNumHandlesSameVector_dict_full[vector_string] = num_handles_same_vector
                vector_string_toHandleList_dict_full[vector_string] = handle_list

                for handle in handle_list:
                    if handle in handles_inFreqBucket_list_high:

                        if vector_string in vector_string_toNumHandlesSameVector_dict_high:
                            vector_string_toNumHandlesSameVector_dict_high[vector_string] = vector_string_toNumHandlesSameVector_dict_high[vector_string] + 1
                        else:
                            vector_string_toNumHandlesSameVector_dict_high[vector_string] = 1
                        if vector_string in vector_string_toHandleList_dict_high:
                            vector_string_toHandleList_dict_high[vector_string].append(handle)
                        else:
                            vector_string_toHandleList_dict_high[vector_string] = []
                            vector_string_toHandleList_dict_high[vector_string].append(handle)

                    elif handle in handles_inFreqBucket_list_med:

                        if vector_string in vector_string_toNumHandlesSameVector_dict_med:
                            vector_string_toNumHandlesSameVector_dict_med[vector_string] = vector_string_toNumHandlesSameVector_dict_med[vector_string] + 1
                        else:
                            vector_string_toNumHandlesSameVector_dict_med[vector_string] = 1
                        if vector_string in vector_string_toHandleList_dict_med:
                            vector_string_toHandleList_dict_med[vector_string].append(handle)
                        else:
                            vector_string_toHandleList_dict_med[vector_string] = []
                            vector_string_toHandleList_dict_med[vector_string].append(handle)

                    elif handle in handles_inFreqBucket_list_low:

                        if vector_string in vector_string_toNumHandlesSameVector_dict_low:
                            vector_string_toNumHandlesSameVector_dict_low[vector_string] = vector_string_toNumHandlesSameVector_dict_low[vector_string] + 1
                        else:
                            vector_string_toNumHandlesSameVector_dict_low[vector_string] = 1
                        if vector_string in vector_string_toHandleList_dict_low:
                            vector_string_toHandleList_dict_low[vector_string].append(handle)
                        else:
                            vector_string_toHandleList_dict_low[vector_string] = []
                            vector_string_toHandleList_dict_low[vector_string].append(handle)

    numHandlesSameVector_value_list_full = []
    numHandlesSameVector_value_list_high = []
    numHandlesSameVector_value_list_med = []
    numHandlesSameVector_value_list_low = []

    for vector_string, numHandlesSameVector in sorted(vector_string_toNumHandlesSameVector_dict_full.items(), key=operator.itemgetter(1), reverse = False):
        numHandlesSameVector_value_list_full.append(numHandlesSameVector)

    for vector_string, numHandlesSameVector in sorted(vector_string_toNumHandlesSameVector_dict_high.items(), key=operator.itemgetter(1), reverse = False):
        numHandlesSameVector_value_list_high.append(numHandlesSameVector)

    for vector_string, numHandlesSameVector in sorted(vector_string_toNumHandlesSameVector_dict_med.items(), key=operator.itemgetter(1), reverse = False):
        numHandlesSameVector_value_list_med.append(numHandlesSameVector)

    for vector_string, numHandlesSameVector in sorted(vector_string_toNumHandlesSameVector_dict_low.items(), key=operator.itemgetter(1), reverse = False):
        numHandlesSameVector_value_list_low.append(numHandlesSameVector)

# does every vector have one collision, itelsf?
# yes, this is the case. every vector has at least k=1, one 'colllision' on itself
    # in my graphs, k=1 should be turned into k=0, etc.

    num_high_handles = len(handles_inFreqBucket_list_high)
    num_med_handles = len(handles_inFreqBucket_list_med)
    num_low_handles = len(handles_inFreqBucket_list_low)

    total_handle_num = 5626 # this includes handles with all-zero vectors
    # print 'All Handles', str(handles_with_nonUniqueVectors_num_full + num_full)
    # print 'handles_with_nonUniqueVectors_num_full', str(handles_with_nonUniqueVectors_num_full)
    print 'num_high_handles', str(num_high_handles)
    print 'num_med_handles', str(num_med_handles)
    print 'num_low_handles', str(num_low_handles)

    print 'handles_with_nonUniqueVectors_num_full:', handles_with_nonUniqueVectors_num_full
    print 'handles_with_nonUniqueVectors_num_high:', handles_with_nonUniqueVectors_num_high
    print 'handles_with_nonUniqueVectors_num_med:', handles_with_nonUniqueVectors_num_med
    print 'handles_with_nonUniqueVectors_num_low:', handles_with_nonUniqueVectors_num_low

    print 'handles_with_allZeroVector_count_full', handles_with_allZeroVector_count_full
    print 'handles_with_allZeroVector_count_high', handles_with_allZeroVector_count_high
    print 'handles_with_allZeroVector_count_med', handles_with_allZeroVector_count_med
    print 'handles_with_allZeroVector_count_low', handles_with_allZeroVector_count_low

    num_uniqueVectors_full = len(numHandlesSameVector_value_list_full) # non unique vector is a vector with at least one handle,
    num_uniqueVectors_high = len(numHandlesSameVector_value_list_high)
    num_uniqueVectors_med = len(numHandlesSameVector_value_list_med)
    num_uniqueVectors_low = len(numHandlesSameVector_value_list_low)

    print 'num_uniqueVectors_full', num_uniqueVectors_full
    print 'num_uniqueVectors_high', num_uniqueVectors_high
    print 'num_uniqueVectors_med', num_uniqueVectors_med
    print 'num_uniqueVectors_low', num_uniqueVectors_low

    handles_with_nonUniqueVectors_num_full_2 =  sum(numHandlesSameVector_value_list_full)

    # num handles that correspond to all zero vector + num vectors that have more than one corresponding handle + num of collisions
    total_full = handles_with_allZeroVector_count_full + num_uniqueVectors_full + handles_with_nonUniqueVectors_num_full
    print 'handles_with_allZeroVector_count_full + num_uniqueVectors_full + handles_with_nonUniqueVectors_num_full:', str(total_full)

    total_high = handles_with_allZeroVector_count_high + num_uniqueVectors_high + handles_with_nonUniqueVectors_num_high
    total_med = handles_with_allZeroVector_count_med + num_uniqueVectors_med + handles_with_nonUniqueVectors_num_med
    total_low = handles_with_allZeroVector_count_low + num_uniqueVectors_low + handles_with_nonUniqueVectors_num_low

    print 'total_high', total_high
    print 'total_med', total_med
    print 'total_low', total_low

    # exclude handles with all-zero vector (already done)
    # add handles that do not have unique vectors (i.e. num_collisions) to output list
    # add handles that are unique to output list, because they are not currently there
        # this entails add n '0's to each k_value_list, where n is the number of

    # num_zeroColissionHandles_full = num_zeroColission_handles - num_high_handles - num_med_handles - num_low_handles
    # num_zeroColissionHandles_high = num_zeroColission_handles - num_high_handles - num_med_handles - num_low_handles
    # num_zeroColissionHandles_med = num_zeroColission_handles - num_high_handles - num_med_handles - num_low_handles
    # num_zeroColissionHandles_low = num_zeroColission_handles - num_high_handles - num_med_handles - num_low_handles


    num_zeroCollision_handlesToAdd_high = num_high_handles - total_high
    num_zeroCollision_handlesToAdd_med = num_med_handles - total_med
    num_zeroCollision_handlesToAdd_low = num_low_handles - total_low

    # because only added collision vectors from input file, need to add vectors that only had one indivicual corresponding to it
    # add 1 for each handle that has a unique vector
        # number of handles that have a unique vector =  total number of handles - number of handles with non unique vectors

    # subtract Num unique vectors from total vectors
    # THIS IS ACTUALY NUM UNIQUE HANDLES
    num_uniqueVectors_full = total_handle_num - handles_with_nonUniqueVectors_num_full

    num_uniqueVectors_high = len(handles_inFreqBucket_list_high) - handles_with_nonUniqueVectors_num_high
    num_uniqueVectors_med = len(handles_inFreqBucket_list_med) - handles_with_nonUniqueVectors_num_med
    num_uniqueVectors_low = len(handles_inFreqBucket_list_low) - handles_with_nonUniqueVectors_num_low

    for i in range(num_uniqueVectors_full):
        equivalenceClassSizes_list_full.insert(0, 1)
    for i in range(num_uniqueVectors_high):
        equivalenceClassSizes_list_high.insert(0, 1)
    for i in range(num_uniqueVectors_med):
        equivalenceClassSizes_list_med.insert(0, 1)
    for i in range(num_uniqueVectors_low):
        equivalenceClassSizes_list_low.insert(0, 1)

    # add handles with

    # WHY am i plotting handles across x axis? shouldn't it be unique vectors?
    # if unique vectors, then can use full set of unique vectors, sorted by k,
    # if I have unique handles as my x axis...

    # x_data = ['Equivalancy class size distribution, <br> for Full tweet_freq handles <br> <i>total_num_classes=<i> ' + str(len(numHandlesSameVector_value_list_full)), 'Handle equivalancy class size distribution, <br> for High tweet_freq handles <br> <i>total_num_classes=<i> ' + str(len(numHandlesSameVector_value_list_high)),
      # 'Handle equivalancy class size distribution, <br> for Med tweet_freq handles <br> <i>total_num_classes=<i> ' + str(len(numHandlesSameVector_value_list_med)), 'Handle equivalancy class size distribution, <br> for Low tweet_freq handles <br> <i>total_num_classes=<i> ' + str(len(numHandlesSameVector_value_list_low)),]

    # y_data = [numHandlesSameVector_value_list_full, numHandlesSameVector_value_list_high, numHandlesSameVector_value_list_med, numHandlesSameVector_value_list_low,]

    # colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)']
    # str(len(numHandlesSameVector_value_list_full)) - former total_num_classes variable

    # trace_full = go.Box(
    #     y = equivalenceClassSizes_list_full,
    #     name='All <br> <i>n=<i> ' + str(num_uniqueVectors_full) + '<br>  <i>z=<i> ' + str(handles_with_allZeroVector_count_full),
    #     boxpoints='all',
    #     jitter=0.5,
    #     whiskerwidth=0.2,
    #     fillcolor='rgb(93, 164, 214)',
    #     marker=dict(
    #         size = 2,
    #         color='rgb(93, 164, 214)',
    #     ),
    #     line=dict(width=1),
    #     boxmean='sd'
    # )
    trace_high = go.Box(
        y = equivalenceClassSizes_list_high,
        name='High <br> <i>protected=<i> ' + str(num_uniqueVectors_high) + '<br>  <i>all-zero=<i> ' + str(handles_with_allZeroVector_count_high),
        boxpoints='all',
        jitter=0.5,
        whiskerwidth=0.2,
        fillcolor='rgb(255, 144, 14)',
        marker=dict(
            size = 2,
            color='rgb(255, 144, 14)',
        ),
        line=dict(width=1),
        boxmean='sd'
    )
    trace_med = go.Box(
        y = equivalenceClassSizes_list_med,
        name='Moderate <br> <i>protected=<i> ' + str(num_uniqueVectors_med) + '<br>  <i>all-zero=<i> ' + str(handles_with_allZeroVector_count_med),
        boxpoints='all',
        jitter=0.5,
        whiskerwidth=0.2,
        fillcolor='rgb(44, 160, 101)',
        marker=dict(
            size = 2,
            color='rgb(44, 160, 101)',
        ),
        line=dict(width=1),
        boxmean='sd'
    )
    trace_low = go.Box(
        y = equivalenceClassSizes_list_low,
        name='Low <br> <i>protected=<i> ' + str(num_uniqueVectors_low) + '<br>  <i>all-zero=<i> ' + str(handles_with_allZeroVector_count_low),
        boxpoints='all',
        jitter=0.5,
        whiskerwidth=0.2,
        fillcolor='rgb(255, 65, 54)',
        marker=dict(
            size = 2,
            color='rgb(255, 65, 54)',
        ),
        line=dict(width=1),
        boxmean='sd'
    )


    layout = go.Layout(
        annotations=Annotations([
            Annotation(
                x=0.5004254919715793,
                y=-0.12191064079952971,
                showarrow=False,
                text='',
                xref='paper',
                yref='paper'
            ),
            Annotation(
                x=-0.05944728761514841,
                y=0.4714285714285711,
                showarrow=False,
                text='Equivalency class sizes, excluding all zero vector',
                textangle=-90,
                xref='paper',
                yref='paper'
            )
        ]),
        autosize=True,
        title='Distribution of Handle Equivalency Class Sizes, ' + projection_name,
        yaxis=dict(
           zeroline=False
        ),
                # # margin=dict(
        # #     l=40,
        # #     r=30,
        # #     b=80,
        # #     t=100,
        # # ),
        # paper_bgcolor='rgb(243, 243, 243)',
        # plot_bgcolor='rgb(243, 243, 243)',
        showlegend=False
    )

    # traces = [trace_full, trace_high, trace_med, trace_low]
    traces = [trace_high, trace_med, trace_low]


    fig = go.Figure(data=traces, layout=layout)
    py.iplot(fig, filename=projection_name + ' - Equivalancy Class Sizes - 3 Buckets')

    # trace_full = go.Scatter(
    #     x = range(len(k_value_list_full)),
    #     y = k_value_list_full,
    #     mode = 'lines',
    #     name = 'All Handles [n=' + str(total_full) + ']'
    # )

    # trace_high = go.Scatter(
    #     x = range(len(k_value_list_high)),
    #     y = k_value_list_high,
    #     mode = 'lines',
    #     name = 'High Tweet Frequency Handles [n=' + str(num_high_handles) + ']'
    # )

    # trace_med = go.Scatter(
    #     x = range(len(k_value_list_med)),
    #     y = k_value_list_med,
    #     mode = 'lines',
    #     name = 'Med Tweet Frequency Handles [n=' + str(num_med_handles) + ']'
    # )

    # trace_low = go.Scatter(
    #     x = range(len(k_value_list_low)),
    #     y = k_value_list_low,
    #     mode = 'lines',
    #     name = 'Low Tweet Frequency Handles [n=' + str(num_low_handles) + ']'
    # )

    # data = [trace_full, trace_high, trace_med, trace_low]

    # layout = go.Layout(
    #     title = 'Vector-Projection Collisions with other Handles' + projection_name,
    #     xaxis = dict(title = 'Unique vector-projections sorted on collisions, excluding all-zero vector-projection'),
    #     yaxis = dict(title = 'Number collisions'),
    #     )

    # filename = 'Collisions, ' + projection_name
    # fig = dict(data=data, layout=layout)
    # py.iplot(fig, filename=filename)


#  Chart that shows the original k values, and the k values for other representations, when you exclude the all-zero vector
#  load in 4 representation files (actually, just similarity statistics files): bag of words, emotion dict, top 1k vocab, top 1k tf-idf

# tweet_set = {'', 'retweet', 'original'}
def Graph_Similarity( tweet_set = 'retweet', binary=False, min_k=2, tweet_freq_type='' ):

    infile_name_BoW = ''
    infile_name_EmotionDict = ''
    infile_name_topTF_IDF = ''
    infile_name_topVocab = ''
    graph_name = ''

    if binary:
        infile_name_BoW = 'clean_tweets_' + tweet_set + '4000000_BagOfWords_summed_Uniqueness_Metrics_' + tweet_freq_type + 'BINARY.csv'
        infile_name_EmotionDict = 'tweets_MatrixRepresentation_emotionDictionary_' + tweet_set +'4000000_summed_Uniqueness_Metrics_' + tweet_freq_type + 'BINARY.csv'
        infile_name_topTF_IDF = 'tweets_MatrixRepresentation_topTF_IDF_' + tweet_set +'4000000_summed_Uniqueness_Metrics_' + tweet_freq_type + 'BINARY.csv'
        infile_name_topVocab = 'tweets_MatrixRepresentation_topVocab_' + tweet_set + '4000000_summed_Uniqueness_Metrics_' + tweet_freq_type + 'BINARY.csv'

        graph_name = 'Similarity: ' + tweet_set + ' binary=true'

    else:
        infile_name_BoW = 'clean_tweets_' + tweet_set + '4000000_BagOfWords_summed_Uniqueness_Metrics_' + tweet_freq_type + 'EXACT.csv'
        infile_name_EmotionDict = 'tweets_MatrixRepresentation_emotionDictionary_' + tweet_set + '4000000_summed_Uniqueness_Metrics_' + tweet_freq_type + 'EXACT.csv'
        infile_name_topTF_IDF = 'tweets_MatrixRepresentation_topTF_IDF_' + tweet_set + '4000000_summed_Uniqueness_Metrics_' + tweet_freq_type + 'EXACT.csv'
        infile_name_topVocab = 'tweets_MatrixRepresentation_topVocab_' + tweet_set + '4000000_summed_Uniqueness_Metrics_' + tweet_freq_type + 'EXACT.csv'

        graph_name = 'Similarity: ' + tweet_set + '  binary=false ' + 'min_k=' + str(min_k)


    # instantiate data structures

    # for each representation,
            # ( if k greater than 1)
        # a dictionary of  vector_string --> k value
        # a dictionary of vector_string --> handle list

    vector_string_toK_dict_BoW = {}
    vector_string_toHandleList_dict_BoW = {}

    vector_string_toK_dict_EmotionDict = {}
    vector_string_toHandleList_dict_EmotionDict = {}

    vector_string_toK_dict_topTF_IDF = {}
    vector_string_toHandleList_dict_topTF_IDF = {}

    vector_string_toK_dict_topVocab = {}
    vector_string_toHandleList_dict_topVocab = {}


    # for each line (for each file), aquire
    # vector, k value, handles

    # # bag of words
    # with open(infile_name_BoW) as infile_BoW:
    #     line_count = 0

    #     is_all_zero_vector = True # keep track that singular all zero vector for each file is not included in graph
    #     all_zero_vector_occured = False

    #     for line in infile_BoW:
    #         line_count = line_count + 1
    #         print 'BoW: processing line ', line_count

    #         line = unicode(line,'utf-8')
    #         line = line.encode('unicode-escape')

    #         # line = line.lower()

    #         line = line.strip('\n')
    #         line = line.strip('\\n')
    #         line = line.strip('\r')

    #         # line_elements = line.split(' ')

    #         # use regex to extract components
    #         vector_string = re.findall('\[.+\]', line)[0]
    #         vector = ast.literal_eval(vector_string)

    #         if all_zero_vector_occured: # only have to check if vector is not all zero, if all zero vector has not occured yet
    #             is_all_zero_vector = False
    #         else:
    #             is_all_zero_vector = True # keep track that singular all zero vector for each file is not included in graph

    #             for value in vector:
    #                 if value != 0:
    #                     is_all_zero_vector = False

    #         k = int(re.findall('k=(\d+)', line)[0])

    #         handle_list = re.findall('\w{2,}', line)[1:]

    #         # exclude the all-zero vector
    #         if k > min_k and not is_all_zero_vector:
    #             vector_string_toK_dict_BoW[vector_string] = k
    #             vector_string_toHandleList_dict_BoW[vector_string] = handle_list



    # emotion Dict
    with open(infile_name_EmotionDict) as infile_EmotionDict:
        line_count = 0

        is_all_zero_vector = True # keep track that singular all zero vector for each file is not included in graph
        all_zero_vector_occured = False

        for line in infile_EmotionDict:
            line_count = line_count + 1
            print 'EmotionDict: processing line ', line_count

            line = unicode(line,'utf-8')
            line = line.encode('unicode-escape')

            # line = line.lower()

            line = line.strip('\n')
            line = line.strip('\\n')
            line = line.strip('\r')

            # line_elements = line.split(' ')

            # use regex to extract components
            vector_string = re.findall('\[.+\]', line)[0]
            vector = ast.literal_eval(vector_string)

            if all_zero_vector_occured: # only have to check if vector is not all zero, if all zero vector has not occured yet
                is_all_zero_vector = False
            else:
                is_all_zero_vector = True # keep track that singular all zero vector for each file is not included in graph

                for value in vector:
                    if value != 0:
                        is_all_zero_vector = False

            k = int(re.findall('k=(\d+)', line)[0])

            handle_list = re.findall('\w{2,}', line)[1:]

            # exclude the all-zero vector
            if k > min_k and not is_all_zero_vector:
                vector_string_toK_dict_EmotionDict[vector_string] = k
                vector_string_toHandleList_dict_EmotionDict[vector_string] = handle_list

    # top tf-idf
    with open(infile_name_topTF_IDF) as infile_topTF_IDF:
        line_count = 0

        is_all_zero_vector = True # keep track that singular all zero vector for each file is not included in graph
        all_zero_vector_occured = False

        for line in infile_topTF_IDF:
            line_count = line_count + 1
            print 'topTF_IDF: processing line ', line_count

            line = unicode(line,'utf-8')
            line = line.encode('unicode-escape')

            # line = line.lower()

            line = line.strip('\n')
            line = line.strip('\\n')
            line = line.strip('\r')

            # line_elements = line.split(' ')

            # use regex to extract components
            vector_string = re.findall('\[.+\]', line)[0]
            vector = ast.literal_eval(vector_string)

            if all_zero_vector_occured: # only have to check if vector is not all zero, if all zero vector has not occured yet
                is_all_zero_vector = False
            else:
                is_all_zero_vector = True # keep track that singular all zero vector for each file is not included in graph

                for value in vector:
                    if value != 0:
                        is_all_zero_vector = False

            k = int(re.findall('k=(\d+)', line)[0])

            handle_list = re.findall('\w{2,}', line)[1:]

            # exclude the all-zero vector
            if k > min_k and not is_all_zero_vector:
                vector_string_toK_dict_topTF_IDF[vector_string] = k
                vector_string_toHandleList_dict_topTF_IDF[vector_string] = handle_list

    # top vocab
    with open(infile_name_topVocab) as infile_topVocab:
        line_count = 0

        is_all_zero_vector = True # keep track that singular all zero vector for each file is not included in graph
        all_zero_vector_occured = False

        for line in infile_topVocab:
            line_count = line_count + 1
            print 'topVocab: processing line ', line_count

            line = unicode(line,'utf-8')
            line = line.encode('unicode-escape')

            # line = line.lower()

            line = line.strip('\n')
            line = line.strip('\\n')
            line = line.strip('\r')

            # line_elements = line.split(' ')

            # use regex to extract components
            vector_string = re.findall('\[.+\]', line)[0]
            vector = ast.literal_eval(vector_string)

            if all_zero_vector_occured: # only have to check if vector is not all zero, if all zero vector has not occured yet
                is_all_zero_vector = False
            else:
                is_all_zero_vector = True # keep track that singular all zero vector for each file is not included in graph

                for value in vector:
                    if value != 0:
                        is_all_zero_vector = False

            k = int(re.findall('k=(\d+)', line)[0])

            handle_list = re.findall('\w{2,}', line)[1:]

            # exclude the all-zero vector
            if k >= min_k and not is_all_zero_vector:
                vector_string_toK_dict_topVocab[vector_string] = k
                vector_string_toHandleList_dict_topVocab[vector_string] = handle_list

    # sort each of the vectors (with k greater than 1) on k, aquire list of these k values
    k_value_list_BoW = []
    k_value_list_EmotionDict = []
    k_value_list_topTF_IDF = []
    k_value_list_topVocab = []

    # for vector_string, k in sorted(vector_string_toK_dict_BoW.items(), key=operator.itemgetter(1), reverse = False):
    #     k_value_list_BoW.append(k)

    for vector_string, k in sorted(vector_string_toK_dict_EmotionDict.items(), key=operator.itemgetter(1), reverse = False):
        k_value_list_EmotionDict.append(k)

    for vector_string, k in sorted(vector_string_toK_dict_topTF_IDF.items(), key=operator.itemgetter(1), reverse = False):
        k_value_list_topTF_IDF.append(k)

    for vector_string, k in sorted(vector_string_toK_dict_topVocab.items(), key=operator.itemgetter(1), reverse = False):
        k_value_list_topVocab.append(k)

    # trace_BoW = go.Scatter(
    #     x = range(len(k_value_list_BoW)),
    #     y = k_value_list_BoW,
    #     mode = 'lines',
    #     name = 'BoW'
    # )

    trace_EmotionDict = go.Scatter(
        x = range(len(k_value_list_EmotionDict)),
        y = k_value_list_EmotionDict,
        mode = 'lines',
        name = 'Emotion'
    )

    trace_topTF_IDF = go.Scatter(
        x = range(len(k_value_list_topTF_IDF)),
        y = k_value_list_topTF_IDF,
        mode = 'lines',
        name = 'topTF_IDF'
    )

    trace_topVocab = go.Scatter(
        x = range(len(k_value_list_topVocab)),
        y = k_value_list_topVocab,
        mode = 'lines',
        name = 'topVocab'
    )

    # data = [trace_BoW, trace_EmotionDict, trace_topTF_IDF, trace_topVocab]
    data = [trace_EmotionDict, trace_topTF_IDF, trace_topVocab]

    layout = dict(title = graph_name + ' (all-zero vector excluded)',
              xaxis = dict(title = 'Unique Vectors, min_k=' + str(min_k)),
              yaxis = dict(title = 'K [num_vector_collisions]'),
              )

    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename=graph_name)

    # graph with 4 lines

    # sort handles according to



def Num_Lines(tweets_fileName):
    with open(tweets_fileName) as tweets_file:

        line_count = 0

        for line in tweets_file:
            line_count = line_count + 1
            print ' processing line ', line_count

    print line_count, 'lines in', tweets_fileName

def Num_UniqueHandles(handles_fileName):
    with open(handles_fileName) as handles_file:

        unique_handle_dict = {}

        line_count = 0
        for line in handles_file:
            line_count = line_count + 1
            # print 'processing line ', line_count

            line = line.strip('\n')
            line = line.strip('\\n')
            line = line.strip('\r')
            handle = line

            if handle not in unique_handle_dict:
                unique_handle_dict[handle] = line_count

    print len(unique_handle_dict), 'unqiue handles in', handles_fileName


# results inputs are in the format: [parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list]
    # parameter_value_list should be same for all inputs

    # Anomaly_Detection_LOF - n_neighbors
    # Hierarchical - cluster_num
    # K_Means - cluster_num
def Utility_Performance_forProjection_Graph( results_K_Means_exact, results_K_Means_binary, results_Hierarchical_exact, results_Hierarchical_binary, projection_name ):

    # graph  parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list for each method

    # what kind of graph do I want?
        # line? each x value is a param value, each y value a result value
            # solid line for eps
            # dashed line for percentage max entropy

    parameter_value_list_K_Means_EXACT = results_K_Means_exact[0]
    sil_score_list_K_Means_EXACT = results_K_Means_exact[1]
    percentageMaxPossilbeEntropy_list_K_Means_EXACT = results_K_Means_exact[2]

    parameter_value_list_K_Means_BINARY = results_K_Means_binary[0]
    sil_score_list_K_Means_BINARY = results_K_Means_binary[1]
    percentageMaxPossilbeEntropy_list_K_Means_BINARY = results_K_Means_binary[2]


    parameter_value_list_Hierarchical_EXACT = results_Hierarchical_exact[0]
    sil_score_list_Hierarchical_EXACT = results_Hierarchical_exact[1]
    percentageMaxPossilbeEntropy_list_Hierarchical_EXACT = results_Hierarchical_exact[2]

    parameter_value_list_Hierarchical_BINARY = results_Hierarchical_binary[0]
    sil_score_list_Hierarchical_BINARY = results_Hierarchical_binary[1]
    percentageMaxPossilbeEntropy_list_Hierarchical_BINARY = results_Hierarchical_binary[2]


    # Create and style traces
    trace_K_Means_sil_score_EXACT = go.Scatter(
        x = parameter_value_list_K_Means_EXACT,
        y = sil_score_list_K_Means_EXACT,
        name = 'exact K_Means silhouette score',
        mode = 'lines+markers',
        line = dict(
            color = ('rgb(184, 247, 212)'),
            width = 4)
    )
    trace_K_Means_percMaxEntropy_EXACT = go.Scatter(
        x = parameter_value_list_K_Means_EXACT,
        y = percentageMaxPossilbeEntropy_list_K_Means_EXACT,
        name = 'exact K_Means percentage max entropy',
        mode = 'lines+markers',
        line = dict(
            color = ('rgb(184, 247, 212)'),
            width = 4,
            dash = 'dash')
    )
    trace_K_Means_sil_score_BINARY = go.Scatter(
        x = parameter_value_list_K_Means_BINARY,
        y = sil_score_list_K_Means_BINARY,
        name = 'binary K_Means silhouette score',
        mode = 'lines+markers',
        line = dict(
            color = ('rgb(111, 231, 219)'),
            width = 4)
    )
    trace_K_Means_percMaxEntropy_BINARY = go.Scatter(
        x = parameter_value_list_K_Means_BINARY,
        y = percentageMaxPossilbeEntropy_list_K_Means_BINARY,
        name = 'binary K_Means percentage max entropy',
        mode = 'lines+markers',
        line = dict(
            color = ('rgb(111, 231, 219)'),
            width = 4,
            dash = 'dash')
    )


    trace_Hierarchical_sil_score_EXACT = go.Scatter(
        x = parameter_value_list_Hierarchical_EXACT,
        y = sil_score_list_Hierarchical_EXACT,
        name = 'exact Hierarchical silhouette score',
        mode = 'lines+markers',
        line = dict(
            color = ('rgb(127, 166, 238)'),
            width = 4)
    )
    trace_Hierarchical_percMaxEntropy_EXACT = go.Scatter(
        x = parameter_value_list_Hierarchical_EXACT,
        y = percentageMaxPossilbeEntropy_list_Hierarchical_EXACT,
        name = 'exact Hierarchical percentage max entropy',
        mode = 'lines+markers',
        line = dict(
            color = ('rgb(127, 166, 238)'),
            width = 4,
            dash = 'dash')
    )
    trace_Hierarchical_sil_score_BINARY = go.Scatter(
        x = parameter_value_list_Hierarchical_BINARY,
        y = sil_score_list_Hierarchical_BINARY,
        name = 'binary Hierarchical silhouette score',
        mode = 'lines+markers',
        line = dict(
            color = ('rgb(131, 90, 241)'),
            width = 4)
    )
    trace_Hierarchical_percMaxEntropy_BINARY = go.Scatter(
        x = parameter_value_list_Hierarchical_BINARY,
        y = percentageMaxPossilbeEntropy_list_Hierarchical_BINARY,
        name = 'binary Hierarchical percentage max entropy',
        mode = 'lines+markers',
        line = dict(
            color = ('rgb(131, 90, 241)'),
            width = 4,
            dash = 'dash')
    )
    # trace_LOF_sil_score = go.Scatter(
    #     x = month,
    #     y = high_2014,
    #     name = 'Local_Outlier_Factor silhouette score',
    #     line = dict(
    #         color = ('rgb(127, 166, 238))'),
    #         width = 4)
    # )
    # trace_LOF_percMaxEntropy = go.Scatter(
    #     x = month,
    #     y = low_2014,
    #     name = 'Local_Outlier_Factor percentage max entropy',
    #     line = dict(
    #         color = ('rgb(127, 166, 238)'),
    #         width = 4,
    #         dash = 'dash')
    # )

    data = [trace_K_Means_sil_score_EXACT, trace_K_Means_percMaxEntropy_EXACT, trace_K_Means_sil_score_BINARY, trace_K_Means_percMaxEntropy_BINARY,trace_Hierarchical_sil_score_EXACT,trace_Hierarchical_percMaxEntropy_EXACT,trace_Hierarchical_sil_score_BINARY, trace_Hierarchical_percMaxEntropy_BINARY]

    # Edit the layout
    layout = dict(title = 'Utility Metrics Compuation, ' + projection_name,
                  xaxis = dict(title = 'num_of_clusters parameter values'),
                  yaxis = dict(title = 'Utility Metrics: silhouette score, max possilbe entropy'),
                  )

    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename='Utility Computation: ' + projection_name)

def Frequent_Item_Sets_Utility_Metics(report_projection, report_BASELINE):
    # need to implement funtionality for Frequent_Item_Sets method to work on projections other than BoW

    itemsets_projection = frozenset(report_projection.items())
    itemsets_BASELINE = frozenset(report_BASELINE.items())


    # x.difference(y)  -->   returngs the items that are in x, but not in y

    # Proportion of sets absent in projection that were present in frequent item set of baseline
        # = number of sets not in itemsets_projection that are in itemsets_BASELINE / len(itemsets_projection)
    # proportion_1 = len(itemsets_BASELINE.difference(itemsets_projection)) / float(len(report_BASELINE))

    # proportion of sets that are now present in frequent item set of projection, but were not in the baseline
        # = number of sets in itemsets_projection not in itemsets_BASELINE / len(itemsets_projection)
    # proportion_2 = len(itemsets_projection.difference(itemsets_BASELINE)) / float(len(report_BASELINE))

    # the number that are present that should be present
        # = number of set in itemsets_projection that are in itemsets_BASELINE / itemsets_BASLINE

    proportion_1 = len(itemsets_projection.intersection(itemsets_BASELINE)) / float(len(report_BASELINE)) # recall
    proportion_2 = len(itemsets_projection.intersection(itemsets_BASELINE)) / float(len(report_projection)) # precision

    metric = (proportion_1 + proportion_2) / 2

    print 'Frequent Itemset metric: ', str(metric), 'num item sets: ', str(len(report_projection))
    return metric


# IMPORTANT: handle_list needs to correspond to order of labels for both inputs
def Clustering_Utility_Metrics(handle_list_projection, handle_list_BASELINE, labels_projection_list, labels_BASELINE_list):
    print


    if handle_list_projection != handle_list_BASELINE:
        print 'ERROR: handle lists are not the same for the two inputs'

    # for each cluster C
        # count number of handles in both C_BASELINE and C_projeciton_list

    clusterID_toSize_BASELINE_dict = {}
    clusterID_toCountSharedHandles_dict = {}

    unqiue_cluster_labels = []

    for label in labels_projection_list:
        if label not in unqiue_cluster_labels:
            unqiue_cluster_labels.append(label)

    for label in labels_BASELINE_list:
        if label not in unqiue_cluster_labels:
            unqiue_cluster_labels.append(label)

        if label in clusterID_toSize_BASELINE_dict:
            clusterID_toSize_BASELINE_dict[label] = clusterID_toSize_BASELINE_dict[label] + 1
        else:
            clusterID_toSize_BASELINE_dict[label] = 1

    # for label in unqiue_cluster_labels:
    for label_projection, label_BASELINE in izip(labels_projection_list, labels_BASELINE_list):

        if label_projection == label_BASELINE: # increment count of shared handles for that label

            if label_BASELINE in clusterID_toCountSharedHandles_dict:
                clusterID_toCountSharedHandles_dict[label_BASELINE] = clusterID_toCountSharedHandles_dict[label_BASELINE] + 1
            else:
                clusterID_toCountSharedHandles_dict[label_BASELINE] = 1

    metric_unNormalized = []
    # examine: clusterID not in clusterID_toCountSharedHandles_dict

    num_shared_handles = 0

    for clusterID, sizeCluster in clusterID_toSize_BASELINE_dict.iteritems():

        if clusterID in clusterID_toCountSharedHandles_dict:
            num_shared_handles = clusterID_toCountSharedHandles_dict[clusterID]
        else:
            num_shared_handles = 0


        clusterID_proportionShared = num_shared_handles / float(sizeCluster)

        metric_unNormalized.append(clusterID_proportionShared)


    metric = sum(metric_unNormalized) / len(clusterID_toSize_BASELINE_dict)
    print 'clustering utility metric: ', str(metric)

    return metric

def LOF_Utility_Metrics( handle_list_projection, handle_list_BASELINE, rankings_projection_list, rankings_BASELINE_list ):

    # kendall tau
    #  [(number of matching ranking pairs) - (number of non-matchig ranking pairs)] / [(n)(n-1)/2]


    num_nonMatching_pairs = 0
    num_matching_pairs = 0

    if handle_list_projection != handle_list_BASELINE:
        print 'ERROR: handle lists are not the same for the two inputs'


    # sort handle lists for each projection accoring to ranking lists

    # handle_list_projection_SORTED = [handle for _, handle in sorted(zip(rankings_projection_list, handle_list_projection))]

    # handle_list_BASELINE_SORTED = [handle for _, handle in sorted(zip(rankings_BASELINE_list, handle_list_BASELINE))]

    # for ranked_handle_projection, ranked_handle_BASLINE in izip(handle_list_projection_SORTED, handle_list_BASELINE_SORTED):

    #     if ranked_handle_projection == ranked_handle_BASLINE:
    #         num_matching_pairs = num_matching_pairs + 1
    #     else:
    #         num_nonMatching_pairs = num_nonMatching_pairs + 1

    kendall_tau = stats.kendalltau(rankings_projection_list, rankings_BASELINE_list)
    # n = len(handle_list_projection)
    # kendall_tau = (num_matching_pairs - num_nonMatching_pairs) / ( (n) * (n-1) / 2 )

    print 'LOF kendall_tau ', str(kendall_tau)

    return kendall_tau



def graph_similarity_k():

    # 15
    x = ['Bag of Words', 'Import 1k', 'Import 500', 'Import 100', 'Import 50', 'Import 10', 'Import 5', 'Freq 1k', 'Freq 500', 'Freq 100', 'Freq 50', 'Freq 10', 'Freq 5', 'Emotion 333', 'Emotion 5']


    # x = ['Product A', 'Product B', 'Product C']
    # y = [20, 14, 23]

    y = [1, 1, 1, 1, 1, 1, 53, 1, 1, 1, 1, 1, 1, 1, 1]

    data = [go.Bar(
                x=x,
                y=y,
                text=y,
                textposition = 'auto',
                marker=dict(
                    color='rgb(158,202,225)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                )
            )]



    layout = dict(title = 'k for Least Private Invidual(s) Across Projections',
          xaxis = dict(title = ''),
          yaxis = dict(title = 'Lowest k'),
          )

    fig = dict(data=data, layout=layout)

    py.iplot(fig, filename='k-bar-labels')


def graph_similarity_privacyMetric():
    x = ['Bag of Words', 'Import 1k', 'Import 500', 'Import 100', 'Import 50', 'Import 10', 'Import 5', 'Freq 1k', 'Freq 500', 'Freq 100', 'Freq 50', 'Freq 10', 'Freq 5', 'Emotion 333', 'Emotion 5']


    # x = ['Product A', 'Product B', 'Product C']
    # y = [20, 14, 23]

    y = [0.010, 0.171, 0.215, 0.624, 0.872, 0.975, 1.0, 0.026, 0.037, 0.362, .500, 0.988, 0.999, 0.186, 0.999]

    data = [go.Bar(
                x=x,
                y=y,
                text=y,
                textposition = 'auto',
                marker=dict(
                    color='rgb(158,202,225)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                )
            )]

    layout = dict(title = 'Proportion of Private Handles Across Projections',
              xaxis = dict(title = ''),
              yaxis = dict(title = 'Proportion of Handles with k >= 2'),
              )

    fig = dict(data=data, layout=layout)

    py.iplot(fig, filename='utility-bar-labels')


def graph_utility_privacy(task):

    # 6 elements
    trace_clustering_import_UTILITY = [0.241, 0.160, 0.170, 0.164, 0.165, 0.167]
    trace_itemsets_import_UTILITY = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    trace_lof_import_UTILITY = [  -0.003 ,0.011, 0.012, 0.003, 0.004, 0.006]
    trace_import_PRIVACY = [0.171, 0.215, 0.624, 0.872, 0.975, 1.0]

    # 6 elements
    trace_clustering_freq_UTILITY = [0.180, 0.162, 0.174, 0.166, 0.162, 0.179]
    trace_itemsets_freq_UTILITY = [ 0.814, 0.542, 0.581, 0.539, 0.366, 0.125]
    trace_lof_freq_UTILITY = [ 0.004, 0.002, 0.004, 0.002, 0.002, 0.021]
    trace_freq_PRIVACY = [0.026, 0.037, 0.362, .500, 0.988, 0.999]

    # 2 elements
    trace_clustering_emotion_UTILITY = [0.162, 0.164 ]
    trace_itemsets_emotion_UTILITY = [ 0.0, 0.0 ]
    trace_lof_emotion_UTILITY = [ 0.014, 0.006 ]
    trace_emotion_PRIVACY = [0.186, 0.999]

    # # Create and style traces
    # trace_clustering_bow = go.Scatter(
    #     x = month,
    #     y = high_2014,
    #     name = 'bow clustering',
    #     line = dict(
    #         color = ('rgb(205, 12, 24)'),
    #         width = 4)
    # )
    # trace_itemsets_bow = go.Scatter(
    #     x = month,
    #     y = low_2014,
    #     name = 'bow itemsets',
    #     line = dict(
    #         color = ('rgb(22, 96, 167)'),
    #         width = 4,
    #         dash = 'dot')
    # )
    # trace_lof_bow = go.Scatter(
    #     x = month,
    #     y = high_2007,
    #     name = 'bow lof',
    #     line = dict(
    #         color = ('rgb(205, 12, 24)'),
    #         width = 4,
    #         dash = 'dash') # dash options include 'dash', 'dot', and 'dashdot'
    # )


    trace_clustering_import = go.Scatter(
        x = trace_clustering_import_UTILITY,
        y = trace_import_PRIVACY,
        name = 'import clustering',
        mode='lines+markers+text',
        text=['  1000 features', '  500 features', '  100 features', '  50 features', '  10 features', '  5 features'],
        textposition='right',
        line = dict(
            color = ('rgb(31,120,180)'),
            width = 4)
    )

    trace_itemsets_import = go.Scatter(
        x = trace_itemsets_import_UTILITY,
        y = trace_import_PRIVACY,
        name = 'import itemsets',
        mode='lines+markers+text',
        text=['  1000 features', '  500 features', '  100 features', '  50 features', '  10 features', '  5 features'],
        textposition='top right',
        line = dict(
            color = ('rgb(31,120,180)'),
            width = 4,
            dash = 'dot')
    )

    trace_lof_import = go.Scatter(
        x = trace_lof_import_UTILITY,
        y = trace_import_PRIVACY,
        name = 'import lof',
        mode='lines+markers+text',
        text=['  1000 features', '  500 features', '  100 features', '  50 features', '  10 features', '  5 features'],
        textposition='right',
        line = dict(
            color = ('rgb(31,120,180)'),
            width = 4,
            dash = 'dash') # dash options include 'dash', 'dot', and 'dashdot'
    )


    trace_clustering_freq = go.Scatter(
        x = trace_clustering_freq_UTILITY,
        y = trace_freq_PRIVACY,
        name = 'freq clustering',
        mode='lines+markers+text',
        text=['  1000 features', '  500 features', '  100 features', '  50 features', '  10 features', '  5 features'],
        textposition='right',
        line = dict(
            color = ('rgb(51,160,44)'),
            width = 4)
    )

    trace_itemsets_freq = go.Scatter(
        x = trace_itemsets_freq_UTILITY,
        y = trace_freq_PRIVACY,
        name = 'freq itemsets',
        mode='lines+markers+text',
        text=['  1000 features', '  500 features', '  100 features', '  50 features', '  10 features', '  5 features'],
        textposition='top right',
        line = dict(
            color = ('rgb(51,160,44)'),
            width = 4,
            dash = 'dot')
    )

    trace_lof_freq = go.Scatter(
        x = trace_lof_freq_UTILITY,
        y = trace_freq_PRIVACY,
        name = 'freq lof',
        mode='lines+markers+text',
        text=['  1000 features', '  500 features', '  100 features', '  50 features', '  10 features', '  5 features'],
        textposition='right',
        line = dict(
            color = ('rgb(51,160,44)'),
            width = 4,
            dash = 'dash') # dash options include 'dash', 'dot', and 'dashdot'
    )


    trace_clustering_emotion = go.Scatter(
        x = trace_clustering_emotion_UTILITY,
        y = trace_emotion_PRIVACY,
        name = 'emotion clustering',
        mode='lines+markers+text',
        text=['333 features  ', '5 features  '],
        textposition='left',
        line = dict(
            color = ('rgb(227,26,28)'),
            width = 4)
    )

    trace_itemsets_emotion = go.Scatter(
        x = trace_itemsets_emotion_UTILITY,
        y = trace_emotion_PRIVACY,
        name = 'emotion itemsets',
        mode='lines+markers+text',
        text=['333 features  ', '5 features'  ],
        textposition='top left',
        line = dict(
            color = ('rgb(227,26,28)'),
            width = 4,
            dash = 'dot')
    )

    trace_lof_emotion = go.Scatter(
        x = trace_lof_emotion_UTILITY,
        y = trace_emotion_PRIVACY,
        name = 'emotion lof',
        mode='lines+markers+text',
        text=['333 features  ', '5 features  '],
        textposition='left',
        line = dict(
            color = ('rgb(227,26,28)'),
            width = 4,
            dash = 'dash') # dash options include 'dash', 'dot', and 'dashdot'
    )

    # data = [trace_clustering_import, trace_itemsets_import, trace_lof_import, trace_clustering_freq, trace_itemsets_freq, trace_lof_freq, trace_clustering_emotion, trace_itemsets_emotion, trace_lof_emotion]

    data = []

    if task == 'clustering':
        data = [trace_clustering_import, trace_clustering_freq, trace_clustering_emotion]
    elif task == 'frequent itemsets':
        data = [trace_itemsets_import, trace_itemsets_freq, trace_itemsets_emotion]
    elif task == 'anomaly detection':
        data = [trace_lof_import, trace_lof_freq, trace_lof_emotion]

    # if projection == 'Importance':
    #     data = [trace_clustering_import, trace_itemsets_import, trace_lof_import]
    # elif projection == 'Frequency':
    #     data = [trace_clustering_freq, trace_clustering_freq, trace_lof_freq]
    # elif projection == 'Emotion':
    #     data = [trace_clustering_emotion, trace_itemsets_emotion, trace_lof_emotion]

    title = 'Privacy Utility Tradeoffs, ' + task + ' task'
    file_name = 'privacy-utility-tradeoff ' + task

    # title = 'Privacy Utility Tradeoffs, ' + projection + ' task'
    # file_name = 'privacy-utility-tradeoff ' + projection
    # Edit the layout
    layout = dict(title = title,
                  xaxis = dict(title = 'Utility Retained From Baseline',
                               range=[0, 0.1]),
                  yaxis = dict(title = 'Proportion of Handles with k >= 2'),
                  )

    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename=file_name)


if __name__ == '__main__':



    # task = 'clustering'
    # task = 'frequent itemsets'
    task = 'anomaly detection'

    # projection = 'Emotion'
    # projection = 'Frequency'
    # projection = 'Importance'

    graph_utility_privacy(task)

    pdb.set_trace()

    # graph_similarity_k()
    # graph_similarity_privacyMetric()

    # pdb.set_trace()

    # WHY is data set not larger? Examine tweet_processing

    dataSet_size = 4000000
    # tweet_representation_fileName = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/tweets_MatrixRepresentation_topVocab.csv'
    # filename_handles = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/handles' + tweet_setType + str(dataSet_size) + '.txt'


# SPECIFY TYPE OF TWEETS HERE
    tweet_setType = ''
    # tweet_setType = 'original'
    # tweet_setType = 'retweet'

    tf_idf_wordList_filename = 'tf_idf_wordList.txt'
    uncleaned_tweets_filename = 'tweets.txt'

    cleanedTweets_filename = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/clean_tweets_' + tweet_setType + str(dataSet_size) + '.txt'
    filename_handles = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/handles' + tweet_setType + str(dataSet_size) + '.txt'
    if tweet_setType == '':
        filename_handles = 'handles.txt'
    if tweet_setType == '':
        cleanedTweets_filename = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/clean_tweets_' + str(dataSet_size) + '.txt'

    # tweet_representation_fileName = 'tweets_MatrixRepresentation_topVocab_original4000000.csv'
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_topVocab_original4000000_summed.txt'
    # tweet_representation_fileName_summed = 'clean_tweets_4000000_BagOfWords_summed_wordCounts.txt'

    # BagOfWords
    # tweet_representation_fileName_summed = 'clean_tweets_' + tweet_setType + '4000000_BagOfWords_summed.txt'
    # tweet_representation_fileName = 'clean_tweets_' + tweet_setType + '4000000_BagOfWords_summed.txt'
    tweet_representation_fileName_sparse = 'handle_vectors_BagOfWords_sparse.dat'


    # param_value_list = [6,8,10]

    # importance
    # projection_name = 'binary Top1k_TF_IDF'
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_topTF_IDF_' + tweet_setType + '4000000_summed.txt'
    # tweet_representation_fileName = 'tweets_MatrixRepresentation_topTF_IDF_' + tweet_setType + '4000000.csv'
    # emot = False

    # num_features_ToKeep_list = [500, 100, 50, 10]

    # labels_BASELINE_list, handle_list_BASELINE = Clustering_K_Means_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, num_clusters = 6, param_testing = False, binary=False)
    # rankings_BASELINE_list, handle_list_BASELINE = Anomaly_Detection_LOF_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, n_neighbors = 20, param_testing = False, binary=True )
    # report_BASELINE = Frequent_ItemSets(cleanedTweets_filename, tweet_representation_fileName, projection_input=False, emotion=emot, num_features_ToKeep = 1000 )

    # for num_features_ToKeep in num_features_ToKeep_list:
    #     print projection_name, ' num features kept: ', str(num_features_ToKeep)
    #     # frequent item set mining
    #     report_projection = Frequent_ItemSets(cleanedTweets_filename, tweet_representation_fileName, projection_input=True, emotion=emot, sup_div = 100, num_features_ToKeep = num_features_ToKeep )
    #     Frequent_Item_Sets_Utility_Metics(report_projection, report_BASELINE)

        # # clustering
        # labels_projection_list, handle_list_projection = Clustering_K_Means ( tweet_representation_fileName_summed, param_value_list, num_clusters = 6, param_testing = False, binary=True, num_features_ToKeep=num_features_ToKeep )
        # Clustering_Utility_Metrics(handle_list_projection, handle_list_BASELINE, labels_projection_list, labels_BASELINE_list)

        # # abnormality detection
        # rankings_projection_list, handle_list_projection = Anomaly_Detection_LOF ( tweet_representation_fileName_summed, param_value_list, n_neighbors = 20, param_testing=False, binary=True, num_features_ToKeep = num_features_ToKeep )
        # LOF_Utility_Metrics( handle_list_projection, handle_list_BASELINE, rankings_projection_list, rankings_BASELINE_list )


    # frequency
    # projection_name = 'binary Top1k_Vocab'
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_topVocab_' + tweet_setType + '4000000_summed.txt'
    # tweet_representation_fileName = 'tweets_MatrixRepresentation_topVocab_' + tweet_setType + '4000000.csv'
    # emot = False

    # num_features_ToKeep_list = [500, 100, 50, 10]

    # rankings_BASELINE_list, handle_list_BASELINE = Anomaly_Detection_LOF_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, n_neighbors = 20, param_testing = False, binary=True )
    # labels_BASELINE_list, handle_list_BASELINE = Clustering_K_Means_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, num_clusters = 6, param_testing = False, binary=False)
    # report_BASELINE = Frequent_ItemSets(cleanedTweets_filename, tweet_representation_fileName, projection_input=False, emotion=emot, num_features_ToKeep = 1000 )

    # for num_features_ToKeep in num_features_ToKeep_list:
    #     print projection_name, ' num features kept: ', str(num_features_ToKeep)

        # frequent item set mining
        # report_projection = Frequent_ItemSets(cleanedTweets_filename, tweet_representation_fileName, projection_input=True, emotion=emot, sup_div = 100, num_features_ToKeep = num_features_ToKeep )
        # Frequent_Item_Sets_Utility_Metics(report_projection, report_BASELINE)

        # # clustering
        # labels_projection_list, handle_list_projection = Clustering_K_Means ( tweet_representation_fileName_summed, param_value_list, num_clusters = 6, param_testing = False, binary=True, num_features_ToKeep=num_features_ToKeep )
        # Clustering_Utility_Metrics(handle_list_projection, handle_list_BASELINE, labels_projection_list, labels_BASELINE_list)

        # # abnormality detection
        # rankings_projection_list, handle_list_projection = Anomaly_Detection_LOF ( tweet_representation_fileName_summed, param_value_list, n_neighbors = 20, param_testing=False, binary=True, num_features_ToKeep = num_features_ToKeep )
        # LOF_Utility_Metrics( handle_list_projection, handle_list_BASELINE, rankings_projection_list, rankings_BASELINE_list )


    # # emotion


    # projection_name = 'binary Emotion_333'
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_emotionDictionary_' + tweet_setType + '4000000_summed.txt'
    # tweet_representation_fileName = 'tweets_MatrixRepresentation_emotionDictionary_' + tweet_setType + '4000000.csv'
    # emot = True

    # num_features_ToKeep = 333

    # print projection_name, ' num features kept: ', str(num_features_ToKeep)

    # frequent item set mining
    # report_projection = Frequent_ItemSets(cleanedTweets_filename, tweet_representation_fileName, projection_input=True, emotion=emot, sup_div = 100, num_features_ToKeep = num_features_ToKeep )
    # report_BASELINE = Frequent_ItemSets(cleanedTweets_filename, tweet_representation_fileName, projection_input=False, emotion=emot, num_features_ToKeep = 1000 )
    # Frequent_Item_Sets_Utility_Metics(report_projection, report_BASELINE)

    # # clustering
    # labels_BASELINE_list, handle_list_BASELINE = Clustering_K_Means_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, num_clusters = 6, param_testing = False, binary=False)
    # labels_projection_list, handle_list_projection = Clustering_K_Means ( tweet_representation_fileName_summed, param_value_list, num_clusters = 6, param_testing = False, binary=True, num_features_ToKeep=num_features_ToKeep )
    # Clustering_Utility_Metrics(handle_list_projection, handle_list_BASELINE, labels_projection_list, labels_BASELINE_list)

    # # abnormality detection
    # rankings_projection_list, handle_list_projection = Anomaly_Detection_LOF ( tweet_representation_fileName_summed, param_value_list, n_neighbors = 20, param_testing=False, binary=True, num_features_ToKeep = num_features_ToKeep )
    # rankings_BASELINE_list, handle_list_BASELINE = Anomaly_Detection_LOF_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, n_neighbors = 20, param_testing = False, binary=True )
    # LOF_Utility_Metrics( handle_list_projection, handle_list_BASELINE, rankings_projection_list, rankings_BASELINE_list )




    # projection_name = 'binary EmotionDict_5'
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_emotionDictionary5_4000000_summed_5.txt'
    # tweet_representation_fileName = 'tweets_MatrixRepresentation_emotionDictionary5_4000000.csv'
    # emot = True

    # print projection_name, ' num features kept: ', str(num_features_ToKeep)

    # frequent item set mining
    # report_projection = Frequent_ItemSets(cleanedTweets_filename, tweet_representation_fileName, projection_input=True, emotion=emot, sup_div = 100, num_features_ToKeep = num_features_ToKeep )
    # report_BASELINE = Frequent_ItemSets(cleanedTweets_filename, tweet_representation_fileName, projection_input=False, nemotion=emot, um_features_ToKeep = 1000 )
    # Frequent_Item_Sets_Utility_Metics(report_projection, report_BASELINE)

    # clustering
    # labels_BASELINE_list, handle_list_BASELINE = Clustering_K_Means_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, num_clusters = 6, param_testing = False, binary=False)
    # labels_projection_list, handle_list_projection = Clustering_K_Means ( tweet_representation_fileName_summed, param_value_list, num_clusters = 6, param_testing = False, binary=True, num_features_ToKeep=num_features_ToKeep )
    # Clustering_Utility_Metrics(handle_list_projection, handle_list_BASELINE, labels_projection_list, labels_BASELINE_list)

    # # abnormality detection
    # rankings_projection_list, handle_list_projection = Anomaly_Detection_LOF ( tweet_representation_fileName_summed, param_value_list, n_neighbors = 20, param_testing=False, binary=True, num_features_ToKeep = num_features_ToKeep )
    # rankings_BASELINE_list, handle_list_BASELINE = Anomaly_Detection_LOF_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, n_neighbors = 20, param_testing = False, binary=True )
    # LOF_Utility_Metrics( handle_list_projection, handle_list_BASELINE, rankings_projection_list, rankings_BASELINE_list )


    # projection_name = 'binary Bag of Words'
    # print projection_name

    # Handle_Uniqueness_Metrics_BOW('clean_tweets_4000000_BagOfWords_summed_wordLists.txt', projection_name, binary=True)

    # # pdb.set_trace()
    # projection_name = 'exact Bag of Words'

    # print projection_name
    # Handle_Uniqueness_Metrics_BOW('clean_tweets_4000000_BagOfWords_summed_wordLists.txt', projection_name, binary=False)

    # pdb.set_trace()
    # num_features_ToKeep = 1000


    #  topVocab
    # projection_name = 'exact Top1k_Vocab'
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_topVocab_' + tweet_setType + '4000000_summed.txt'
    # tweet_representation_fileName = 'tweets_MatrixRepresentation_topVocab_' + tweet_setType + '4000000.csv'
    # emot = False

    # topVocab
    # projection_name = 'binary Top1k_Vocab'
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_topVocab_' + tweet_setType + '4000000_summed.txt'
    # tweet_representation_fileName = 'tweets_MatrixRepresentation_topVocab_' + tweet_setType + '4000000.csv'
    # emot = False

    # topTF_IDF
    # projection_name = 'exact Top1k_TF_IDF'
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_topTF_IDF_' + tweet_setType + '4000000_summed.txt'
    # tweet_representation_fileName = 'tweets_MatrixRepresentation_topTF_IDF_' + tweet_setType + '4000000.csv'
    # emot = False

    # topTF_IDF
    projection_name = 'binary Top1k_TF_IDF'
    tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_topTF_IDF_' + tweet_setType + '4000000_summed.txt'
    tweet_representation_fileName = 'tweets_MatrixRepresentation_topTF_IDF_' + tweet_setType + '4000000.csv'
    emot = False

    # Emotion
    # projection_name = 'binary Emotion_333'
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_emotionDictionary_' + tweet_setType + '4000000_summed.txt'
    # tweet_representation_fileName = 'tweets_MatrixRepresentation_emotionDictionary_' + tweet_setType + '4000000.csv'
    # emot = True


    # projection_name = 'binary EmotionDict_5'
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_emotionDictionary5_4000000_summed_5.txt'
    # tweet_representation_fileName = 'tweets_MatrixRepresentation_emotionDictionary5_4000000.csv'
    # emot = True

    # projection_name = 'exact EmotionDict_5'
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_emotionDictionary5_4000000_summed_5.txt'
    # tweet_representation_fileName = 'tweets_MatrixRepresentation_emotionDictionary5_4000000.csv'
    # emot = True

    # print projection_name

    # tf - idf --> 1000, 500, 100, 50, 10, 5
    # top vocab -->

    # kept_features_num_list = [1000, 500, 100, 50, 10, 5]

    # binary, exact
    # full, smallest reduced
    # get BOW too


    num_features_ToKeep = 5

    print 'num_features_ToKeep: ', str(num_features_ToKeep)
    Handle_Uniqueness_Metrics(tweet_representation_fileName_summed, filename_handles, projection_name, binary=True, bag_of_words=False, reduced_runtime=True,tweet_freq_type='', num_features_ToKeep=num_features_ToKeep)

    pdb.set_trace()
    # for num_features_ToKeep in kept_features_num_list:
    #     print 'num_features_ToKeep: ', str(num_features_ToKeep)
    #     print 'exact'
    #     Handle_Uniqueness_Metrics(tweet_representation_fileName_summed, filename_handles, projection_name, binary=False, bag_of_words=False, reduced_runtime=True,tweet_freq_type='', num_features_ToKeep=num_features_ToKeep)

    # num_features_ToKeep = 5

    # pdb.set_trace()

    # Distortion_Metric ('clean_tweets_retweet4000000_BagOfWords_summed_wordLists.txt', tweet_representation_fileName_summed, tweet_representation_fileName, emot=emot, num_features_ToKeep = num_features_ToKeep)

    # pdb.set_trace()
    # print tweet_representation_fileName

    # report_projection = Frequent_ItemSets(cleanedTweets_filename, tweet_representation_fileName, projection_input=True, emotion= emot, sup_div = 5000)


    # metric = Frequent_Item_Sets_Utility_Metics(report_projection, report_BASELINE)
    # print

    # report_projection = Frequent_ItemSets(cleanedTweets_filename, tweet_representation_fileName, projection_input=True, emotion= emot, sup_div = 3000)


    # metric = Frequent_Item_Sets_Utility_Metics(report_projection, report_BASELINE)

    # projection_name = 'Emotion_333'
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_emotionDictionary_' + tweet_setType + '4000000_summed.txt'
    # tweet_representation_fileName = 'tweets_MatrixRepresentation_emotionDictionary_' + tweet_setType + '4000000.csv'
    # emot = True

    # num_features_ToKeep = 10

    # print projection_name

    # report_projection = Frequent_ItemSets(cleanedTweets_filename, tweet_representation_fileName, projection_input=True, emotion= emot, num_features_ToKeep = num_features_ToKeep )
    # report_BASELINE = Frequent_ItemSets(cleanedTweets_filename, tweet_representation_fileName, projection_input=False, emotion = False, num_features_ToKeep = 1000 )

    # metric = Frequent_Item_Sets_Utility_Metics(report_projection, report_BASELINE)
    # print

    # pdb.set_trace()

    # pdb.set_trace()
    # print projection_name


    # Apriori_AssociationRules( cleaned_tweets_fileName )

    # Zero_Incidence_Metrics( cleanedTweets_filename, filename_handles, tweet_representation_fileName_summed, bag_of_words=True, tweet_freq_type='' )

    # isHashtag(uncleaned_tweets_filename, tf_idf_wordList_filename, dataSet_size)

    # Aggregate_Similarity(tweet_representation_fileName_summed)

    # Clustering_K_Means ( tweet_representation_fileName_summed, num_clusters = 2, param_testing = True, binary=False )
    # Clustering_K_Means_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, num_clusters = 2, param_testing = True, binary=False )

    # Clustering_Hierarchical( tweet_representation_fileName_summed, num_clusters = 2, param_testing = True, binary=False )
    # Clustering_Hierarchical_2( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, num_clusters = 2, param_testing = True, binary=False )

    # Clustering_DB_Scan(tweet_representation_fileName_summed, param_testing = True, binary=False)
    # Clustering_DB_Scan_2( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, param_testing = True, binary=False)

    # num_features_ToKeep = 5


    # report_projection = Frequent_ItemSets(cleanedTweets_filename, tweet_representation_fileName, projection_input=True, num_features_ToKeep = num_features_ToKeep )
    # report_BASELINE = Frequent_ItemSets(cleanedTweets_filename, tweet_representation_fileName, projection_input=False, num_features_ToKeep = 1000 )

    # metric = Frequent_Item_Sets_Utility_Metics(report_projection, report_BASELINE)

    # pdb.set_trace()


    # param_value_list = [5,10,20,40,100,200]
    # param_value_list = [6,8,10]
    param_value_list = [5,6,7,8,9,10,15,20,30,40,50]


    num_features_ToKeep = 5

    # print 'exact'
    # labels_BASELINE_list, handle_list_BASELINE = Clustering_K_Means_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, num_clusters = 8, param_testing = False, binary=False, num_features_ToKeep=5 )
    # labels_projection_list, handle_list_projection = Clustering_K_Means ( tweet_representation_fileName_summed, param_value_list, num_clusters = 8, param_testing = False, binary=False )

    # print 'binary'
    # labels_BASELINE_list, handle_list_BASELINE = Clustering_K_Means_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, num_clusters = 6, param_testing = False, binary=False)
    # labels_projection_list, handle_list_projection = Clustering_K_Means ( tweet_representation_fileName_summed, param_value_list, num_clusters = 6, param_testing = False, binary=True, num_features_ToKeep=num_features_ToKeep )

    # print 'exact'
    # labels_BASELINE_list, handle_list_BASELINE = Clustering_Hierarchical_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, num_clusters = 9, param_testing = True, binary=False )
    # labels_projection_list, handle_list_projection = Clustering_Hierarchical ( tweet_representation_fileName_summed, param_value_list, num_clusters = 9, param_testing = False, binary=False )

    # print 'binary'
    # labels_BASELINE_list, handle_list_BASELINE = Clustering_Hierarchical_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, num_clusters = 8, param_testing = True, binary=True )
    # labels_projection_list, handle_list_projection = Clustering_Hierarchical ( tweet_representation_fileName_summed, param_value_list, num_clusters = 8, param_testing = False, binary=True )

    # labels_BASELINE_list, handle_list_BASELINE = Clustering_Hierarchical_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, num_clusters = 9, param_testing = True, binary=True )

    # Clustering_Utility_Metrics(handle_list_projection, handle_list_BASELINE, labels_projection_list, labels_BASELINE_list)

    # pdb.set_trace()

    # print 'exact'
    # rankings_projection_list, handle_list_projection = Anomaly_Detection_LOF ( tweet_representation_fileName_summed, param_value_list, n_neighbors=20, param_testing=False, binary=False )
    # rankings_BASELINE_list, handle_list_BASELINE = Anomaly_Detection_LOF_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, n_neighbors = 20, param_testing = False, binary=False )

    # print 'binary'
    # rankings_projection_list, handle_list_projection = Anomaly_Detection_LOF ( tweet_representation_fileName_summed, param_value_list, n_neighbors = 20, param_testing=False, binary=True, num_features_ToKeep = num_features_ToKeep )
    # rankings_BASELINE_list, handle_list_BASELINE = Anomaly_Detection_LOF_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, n_neighbors = 20, param_testing = False, binary=True )

    # metric = LOF_Utility_Metrics( handle_list_projection, handle_list_BASELINE, rankings_projection_list, rankings_BASELINE_list )

    pdb.set_trace()

    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_topTF_IDF10_4000000_summed_10.txt'

    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_emotionDictionary5_4000000_summed_5.txt'

    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_topVocab10_4000000_summed_10.txt'


    # print 'binary'
    # Handle_Uniqueness_Metrics(tweet_representation_fileName_summed, filename_handles, binary=True, bag_of_words=False, reduced_runtime=True,tweet_freq_type='')

    # pdb.set_trace()
    # pdb.set_trace()
    # that was for non binary; what about binary

    # Num_UniqueHandles(filename_handles)

    # Num_Lines('/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/clean_tweets_4000000.txt')



    # if Bag_of_Words, tweet_representation_fileName should be one for one of the others, such as Top1k_TF_IDF

    # Zero_Incidence_Metrics_Graph( tweet_representation_fileName, filename_handles, tweet_representation_fileName_summed, graph_name=projection_name)

    # Graph_Similarity()


    # binary Bag_of_Words
    # projection_name = 'binary Bag_of_Words'
    # BINARY_similarity_projection_file_name = 'clean_tweets_4000000_BagOfWords_summed_Uniqueness_Metrics_BINARY.csv'

    # Graph_HandleEquivalancyClassSize_Distributions_SingleProjection( BINARY_similarity_projection_file_name, min_num_handles_same_vector=0, projection_name=projection_name)

    # # binary Emotion_306
    # projection_name = 'binary Emotion_333'
    # BINARY_similarity_projection_file_name = 'tweets_MatrixRepresentation_emotionDictionary_4000000_summed_Uniqueness_Metrics_BINARY.csv'

    # Graph_HandleEquivalancyClassSize_Distributions_SingleProjection( BINARY_similarity_projection_file_name, min_num_handles_same_vector=0, projection_name=projection_name)

    # # binary Top1k_Vocab
    # projection_name = 'binary Top1k_Vocab'
    # BINARY_similarity_projection_file_name = 'tweets_MatrixRepresentation_topVocab_4000000_summed_Uniqueness_Metrics_BINARY.csv'

    # Graph_HandleEquivalancyClassSize_Distributions_SingleProjection( BINARY_similarity_projection_file_name, min_num_handles_same_vector=0, projection_name=projection_name)

    # # binary Top1k_TF_IDF
    # projection_name = 'binary Top1k_TF_IDF'
    # BINARY_similarity_projection_file_name = 'tweets_MatrixRepresentation_topTF_IDF_4000000_summed_Uniqueness_Metrics_BINARY.csv'

    # Graph_HandleEquivalancyClassSize_Distributions_SingleProjection( BINARY_similarity_projection_file_name, min_num_handles_same_vector=0, projection_name=projection_name)


    # exact Bag_of_Words
    # projection_name = 'exact Bag_of_Words'
    # EXACT_similarity_projection_file_name = 'clean_tweets_4000000_BagOfWords_summed_Uniqueness_Metrics_EXACT.csv'

    # projection_name = 'exact Emotion_333'
    # EXACT_similarity_projection_file_name = 'tweets_MatrixRepresentation_emotionDictionary_4000000_summed_Uniqueness_Metrics_EXACT.csv'

    # projection_name = 'exact Top1k_Vocab'
    # EXACT_similarity_projection_file_name = 'tweets_MatrixRepresentation_topVocab_4000000_summed_Uniqueness_Metrics_EXACT.csv'

    # projection_name = 'exact Top1k_TF_IDF'
    # EXACT_similarity_projection_file_name = 'tweets_MatrixRepresentation_topTF_IDF_4000000_summed_Uniqueness_Metrics_EXACT.csv'



    projection_name = 'binary Top5_Vocab'
    BINARY_similarity_projection_file_name = 'tweets_MatrixRepresentation_topVocab10_4000000_summed_5_Uniqueness_Metrics_BINARY.csv'

    # projection_name = 'binary Top10_TF_IDF'
    # BINARY_similarity_projection_file_name = 'tweets_MatrixRepresentation_topTF_IDF10_4000000_summed_10_Uniqueness_Metrics_BINARY.csv'

    # projection_name = 'binary Emotion_Dict_5'
    # BINARY_similarity_projection_file_name = 'tweets_MatrixRepresentation_emotionDictionary5_4000000_summed_5_Uniqueness_Metrics_BINARY.csv'

    Graph_HandleEquivalancyClassSize_Distributions_SingleProjection( BINARY_similarity_projection_file_name, min_num_handles_same_vector=0, projection_name=projection_name)
    # pdb.set_trace()
    # Graph_CollisionsOnCollisionVectors_Distributions_SingleProjection( EXACT_similarity_projection_file_name, min_k=1, projection_name=projection_name)
    pdb.set_trace()



    # ~~~ Utility Metrics Computation, Graphing ~~~
    param_value_list = [2, 3, 4, 5, 6, 8, 10]
    # param_value_list = [2, 3]

    # projection_name = 'binary Top1k_TF_IDF'
    # projection_name = 'binary Top1k_Vocab'
    # projection_name = 'binary Emotion_306'
    # projection_name = 'binary Bag_of_Words'

    # projection_name = 'exact Top1k_TF_IDF'
    # projection_name = 'exact Top1k_Vocab'
    # projection_name = 'exact Emotion_306'
    # projection_name = 'exact Bag_of_Words'



    # BagOfWords
    tweet_representation_fileName_summed = 'clean_tweets_' + tweet_setType + '4000000_BagOfWords_summed.txt'
    tweet_representation_fileName = 'clean_tweets_' + tweet_setType + '4000000_BagOfWords_summed.txt'
    tweet_representation_fileName_sparse = 'handle_vectors_BagOfWords_sparse.dat'
    projection_name = 'Bag_of_Words'

    # topVocab
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_topVocab_' + tweet_setType + '4000000_summed.txt'
    # tweet_representation_fileName = 'tweets_MatrixRepresentation_topVocab_' + tweet_setType + '4000000.csv'
    # projection_name = 'Top1k_Vocab'

    # topTF_IDF
    tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_topTF_IDF_' + tweet_setType + '4000000_summed.txt'
    tweet_representation_fileName = 'tweets_MatrixRepresentation_topTF_IDF_' + tweet_setType + '4000000.csv'
    projection_name = 'Top1k_TF_IDF'

    # Emotion_333
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_emotionDictionary_' + tweet_setType + '4000000_summed.txt'
    # tweet_representation_fileName = 'tweets_MatrixRepresentation_emotionDictionary_' + tweet_setType + '4000000.csv'
    # projection_name = 'Emotion_333'





     # NOTE: if bag of words, use <function>_2()

    # print tweet_representation_fileName_summed

    # parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list = Clustering_K_Means ( tweet_representation_fileName_summed, param_value_list, num_clusters = 2, param_testing = True, binary=False )
    # parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list = Clustering_K_Means_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, num_clusters = 2, param_testing = True, binary=False )
    # results_K_Means_exact = [parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list]

    # parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list = Clustering_K_Means ( tweet_representation_fileName_summed, param_value_list, num_clusters = 2, param_testing = True, binary=True )
    # parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list = Clustering_K_Means_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, num_clusters = 2, param_testing = True, binary=True )
    # results_K_Means_binary = [parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list]


    # parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list = Clustering_Hierarchical( tweet_representation_fileName_summed, param_value_list, num_clusters = 2, param_testing = True, binary=False )
    # parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list = Clustering_Hierarchical_2( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, num_clusters = 2, param_testing = True, binary=True )
    # results_Hierarchical_exact = [parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list]

    # parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list = Clustering_Hierarchical( tweet_representation_fileName_summed, param_value_list, num_clusters = 2, param_testing = True, binary=True )
    # parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list = Clustering_Hierarchical_2( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, num_clusters = 2, param_testing = True, binary=True )
    # results_Hierarchical_binary = [parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list]


    # parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list = Anomaly_Detection_LOF ( summed_representation_fileName, n_neighbors=20 parameter_value_list, param_testing=True )
    parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list = Anomaly_Detection_LOF_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, n_neighbors = 20, param_testing = False, binary=False )
    # projection_name = [parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list]

    # parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list = Anomaly_Detection_LOF ( summed_representation_fileName, parameter_value_list, param_testing=True )
    # parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list = Anomaly_Detection_LOF_2 ( tweet_representation_fileName_sparse, tweet_representation_fileName_summed, param_value_list, n_neighbors = 20, param_testing = False, binary=True )
    # projection_name = [parameter_value_list, sil_score_list, percentageMaxPossilbeEntropy_list]

    # do: incorporate binary evaluation into graph

    # Utility_Performance_forProjection_Graph( results_K_Means_exact, results_K_Means_binary, results_Hierarchical_exact, results_Hierarchical_binary, projection_name )



