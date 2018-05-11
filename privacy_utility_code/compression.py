import operator
import pdb
import csv
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from itertools import izip
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import pywt

import plotly.plotly as py
import plotly.graph_objs as go

import ast

from scipy.sparse import csr_matrix
import cPickle







# filename_top1kWords_perUser = '/home/kr623/top_1k_words_perHandle.txt'
# filename_emotionDictHits_perHandle =

def save_pickle(matrix, filename):
    with open(filename, 'wb') as outfile:
        cPickle.dump(matrix, outfile, cPickle.HIGHEST_PROTOCOL)

def word_tf(word, document):
    if isinstance(document, basestring):
        document = tokenize(document)

    return float(document.count(word)) / len(document)

def tf_idf(word, document):
    # If not tokenized
    if isinstance(document, basestring):
        document = tokenize(document)

    if word not in word_index:
        return .0

    return word_tf(word, document) * word_idf[word_index[word]]


def count ( specified_word, document ):
    # where document is a list of words

    count = 0

    for word in document:
        if specified_word == word:
            count = count + 1

    return count

def csr_vappend(a,b):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one.
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""

    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    return a


def Aquire_HandlePoints ( cleanedTweets_filename, filename_handles):

    # handle --> [count for each of the feature in the feature space ]
    handle_toTweetsDict = {}
    unique_handle_count = 0

    # load handles and tweets in
    with open(cleanedTweets_filename) as file_tweets, open(filename_handles) as file_handles:
        line_count = 0

        # issue: first line in tweet file is header, but first line in handle file corresponds to second line in tweet file
        header_line = file_tweets.readline()
        header_line = header_line.strip('\n')
        header_line = header_line.strip('\r')
        header_elements = header_line.split(',')


        # i can just modify the feature space here to get the new projections

        # header_elements = header_elements[:100]

        print 'size feature space: ', str(len(header_elements)), '  ', cleanedTweets_filename

        for tweet, handle in izip(file_tweets, file_handles):
            print 'processing tweet ', line_count

            tweet = tweet.strip('\n')
            tweet = tweet.strip('\r')
            tweet_elements = tweet.split(',')

            handle = handle.strip('\n')
            handle = handle.strip('\r')

            # populates handle to feature space count dict
            if handle in handle_toTweetsDict:
                row_index = 0
                for feature_value in tweet_elements:
                    handle_toTweetsDict[handle][row_index] = handle_toTweetsDict[handle][row_index] + int(feature_value)
                    row_index = row_index + 1
            else:
                unique_handle_count = unique_handle_count + 1
                newList = []
                for feature_value in tweet_elements:
                    newList.append( int(feature_value) )
                handle_toTweetsDict[handle] = newList

            line_count = line_count + 1


    output_fileName = cleanedTweets_filename[:-4] # remove '.txt'
    output_fileName = output_fileName + '_summed_' +  str(len(header_elements)) + '.txt'
    outfile = open(output_fileName, 'w')
    for handle, tweet_representation in handle_toTweetsDict.iteritems():
        output_line = handle
        for element in tweet_representation:
            output_line = output_line + ',' + str(element)
        output_line = output_line + '\n'
        outfile.write(output_line)
    outfile.close()
    print 'Finished aquisition of handle points for ', str( unique_handle_count ), 'handles'
    print 'File: ', cleanedTweets_filename



    # write out to file
    # how to deal with whether original / retweet / full ? when naming. Should also specifcy size of set? i.e. 4k

def derive_Bag_of_Words_sparse ( cleanedTweets_filename, handles_filename ):

    handleTo_WordCountDict_dict = {}
    word_toFeatureSpaceIndex = {}
    unqiue_word_count = 0
    # outfile_line_featureList = ''
    # first_word = True

    handle_list = []

    # store all tweets corresponding to each handle
    with open(cleanedTweets_filename) as file_tweets, open(filename_handles) as file_handles:

        line_count = 0
        for tweet, handle in izip(file_tweets, file_handles):
            # print 'Processing tweet', line_count
            line_count = line_count + 1

            handle = handle.strip('\n')
            handle = handle.strip('\r')
            tweet = tweet.strip('\n')
            tweet = tweet.strip('\r')

            if handle not in handle_list:
                handle_list.append(handle)

            if handle not in handleTo_WordCountDict_dict:
                new_WordCountDict = {}
                handleTo_WordCountDict_dict[handle] = new_WordCountDict

            word_list = re.sub(r'[^\w\'] ', " ",  tweet).split()
            for word in word_list:
                if word not in word_toFeatureSpaceIndex:
                    # if first_word:
                    #     outfile_line_featureList = word
                    #     first_word = False
                    # else:
                    #     # outfile_line_featureList = outfile_line_featureList + ',' + word

                    unqiue_word_count = unqiue_word_count + 1
                    word_toFeatureSpaceIndex[word] = unqiue_word_count
                    print word, ' ', unqiue_word_count

                if word not in handleTo_WordCountDict_dict[handle]:
                    handleTo_WordCountDict_dict[handle][word] = 1
                else:
                    handleTo_WordCountDict_dict[handle][word] = handleTo_WordCountDict_dict[handle][word] + 1

    # because can't initilaize empty csr matrix, need to get first vector projection
    WordCountDict_0 = handleTo_WordCountDict_dict[ handle_list[0] ]

    vector_representation = [0] * unqiue_word_count
    for word, count in WordCountDict_0.iteritems():
        vector_representation[ word_toFeatureSpaceIndex[word] ] = count

    handle_vectors_sparse = csr_matrix(vector_representation)

    print 'projected handle 1 out of', str(len(handle_list))

    # ignore first handle, which has already been converted
    handle_count  = 2
    for handle in handle_list[1:]:
        WordCountDict = handleTo_WordCountDict_dict[handle]

        vector_representation = [0] * (unqiue_word_count + 1)

        for word, count in WordCountDict.iteritems():
            vector_representation[ word_toFeatureSpaceIndex[word] ] = count

        vector_representation_sparse = csr_matrix(vector_representation)

        handle_vectors_sparse = csr_vappend(handle_vectors_sparse,  vector_representation_sparse)

        print 'projected handle', handle_count, 'out of', str(len(handle_list))
        handle_count = handle_count + 1

    fout = open('handle_vectors_BagOfWords_sparse.dat','wb')
    cPickle.dump(handle_vectors_sparse,fout, cPickle.HIGHEST_PROTOCOL)
    fout.close()


def derive_Bag_of_Words ( cleanedTweets_filename, handles_filename ):
    # store as summed directly? yes...

    handleTo_TweetList_dict = {}

# store all tweets corresponding to each handle
    with open(cleanedTweets_filename) as file_tweets, open(filename_handles) as file_handles:

        line_count = 0
        for tweet, handle in izip(file_tweets, file_handles):
            print 'Processing tweet', line_count
            line_count = line_count + 1

            handle = handle.strip('\n')
            handle = handle.strip('\r')
            tweet = tweet.strip('\n')
            tweet = tweet.strip('\r')

            if handle not in handleTo_TweetList_dict:
                new_tweetList = []
                new_tweetList.append(tweet)
                handleTo_TweetList_dict[handle] = new_tweetList
            else:
                handleTo_TweetList_dict[handle].append(tweet)

    # for each handle, aquire counts for each word that occurs for that handle
        # output to two files: word_list, count_list (count_list_file is what we would be passing to utility functions,
            # and word_list_file is what would be opened up, once the summed representation was realized to be bag of words)
            # update utility code to be compatable with this (and mathmatical copmressions, like PCA and Wavlet)

    # if first line = 'bag_of_words', skip to next line and use bag_of_words representation parsing
        # file_tweets.readline()


    # if doing full vector size comparison of handle reprsenations,
    # need to write out the entire feature space, so it doesnt have to be calculated again later

    output_fileName_wordList = cleanedTweets_filename[:-4]
    output_fileName_countList = cleanedTweets_filename[:-4]
    outfput_fileName_featureList = cleanedTweets_filename[:-4]

    output_fileName_wordList = output_fileName_wordList + '_BagOfWords_summed_wordLists.txt'
    output_fileName_countList = output_fileName_countList + '_BagOfWords_summed.txt'
    outfput_fileName_featureList = outfput_fileName_featureList + '_BagOfWords_featureSpace.txt'

    outfile_wordList = open(output_fileName_wordList, 'w')
    outfile_countList = open(output_fileName_countList, 'w')
    outfile_featureList = open(outfput_fileName_featureList, 'w')

    unique_word_dict = {}
    unqiue_word_count = 0

    outfile_line_featureList = ''
    first_word = True

    for handle, tweetList in handleTo_TweetList_dict.iteritems():
        wordTo_OccurenceCount_dict = {}

        for tweet in tweetList:
            tweet = tweet.strip('\n')
            tweet = tweet.strip('\r')

            word_list = re.sub(r'[^\w\'] ', " ",  tweet).split()
            for word in word_list:

                if word in wordTo_OccurenceCount_dict:
                    wordTo_OccurenceCount_dict[word] = wordTo_OccurenceCount_dict[word] + 1
                else:
                    wordTo_OccurenceCount_dict[word] = 1

                if word not in unique_word_dict:
                    if first_word:
                        outfile_line_featureList = word
                        first_word = False

                    outfile_line_featureList = outfile_line_featureList + ',' + word

                    unqiue_word_count = unqiue_word_count + 1
                    unique_word_dict[word] = unqiue_word_count
                    print word, ' ', unqiue_word_count

        # create two outfile_lines: word_list_file & count_list_file
            # add handles to both
        outfile_line_wordList = handle
        outfile_line_countList = handle

        for word, count in sorted(wordTo_OccurenceCount_dict.items(), key=operator.itemgetter(1), reverse = True):
            # add to both word and count outfile_lines
            outfile_line_wordList = outfile_line_wordList + ',' + word
            outfile_line_countList = outfile_line_countList + ',' + str ( count )

        outfile_line_wordList = outfile_line_wordList + '\n'
        outfile_line_countList = outfile_line_countList + '\n'
        outfile_line_featureList = outfile_line_featureList + '\n'

        # write both outfile lines to their respective files
        outfile_wordList.write(outfile_line_wordList)
        outfile_countList.write(outfile_line_countList)

    outfile_featureList.write(outfile_line_featureList)

    outfile_wordList.close()
    outfile_countList.close()
    outfile_featureList.close()

    # close file


# the bag of words representation should involve a feature space that encompasses all words in the corpus (all tweets),
# and we discussed a technique for reducing the size of such a representation (in terms of memory),
# which involved writing out this bag of words representation to a file in a manner that involved keeping a count (summed across all tweets for a user)
# of the occurrence counts of the words that occur for a given user, rather than storing the number of occurrences for every word in the feature space (i.e. all words in the corpus) for each user (which would involve many, many zeros on average)


#     do intersection of two lists: words corresponding to tweet, and words in feature space

# So store all words (and counts corresponding to those words for each handle)
# then load in this summed bag of words model for each user

# when acquiring this bag of words to write out to file, do on per user basis, and then clear memory after each user.



def derive_MAX_tf_idf_ofWordList_ALLWORDS (  ):

    word_toHandleCount = {}
    handleCount = 0
    unique_handleList = []
    COLLECTION_handleToTweets_dict = {}
    word_toMAX_tf_idf = {}

    # make more efficent by already have established a dict of word --> [ handle --> count of occurances ]  ?
    word_to_handlesWordAppearsIn_To_numOccuranceInHandle_DictOfDicts = {}


    # aquire count of number of documents that word occurs in

    file_size = 0
    with open(cleanedTweets_filename) as file_tweets:
        for tweet in file_tweets:
            file_size = file_size + 1

    # CONSTRUCT COLLECTION: a container for all the documents --> handle to tweet dictionary
    with open(cleanedTweets_filename) as file_tweets, open(filename_handles) as file_handles:

        line_count = 0
        for tweet, handle in izip(file_tweets, file_handles):
            print 'processing line: ', line_count
            line_count = line_count + 1

            tweet = tweet.strip('\n')
            handle = handle.strip('\n')
            wordsInTweet_list = tweet.split()

            for word in wordsInTweet_list: # count # of handles for which a word in top1kWords_list appears

                # NEEDS to record: number of handles a word corresponds to
                if word in word_toHandleCount:
                    word_toHandleCount[word].append(handle)
                else:
                    word_toHandleCount[word] = []
                    word_toHandleCount[word].append(handle)

                # # keep track of number of times word occurs for each handle
                if word in word_to_handlesWordAppearsIn_To_numOccuranceInHandle_DictOfDicts: # if word already in dict
                    if handle in word_to_handlesWordAppearsIn_To_numOccuranceInHandle_DictOfDicts[word]: # if handle already in sub dict
                        word_to_handlesWordAppearsIn_To_numOccuranceInHandle_DictOfDicts[word][handle] = word_to_handlesWordAppearsIn_To_numOccuranceInHandle_DictOfDicts[word][handle] + 1 # increment count
                    else: # handle not already in sub dict
                        word_to_handlesWordAppearsIn_To_numOccuranceInHandle_DictOfDicts[word][handle] = 1
                else: # word not already in dict
                    word_to_handlesWordAppearsIn_To_numOccuranceInHandle_DictOfDicts[word] = {}
                    word_to_handlesWordAppearsIn_To_numOccuranceInHandle_DictOfDicts[word][handle] = 1


            if handle in COLLECTION_handleToTweets_dict:
                for word in wordsInTweet_list:
                    COLLECTION_handleToTweets_dict[handle].append(word)
            else:
                handleCount = handleCount + 1
                newList = []
                unique_handleList.append(handle)
                COLLECTION_handleToTweets_dict[handle] = newList
                for word in wordsInTweet_list:
                    COLLECTION_handleToTweets_dict[handle].append(word)

            if line_count > file_size: # keep tf-idf extraction limited to total tweets in current data set
                break

    # calculate idf for each word
        # idf = log ( len ( collection ) / count (document_containing_term, collection ) )

    word_toIDF_dict = {}
    for word, handle_list in word_toHandleCount.iteritems():
        word_toMAX_tf_idf[word] = 0
        word_toIDF_dict[word] = math.log ( handleCount / float( len ( handle_list ) ) )

    # now go through each handle,
        # calculate tf for all words

    # make more efficent by already have established a dict of word --> [ handle --> count number of tweets word showed up in ]  ?

    line_count = 0
    for handle in unique_handleList:
        print 'calculating tf for words corresponding to handle ', line_count, ' out of ', len( unique_handleList )
        for word, handle_list in word_toHandleCount.iteritems():  # go through every word  (this is inefficent because not every word is represented by every handle)
            # tf = count ( word, COLLECTION_handleToTweets_dict[handle] )

            # pdb.set_trace()

            tf = 0
            if handle in word_to_handlesWordAppearsIn_To_numOccuranceInHandle_DictOfDicts[word]:
                tf = word_to_handlesWordAppearsIn_To_numOccuranceInHandle_DictOfDicts[word][handle]

            idf = word_toIDF_dict[word]

            tf_idf = tf * idf

            if word_toMAX_tf_idf[word] < tf_idf:
                word_toMAX_tf_idf[word] = tf_idf
        line_count = line_count + 1

    print 'completed calculation of tf-idf for all words'
    return word_toMAX_tf_idf

# # for each word in the wordList, calculate the tf-idf for all docuements, and keep track of the MAX tf-idf
# # MADE OBSOLETE BY derive_MAX_tf_idf_ofWordList_ALLWORDS()
# def derive_MAX_tf_idf_ofWordList ( wordList ):

#     word_toHandleCount = {}
#     handleCount = 0
#     unique_handleList = []
#     COLLECTION_handleToTweets_dict = {}
#     word_toMAX_tf_idf = {}

#     for word in wordList: # intilize to 0
#         word_toHandleCount[word] = 0

#     # aquire count of number of documents that word occurs in

#     # CONSTRUCT COLLECTION: a container for all the documents --> handle to tweet dictionary
#     # need to store all words, but only need to do calulations for words in top1k Vocab
#     with open(cleanedTweets_filename) as file_tweets, open(filename_handles) as file_handles:

#         line_count = 0
#         for tweet, handle in izip(file_tweets, file_handles):
#             print 'processing line: ', line_count
#             line_count = line_count + 1

#             tweet = tweet.strip('\n')
#             handle = handle.strip('\n')
#             wordsInTweet_list = tweet.split()

#             for word in wordsInTweet_list: # count # of handles for which a word in top1kWords_list appears
#                 if word in wordList:
#                     word_toHandleCount[word] = word_toHandleCount[word] + 1

#             if handle in COLLECTION_handleToTweets_dict:
#                 for word in wordsInTweet_list:
#                     COLLECTION_handleToTweets_dict[handle].append(word)
#             else:
#                 handleCount = handleCount + 1
#                 newList = []
#                 unique_handleList.append(handle)
#                 COLLECTION_handleToTweets_dict[handle] = newList
#                 for word in wordsInTweet_list:
#                     COLLECTION_handleToTweets_dict[handle].append(word)

#     # calculate idf for each word in top1kWords_list
#         # idf = log ( len ( collection ) / count (document_containing_term, collection ) )

#     word_toIDF_dict = {}
#     for word in wordList:
#         if word_toHandleCount[word] == 0: # fix for when number of tweets are abrubtly reduced
#             word_toIDF_dict[word] = 0
#         else:
#             word_toIDF_dict[word] = math.log (  handleCount / float(word_toHandleCount[word]) )

#     for word in wordList:
#         word_toMAX_tf_idf[word] = 0

#     # now go through each handle,
#         # calculate tf for all words in wordList
#     for handle in unique_handleList:

#         for word in wordList:
#             tf = count ( word, COLLECTION_handleToTweets_dict[handle] )
#             idf = word_toIDF_dict[word]

#             tf_idf = tf * idf

#             if word_toMAX_tf_idf[word] < tf_idf:
#                 word_toMAX_tf_idf[word] = tf_idf

#     return word_toMAX_tf_idf



def derive_tweetMatrixRepresentation_topTF_IDF ( numDimensions, cleanedTweets_filename ):
    print 'in TF IDF'
    outputFile_name = 'tweets_MatrixRepresentation_topTF_IDF_' + tweet_setType + str(dataSet_size) + '.csv'
    outFile = open(outputFile_name, 'w')
    writer = csv.writer(outFile)
    word_toIndexInMatrix_dict = {}

    wordList = []
    words_to_MAX_TF_IDF_dict = {}

    vocabDimensions_list = []

    # with open (filename_top1kWords_TOTAL) as file:
    #     line_count = 0
    #     # load in top1kWords_Vocab
    #     for line in file:
    #         # word_tokens = re.sub(r'[^\w\'] ', " ",  line).split() # convert into list of words
    #         line_count = line_count + 1
    #         line = line.strip('\n')
    #         word_tokens = line.split(',')

    #         word = word_tokens[0]
    #         wordFreq = word_tokens[1]

    #         wordList.append(word)

    words_to_MAX_TF_IDF_dict = derive_MAX_tf_idf_ofWordList_ALLWORDS ( )

    # top 100 tf-idf words
    # words_to_MAX_TF_IDF_dict = {'vivaheug': 9, 'seokjin': 93, 'contentchat': 8, 'lifewithboys': 18, 'vintagebrooches': 51, 'ttcwithstorkotc': 68, 'asthma': 87, 'northof': 36, 'datemydad': 66, 'wear_its_hat': 57, 'sens': 74, 'dmdcontest': 53, 'weallgrow': 65, 'boysclothing': 78, 'blff': 20, 'robinpagoria': 80, 'runtastic': 29, 'sextoys': 2, 'incredipizza': 82, 'bringbackthebrooch': 17, 'usethewholewatermelon': 47, 'bookbuzzr': 26, 'trouv': 11, 'jokes': 84, 'fiverr': 85, 'nairn': 32, 'marketingnw': 62, 'establishkhilafah': 43, 'imaginationplayground': 63, 'sahaf': 25, 'gostei': 96, 'tadcummins': 91, 'westlakecharter': 48, 'allin': 86, 'italiamac': 71, 'openvisconf': 12, 'mplusrewards': 23, 'longlivedahighest': 19, 'costumejewelry': 59, 'cnc': 45, 'findyourrhythm': 49, 'memes': 98, 'bloggycon': 21, 'jakksxttpm': 94, 'grainfree': 5, 'tbsarkday': 92, 'peggy': 55, 'pami': 30, 'egifter': 50, 'sgtp': 54, 'ednet': 72, 'ksedchat': 79, 'attrap': 39, 'nptcs': 88, 'coparenting': 35, 'forsale': 41, 'naadu': 22, 'alliance': 60, 'confirmgorsuch': 38, 'fastmn': 27, 'asuggestion': 73, 'outfittheday': 70, 'lovemyschoolday': 52, 'corporal': 56, 'wizebot': 16, 'splendazero': 44, 'bconnectedto': 58, 'masturbating': 3, 'schoolcorporalpunishment': 4, 'bestsuggestor': 37, 'isedchat': 61, 'funy': 46, 'kingdomhearts': 15, 'dartmoor': 83, 'trecru': 10, 'teamlife': 69, 'maximfitpaper': 6, 'jurassicworld': 64, 'letrjack': 76, 'egelem': 89, 'toilet': 90, 'nhma': 67, 'bbrcoffee': 34, 'fashionaddic': 33, 'escargophone': 0, 'egpride': 31, 'votemoose': 1, 'ingredients': 99, 'neurolanguage': 75, 'dadsasprincipals': 77, 'tennessees': 81, 'fibaonstartimes': 95, 'athomedaddy': 14, 'dresses': 42, 'mhaconf': 40, 'paleo': 13, 'noddykidswear': 24, 'maximfitcontest': 28, 'hooterholsterlove': 97, 'masturbation': 7}

    sorted_words_to_MAX_TF_IDF_dict = sorted(words_to_MAX_TF_IDF_dict.items(), key=operator.itemgetter(1), reverse = True)

    X = 0
    for word, count in sorted_words_to_MAX_TF_IDF_dict:
        if X < numDimensions:
            vocabDimensions_list.append(word)
            word_toIndexInMatrix_dict[word] = ( len(vocabDimensions_list) -1 )
        else:
            break
        X = X + 1

    # top 100 tf_idf words
    # vocabDimensions_list = ['escargophone', 'votemoose', 'sextoys', 'masturbating', 'schoolcorporalpunishment', 'grainfree', 'maximfitpaper', 'masturbation', 'contentchat', 'vivaheug', 'trecru', 'trouv', 'openvisconf', 'paleo', 'athomedaddy', 'kingdomhearts', 'wizebot', 'bringbackthebrooch', 'lifewithboys', 'longlivedahighest', 'blff', 'bloggycon', 'naadu', 'mplusrewards', 'noddykidswear', 'sahaf', 'bookbuzzr', 'fastmn', 'maximfitcontest', 'runtastic', 'pami', 'egpride', 'nairn', 'fashionaddic', 'bbrcoffee', 'coparenting', 'northof', 'bestsuggestor', 'confirmgorsuch', 'attrap', 'mhaconf', 'forsale', 'dresses', 'establishkhilafah', 'splendazero', 'cnc', 'funy', 'usethewholewatermelon', 'westlakecharter', 'findyourrhythm', 'egifter', 'vintagebrooches', 'lovemyschoolday', 'dmdcontest', 'sgtp', 'peggy', 'corporal', 'wear_its_hat', 'bconnectedto', 'costumejewelry', 'alliance', 'isedchat', 'marketingnw', 'imaginationplayground', 'jurassicworld', 'weallgrow', 'datemydad', 'nhma', 'ttcwithstorkotc', 'teamlife', 'outfittheday', 'italiamac', 'ednet', 'asuggestion', 'sens', 'neurolanguage', 'letrjack', 'dadsasprincipals', 'boysclothing', 'ksedchat', 'robinpagoria', 'tennessees', 'incredipizza', 'dartmoor', 'jokes', 'fiverr', 'allin', 'asthma', 'nptcs', 'egelem', 'toilet', 'tadcummins', 'tbsarkday', 'seokjin', 'jakksxttpm', 'fibaonstartimes', 'gostei', 'hooterholsterlove', 'memes', 'ingredients']

    csv_row_summary = vocabDimensions_list
    # vocabDimensions_list = emotionDictionaryWords_list
    writer.writerow(csv_row_summary)

    numEmotionVectorHits_total = 0
    numTweets_hitEmotionVector = 0

    with open(cleanedTweets_filename) as file:

        line_count = 0
        for line in file:
            print 'compressing line ', line_count
            tweetRepresentation_list = [0] * len(vocabDimensions_list)
            line_count = line_count + 1
            line = line.strip('\n')
            # print line
            word_tokens = line.split()
            # print 'line before: ', line

            # pdb.set_trace()

            # pdb.set_trace()
            for word in word_tokens:

                alreadyHitVocab = False

                if word in vocabDimensions_list:
                    # print 'word hit', word

                    numEmotionVectorHits_total = numEmotionVectorHits_total + 1
                    if not alreadyHitVocab:
                        numTweets_hitEmotionVector = numTweets_hitEmotionVector + 1
                        alreadyHitVocab = True

                    index_ofWord_inVector = word_toIndexInMatrix_dict[word]

                    # increment count in list
                    tweetRepresentation_list[index_ofWord_inVector] = tweetRepresentation_list[index_ofWord_inVector] + 1
                    # print 'the value of index ', index_ofWord_inVector, 'is now ', tweetRepresentation_list[index_ofWord_inVector]

            # print
            # if 1 in tweetRepresentation_list:
            #     print 'tweet: ', line
            # print line

                    # pdb.set_trace()

            for i in range(len(tweetRepresentation_list)):
                tweetRepresentation_list[i] = str(tweetRepresentation_list[i])

            csv_row_summary = tweetRepresentation_list
            # csv_row_summary_2 = ['a', 'b', 'c']

            # pdb.set_trace()
            # print 'converting tweet: ', line_count
            writer.writerow(csv_row_summary)


def output_list_from_csv(emotion):
    filename = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/emotion_dictionary/' + emotion + '.txt'

    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        synonym_list = list(reader)[0]
    return synonym_list



# aquire the most frequent 1k words from all the tweets represented, with counts = vocabulary
def aquireVocab( filename_topXWords_perUser ):

    # highFreqWords_list = []
    print 'Running aquireVocab for ', filename_topXWords_perUser

    wordToFreq_dict = {}
    with open(filename_topXWords_perUser) as file:
        handle = ''

        line_count = 0
        for line in file:
            # word_tokens = re.sub(r'[^\w\'] ', " ",  line).split() # convert into list of words
            line_count = line_count + 1
            line = line.strip('\n')

            # print line

            word_tokens = line.split(',')

            if word_tokens[0] == 'handle':
                handle = word_tokens[1]
            else:
                word = word_tokens[0]
                # print word_tokens[1]
                # if int(word_tokens[1]) >  1000:
                    # print word_tokens[0], word_tokens[1]
                    # highFreqWords_list.append(word_tokens[0])

                word_freq_forHandle = int(word_tokens[1])

                if word in wordToFreq_dict:
                    wordToFreq_dict[word] = wordToFreq_dict[word] + word_freq_forHandle
                else:
                    wordToFreq_dict[word] = word_freq_forHandle

    # remove numbers from top 1k?
    ranked_wordToFreq_dict = sorted(wordToFreq_dict.items(), key=operator.itemgetter(1), reverse = True)

    outfile = open('top_1k_words_Vocabulary_'  + tweet_setType + str(dataSet_size) + '.txt', 'w')


    # pdb.set_trace()

    # for word in highFreqWords_list:
    #     print word, wordToFreq_dict[word]


    X = 0
    for word, count in ranked_wordToFreq_dict:
        X = X + 1

        if X < 1000:
            outfile.write(word + ',' + str(count) + "\n")
        else:
            break


# calculate tf-idf of words in vocabulary, and return top 100 words (Based on tf-idf), with tf-idf score



# tweet matrix projection

# load in vocabulary vector
    # store as list, and write out header, with words in order of frequency
    # have dictionary of word --> index of word in vector

# for each tweet,
    #create dict of word --> word freq
    # intilize list tweetRepresentation_list of length 1k, with 0 as default value.
    # for each word:
        # if word in vocabularly list,
        # reference vocabWord_toVectorIndex_dict to obtain index at which to store word freq
        # store word freq at this index

    # write tweetRepresentation_list as a line to a file. with each element in the list being comma delimited


def derive_tweetMatrixRepresentation_topVocab( cleanedTweets_filename, filename_handles):

    vocabHeader_list = []
    word_toIndexInVocabMatrix_dict = {}

    handle_toTweetsDict = {}

    # filename_top1kWords_perUser = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/top_1k_words_perHandle_'  + tweet_setType + str(dataSet_size) + '.txt'
    outputFile_name = 'tweets_MatrixRepresentation_topVocab_' + tweet_setType + str(dataSet_size) + '.csv'
    outFile = open(outputFile_name, 'w')
    writer = csv.writer(outFile)

    with open('top_1k_words_Vocabulary_'  + tweet_setType + str(dataSet_size) + '.txt', 'r') as file:

        line_count = 0

        for line in file:
            # word_tokens = re.sub(r'[^\w\'] ', " ",  line).split() # convert into list of words
            line_count = line_count + 1
            line = line.strip('\n')
            word_tokens = line.split(',')

            word = word_tokens[0]
            wordFreq = word_tokens[1]

            if line_count > 1000:
                break
            else:
                vocabHeader_list.append(word)

            word_toIndexInVocabMatrix_dict[word] = (len(vocabHeader_list) - 1)

        # csv_Data_summary = []
        csv_row_summary = vocabHeader_list
        writer.writerow(csv_row_summary)

    numVocabVectorHits_total = 0
    numTweets_hitVocabVector = 0

    with open(cleanedTweets_filename) as file_tweets, open(filename_handles) as file_handles:

        line_count = 0
        for tweet, handle in izip(file_tweets, file_handles):
            print 'Processing tweet', line_count
            handle = handle.strip('\n')
            handle = handle.strip('\r')

            tweetRepresentation_list = [0] * 1000
            line_count = line_count + 1
            tweet = tweet.strip('\n')
            # print line
            word_tokens = tweet.split()

            # pdb.set_trace()
            for word in word_tokens:

                alreadyHitVocab = False

                if word in vocabHeader_list:
                    # print 'word hit', word

                    numVocabVectorHits_total = numVocabVectorHits_total + 1
                    if not alreadyHitVocab:
                        numTweets_hitVocabVector = numTweets_hitVocabVector + 1
                        alreadyHitVocab = True

                    index_ofWord_inVector = word_toIndexInVocabMatrix_dict[word]

                    # increment count in list
                    tweetRepresentation_list[index_ofWord_inVector] = tweetRepresentation_list[index_ofWord_inVector] + 1
                    # print 'the value of index ', index_ofWord_inVector, 'is now ', tweetRepresentation_list[index_ofWord_inVector]

            # print
            # if 1 in tweetRepresentation_list:
            #     print 'tweet: ', line
            # print line

                    # pdb.set_trace()

            for i in range(len(tweetRepresentation_list)):
                tweetRepresentation_list[i] = str(tweetRepresentation_list[i])

            csv_row_summary = tweetRepresentation_list
            # csv_row_summary_2 = ['a', 'b', 'c']

            if handle in handle_toTweetsDict:
                handle_toTweetsDict[handle].append(csv_row_summary)
            else:
                handle_toTweetsDict[handle] = []
                handle_toTweetsDict[handle].append(csv_row_summary)

            # pdb.set_trace()
            # print 'converting tweet: ', line_count
            writer.writerow(csv_row_summary)

            # if line_count >= 10000:
            #     break


    # outputFile_name = 'tweets_MatrixRepresentation_topVocab_Handle_toVectorSpace.csv'
    # outFile = open(outputFile_name, 'w')
    # writer = csv.writer(outFile)
    # writer.writerow(vocabHeader_list)
    # count = 0

    # for handle, tweets in handle_toTweetsDict.iteritems():
    #     if count > 3:
    #         break
    #     writer.writerow(handle)
    #     for tweet in tweets:
    #         writer.writerow(tweet)
    #     writer.writerow('\n')
    #     writer.writerow('\n')
    #     count = count + 1


# emotion dictionary compression
# go through each emotion dictionary, and establish list emotionVocab_list that contains all of the words in the diciontaries
# have dictionary of word --> index of word in vector

# for each tweet,
    #create dict of word --> word freq
    # intilize list tweetRepresentation_list of length = emotionVocab_list
        # for each word:
        # if word in emotionVocab_list list,
        # reference emotionVocabWord_toVectorIndex_dict to obtain index at which to store word freq
        # store word freq at this index

    # write tweetRepresentation_list as a line to a file. with each element in the list being comma delimited


# UPDATE: change to keep words from vocab that are in emotional dictionary, rather than just using all of the words in the emotional dictionary
def derive_tweetMatrixRepresentation_emotionDict( cleanedTweets_filename, handles_filename ):

    vocabDimensions_list = []
    word_toIndexInMatrix_dict = {}
    top1kWords_Vocab = []

    FunSpanishTchr_tweetList = []

    outputFile_name = 'tweets_MatrixRepresentation_emotionDictionary' + tweet_setType + str(dataSet_size) + '.csv'
    outFile = open(outputFile_name, 'w')
    writer = csv.writer(outFile)

    # with open('top_1k_words_Vocabulary_' +  tweet_setType + str(dataSet_size) + '.txt', 'r') as file:
    #     line_count = 0
    #     for line in file:
    #         # word_tokens = re.sub(r'[^\w\'] ', " ",  line).split() # convert into list of words
    #         line_count = line_count + 1
    #         line = line.strip('\n')
    #         word_tokens = line.split(',')
    #         word = word_tokens[0]
    #         wordFreq = word_tokens[1]
    #         if line_count > 1000:
    #             break
    #         else:
    #             top1kWords_Vocab.append(word)

    emotion_list = ['anger_disgust', 'fear', 'joy_love', 'sadness', 'surprise']
    emotionDictionaryWords_list = []

    # vocabDimensions_list = ['sadnessemotionality', 'joyloveemotionality', 'angerdisgustemotionality', 'surpriseemotionality', 'fearemotionality']

    for emotion in emotion_list:
            snyonym_list = output_list_from_csv(emotion)
            for word in snyonym_list:
                if word not in vocabDimensions_list: # add unique words to feature space
                    vocabDimensions_list.append(word)
                    # word_toIndexInMatrix_dict[word] = (len(vocabDimensions_list) - 1) # specify feature number

                    word_toIndexInMatrix_dict[word] = (len(word_toIndexInMatrix_dict) )

    # for word in top1kWords_Vocab:
    #     if word in emotionDictionaryWords_list:
    #         vocabDimensions_list.append(word)
    #         # word_toIndexInMatrix_dict[word] = (len(vocabDimensions_list) - 1)
    #         word_toIndexInMatrix_dict[word] = (len(word_toIndexInMatrix_dict) )


    # for word in vocabDimensions_list:
    #     word_toIndexInMatrix_dict[word] = (len(word_toIndexInMatrix_dict))

    csv_row_summary = vocabDimensions_list
    # vocabDimensions_list = emotionDictionaryWords_list
    writer.writerow(csv_row_summary)

    numEmotionVectorHits_total = 0
    numTweets_hitEmotionVector = 0

    with open(cleanedTweets_filename) as file:

        line_count = 0

        for line in file:
            # print line
            tweetRepresentation_list = [0] * len(vocabDimensions_list)
            line_count = line_count + 1
            line = line.strip('\n')
            # print line
            word_tokens = line.split()
            # print 'line before: ', line
            # pdb.set_trace()
            for word in word_tokens:

                alreadyHitVocab = False

                if word in vocabDimensions_list:
                    print 'word hit', word

                    numEmotionVectorHits_total = numEmotionVectorHits_total + 1
                    if not alreadyHitVocab:
                        numTweets_hitEmotionVector = numTweets_hitEmotionVector + 1
                        alreadyHitVocab = True

                    index_ofWord_inVector = word_toIndexInMatrix_dict[word]

                    # increment count in list

                    # if line_count == 15:
                    # pdb.set_trace()

                    tweetRepresentation_list[index_ofWord_inVector] = tweetRepresentation_list[index_ofWord_inVector] + 1
                    # print 'the value of index ', index_ofWord_inVector, 'is now ', tweetRepresentation_list[index_ofWord_inVector]

            # print
            # if 1 in tweetRepresentation_list:
            #     print 'tweet: ', line
            # print line

                    # pdb.set_trace()

            for i in range(len(tweetRepresentation_list)):
                tweetRepresentation_list[i] = str(tweetRepresentation_list[i])

            csv_row_summary = tweetRepresentation_list
            # csv_row_summary_2 = ['a', 'b', 'c']

            # pdb.set_trace()
            print 'converting tweet: ', line_count
            writer.writerow(csv_row_summary)

    print 'Completed aquisition of tweetMatrixRepresentation_emotionDict for ', cleanedTweets_filename


def PCA_compression ( representation_fileName, number_of_components  ):

    # create dict of column name --> column elements

    features = []
    raw_data = {}

    with open(representation_fileName) as file:

        line_count = 0
        for line in file: # construct raw data dictionary for dataframe
            # print line

            line = line.strip('\n')
            line = line.strip('\r')
            line_elements = line.split(',')

            if line_count == 0: # first row, which is columns information (header)
                features = line_elements
                for element in features:
                    emptyList = []
                    raw_data[element] = emptyList
            else:
                for i in range(len(features)):
                    raw_data[ features[i] ].append(   float(line_elements[i])   )  # for each column, add the value that that tweet has


            line_count = line_count + 1

        df = pd.DataFrame(raw_data, columns = features)

        # normalize
        x = df.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)

        # transform data
        pca = PCA( n_components = number_of_components )
        principalComponents = pca.fit_transform(x)

        component_headerList = []
        # create column list based on dimensionality being reduced to.
        for i in range(number_of_components):
            currentString = 'principal component ' +  str(i)
            component_headerList.append(currentString)



        # pdb.set_trace()

        df_final = pd.DataFrame(data = principalComponents
             , columns = component_headerList)

        outputFile_name = representation_fileName[:-3]
        outputFile_name = outputFile_name + '_PCA_' + str(number_of_components) + '.csv'

        df_final.to_csv(outputFile_name)


# SINGLE LEVEL DECOMPOSITION
def Wavelet_Compression ( representation_fileName, handle_fileName, header=True ):

    features = []
    raw_data = {}

    # handle --> [2d python list, line from tweet file of each tweet in python list]
    handle_toTweetsDict = {}

    # construct list of all tweets for a handle, compress those tweets,
    # then put back in order at which the tweets came in

    with open(representation_fileName) as file_tweets, open(filename_handles) as file_handles:

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

            # populates dictionary  of handle --> [2d python list, list of line locations for tweets in python list]
            if handle in handle_toTweetsDict:
                handle_toTweetsDict[handle][0].append( tweet_elements )  # add new row to 2d array
                handle_toTweetsDict[handle][1].append( line_count ) # append line location of new tweet added
            else:
                handle_toTweetsDict[handle] = []
                newList = []
                handle_toTweetsDict[handle].append( newList )  #
                handle_toTweetsDict[handle][0].append( tweet_elements )  #
                newList = []
                handle_toTweetsDict[handle].append( newList )
                handle_toTweetsDict[handle][1].append( line_count )

            line_count = line_count + 1

        # intilize line_count size output list
        output_list = [None] * line_count

        # now, do a 2d transformation for each handle
            # then update the represenation of that handle in the dict
        for handle, Data in handle_toTweetsDict.iteritems():
            print 'compressing: ', handle

            x = np.array( Data[0] )

            # single level 2D discrete Wavelet transform
            # http://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html
            coeffs = pywt.dwt2(x, 'haar') # obtain coefficents
            # drop low coefficients here ?
            result = pywt.idwt2(coeffs, 'haar') # inverse transform
            result = result.tolist()

            # then go through each tweet in handle and put tweets in indexes based off of their location in the file ( reference handle_toTweetsDict[handle][1] )
            for tweet_num in range( len ( Data[1] ) ):
                tweet = result[tweet_num]
                # print tweet_num, len(Data[1])
                tweet_index = Data[1][tweet_num]
                output_list[tweet_index] = tweet

        # RE-INCORPORATE FEATURE HEADING
        output_list.insert(0, header_elements)

        outputFile_name = representation_fileName[:-3]
        outputFile_name = outputFile_name + '_Wavelet_' + '.csv'

        outfile = open(outputFile_name, 'w')

        # double check no None field in output_list
        # output compressed tweets to file
        for line_list in output_list:
            if line_list == None:
                print 'NONE VALUE FOUND'

            for i in range( len ( line_list )): #convert to string
                line_list[i] = str ( line_list[i] )

            outputString = ",".join( line_list )
            outputString = outputString + "\n"
            outfile.write(outputString)


def Wavelet_Compression_New( representation_fileName ):
    features = []
    raw_data = {}

    # handle --> [2d python list, line from tweet file of each tweet in python list]
    handle_toTweetsDict = {}

    # construct list of all tweets for a handle, compress those tweets,
    # then put back in order at which the tweets came in

    with open(representation_fileName) as file_tweets:

        line_count = 0

        # issue: first line in tweet file is header, but first line in handle file corresponds to second line in tweet file
        header_line = file_tweets.readline()
        header_line = header_line.strip('\n')
        header_line = header_line.strip('\r')
        header_elements = header_line.split(',')

        outputFile_name = representation_fileName[:-4]
        outputFile_name = outputFile_name + '_Wavelet' + '.csv'
        outfile = open(outputFile_name, 'w')

        for line in file_tweets:
            print 'processing line ', line_count
            line_count = line_count + 1

            line = line.strip('\n')
            line = line.strip('\r')
            tweet_elements = line.split(',')

            handle = tweet_elements[0]
            feature_values = tweet_elements[1:]

            x = np.array( feature_values )

            coefficents_approximation, coefficents_detail = pywt.dwt(x, 'haar')
            coefficents_approximation = list( coefficents_approximation )
            coefficents_detail = list ( coefficents_detail )

            for i in range( len ( coefficents_approximation )): #convert to string
                coefficents_approximation[i] = str ( coefficents_approximation[i] )
                coefficents_detail[i] = str ( coefficents_detail[i] )

            outfile.write(handle + '\n')
            outputString = 'Approximation Coefficients: ' + ", ".join( coefficents_approximation ) + '\n'
            outfile.write(outputString)
            outputString = 'Detail Coefficients: ' + ", ".join( coefficents_detail ) + '\n'
            outfile.write(outputString)
            outfile.write('\n')


def metrics_WordCount_UserLevel( representation_fileName, handles_fileName, header=True ):

    handle_toWordCount_dict = {}

    with open( representation_fileName ) as representation_FILE, open (handles_fileName) as handles_FILE :

        # account for header on representation potentially adding an extra line at top
        if header:
            header = representation_FILE.readline()
            header = header.strip('\n')
            header = header.strip('\r')
            header_elements = header.split(',')

        for tweet, handle in izip(representation_FILE, handles_fileName):

            tweet = tweet.strip('\n')
            tweet = tweet.strip('\r')
            tweet_elements = tweet.split(',')

            handle = handle.strip('\n')
            handle = handle.strip('\r')

            if handle not in handle_toWordCount_dict:
                newList = []
                handle_toWordCount_dict[handle] = 0

            count = handle_toWordCount_dict[handle]
            for word in tweet_elements:
                count = count + 1
            handle_toWordCount_dict[handle] = count

    outputFile_name = representation_fileName[:-3]
    outputFile_name = outputFile_name + '_wordCount_perHandle_' + '.csv'

    outfile = open(outputFile_name, 'w')

    for handle, count in handle_toWordCount_dict.iteritems():
        output_line = handle + ',' + str(count) + '\n'
        outfile.write(output_line)


def utf8len(s):
    return len(s.encode('utf-8'))

def metrics_ByteSize_UserLevel( representation_fileName, handles_fileName ):

    handle_toWordCount_dict = {}

    with open( representation_fileName ) as representation_FILE, open (handles_fileName) as handles_FILE:

        # account for header on representation potentially adding an extra line at top
        if header:
            header = representation_FILE.readline()
            header = header.strip('\n')
            header = header.strip('\r')
            header_elements = header.split(',')

        for tweet, handle in izip(representation_FILE, handles_fileName):

            tweet = tweet.strip('\n')
            tweet = tweet.strip('\r')

            handle = handle.strip('\n')
            handle = handle.strip('\r')

            if handle not in handle_toWordCount_dict:
                newList = []
                handle_toWordCount_dict[handle] = 0

            handle_toWordCount_dict[handle] = handle_toWordCount_dict[handle] + utf8len(tweet) # calculate byte size of line

    outputFile_name = representation_fileName[:-3]
    outputFile_name = outputFile_name + '_byteSize_perHandle_' + '.csv'

    outfile = open(outputFile_name, 'w')

    for handle, count in handle_toWordCount_dict.iteritems():
        output_line = handle + ',' + str(count) + '\n'
        outfile.write(output_line)


def graph_Tweet_Sets( wordToCount_filename_original, wordtToCount_filename_retweet, sortWords_onOriginal=True ):

# EXAMINE EMPTY HANDLE LISTS FOR RETWEETS AND ORIGINAL

# AQUIRE TWO GRAPHS: One sorted on original, one sorted on retweet

    wordToCount_Dict_original = {}
    wordToCount_Dict_retweet = {}
    wordList = []

    # aquire counts for number of tweets in each set --> used for normalization
    size_retweet = 0
    with open('clean_tweets_retweet4000000.txt', 'r') as file:
        for line in file:
            size_retweet = size_retweet + 1

    size_original = 0
    with open('clean_tweets_original4000000.txt', 'r') as file:
        for line in file:
            size_original = size_original + 1

    # open first file
    with open(wordToCount_filename_original, 'r') as file:
        line_count = 0
        for line in file:
            line_count = line_count + 1
            line = line.strip('\n')
            word_tokens = line.split(',')

            word = word_tokens[0]
            wordFreq = word_tokens[1]

            wordToCount_Dict_original[word] = int(wordFreq)

        # open first file
    with open(wordtToCount_filename_retweet, 'r') as file:
        line_count = 0
        for line in file:
            line_count = line_count + 1
            line = line.strip('\n')
            word_tokens = line.split(',')

            word = word_tokens[0]
            wordFreq = word_tokens[1]

            wordToCount_Dict_retweet[word] = int(wordFreq)

    freq_ofWord_original = []


    if sortWords_onOriginal:
        # acquire list of words that occur in both dictionaries
        for word, count in wordToCount_Dict_original.iteritems():
            if word in wordToCount_Dict_retweet:
                wordList.append(word)
                freq_ofWord_original.append(count)
    else:
        for word, count in wordToCount_Dict_retweet.iteritems():
            if word in wordToCount_Dict_original:
                wordList.append(word)
                freq_ofWord_original.append(count)

    print 'Number of co-occuring words between original and retweet tweet sets: ', len(wordList)
    print 'Number of original tweets: ', size_original
    print 'Number of retweets: ', size_retweet

    # sort words in wordList on x_axis according to frequency of said words in original set
    sorted_wordList = [x for _, x in sorted(zip(freq_ofWord_original, wordList ), reverse = True )]

    # get top 100 co-occuring words
    sorted_wordList = sorted_wordList[:100]

    values_original = []
    values_retweet = []
    for word in sorted_wordList:
        # pdb.set_trace()
        values_original.append( wordToCount_Dict_original[word] / float ( size_original ) )
        values_retweet.append( wordToCount_Dict_retweet[word] / float ( size_retweet ) )

    trace_original = go.Scatter(
        x = sorted_wordList,
        y = values_original,
        name = 'Original Tweets',
        line = dict(
            color = ('rgb(205, 12, 24)'),
            width = 4)
    )

    trace_retweet = go.Scatter(
        x = sorted_wordList,
        y = values_retweet,
        name = 'Retweet Tweets',
        line = dict(
            color = ('rgb(22, 96, 167)'),
            width = 4)
    )

    data = [trace_original, trace_retweet]

    if sortWords_onOriginal:
        layout = dict(title = 'Incidence for Top 100 co-occuring words for Original and Retweet Tweet Sets, words sorted on freq in original tweet set',
              xaxis = dict(title = 'Word', range=[0, 99]),
              yaxis = dict(title = 'Incidence Across Tweet Set', range=[0, 1]),
              )
    else:
        layout = dict(title = 'Incidence for Top 100 co-occuring words for Original and Retweet Tweet Sets, words sorted on freq in retweet tweet set',
              xaxis = dict(title = 'Word', range=[0, 99]),
              yaxis = dict(title = 'Incidence Across Tweet Set', range=[0, 1]),
              )

    fig = dict(data=data, layout=layout)

    if sortWords_onOriginal:
        py.iplot(fig, filename='Top100 Words, sorted on word freq in original set')
    else:
        py.iplot(fig, filename='Top100 Words, sorted on word freq in retweet set')



def Aquire_Element_Number (file_name):

    elements_list = []
    elements_unqiue_list = []

    with open(file_name) as file:
        for line in file:

            line = line.strip('\n')
            line = line.strip('\r')

            line_list = line.split(',')

            for element in line_list:
                print element
                elements_list.append(elements_list)
                if element not in elements_list:
                    elements_unqiue_list.append(elements_list)

    print 'elements_list', str(len(elements_list))
    print 'elements_unqiue_list', str(len(elements_unqiue_list))




def Aquire_Handle_toTweets_dict (cleanedTweets_filename, filename_handles, rawTweets_filename, tweets_as_vectors=True):

    Handle_toRawTweets_dict = {}
    Handle_toCleanedTweets_dict = {}

    feature_space = ''

    handle_list = ['thediaryblog', 'FundMatopos', 'thejonnydiesel', 'FuntymeDesigns', 'dianawhitney31', 'ackermanrach', 'Daniela78340481', 'FTM_problems', 'PatriotUsa74', 'PhilaDomphia', 'lauriehuske1', 'Laura_White', 'TheGirFan4Eva', 'MissJade562', 'missadobeah', 'Lbanting', 'LidiaVega5']

    if tweets_as_vectors:
        with open(cleanedTweets_filename) as file_tweets, open(filename_handles) as file_handles:

            if tweets_as_vectors:
                header_line = file_tweets.readline()
                header_line = header_line.strip('\n')
                header_line = header_line.strip('\r')
                feature_space = header_line

            line_count = 0
            for tweet, handle in izip(file_tweets, file_handles):
                line_count = line_count + 1
                print 'Processing tweet', line_count

                handle = handle.strip('\n')
                handle = handle.strip('\r')

                tweet = tweet.strip('\n')
                tweet = tweet.strip('\r')

                # if handle == 'FunSpanishTchr':
                #     if handle not in Handle_toCleanedTweets_dict:
                #         Handle_toCleanedTweets_dict[handle] = []
                #         Handle_toCleanedTweets_dict[handle].append(tweet)
                #     else:
                #         Handle_toCleanedTweets_dict[handle].append(tweet)

                if handle in handle_list:
                    if handle not in Handle_toCleanedTweets_dict:
                        Handle_toCleanedTweets_dict[handle] = []
                        Handle_toCleanedTweets_dict[handle].append(tweet)
                    else:
                        Handle_toCleanedTweets_dict[handle].append(tweet)

    if not tweets_as_vectors:
        with open(rawTweets_filename) as file_tweets, open(filename_handles) as file_handles:

            line_count = 0
            for tweet, handle in izip(file_tweets, file_handles):
                if line_count >= 4000000:
                    break

                line_count = line_count + 1
                print 'Processing tweet', line_count

                handle = handle.strip('\n')
                handle = handle.strip('\r')

                tweet = tweet.strip('\n')

                tweet = unicode(tweet,'utf-8')
                tweet = tweet.encode('unicode-escape')

                # if handle == 'FunSpanishTchr':
                #     if handle not in Handle_toRawTweets_dict:
                #         Handle_toRawTweets_dict[handle] = []
                #         Handle_toRawTweets_dict[handle].append(tweet)
                #     else:
                #         Handle_toRawTweets_dict[handle].append(tweet)

                if handle in handle_list:
                    if handle not in Handle_toCleanedTweets_dict:
                        Handle_toCleanedTweets_dict[handle] = []
                        Handle_toCleanedTweets_dict[handle].append(tweet)
                    else:
                        Handle_toCleanedTweets_dict[handle].append(tweet)


        # print '1:', Handle_toRawTweets_dict['FunSpanishTchr'][1]


        # print '5:', Handle_toRawTweets_dict['FunSpanishTchr'][5]


        # print '20:', Handle_toRawTweets_dict['FunSpanishTchr'][20]

    pdb.set_trace()

    if tweets_as_vectors:

        feature_space = feature_space.split(',')

        print '1:',
        for feature_num in range(len(Handle_toCleanedTweets_dict['FunSpanishTchr'][1].split(','))-1): # go through each feature index
            feature_value = Handle_toCleanedTweets_dict['FunSpanishTchr'][1].split(',')[feature_num] # get the value of the vector at current index
            if int(feature_value) != 0:
                print feature_space[feature_num], '',
        print

        print '5:',
        for feature_num in range(len(Handle_toCleanedTweets_dict['FunSpanishTchr'][5].split(','))-1):
            feature_value = Handle_toCleanedTweets_dict['FunSpanishTchr'][5].split(',')[feature_num]
            if int(feature_value) != 0:
                print feature_space[feature_num], '',
        print

        print '20:',
        for feature_num in range(len(Handle_toCleanedTweets_dict['FunSpanishTchr'][20].split(','))-1):
            feature_value = Handle_toCleanedTweets_dict['FunSpanishTchr'][20].split(',')[feature_num]
            if int(feature_value) != 0:
                print feature_space[feature_num], '',
        print


    pdb.set_trace()

    return Handle_toRawTweets_dict, Handle_toCleanedTweets_dict



if __name__ == '__main__':
    # NEED TO MAKE SURE: HANDLE AND TWEET FILES MATCH UP (in terms of subset [retweet, original, etc.])
        # FINISH INCORPORATE THIS APPROACH for all file addresses
    tweet_setType = ''
    # tweet_setType = 'original'
    # tweet_setType = 'retweet'


    dataSet_size = 4000000
    filename_top1kWords_perUser = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/top_1k_words_perHandle_'  + tweet_setType + str(dataSet_size) + '.txt'
    cleanedTweets_filename = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/clean_tweets_' + tweet_setType + str(dataSet_size) + '.txt'
    filename_handles = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/handles' + tweet_setType + str(dataSet_size) + '.txt'
    if tweet_setType == '':
        filename_handles = 'handles.txt'


    rawTweets_filename = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/tweets.txt'
    top_1k_words_Vocabulary_filename_original = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/top_1k_words_Vocabulary_original4000000.txt'
    top_1k_words_Vocabulary_filename_retweet = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/top_1k_words_Vocabulary_retweet4000000.txt'
    # filename_top1kWords_TOTAL = 'top_1k_words_Vocabulary_'   + str(dataSet_size) + '.txt'

    # topVocab
    tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_topVocab_' + tweet_setType + '4000000.csv'

    # topTF_IDF
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_topTF_IDF_' + tweet_setType + '4000000.csv'

    # emotionDictionary
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_emotionDictionary_' + tweet_setType + '4000000.csv'

    # Aquire_Element_Number('emotions_synonyms.txt')

    # BOW
    # Aquire_Handle_toTweets_dict(tweet_representation_fileName_summed, filename_handles, rawTweets_filename, tweets_as_vectors=False)

    # non BOW


    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_emotionDictionary5_4000000.csv'
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_topVocab10_4000000.csv'
    # tweet_representation_fileName_summed = 'tweets_MatrixRepresentation_topTF_IDF10_4000000.csv'

    # Aquire_Handle_toTweets_dict(tweet_representation_fileName_summed, filename_handles, rawTweets_filename, tweets_as_vectors=True)

    # derive_Bag_of_Words(cleanedTweets_filename, filename_handles)
    # aquireVocab( filename_top1kWords_perUser )

    derive_tweetMatrixRepresentation_topVocab( cleanedTweets_filename, filename_handles )
    # derive_tweetMatrixRepresentation_topTF_IDF( 1000, cleanedTweets_filename )

    # derive_Bag_of_Words_sparse(cleanedTweets_filename, filename_handles)

# NEED TO DO ONCE I HAVE EMOTION DICT (emotion_print.py completes)
    # cleanedTweets_filename = 'clean_tweets_EMOTION4000000.txt'
    # derive_tweetMatrixRepresentation_emotionDict( cleanedTweets_filename, filename_handles )


    Aquire_HandlePoints ( tweet_representation_fileName_summed, filename_handles)

    # PCA_compression('/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/tweets_MatrixRepresentation_emotionDictionary.csv', 5)
    # Wavelet_Compression_New( 'tweets_MatrixRepresentation_topVocab_original4000000_summed.txt')

    # sortWords_onOriginal = True
    # graph_Tweet_Sets(top_1k_words_Vocabulary_filename_original, top_1k_words_Vocabulary_filename_retweet, sortWords_onOriginal)

