import tweet_processing
import emotive_dict_count
import pdb
import csv
import random
import sys
import operator
from itertools import izip
import statistics

# from __future__ import print_function



# def calculate_handleStats( tweets_toHandle_dict, tweet_list ):

#     handle_totalTweetCount = {}
#     handleTo_numLabelledAs_anger_disgust = {}
#     handleTo_numLabelledAs_fear = {}
#     handleTo_numLabelledAs_joy = {}
#     handleTo_numLabelledAs_sadness = {}
#     handleTo_numLabelledAs_surprise = {}
#     handleTo_numLabelledAs_none = {}

#     unique_handleList = []

#     for tweet in tweet_list:
#         handle = tweets_toHandle_dict[tweet]

#         if handle not in unique_handleList:
#             unique_handleList.append(handle)


#         if handle in handle_totalTweetCount:
#             handle_totalTweetCount[handle] = handle_totalTweetCount[handle] + 1
#         else:
#             handle_totalTweetCount[handle] = 1

#         label, emotion_word_count, word_count, dictionary_hit_count, emotion_list = emotive_dict_count.classify_tweet(tweet)

#         # update totals
#         # total_emotion_words = total_emotion_words + emotion_word_count
#         # total_words = total_words + word_count
#         # total_emojis = total_emojis + emojiCount_perTweet[x]
#         # total_tweets = total_tweets + 1

#         for emotion in emotion_list:
#             hit_count = dictionary_hit_count[emotion]
#             if hit_count > 0:
#                 if emotion == 'anger_disgust':
#                     if handle in handleTo_numInvolved_anger_disgust:
#                         handleTo_numInvolved_anger_disgust[handle] = handleTo_numInvolved_anger_disgust[handle] + 1
#                     else:
#                         handleTo_numInvolved_anger_disgust[handle] = 1
#                 elif emotion == 'fear':
#                     if handle in handleTo_numInvolved_fear:
#                         handleTo_numInvolved_fear[handle] = handleTo_numInvolved_fear[handle] + 1
#                     else:
#                         handleTo_numInvolved_fear[handle] = 1
#                 elif emotion == 'joy_love':
#                     if handle in handleTo_numInvolved_joy_love:
#                         handleTo_numInvolved_joy_love[handle] = handleTo_numInvolved_joy_love[handle] + 1
#                     else:
#                         handleTo_numInvolved_joy_love[handle] = 1
#                 elif emotion == 'sadness':
#                     if handle in handleTo_numInvolved_sadness:
#                         handleTo_numInvolved_sadness[handle] = handleTo_numInvolved_sadness[handle] + 1
#                     else:
#                         handleTo_numInvolved_sadness[handle] = 1
#                 elif emotion == 'surprise':
#                     if handle in handleTo_numInvolved_surprise:
#                         handleTo_numInvolved_surprise[handle] = handleTo_numInvolved_surprise[handle] + 1
#                     else:
#                         handleTo_numInvolved_surprise[handle] = 1
#             else:
#                 if (emotion == 'anger_disgust') & (handle not in handleTo_numInvolved_anger_disgust):
#                     handleTo_numInvolved_anger_disgust[handle] = 0
#                 elif (emotion == 'fear') & (handle not in handleTo_numInvolved_fear):
#                     handleTo_numInvolved_fear[handle] = 0
#                 elif (emotion == 'joy_love') & (handle not in handleTo_numInvolved_joy_love):
#                     handleTo_numInvolved_joy_love[handle] = 0
#                 elif (emotion == 'sadness') & (handle not in handleTo_numInvolved_sadness):
#                     handleTo_numInvolved_sadness[handle] = 0
#                 elif (emotion == 'surprise') & (handle not in handleTo_numInvolved_surprise):
#                     handleTo_numInvolved_surprise[handle] = 0

#         # update dicts
#         # if emojiCount_perTweet[x] > 0:
#         if emojiCount > 0:
#             if handle in handleTo_numTweetsWithEmoji:
#                 handleTo_numTweetsWithEmoji[handle] = handleTo_numTweetsWithEmoji[handle] + 1
#             else:
#                 handleTo_numTweetsWithEmoji[handle] = 1
#         else:
#             handleTo_numTweetsWithEmoji[handle] = 0

#         if handle in handle_totalEmotionWordCount:
#             handle_totalEmotionWordCount[handle] = handle_totalEmotionWordCount[handle] + emotion_word_count
#         else:
#             handle_totalEmotionWordCount[handle] = emotion_word_count

#         if handle in handle_totalWordCount:
#             handle_totalWordCount[handle] = handle_totalWordCount[handle] + word_count
#         else:
#             handle_totalWordCount[handle] = word_count

#         if handle in handle_totalEmojiCount:
#             # handle_totalEmojiCount[handle] = handle_totalEmojiCount[handle] + emojiCount_perTweet[x]
#             handle_totalEmojiCount[handle] = handle_totalEmojiCount[handle] + emojiCount
#         else:
#             # handle_totalEmojiCount[handle] = emojiCount_perTweet[x]
#             handle_totalEmojiCount[handle] = emojiCount

#         if label == 'anger_disgust':
#             if handle in handleTo_numLabelledAs_anger_disgust:
#                 handleTo_numLabelledAs_anger_disgust[handle] = handleTo_numLabelledAs_anger_disgust[handle] + 1
#             else:
#                 handleTo_numLabelledAs_anger_disgust[handle] = 1
#         elif label == 'fear':
#             if handle in handleTo_numLabelledAs_fear:
#                 handleTo_numLabelledAs_fear[handle] = handleTo_numLabelledAs_fear[handle] + 1
#             else: handleTo_numLabelledAs_fear[handle] = 1
#         elif label == 'joy_love':
#             if handle in handleTo_numLabelledAs_joy:
#                 handleTo_numLabelledAs_joy[handle] = handleTo_numLabelledAs_joy[handle] + 1
#             else:
#                 handleTo_numLabelledAs_joy[handle] = 1
#         elif label == 'sadness':
#             if handle in handleTo_numLabelledAs_sadness:
#                 handleTo_numLabelledAs_sadness[handle] = handleTo_numLabelledAs_sadness[handle] + 1
#             else:
#                 handleTo_numLabelledAs_sadness[handle] = 1
#         elif label == 'surprise':
#             if handle in handleTo_numLabelledAs_surprise:
#                 handleTo_numLabelledAs_surprise[handle] = handleTo_numLabelledAs_surprise[handle] + 1
#             else:
#                 handleTo_numLabelledAs_surprise[handle] = 1
#         elif label == 'none':
#             if handle in handleTo_numLabelledAs_none:
#                 handleTo_numLabelledAs_none[handle] = handleTo_numLabelledAs_none[handle] + 1
#             else:
#                 handleTo_numLabelledAs_none[handle] = 1

#     return handle_totalTweetCount, handleTo_numLabelledAs_anger_disgust, handleTo_numLabelledAs_fear, handleTo_numLabelledAs_joy, handleTo_numLabelledAs_sadness, handleTo_numLabelledAs_surprise, handleTo_numLabelledAs_none, unique_handleList

def calculate_handleStats( tweet_list ): # take a set of tweets from a specific handle and calculates stats

    totalTweetCount = 0
    numLabelledAs_anger_disgust = 0
    numLabelledAs_fear = 0
    numLabelledAs_joy = 0
    numLabelledAs_sadness = 0
    numLabelledAs_surprise = 0
    numLabelledAs_none = 0

    unique_handleList = []

    for tweet in tweet_list:

        totalTweetCount = totalTweetCount + 1

        label, emotion_word_count, word_count, dictionary_hit_count, emotion_list = emotive_dict_count.classify_tweet(tweet)

        # update totals
        # total_emotion_words = total_emotion_words + emotion_word_count
        # total_words = total_words + word_count
        # total_emojis = total_emojis + emojiCount_perTweet[x]
        # total_tweets = total_tweets + 1

        # for emotion in emotion_list:
        #     hit_count = dictionary_hit_count[emotion]
        #     if hit_count > 0:
        #         if emotion == 'anger_disgust':
        #             numInvolved_anger_disgust = numInvolved_anger_disgust + 1
        #         elif emotion == 'fear':
        #             numInvolved_fear = numInvolved_fear + 1
        #         elif emotion == 'joy_love':
        #             numInvolved_joy_love = numInvolved_joy_love + 1
        #         elif emotion == 'sadness':
        #             numInvolved_sadness = numInvolved_sadness + 1
        #         elif emotion == 'surprise':
        #             numInvolved_surprise = numInvolved_surprise + 1
        #     else:
        #         if (emotion == 'anger_disgust'):
        #             numInvolved_anger_disgust = 0
        #         elif (emotion == 'fear'):
        #             numInvolved_fear = 0
        #         elif (emotion == 'joy_love'):
        #             numInvolved_joy_love = 0
        #         elif (emotion == 'sadness'):
        #             numInvolved_sadness = 0
        #         elif (emotion == 'surprise'):
        #             numInvolved_surprise = 0

        # update dicts
        # if emojiCount_perTweet[x] > 0:
        # if emojiCount > 0:
        #     if handle in handleTo_numTweetsWithEmoji:
        #         handleTo_numTweetsWithEmoji[handle] = handleTo_numTweetsWithEmoji[handle] + 1
        #     else:
        #         handleTo_numTweetsWithEmoji[handle] = 1
        # else:
        #     handleTo_numTweetsWithEmoji[handle] = 0

        # if handle in handle_totalEmotionWordCount:
        #     handle_totalEmotionWordCount[handle] = handle_totalEmotionWordCount[handle] + emotion_word_count
        # else:
        #     handle_totalEmotionWordCount[handle] = emotion_word_count

        # if handle in handle_totalWordCount:
        #     handle_totalWordCount[handle] = handle_totalWordCount[handle] + word_count
        # else:
        #     handle_totalWordCount[handle] = word_count

        # if handle in handle_totalEmojiCount:
        #     # handle_totalEmojiCount[handle] = handle_totalEmojiCount[handle] + emojiCount_perTweet[x]
        #     handle_totalEmojiCount[handle] = handle_totalEmojiCount[handle] + emojiCount
        # else:
        #     # handle_totalEmojiCount[handle] = emojiCount_perTweet[x]
        #     handle_totalEmojiCount[handle] = emojiCount

        if label == 'anger_disgust':
            numLabelledAs_anger_disgust = numLabelledAs_anger_disgust + 1
        elif label == 'fear':
            numLabelledAs_fear = numLabelledAs_fear + 1
        elif label == 'joy_love':
            numLabelledAs_joy = numLabelledAs_joy + 1
        elif label == 'sadness':
            numLabelledAs_sadness = numLabelledAs_sadness + 1
        elif label == 'surprise':
            numLabelledAs_surprise = numLabelledAs_surprise + 1
        elif label == 'none':
            numLabelledAs_none = numLabelledAs_none + 1

    return totalTweetCount, numLabelledAs_anger_disgust, numLabelledAs_fear, numLabelledAs_joy, numLabelledAs_sadness, numLabelledAs_surprise, numLabelledAs_none


print 'Process Started'

# filename_tweets = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/tweets.txt'

tweet_setType = ''
# tweet_setType = 'original'
# tweet_setType = 'retweet'

total_tweets = 4000000
total_handles = total_tweets

# total_tweets = 2000000

filename_handles = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/handles' + tweet_setType + str(total_tweets) + '.txt'
filename_tweets = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/clean_tweets_' + tweet_setType + str(total_tweets) + '.txt'
filename_emojiCounts = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/emojiCount_perTweet_' + tweet_setType + str(total_tweets) + '.txt'

print filename_handles
print filename_tweets
print filename_emojiCounts

if tweet_setType == '':
    filename_handles = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/handles' + tweet_setType +'.txt'
    filename_tweets = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/clean_tweets_' + tweet_setType + str(total_tweets) + '.txt'
    filename_emojiCounts = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/emojiCount_perTweet_' + tweet_setType + str(total_tweets) + '.txt'


# total_handles = 1000
# total_tweets = 1000


# filename_tweets = '/home/kr623/tweets.txt'
# filename_handles = '/home/kr623/handles.txt'

# edit this here? or try to run on server?
# see what format the data file is in

# mylist = ['a', 'b', 'c']
# print mylist[z]


# tweetList, emojiCount_perTweet = tweet_processing.clean(filename_tweets)



# handleList = []

# N= 1200
# f=open(filename_handles)
# for i in range(N):
#     line=f.next().strip()
#     handleList.append(line)
# f.close()

# need to incorporate emotive_dict_count
    # just pass in tweet to emotive_dict_count
    # convert emotive_dict_count such that it can classify a single tweet
    # update emotive_dict_count from email

# for each individual,
# total # emotion words (DictHit), total # words, total # emojis, how many tweets, counts for # of tweets deemed as belonging to each of our 6 emotional categories (turn into fraction)
# how many emotion words did they have, how many emojis, how many tweets they had
# Come up with average from all of these results, such that we can examine the degree observed deviation for an individual Twitter user from this mean.

# Dict #
# handle --> count emotion words
handle_totalEmotionWordCount = {}
# handle --> total # of words
handle_totalWordCount = {}
# handle --> total # of emojis
handle_totalEmojiCount = {}
# handle --> how many tweets
handle_totalTweetCount = {}
# handle --> counts for # of tweets deemed as belonging to each emotion (turned into fraction)
handleTo_numInvolved_anger_disgust = {}
handleTo_numInvolved_fear = {}
handleTo_numInvolved_joy_love = {}
handleTo_numInvolved_sadness = {}
handleTo_numInvolved_surprise = {}

handleTo_numLabelledAs_anger_disgust = {}
handleTo_numLabelledAs_fear = {}
handleTo_numLabelledAs_joy = {}
handleTo_numLabelledAs_sadness = {}
handleTo_numLabelledAs_surprise = {}
handleTo_numLabelledAs_none = {}

handle_emotionProportions = {}

handleTo_numTweetsWithEmoji = {}

total_emotion_words = 0
total_words = 0
total_emojis = 0


unique_handleList = []
# for each tweet

#creat
# dictionary of handle to dictionary of word to count



# for each tweet
# tokenize
# go through each word and
    # if not in dictionary, add, with count of 0
    # if in dicitonary, update count


handleTo_wordCountDict = {}


# possible to only load in one tweet and handle at a time?


# aquire lsit of tweet counts for each tweet
# has_50_Tweets_Handlelist = []
# for x in range(len(tweetList)):
#     handle = handleList[x]

#     if handle not in handle_totalTweetCount:
#         total_handles = total_handles + 1

#     if handle in handle_totalTweetCount:
#         handle_totalTweetCount[handle] = handle_totalTweetCount[handle] + 1
#     else:
#         handle_totalTweetCount[handle] = 1

# if len(tweetList) != len(handleList):
#     print 'ERROR: mismatch in length of handle file and tweet file'

tweetToHandle_dict = {}
handleToEmotionDictHit = {} # dictionary of dictionaries that maps a handle to the total dict hits for each emotion
handle_toTweetList_dict = {}

# for x in range( len(file_handles) ):




line_counter = 0
with open(filename_tweets) as file_tweets, open(filename_handles) as file_handles, open(filename_emojiCounts) as file_emojiCounts:

    for tweet, handle, emojiCount in izip(file_tweets, file_handles, file_emojiCounts):

        print tweet
        tweet = tweet.strip('\n')
        handle = handle.strip('\n')
        emojiCount = emojiCount.strip('\n')

        if handle in handle_toTweetList_dict:
            handle_toTweetList_dict[handle].append(tweet)
        else:
            handle_toTweetList_dict[handle] = []
            handle_toTweetList_dict[handle].append(tweet)

        print 'processing line: ', line_counter
        line_counter = line_counter + 1


        if handle in handle_totalTweetCount:
            handle_totalTweetCount[handle] = handle_totalTweetCount[handle] + 1
        else:
            handle_totalTweetCount[handle] = 1

        tweetToHandle_dict[tweet] = handle


        wordsInTweet_list = tweet.split()

        if handle not in handleTo_wordCountDict:
            handleTo_wordCountDict[handle] = {}

        for word in wordsInTweet_list:
            if word in handleTo_wordCountDict[handle]:
                handleTo_wordCountDict[handle][word] = handleTo_wordCountDict[handle][word] + 1
                # print 'incrementing word count for ', word, ' to be ', handleTo_wordCountDict[handle][word]
            else:
                handleTo_wordCountDict[handle][word] = 1
                # print 'word count for ', word, ' is ', handleTo_wordCountDict[handle][word]


        if handle not in unique_handleList:
            unique_handleList.append(handle)

        # acquire data
        label, emotion_word_count, word_count, dictionary_hit_count, emotion_list = emotive_dict_count.classify_tweet(tweet)

        # update totals
        total_emotion_words = total_emotion_words + emotion_word_count
        total_words = total_words + word_count
        # total_emojis = total_emojis + emojiCount_perTweet[x]
        total_emojis = total_emojis + int(emojiCount)
        # total_tweets = total_tweets + 1

        if handle not in handleToEmotionDictHit:
            handleToEmotionDictHit[handle] = {}

        for emotion in emotion_list:
            hit_count = dictionary_hit_count[emotion]
            # print 'hit_count: ', hit_count
            if hit_count > 0:
                if emotion == 'anger_disgust':
                    if handle in handleTo_numInvolved_anger_disgust: # num tweets for this handle that involved anger_disgust
                        handleTo_numInvolved_anger_disgust[handle] = handleTo_numInvolved_anger_disgust[handle] + 1
                    else:
                        handleTo_numInvolved_anger_disgust[handle] = 1
                elif emotion == 'fear':
                    if handle in handleTo_numInvolved_fear:
                        handleTo_numInvolved_fear[handle] = handleTo_numInvolved_fear[handle] + 1
                    else:
                        handleTo_numInvolved_fear[handle] = 1
                elif emotion == 'joy_love':
                    if handle in handleTo_numInvolved_joy_love:
                        handleTo_numInvolved_joy_love[handle] = handleTo_numInvolved_joy_love[handle] + 1
                    else:
                        handleTo_numInvolved_joy_love[handle] = 1
                elif emotion == 'sadness':
                    if handle in handleTo_numInvolved_sadness:
                        handleTo_numInvolved_sadness[handle] = handleTo_numInvolved_sadness[handle] + 1
                    else:
                        handleTo_numInvolved_sadness[handle] = 1
                elif emotion == 'surprise':
                    if handle in handleTo_numInvolved_surprise:
                        handleTo_numInvolved_surprise[handle] = handleTo_numInvolved_surprise[handle] + 1
                    else:
                        handleTo_numInvolved_surprise[handle] = 1

                # number of tweets that hit a given dictionary, for each user
                if emotion in handleToEmotionDictHit[handle]:
                    handleToEmotionDictHit[handle][emotion] = handleToEmotionDictHit[handle][emotion] + 1
                else:
                    handleToEmotionDictHit[handle][emotion] = 1


            else:
                if (emotion == 'anger_disgust') & (handle not in handleTo_numInvolved_anger_disgust):
                    handleTo_numInvolved_anger_disgust[handle] = 0
                elif (emotion == 'fear') & (handle not in handleTo_numInvolved_fear):
                    handleTo_numInvolved_fear[handle] = 0
                elif (emotion == 'joy_love') & (handle not in handleTo_numInvolved_joy_love):
                    handleTo_numInvolved_joy_love[handle] = 0
                elif (emotion == 'sadness') & (handle not in handleTo_numInvolved_sadness):
                    handleTo_numInvolved_sadness[handle] = 0
                elif (emotion == 'surprise') & (handle not in handleTo_numInvolved_surprise):
                    handleTo_numInvolved_surprise[handle] = 0



        # update dicts
        # if emojiCount_perTweet[x] > 0:
        if emojiCount > 0:
            if handle in handleTo_numTweetsWithEmoji:
                handleTo_numTweetsWithEmoji[handle] = handleTo_numTweetsWithEmoji[handle] + 1
            else:
                handleTo_numTweetsWithEmoji[handle] = 1
        else:
            handleTo_numTweetsWithEmoji[handle] = 0

        if handle in handle_totalEmotionWordCount:
            handle_totalEmotionWordCount[handle] = handle_totalEmotionWordCount[handle] + emotion_word_count
        else:
            handle_totalEmotionWordCount[handle] = emotion_word_count

        if handle in handle_totalWordCount:
            handle_totalWordCount[handle] = handle_totalWordCount[handle] + word_count
        else:
            handle_totalWordCount[handle] = word_count

        if handle in handle_totalEmojiCount:
            # handle_totalEmojiCount[handle] = handle_totalEmojiCount[handle] + emojiCount_perTweet[x]
            handle_totalEmojiCount[handle] = handle_totalEmojiCount[handle] + emojiCount
        else:
            # handle_totalEmojiCount[handle] = emojiCount_perTweet[x]
            handle_totalEmojiCount[handle] = emojiCount


        if label == 'anger_disgust':
            if handle in handleTo_numLabelledAs_anger_disgust:
                handleTo_numLabelledAs_anger_disgust[handle] = handleTo_numLabelledAs_anger_disgust[handle] + 1
            else:
                handleTo_numLabelledAs_anger_disgust[handle] = 1
        elif label == 'fear':
            if handle in handleTo_numLabelledAs_fear:
                handleTo_numLabelledAs_fear[handle] = handleTo_numLabelledAs_fear[handle] + 1
            else: handleTo_numLabelledAs_fear[handle] = 1
        elif label == 'joy_love':
            if handle in handleTo_numLabelledAs_joy:
                handleTo_numLabelledAs_joy[handle] = handleTo_numLabelledAs_joy[handle] + 1
            else:
                handleTo_numLabelledAs_joy[handle] = 1
        elif label == 'sadness':
            if handle in handleTo_numLabelledAs_sadness:
                handleTo_numLabelledAs_sadness[handle] = handleTo_numLabelledAs_sadness[handle] + 1
            else:
                handleTo_numLabelledAs_sadness[handle] = 1
        elif label == 'surprise':
            if handle in handleTo_numLabelledAs_surprise:
                handleTo_numLabelledAs_surprise[handle] = handleTo_numLabelledAs_surprise[handle] + 1
            else:
                handleTo_numLabelledAs_surprise[handle] = 1
        elif label == 'none':
            if handle in handleTo_numLabelledAs_none:
                handleTo_numLabelledAs_none[handle] = handleTo_numLabelledAs_none[handle] + 1
            else:
                handleTo_numLabelledAs_none[handle] = 1

# do sums and subsequent calculations for averages
# average emotional words
avg_emotional_words = total_emotion_words / float( total_tweets )
# average total # words
avg_words = total_words / float( total_tweets )
# average total # emojis
avg_emojis = total_emojis / float( total_tweets )
# average how many tweets per user
avg_tweets_per_user = total_tweets / float( len(unique_handleList) )
# counts for # of tweets deemd as belonging to each of our 6 emotional categories (turn into fraction)



# write out emotion_dictHit data
outfile = open("emotionDictHits_perHandle.txt", 'w')
for handle, emotionDictHits_dict in handleToEmotionDictHit.iteritems():
    outfile.write('handle,' + handle + "\n")

    for emotion, hit_count in handleToEmotionDictHit.iteritems():
        outfile.write(emotion + ',' + str(hit_count) + "\n")

outfile.close()
handleToEmotionDictHit = {}


# write out top1k words per user data
outfile = open('top_1k_words_perHandle_' + tweet_setType + str(total_tweets) + '.txt', 'w')
for handle, wordCount_dict in handleTo_wordCountDict.iteritems():

    outfile.write('handle,' + handle + "\n")
    ranked_wordCount_dict = sorted(wordCount_dict.items(), key=operator.itemgetter(1), reverse = True)

    X = 0
    for word, count in ranked_wordCount_dict:
        X = X + 1

        if X <= 1000:
            outfile.write(word + ',' + str(count) + "\n")
        else:
            break

outfile.close()
print 'Created top_1k_words_perHandle ...'
pdb.set_trace()

handleTo_wordCountDict = {}


# output file with list of handles to top words with counts

anger_disgust_rankSum = 0
fear_rankSum = 0
joy_love_rankSum = 0
sadness_rankSum = 0
surprise_rankSum = 0
rank_count = 0

csv_row_total = ['handle', 'total emotion word count [# number dict hits]', 'total word count', 'total emoji count', 'total tweet count',
    'anger_disgust [proportion labeled as]', 'fear [proportion labeled as]', 'joy_love [proportion labeled as]', 'sadness [proportion labeled as]',
    'surprise [proportion labeled as]', 'none [proportion labeled as]', 'anger_disgust rank', 'fear rank', 'joy_love rank', 'sadness rank',
    'surprise rank', 'num tweets w/ anger_disgust (corresponding dict hit)', 'num tweets w/ fear (corresponding dict hit)', 'num tweets w/ joy_love (corresponding dict hit)',
    'num tweets w/ sadness (corresponding dict hit)', 'num tweets w/ surprise (corresponding dict hit)', 'num tweets with at least one emoji', 'num tweet with no label' ]
# with myFile:
# writer = csv.writer(myFile)
# writer.writerows(csv_row)

csv_Data_total = []
csv_Data_total.append(csv_row_total)

handle_bucket_highTweets = []
handle_bucket_medTweets = []
handle_bucket_lowTweets = []

# write out file for handles in each bucket

lowHandle_count = 0
medHandle_count = 0
highHandle_count = 0

outfile_line_lowHandles = ''
outfile_line_medHandles = ''
outfile_line_highHandles = ''

print ''# data for each handle
for handle in unique_handleList:

    outfile_line = handle + ','
    # assign handles to a bucket based on # of tweets
    if (handle_totalTweetCount[handle] >= 20) & (handle_totalTweetCount[handle] <= 200):
        handle_bucket_lowTweets.append(handle)
        outfile_line_lowHandles = outfile_line_lowHandles + outfile_line
        lowHandle_count = lowHandle_count + 1
    elif (handle_totalTweetCount[handle] > 200) & (handle_totalTweetCount[handle] <= 2000):
        handle_bucket_medTweets.append(handle)
        outfile_line_medHandles = outfile_line_medHandles + outfile_line
        medHandle_count = medHandle_count + 1
    elif (handle_totalTweetCount[handle] > 2000):
        handle_bucket_highTweets.append(handle)
        outfile_line_highHandles = outfile_line_highHandles + outfile_line
        highHandle_count = highHandle_count + 1

    anger_disgust_proportion = 0
    fear_proportion = 0
    joy_love_proportion = 0
    sadness_proportion = 0
    surprise_proportion = 0
    none_proportion = 0

    if handle in handleTo_numLabelledAs_anger_disgust:
        anger_disgust_proportion = handleTo_numLabelledAs_anger_disgust[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_fear:
        fear_proportion = handleTo_numLabelledAs_fear[handle] / float ( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_joy:
        joy_love_proportion = handleTo_numLabelledAs_joy[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_sadness:
        sadness_proportion = handleTo_numLabelledAs_sadness[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_surprise:
        surprise_proportion = handleTo_numLabelledAs_surprise[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_none:
        none_proportion = handleTo_numLabelledAs_none[handle] / float( handle_totalTweetCount[handle] )

    rank_anger_disgust = 0
    rank_fear = 0
    rank_joy_love = 0
    rank_sadness = 0
    rank_surprise = 0

    emotion_to_proportion_dict = {}
    emotion_to_proportion_dict['anger_disgust'] = anger_disgust_proportion
    emotion_to_proportion_dict['fear'] = fear_proportion
    emotion_to_proportion_dict['joy_love'] = joy_love_proportion
    emotion_to_proportion_dict['sadness'] = sadness_proportion
    emotion_to_proportion_dict['surprise'] = surprise_proportion

    rank_list = sorted(emotion_to_proportion_dict, key=emotion_to_proportion_dict.get, reverse=True)

    for x in range(len(rank_list)):
        if rank_list[x] == 'anger_disgust':
            rank_anger_disgust = x + 1
        elif rank_list[x] == 'fear':
            rank_fear = x + 1
        elif rank_list[x] == 'joy_love':
            rank_joy_love = x + 1
        elif rank_list[x] == 'sadness':
            rank_sadness = x + 1
        elif rank_list[x] == 'surprise':
            rank_surprise = x + 1

    # sums for average rank for each emotion
    anger_disgust_rankSum = anger_disgust_rankSum + rank_anger_disgust
    fear_rankSum = fear_rankSum + rank_fear
    joy_love_rankSum = joy_love_rankSum + rank_joy_love
    sadness_rankSum = sadness_rankSum + rank_sadness
    surprise_rankSum = surprise_rankSum + rank_surprise
    rank_count = rank_count + 1

    # COMBINE EMOTIONS

    # print handle
    # print 'total emotion word count [# number dict hits]:', handle_totalEmotionWordCount[handle]
    # print 'total word count:', handle_totalWordCount[handle]
    # print 'total emoji count:', handle_totalEmojiCount[handle]
    # print 'total tweet count:', handle_totalTweetCount[handle]

    # print 'anger_disgust [proportion labeled as]:', anger_disgust_proportion
    # print 'fear [proportion labeled as]:', fear_proportion
    # print 'joy_love [proportion labeled as]:', joy_love_proportion
    # print 'sadness [proportion labeled as]:', sadness_proportion
    # print 'surprise [proportion labeled as]:', surprise_proportion
    # print 'none [proportion labeled as]:', none_proportion

    # print 'anger_disgust rank:', rank_anger_disgust
    # print 'fear rank:', rank_fear
    # print 'joy_love rank:', rank_joy_love
    # print 'sadness rank:', rank_sadness
    # print 'surprise rank:', rank_surprise

    # print 'num tweets w/ anger_disgust (corresponding dict hit):', handleTo_numInvolved_anger_disgust[handle]
    # print 'num tweets w/ fear (corresponding dict hit):', handleTo_numInvolved_fear[handle]
    # print 'num tweets w/ joy_love (corresponding dict hit):', handleTo_numInvolved_joy_love[handle]
    # print 'num tweets w/ sadness (corresponding dict hit):', handleTo_numInvolved_sadness[handle]
    # print 'num tweets w/ surprise (corresponding dict hit):', handleTo_numInvolved_surprise[handle]
    # print 'num tweets with at least one emoji:', handleTo_numTweetsWithEmoji[handle]
    # print 'num tweet with no label: ', handle_totalTweetCount[handle] * none_proportion
    # print

    num_tweets_NoLabel = handle_totalTweetCount[handle] * none_proportion

    csv_row_total = [
        handle,
        handle_totalEmotionWordCount[handle],
        handle_totalWordCount[handle],
        handle_totalEmojiCount[handle],
        handle_totalTweetCount[handle],
        anger_disgust_proportion,
        fear_proportion,
        joy_love_proportion,
        sadness_proportion,
        surprise_proportion,
        none_proportion,
        rank_anger_disgust,
        rank_fear,
        rank_joy_love,
        rank_sadness,
        rank_surprise,
        handleTo_numInvolved_anger_disgust[handle],
        handleTo_numInvolved_fear[handle],
        handleTo_numInvolved_joy_love[handle],
        handleTo_numInvolved_sadness[handle],
        handleTo_numInvolved_surprise[handle],
        handleTo_numTweetsWithEmoji[handle],
        num_tweets_NoLabel
        ]

    csv_Data_total.append(csv_row_total)

    # myFile = open('print_data.csv', 'w+')
    # with myFile:
    # writer = csv.writer(myFile)
    # writer.writerows(csv_row)

    # do csv format with handle as first column

# print 'num in low bucket:', len(handle_bucket_lowTweets)


# write out handle lists
outfile_lowHandles = open('low_handles_' + tweet_setType + str(total_tweets) + '.txt', 'w')
outfile_medHandles = open('med_handles_' + tweet_setType + str(total_tweets) + '.txt', 'w')
outfile_highHandles = open('high_handles_' + tweet_setType + str(total_tweets) + '.txt', 'w')

# removing trailling ','
outfile_line_lowHandles = outfile_line_lowHandles[:-1]
outfile_line_medHandles = outfile_line_medHandles[:-1]
outfile_line_highHandles = outfile_line_highHandles[:-1]

outfile_lowHandles.write(outfile_line_lowHandles)
outfile_medHandles.write(outfile_line_medHandles)
outfile_highHandles.write(outfile_line_highHandles)

outfile_lowHandles.close()
outfile_medHandles.close()
outfile_highHandles.close()

print  'num low handles: ', str(lowHandle_count)
print  'num med handles: ', str(medHandle_count)
print  'num high handles: ', str(highHandle_count)

csv_Data_summary = []
csv_row_summary = []
csv_row_summary = ['avg_emotional_words', 'avg_words', 'avg_emojis', 'avg_tweets_per_user', 'avg anger_disgust rank', 'avg fear rank',
    'avg joy_love rank', 'avg sadness rank', 'avg surprise rank' ]
csv_Data_summary.append(csv_row_summary)

# myFile = open('print_data.csv', 'w+')
# with myFile:
# writer = csv.writer(myFile)
# writer.writerows(csv_row)

# print macro data

# print 'avg_emotional_words', avg_emotional_words
# print 'avg_words', avg_words
# print 'avg_emojis', avg_emojis
print 'total handles:', len(handle_totalTweetCount)
print 'avg_tweets_per_user', avg_tweets_per_user
# print
# print 'avg anger_disgust rank:', anger_disgust_rankSum / float(rank_count)
# print 'avg fear rank:', fear_rankSum / float(rank_count)
# print 'avg joy_love rank:', joy_love_rankSum / float(rank_count)
# print 'avg sadness rank:', sadness_rankSum / float(rank_count)
# print 'avg surprise rank:', surprise_rankSum / float(rank_count)

csv_row_summary = [ avg_emotional_words, avg_words, avg_emojis, avg_tweets_per_user, anger_disgust_rankSum / float(rank_count),
    fear_rankSum / float(rank_count), joy_love_rankSum / float(rank_count), sadness_rankSum / float(rank_count), surprise_rankSum / float(rank_count) ]
csv_Data_summary.append(csv_row_summary)


myFile = open('print_data_total_' + str(total_tweets) + '.csv', 'w')
writer = csv.writer(myFile)
writer.writerows(csv_Data_total)

myFile = open('print_data_summary_' + str(total_tweets) + '.csv', 'w')
writer = csv.writer(myFile)
writer.writerows(csv_Data_summary)


# for each bucket of handles, provide
# proportion of tweets that have emotions
    # sum counts of all tweets that have emotion, then divide by total num tweets
# distrubtions of emotions for each bucket
    # proportion of tweets belonging to each emotion
# range of distrubtions for each emotion across users in a bcuket
    # for each emotion, go through all users and record max and min proportions
    #

# used in the random selection of tweets for different buckets later
tweets_highBucket = []
tweets_medBucket = []
tweets_lowBucket = []

for tweet, handle in tweetToHandle_dict.iteritems():
    if handle in handle_bucket_highTweets:
        tweets_highBucket.append(tweet)
    elif handle in handle_bucket_medTweets:
        tweets_medBucket.append(tweet)
    elif handle in handle_bucket_lowTweets:
        tweets_lowBucket.append(tweet)

comprehensive_bucket_File = open('comprehensive_bucket_data_' + tweet_setType + str(total_tweets) + '.csv', 'w')


# BEGIN low BUCKET ANALYTICS
emotion_tweet_sum = 0
emotion_proportionList_lowBucket = []
anger_disgust_proportionList_lowBucket = []
fear_proportionList_lowBucket = []
joy_love_proportionList_lowBucket = []
sadnes_proportionList_lowBucket = []
surprise_proportionList_lowBucket = []

emotion_maxProportion_lowBucket = 0
anger_disgust_maxProportion_lowBucket = 0
fear_maxProportion_lowBucket = 0
joy_love_maxProportion_lowBucket = 0
sadness_maxProportion_lowBucket = 0
surprise_maxProportion_lowBucket = 0

emotion_minProportion_lowBucket = 1
anger_disgust_minProportion_lowBucket = 1
fear_minProportion_lowBucket = 1
joy_love_minProportion_lowBucket = 1
sadness_minProportion_lowBucket = 1
surprise_minProportion_lowBucket = 1

anger_disgust_proportion_handle = 0
fear_proportion_handle = 0
joy_love_proportion_handle = 0
sadness_proportion_handle = 0
surprise_proportion_handle = 0
none_proportion_handle = 0

totalTweets_lowBucket = 0

for handle in handle_bucket_lowTweets:

    totalTweets_lowBucket = totalTweets_lowBucket + handle_totalTweetCount[handle]

    anger_disgust_proportion = 0
    fear_proportion = 0
    joy_love_proportion = 0
    sadness_proportion = 0
    surprise_proportion = 0
    none_proportion = 0

    if handle in handleTo_numLabelledAs_anger_disgust:
        anger_disgust_proportion_handle = handleTo_numLabelledAs_anger_disgust[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_fear:
        fear_proportion_handle = handleTo_numLabelledAs_fear[handle] / float ( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_joy:
        joy_love_proportion_handle = handleTo_numLabelledAs_joy[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_sadness:
        sadness_proportion_handle = handleTo_numLabelledAs_sadness[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_surprise:
        surprise_proportion_handle = handleTo_numLabelledAs_surprise[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_none:
        none_proportion_handle = handleTo_numLabelledAs_none[handle] / float( handle_totalTweetCount[handle] )

    # proportion of tweets that have emotions for this handle
    none_proportion_handle = handleTo_numLabelledAs_none[handle] / float( handle_totalTweetCount[handle] )
    emotion_proportion_handle = 1 - none_proportion_handle
    emotion_proportionList_lowBucket.append(emotion_proportion_handle)

    # anger_disgust proportion list append
    anger_disgust_proportionList_lowBucket.append(anger_disgust_proportion_handle)

    # fear proportion list append
    fear_proportionList_lowBucket.append(fear_proportion_handle)

    # joy_love proportion list append
    joy_love_proportionList_lowBucket.append(joy_love_proportion_handle)

    # sadness proportion list append
    sadnes_proportionList_lowBucket.append(sadness_proportion_handle)

    # surprise proportion list append
    surprise_proportionList_lowBucket.append(surprise_proportion_handle)

    # update Max and Min proportions if neccesary
    if emotion_maxProportion_lowBucket < emotion_proportion_handle:
        emotion_maxProportion_lowBucket = emotion_proportion_handle
    if anger_disgust_maxProportion_lowBucket < anger_disgust_proportion_handle:
        anger_disgust_maxProportion_lowBucket = anger_disgust_proportion_handle
    if fear_maxProportion_lowBucket < fear_proportion_handle:
        fear_maxProportion_lowBucket = fear_proportion_handle
    if joy_love_maxProportion_lowBucket < joy_love_proportion_handle:
        joy_love_maxProportion_lowBucket = joy_love_proportion_handle
    if sadness_maxProportion_lowBucket < sadness_proportion_handle:
        sadness_maxProportion_lowBucket = sadness_proportion_handle
    if surprise_maxProportion_lowBucket < surprise_proportion_handle:
        surprise_maxProportion_lowBucket = surprise_proportion_handle

    # print 'max proportion emotion: ', emotion_maxProportion_lowBucket

    if emotion_minProportion_lowBucket > emotion_proportion_handle:
        emotion_minProportion_lowBucket = emotion_proportion_handle
    if anger_disgust_minProportion_lowBucket > anger_disgust_proportion_handle:
        anger_disgust_minProportion_lowBucket = anger_disgust_proportion_handle
    if fear_minProportion_lowBucket > fear_proportion_handle:
        fear_minProportion_lowBucket = fear_proportion_handle
    if joy_love_minProportion_lowBucket > joy_love_proportion_handle:
        joy_love_minProportion_lowBucket = joy_love_proportion_handle
    if sadness_minProportion_lowBucket > sadness_proportion_handle:
        sadness_minProportion_lowBucket = sadness_proportion_handle
    if surprise_minProportion_lowBucket > surprise_proportion_handle:
        surprise_minProportion_lowBucket = surprise_proportion_handle

    # print 'min proportion emotion: ', emotion_minProportion_lowBucket


# proportion of tweets that have emotions for this bucket
proportion_emotion_lowBucket = statistics.mean ( emotion_proportionList_lowBucket )
medianProportion_emotion_lowBucket = statistics.median ( emotion_proportionList_lowBucket )

# proportion of anger_disgust tweets that have emotions for this bucket
proportion_anger_disgust_lowBucket = statistics.mean ( anger_disgust_proportionList_lowBucket )
medianProportion_anger_disgust_lowBucket = statistics.median ( anger_disgust_proportionList_lowBucket )

# proportion of fear tweets that have emotions for this bucket
proportion_fear_lowBucket = statistics.mean ( fear_proportionList_lowBucket )
medianProportion_fear_lowBucket = statistics.median ( fear_proportionList_lowBucket )

# proportion of joy_love tweets that have emotions for this bucket
proportion_joy_love_lowBucket = statistics.mean ( joy_love_proportionList_lowBucket )
medianProportion_joy_love_lowBucket = statistics.median ( joy_love_proportionList_lowBucket )

# proportion of sadness tweets that have emotions for this bucket
proportion_sadness_lowBucket = statistics.mean ( sadnes_proportionList_lowBucket )
medianProportion_sadness_lowBucket = statistics.median ( sadnes_proportionList_lowBucket )

# proportion of surprise tweets that have emotions for this bucket
proportion_surprise_lowBucket = statistics.mean ( surprise_proportionList_lowBucket )
medianProportion_surprise_lowBucket = statistics.median ( surprise_proportionList_lowBucket )

# for each bucket of handles, provide
# proportion of tweets that have emotions
    # sum counts of all tweets that have emotion, then divide by total num tweets
# distrubtions of emotions for each bucket
    # proportion of tweets belonging to each emotion
# range of distrubtions for each emotion across users in a bcuket
    # for each emotion, go through all users and record max and min proportions
    #

# write out to csv
csv_Data_lowBucket = []
csv_row_lowBucket = [   '',
                                        'Number of handles in this bucket',
                                        'Number of tweets corresponding to the handles in this bucket',

                                        'Proportion of all tweets in this bucket that are labeled as emotional',
                                        'Highest proportion of tweets labeled as emotional for a handle in this bucket',
                                        'Lowest proportion of tweets labeled as emotional for a handle in this bucket',
                                        'Median proportion of tweets labeled as emotional for a handle in this bucket',

                                        'Proportion of all tweets in this bucket that are labeled as anger_disgust',
                                        'Highest proportion of tweets labeled as anger_disgust for a handle in this bucket',
                                        'Lowest proportion of tweets labeled as anger_disgust for a handle in this bucket',
                                        'Median proportion of tweets labeled as anger_disgust for a handle in this bucket',

                                        'Proportion of all tweets in this bucket that are labeled as fear',
                                        'Highest proportion of tweets labeled as fear for a handle in this bucket',
                                        'Lowest proportion of tweets labeled as fear for a handle in this bucket',
                                        'Median proportion of tweets labeled as fear for a handle in this bucket',

                                        'Proportion of all tweets in this bucket that are labeled as joy_love',
                                        'Highest proportion of tweets labeled as joy_love for a handle in this bucket',
                                        'Lowest proportion of tweets labeled as joy_love for a handle in this bucket',
                                        'Median proportion of tweets labeled as joy_love for a handle in this bucket',

                                        'Proportion of all tweets in this bucket that are labeled as sadness',
                                        'Highest proportion of tweets labeled as sadness for a handle in this bucket',
                                        'Lowest proportion of tweets labeled as sadness for a handle in this bucket',
                                        'Median proportion of tweets labeled as sadness for a handle in this bucket',

                                        'Proportion of all tweets in this bucket that are labeled as surprise',
                                        'Highest proportion of tweets labeled as surprise for a handle in this bucket',
                                        'Lowest proportion of tweets labeled as surprise for a handle in this bucket',
                                        'Median proportion of tweets labeled as surprise for a handle in this bucket'
                                        ]
csv_Data_lowBucket.append(csv_row_lowBucket)

# print 'num in low bucket:', len(handle_bucket_lowTweets)

csv_row_lowBucket = [   '~~~ lowBucket ~~~',
                                        len(handle_bucket_lowTweets),
                                        totalTweets_lowBucket,

                                        proportion_emotion_lowBucket,
                                        emotion_maxProportion_lowBucket,
                                        emotion_minProportion_lowBucket,
                                        medianProportion_emotion_lowBucket,

                                        proportion_anger_disgust_lowBucket,
                                        anger_disgust_maxProportion_lowBucket,
                                        anger_disgust_minProportion_lowBucket,
                                        medianProportion_anger_disgust_lowBucket,

                                        proportion_fear_lowBucket,
                                        fear_maxProportion_lowBucket,
                                        fear_minProportion_lowBucket,
                                        medianProportion_fear_lowBucket,

                                        proportion_joy_love_lowBucket,
                                        joy_love_maxProportion_lowBucket,
                                        joy_love_minProportion_lowBucket,
                                        medianProportion_joy_love_lowBucket,

                                        proportion_sadness_lowBucket,
                                        sadness_maxProportion_lowBucket,
                                        sadness_minProportion_lowBucket,
                                        medianProportion_sadness_lowBucket,

                                        proportion_surprise_lowBucket,
                                        surprise_maxProportion_lowBucket,
                                        surprise_minProportion_lowBucket,
                                        medianProportion_surprise_lowBucket]

csv_Data_lowBucket.append(csv_row_lowBucket)

# myFile = open('print_data_lowBucket.csv', 'w')
# writer = csv.writer(myFile)
# writer.writerows(csv_Data_lowBucket)


# tableName_row = ['~~~ lowBucket ~~~', '', '', '','','','','','','','','','','','','','','','','']
# csv_Data_lowBucket.insert(0, tableName_row)
# csv_Data_lowBucket.append('\n')
writer = csv.writer(comprehensive_bucket_File)
writer.writerows(csv_Data_lowBucket)
# END low BUCKET ANALYTICS

# BEGIN low BUCKET ANALYTICS -  SUBSAMPLE of last 1/10
# numHandles_toSelect_lowSub = len(handle_bucket_lowTweets) / 10
# numHandles_toSelect_lowSub = 25
numHandles_toSelect_lowSub = len(handle_bucket_lowTweets)
# print 'numHandles_toSelect_lowSub: ', numHandles_toSelect_lowSub
handle_bucket_lowTweets_subset =  handle_bucket_lowTweets[-numHandles_toSelect_lowSub:]
# print 'num selected: ', len(handle_bucket_lowTweets_subset)


emotion_tweet_sum = 0
emotion_proportionList_lowBucket = []
anger_disgust_proportionList_lowBucket = []
fear_proportionList_lowBucket = []
joy_love_proportionList_lowBucket = []
sadnes_proportionList_lowBucket = []
surprise_proportionList_lowBucket = []

emotion_maxProportion_lowBucket = 0
anger_disgust_maxProportion_lowBucket = 0
fear_maxProportion_lowBucket = 0
joy_love_maxProportion_lowBucket = 0
sadness_maxProportion_lowBucket = 0
surprise_maxProportion_lowBucket = 0

emotion_minProportion_lowBucket = 1
anger_disgust_minProportion_lowBucket = 1
fear_minProportion_lowBucket = 1
joy_love_minProportion_lowBucket = 1
sadness_minProportion_lowBucket = 1
surprise_minProportion_lowBucket = 1

anger_disgust_proportion_handle = 0
fear_proportion_handle = 0
joy_love_proportion_handle = 0
sadness_proportion_handle = 0
surprise_proportion_handle = 0
none_proportion_handle = 0

totalTweets_lowBucket_sub = 0

for handle in handle_bucket_lowTweets:

    #  gonna pull the last #25 tweets from each user:
    # creat function that: takes list of tweets as input, does set of calculations and returns results
    # anger_disgust_proportion_handle, fear_proportion_handle, joy_love_proportion_handle, sadness_proportion_handle, surprise_proportion_handle, none_proportion_handle

    tweet_list = handle_toTweetList_dict[handle][-20:] # select last 20
    totalTweetCount, numLabelledAs_anger_disgust, numLabelledAs_fear, numLabelledAs_joy, numLabelledAs_sadness, numLabelledAs_surprise, numLabelledAs_none = calculate_handleStats( tweet_list )

    totalTweets_lowBucket_sub = totalTweets_lowBucket_sub + totalTweetCount

    anger_disgust_proportion_handle = numLabelledAs_anger_disgust / float ( totalTweetCount )
    fear_proportion_handle = numLabelledAs_fear / float ( totalTweetCount )
    joy_love_proportion_handle = numLabelledAs_joy / float ( totalTweetCount )
    sadness_proportion_handle = numLabelledAs_sadness / float ( totalTweetCount )
    surprise_proportion_handle = numLabelledAs_surprise / float ( totalTweetCount )
    none_proportion_handle = numLabelledAs_none / float ( totalTweetCount )

    # if handle in handleTo_numLabelledAs_anger_disgust:
    #     anger_disgust_proportion_handle = handleTo_numLabelledAs_anger_disgust[handle] / float( handle_totalTweetCount[handle] )
    # if handle in handleTo_numLabelledAs_fear:
    #     fear_proportion_handle = handleTo_numLabelledAs_fear[handle] / float ( handle_totalTweetCount[handle] )
    # if handle in handleTo_numLabelledAs_joy:
    #     joy_love_proportion_handle = handleTo_numLabelledAs_joy[handle] / float( handle_totalTweetCount[handle] )
    # if handle in handleTo_numLabelledAs_sadness:
    #     sadness_proportion_handle = handleTo_numLabelledAs_sadness[handle] / float( handle_totalTweetCount[handle] )
    # if handle in handleTo_numLabelledAs_surprise:
    #     surprise_proportion_handle = handleTo_numLabelledAs_surprise[handle] / float( handle_totalTweetCount[handle] )
    # if handle in handleTo_numLabelledAs_none:
    #     none_proportion_handle = handleTo_numLabelledAs_none[handle] / float( handle_totalTweetCount[handle] )

    # proportion of tweets that have emotions for this handle
    # none_proportion_handle = handleTo_numLabelledAs_none[handle] / float( handle_totalTweetCount[handle] )
    emotion_proportion_handle = 1 - none_proportion_handle
    emotion_proportionList_lowBucket.append(emotion_proportion_handle)

    # anger_disgust proportion list append
    anger_disgust_proportionList_lowBucket.append(anger_disgust_proportion_handle)

    # fear proportion list append
    fear_proportionList_lowBucket.append(fear_proportion_handle)

    # joy_love proportion list append
    joy_love_proportionList_lowBucket.append(joy_love_proportion_handle)

    # sadness proportion list append
    sadnes_proportionList_lowBucket.append(sadness_proportion_handle)

    # surprise proportion list append
    surprise_proportionList_lowBucket.append(surprise_proportion_handle)

    # update Max and Min proportions if neccesary
    if emotion_maxProportion_lowBucket < emotion_proportion_handle:
        emotion_maxProportion_lowBucket = emotion_proportion_handle
    if anger_disgust_maxProportion_lowBucket < anger_disgust_proportion_handle:
        anger_disgust_maxProportion_lowBucket = anger_disgust_proportion_handle
    if fear_maxProportion_lowBucket < fear_proportion_handle:
        fear_maxProportion_lowBucket = fear_proportion_handle
    if joy_love_maxProportion_lowBucket < joy_love_proportion_handle:
        joy_love_maxProportion_lowBucket = joy_love_proportion_handle
    if sadness_maxProportion_lowBucket < sadness_proportion_handle:
        sadness_maxProportion_lowBucket = sadness_proportion_handle
    if surprise_maxProportion_lowBucket < surprise_proportion_handle:
        surprise_maxProportion_lowBucket = surprise_proportion_handle

    if emotion_minProportion_lowBucket > emotion_proportion_handle:
        emotion_minProportion_lowBucket = emotion_proportion_handle
    if anger_disgust_minProportion_lowBucket > anger_disgust_proportion_handle:
        anger_disgust_minProportion_lowBucket = anger_disgust_proportion_handle
    if fear_minProportion_lowBucket > fear_proportion_handle:
        fear_minProportion_lowBucket = fear_proportion_handle
    if joy_love_minProportion_lowBucket > joy_love_proportion_handle:
        joy_love_minProportion_lowBucket = joy_love_proportion_handle
    if sadness_minProportion_lowBucket > sadness_proportion_handle:
        sadness_minProportion_lowBucket = sadness_proportion_handle
    if surprise_minProportion_lowBucket > surprise_proportion_handle:
        surprise_minProportion_lowBucket = surprise_proportion_handle


# proportion of tweets that have emotions for this bucket
proportion_emotion_lowBucket = statistics.mean ( emotion_proportionList_lowBucket )
medianProportion_emotion_lowBucket = statistics.median ( emotion_proportionList_lowBucket )

# proportion of anger_disgust tweets that have emotions for this bucket
proportion_anger_disgust_lowBucket = statistics.mean ( anger_disgust_proportionList_lowBucket )
medianProportion_anger_disgust_lowBucket = statistics.median ( anger_disgust_proportionList_lowBucket )

# proportion of fear tweets that have emotions for this bucket
proportion_fear_lowBucket = statistics.mean ( fear_proportionList_lowBucket )
medianProportion_fear_lowBucket = statistics.median ( fear_proportionList_lowBucket )

# proportion of joy_love tweets that have emotions for this bucket
proportion_joy_love_lowBucket = statistics.mean ( joy_love_proportionList_lowBucket )
medianProportion_joy_love_lowBucket = statistics.median ( joy_love_proportionList_lowBucket )

# proportion of sadness tweets that have emotions for this bucket
proportion_sadness_lowBucket = statistics.mean ( sadnes_proportionList_lowBucket )
medianProportion_sadness_lowBucket = statistics.median ( sadnes_proportionList_lowBucket )

# proportion of surprise tweets that have emotions for this bucket
proportion_surprise_lowBucket = statistics.mean ( surprise_proportionList_lowBucket )
medianProportion_surprise_lowBucket = statistics.median ( surprise_proportionList_lowBucket )

# for each bucket of handles, provide
# proportion of tweets that have emotions
    # sum counts of all tweets that have emotion, then divide by total num tweets
# distrubtions of emotions for each bucket
    # proportion of tweets belonging to each emotion
# range of distrubtions for each emotion across users in a bcuket
    # for each emotion, go through all users and record max and min proportions
    #

# write out to csv
csv_Data_lowBucket_subset = []
# csv_row_lowBucket_subset = [   'Number of handles in this bucket',
#                                         'Number of tweets corresponding to the handles in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as emotional',
#                                         'Highest proportion of tweets labeled as emotional for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as emotional for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as anger_disgust',
#                                         'Highest proportion of tweets labeled as anger_disgust for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as anger_disgust for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as fear',
#                                         'Highest proportion of tweets labeled as fear for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as fear for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as joy_love',
#                                         'Highest proportion of tweets labeled as joy_love for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as joy_love for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as sadness',
#                                         'Highest proportion of tweets labeled as sadness for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as sadness for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as surprise',
#                                         'Highest proportion of tweets labeled as surprise for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as surprise for a handle in this bucket'
#                                         ]
# csv_Data_lowBucket_subset.append(csv_row_lowBucket_subset)

numberTweets_lowSubsample = totalTweets_lowBucket_sub

csv_row_lowBucket_subset = [   '~~~ lowBucket last 20 tweets per handle subsample ~~~',
                                        len(handle_bucket_lowTweets),
                                        totalTweets_lowBucket_sub,

                                        proportion_emotion_lowBucket,
                                        emotion_maxProportion_lowBucket,
                                        emotion_minProportion_lowBucket,
                                        medianProportion_emotion_lowBucket,

                                        proportion_anger_disgust_lowBucket,
                                        anger_disgust_maxProportion_lowBucket,
                                        anger_disgust_minProportion_lowBucket,
                                        medianProportion_anger_disgust_lowBucket,

                                        proportion_fear_lowBucket,
                                        fear_maxProportion_lowBucket,
                                        fear_minProportion_lowBucket,
                                        medianProportion_fear_lowBucket,

                                        proportion_joy_love_lowBucket,
                                        joy_love_maxProportion_lowBucket,
                                        joy_love_minProportion_lowBucket,
                                        medianProportion_joy_love_lowBucket,

                                        proportion_sadness_lowBucket,
                                        sadness_maxProportion_lowBucket,
                                        sadness_minProportion_lowBucket,
                                        medianProportion_sadness_lowBucket,

                                        proportion_surprise_lowBucket,
                                        surprise_maxProportion_lowBucket,
                                        surprise_minProportion_lowBucket,
                                        medianProportion_surprise_lowBucket]


csv_Data_lowBucket_subset.append(csv_row_lowBucket_subset)

# myFile = open('print_data_lowBucket_subset.csv', 'w')
# writer = csv.writer(myFile)
# writer.writerows(csv_Data_lowBucket_subset)

# tableName_row = ['~~~ lowBucket 1/10 sample ~~~', '', '', '','','','','','','','','','','','','','','','','']
# csv_Data_lowBucket_subset.insert(0, tableName_row)
# csv_Data_lowBucket_subset.append('\n')
writer = csv.writer(comprehensive_bucket_File)
writer.writerows(csv_Data_lowBucket_subset)

# END low BUCKET ANALYTICS - SUBSAMPLE of last 1/10



# BEGIN low BUCKET ANALYTICS - RANDOM 1/10 SUBSAMPLE

# select random 10th
# num_to_select = len(tweets_lowBucket) / 10
# handle_bucket_lowTweets_RANDOMsubset = handle_bucket_lowTweets[:num_to_select]
# tweets_lowBucket_RANDOMsubset =  tweets_lowBucket

# handle_bucket_lowTweets_RANDOMsubset = random.sample(handle_bucket_lowTweets, num_to_select)



# because each handle will have a different set of tweets associated with it, need to redo all the calculations regarding that handle

# determine the number of tweets to select by using the number of tweets present in the previous bucket
# aquire all tweets for all handles in this bucket
# randomly select X tweets from those tweets


# NOT GOOD: tweets_lowBucket is smaller than totalTweets_lowBucket_sub
# tweets_lowBucket_RANDOMsubset = random.sample(tweets_lowBucket, num_to_select)

# print 'num tweets low bucket: ', len(tweets_lowBucket)
# print 'num to select: ', numHandles_toSelect_lowSub
# print 'low bucket random sample: ', tweets_lowBucket_RANDOMsubset


# then do calculations
# on per handle basis

# recalculates stats for each handle, based on the tweets in the random sub sample for this bucket

# handle_totalTweetCount_SUB, handleTo_numLabelledAs_anger_disgust_SUB, handleTo_numLabelledAs_fear_SUB, handleTo_numLabelledAs_joy_SUB, handleTo_numLabelledAs_sadness_SUB, handleTo_numLabelledAs_surprise_SUB, handleTo_numLabelledAs_none_SUB, unique_handleList = calculate_handleStats( tweetToHandle_dict, tweets_lowBucket_RANDOMsubset )

# examine why KeyError: 'dagragdag'
# need to fix for other random samples

emotion_tweet_sum = 0
emotion_proportionList_lowBucket = []
anger_disgust_proportionList_lowBucket = []
fear_proportionList_lowBucket = []
joy_love_proportionList_lowBucket = []
sadnes_proportionList_lowBucket = []
surprise_proportionList_lowBucket = []

emotion_maxProportion_lowBucket = 0
anger_disgust_maxProportion_lowBucket = 0
fear_maxProportion_lowBucket = 0
joy_love_maxProportion_lowBucket = 0
sadness_maxProportion_lowBucket = 0
surprise_maxProportion_lowBucket = 0

emotion_minProportion_lowBucket = 1
anger_disgust_minProportion_lowBucket = 1
fear_minProportion_lowBucket = 1
joy_love_minProportion_lowBucket = 1
sadness_minProportion_lowBucket = 1
surprise_minProportion_lowBucket = 1

anger_disgust_proportion_handle = 0
fear_proportion_handle = 0
joy_love_proportion_handle = 0
sadness_proportion_handle = 0
surprise_proportion_handle = 0
none_proportion_handle = 0

# totalTweets_lowBucket = 0

for handle in handle_bucket_lowTweets:

    tweetList = handle_toTweetList_dict[handle]
    num_to_select = len(tweetList) / 10
    tweetList_RANDOMsubset = random.sample(tweetList, num_to_select)

    totalTweetCount, numLabelledAs_anger_disgust, numLabelledAs_fear, numLabelledAs_joy, numLabelledAs_sadness, numLabelledAs_surprise, numLabelledAs_none = calculate_handleStats( tweetList_RANDOMsubset )

    totalTweets_lowBucket_sub = totalTweets_lowBucket_sub + totalTweetCount

    anger_disgust_proportion_handle = numLabelledAs_anger_disgust / float ( totalTweetCount )
    fear_proportion_handle = numLabelledAs_fear / float ( totalTweetCount )
    joy_love_proportion_handle = numLabelledAs_joy / float ( totalTweetCount )
    sadness_proportion_handle = numLabelledAs_sadness / float ( totalTweetCount )
    surprise_proportion_handle = numLabelledAs_surprise / float ( totalTweetCount )
    none_proportion_handle = numLabelledAs_none / float ( totalTweetCount )

    # proportion of tweets that have emotions for this handle
    # none_proportion_handle = handleTo_numLabelledAs_none_SUB[handle] / float( handle_totalTweetCount_SUB[handle] )

    emotion_proportion_handle = 1 - none_proportion_handle
    emotion_proportionList_lowBucket.append(emotion_proportion_handle)

    # anger_disgust proportion list append
    anger_disgust_proportionList_lowBucket.append(anger_disgust_proportion_handle)

    # fear proportion list append
    fear_proportionList_lowBucket.append(fear_proportion_handle)

    # joy_love proportion list append
    joy_love_proportionList_lowBucket.append(joy_love_proportion_handle)

    # sadness proportion list append
    sadnes_proportionList_lowBucket.append(sadness_proportion_handle)

    # surprise proportion list append
    surprise_proportionList_lowBucket.append(surprise_proportion_handle)

    # update Max and Min proportions if neccesary
    if emotion_maxProportion_lowBucket < emotion_proportion_handle:
        emotion_maxProportion_lowBucket = emotion_proportion_handle
    if anger_disgust_maxProportion_lowBucket < anger_disgust_proportion_handle:
        anger_disgust_maxProportion_lowBucket = anger_disgust_proportion_handle
    if fear_maxProportion_lowBucket < fear_proportion_handle:
        fear_maxProportion_lowBucket = fear_proportion_handle
    if joy_love_maxProportion_lowBucket < joy_love_proportion_handle:
        joy_love_maxProportion_lowBucket = joy_love_proportion_handle
    if sadness_maxProportion_lowBucket < sadness_proportion_handle:
        sadness_maxProportion_lowBucket = sadness_proportion_handle
    if surprise_maxProportion_lowBucket < surprise_proportion_handle:
        surprise_maxProportion_lowBucket = surprise_proportion_handle

    if emotion_minProportion_lowBucket > emotion_proportion_handle:
        emotion_minProportion_lowBucket = emotion_proportion_handle
    if anger_disgust_minProportion_lowBucket > anger_disgust_proportion_handle:
        anger_disgust_minProportion_lowBucket = anger_disgust_proportion_handle
    if fear_minProportion_lowBucket > fear_proportion_handle:
        fear_minProportion_lowBucket = fear_proportion_handle
    if joy_love_minProportion_lowBucket > joy_love_proportion_handle:
        joy_love_minProportion_lowBucket = joy_love_proportion_handle
    if sadness_minProportion_lowBucket > sadness_proportion_handle:
        sadness_minProportion_lowBucket = sadness_proportion_handle
    if surprise_minProportion_lowBucket > surprise_proportion_handle:
        surprise_minProportion_lowBucket = surprise_proportion_handle


# proportion of tweets that have emotions for this bucket
proportion_emotion_lowBucket = statistics.mean ( emotion_proportionList_lowBucket )
medianProportion_emotion_lowBucket = statistics.median ( emotion_proportionList_lowBucket )

# proportion of anger_disgust tweets that have emotions for this bucket
proportion_anger_disgust_lowBucket = statistics.mean ( anger_disgust_proportionList_lowBucket )
medianProportion_anger_disgust_lowBucket = statistics.median ( anger_disgust_proportionList_lowBucket )

# proportion of fear tweets that have emotions for this bucket
proportion_fear_lowBucket = statistics.mean ( fear_proportionList_lowBucket )
medianProportion_fear_lowBucket = statistics.median ( fear_proportionList_lowBucket )

# proportion of joy_love tweets that have emotions for this bucket
proportion_joy_love_lowBucket = statistics.mean ( joy_love_proportionList_lowBucket )
medianProportion_joy_love_lowBucket = statistics.median ( joy_love_proportionList_lowBucket )

# proportion of sadness tweets that have emotions for this bucket
proportion_sadness_lowBucket = statistics.mean ( sadnes_proportionList_lowBucket )
medianProportion_sadness_lowBucket = statistics.median ( sadnes_proportionList_lowBucket )

# proportion of surprise tweets that have emotions for this bucket
proportion_surprise_lowBucket = statistics.mean ( surprise_proportionList_lowBucket )
medianProportion_surprise_lowBucket = statistics.median ( surprise_proportionList_lowBucket )


# for each bucket of handles, provide
# proportion of tweets that have emotions
    # sum counts of all tweets that have emotion, then divide by total num tweets
# distrubtions of emotions for each bucket
    # proportion of tweets belonging to each emotion
# range of distrubtions for each emotion across users in a bcuket
    # for each emotion, go through all users and record max and min proportions
    #

# write out to csv
csv_Data_lowBucket_RANDOMsample = []
# csv_row_lowBucket_RANDOMsample = [   'Number of handles in this bucket',
#                                         'Number of tweets corresponding to the handles in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as emotional',
#                                         'Highest proportion of tweets labeled as emotional for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as emotional for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as anger_disgust',
#                                         'Highest proportion of tweets labeled as anger_disgust for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as anger_disgust for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as fear',
#                                         'Highest proportion of tweets labeled as fear for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as fear for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as joy_love',
#                                         'Highest proportion of tweets labeled as joy_love for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as joy_love for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as sadness',
#                                         'Highest proportion of tweets labeled as sadness for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as sadness for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as surprise',
#                                         'Highest proportion of tweets labeled as surprise for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as surprise for a handle in this bucket'
#                                         ]
# csv_Data_lowBucket_RANDOMsample.append(csv_row_lowBucket_RANDOMsample)

csv_row_lowBucket_RANDOMsample = [   '~~~ lowBucket random 1/10 sample of tweets per handle ~~~',
                                        len(handle_bucket_lowTweets),
                                        totalTweets_lowBucket_sub,

                                        proportion_emotion_lowBucket,
                                        emotion_maxProportion_lowBucket,
                                        emotion_minProportion_lowBucket,
                                        medianProportion_emotion_lowBucket,

                                        proportion_anger_disgust_lowBucket,
                                        anger_disgust_maxProportion_lowBucket,
                                        anger_disgust_minProportion_lowBucket,
                                        medianProportion_anger_disgust_lowBucket,

                                        proportion_fear_lowBucket,
                                        fear_maxProportion_lowBucket,
                                        fear_minProportion_lowBucket,
                                        medianProportion_fear_lowBucket,

                                        proportion_joy_love_lowBucket,
                                        joy_love_maxProportion_lowBucket,
                                        joy_love_minProportion_lowBucket,
                                        medianProportion_joy_love_lowBucket,

                                        proportion_sadness_lowBucket,
                                        sadness_maxProportion_lowBucket,
                                        sadness_minProportion_lowBucket,
                                        medianProportion_sadness_lowBucket,

                                        proportion_surprise_lowBucket,
                                        surprise_maxProportion_lowBucket,
                                        surprise_minProportion_lowBucket,
                                        medianProportion_surprise_lowBucket]

csv_Data_lowBucket_RANDOMsample.append(csv_row_lowBucket_RANDOMsample)

# myFile = open('print_data_lowBucket_RANDOMsubset.csv', 'w')
# writer = csv.writer(myFile)
# writer.writerows(csv_Data_lowBucket_RANDOMsample)

# tableName_row = ['~~~ lowBucket random 1/10 sample ~~~', '', '', '','','','','','','','','','','','','','','','','']
# csv_Data_lowBucket_RANDOMsample.insert(0, tableName_row)
# csv_Data_lowBucket_RANDOMsample.append('\n')
writer = csv.writer(comprehensive_bucket_File)
writer.writerows(csv_Data_lowBucket_RANDOMsample)

# END low BUCKET ANALYTICS - RANDOM 1/10 SUBSAMPLE














# BEGIN med BUCKET ANALYTICS
emotion_tweet_sum = 0
emotion_proportionList_medBucket = []
anger_disgust_proportionList_medBucket = []
fear_proportionList_medBucket = []
joy_love_proportionList_medBucket = []
sadness_proportionList_medBucket = []
surprise_proportionList_medBucket = []

emotion_maxProportion_medBucket = 0
anger_disgust_maxProportion_medBucket = 0
fear_maxProportion_medBucket = 0
joy_love_maxProportion_medBucket = 0
sadness_maxProportion_medBucket = 0
surprise_maxProportion_medBucket = 0

emotion_minProportion_medBucket = 1
anger_disgust_minProportion_medBucket = 1
fear_minProportion_medBucket = 1
joy_love_minProportion_medBucket = 1
sadness_minProportion_medBucket = 1
surprise_minProportion_medBucket = 1

anger_disgust_proportion_handle = 0
fear_proportion_handle = 0
joy_love_proportion_handle = 0
sadness_proportion_handle = 0
surprise_proportion_handle = 0
none_proportion_handle = 0

totalTweets_medBucket = 0

for handle in handle_bucket_medTweets:

    totalTweets_medBucket = totalTweets_medBucket + handle_totalTweetCount[handle]

    anger_disgust_proportion = 0
    fear_proportion = 0
    joy_love_proportion = 0
    sadness_proportion = 0
    surprise_proportion = 0
    none_proportion = 0

    if handle in handleTo_numLabelledAs_anger_disgust:
        anger_disgust_proportion_handle = handleTo_numLabelledAs_anger_disgust[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_fear:
        fear_proportion_handle = handleTo_numLabelledAs_fear[handle] / float ( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_joy:
        joy_love_proportion_handle = handleTo_numLabelledAs_joy[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_sadness:
        sadness_proportion_handle = handleTo_numLabelledAs_sadness[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_surprise:
        surprise_proportion_handle = handleTo_numLabelledAs_surprise[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_none:
        none_proportion_handle = handleTo_numLabelledAs_none[handle] / float( handle_totalTweetCount[handle] )

    # proportion of tweets that have emotions for this handle
    none_proportion_handle = handleTo_numLabelledAs_none[handle] / float( handle_totalTweetCount[handle] )
    emotion_proportion_handle = 1 - none_proportion_handle
    emotion_proportionList_medBucket.append(emotion_proportion_handle)

    # anger_disgust proportion list append
    anger_disgust_proportionList_medBucket.append(anger_disgust_proportion_handle)

    # fear proportion list append
    fear_proportionList_medBucket.append(fear_proportion_handle)

    # joy_love proportion list append
    joy_love_proportionList_medBucket.append(joy_love_proportion_handle)

    # sadness proportion list append
    sadness_proportionList_medBucket.append(sadness_proportion_handle)

    # surprise proportion list append
    surprise_proportionList_medBucket.append(surprise_proportion_handle)

    # update Max and Min proportions if neccesary
    if emotion_maxProportion_medBucket < emotion_proportion_handle:
        emotion_maxProportion_medBucket = emotion_proportion_handle
    if anger_disgust_maxProportion_medBucket < anger_disgust_proportion_handle:
        anger_disgust_maxProportion_medBucket = anger_disgust_proportion_handle
    if fear_maxProportion_medBucket < fear_proportion_handle:
        fear_maxProportion_medBucket = fear_proportion_handle
    if joy_love_maxProportion_medBucket < joy_love_proportion_handle:
        joy_love_maxProportion_medBucket = joy_love_proportion_handle
    if sadness_maxProportion_medBucket < sadness_proportion_handle:
        sadness_maxProportion_medBucket = sadness_proportion_handle
    if surprise_maxProportion_medBucket < surprise_proportion_handle:
        surprise_maxProportion_medBucket = surprise_proportion_handle

    if emotion_minProportion_medBucket > emotion_proportion_handle:
        emotion_minProportion_medBucket = emotion_proportion_handle
    if anger_disgust_minProportion_medBucket > anger_disgust_proportion_handle:
        anger_disgust_minProportion_medBucket = anger_disgust_proportion_handle
    if fear_minProportion_medBucket > fear_proportion_handle:
        fear_minProportion_medBucket = fear_proportion_handle
    if joy_love_minProportion_medBucket > joy_love_proportion_handle:
        joy_love_minProportion_medBucket = joy_love_proportion_handle
    if sadness_minProportion_medBucket > sadness_proportion_handle:
        sadness_minProportion_medBucket = sadness_proportion_handle
    if surprise_minProportion_medBucket > surprise_proportion_handle:
        surprise_minProportion_medBucket = surprise_proportion_handle



# proportion of tweets that have emotions for this bucket
proportion_emotion_medBucket = statistics.mean ( emotion_proportionList_medBucket )
medianProportion_emotion_medBucket = statistics.median ( emotion_proportionList_medBucket )

# proportion of anger_disgust tweets that have emotions for this bucket
proportion_anger_disgust_medBucket = statistics.mean ( anger_disgust_proportionList_medBucket )
medianProportion_anger_disgust_medBucket = statistics.median ( anger_disgust_proportionList_medBucket )

# proportion of fear tweets that have emotions for this bucket
proportion_fear_medBucket = statistics.mean ( fear_proportionList_medBucket )
medianProportion_fear_medBucket = statistics.median ( fear_proportionList_medBucket )

# proportion of joy_love tweets that have emotions for this bucket
proportion_joy_love_medBucket = statistics.mean ( joy_love_proportionList_medBucket )
medianProportion_joy_love_medBucket = statistics.median ( joy_love_proportionList_medBucket )

# proportion of sadness tweets that have emotions for this bucket
proportion_sadness_medBucket = statistics.mean ( sadness_proportionList_medBucket )
medianProportion_sadness_medBucket = statistics.median ( sadness_proportionList_medBucket )

# proportion of surprise tweets that have emotions for this bucket
proportion_surprise_medBucket = statistics.mean ( surprise_proportionList_medBucket )
medianProportion_surprise_medBucket = statistics.median ( surprise_proportionList_medBucket )


# for each bucket of handles, provide
# proportion of tweets that have emotions
    # sum counts of all tweets that have emotion, then divide by total num tweets
# distrubtions of emotions for each bucket
    # proportion of tweets belonging to each emotion
# range of distrubtions for each emotion across users in a bcuket
    # for each emotion, go through all users and record max and min proportions
    #

# write out to csv
csv_Data_medBucket = []
# csv_row_medBucket = [   'Number of handles in this bucket',
#                                         'Number of tweets corresponding to the handles in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as emotional',
#                                         'Highest proportion of tweets labeled as emotional for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as emotional for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as anger_disgust',
#                                         'Highest proportion of tweets labeled as anger_disgust for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as anger_disgust for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as fear',
#                                         'Highest proportion of tweets labeled as fear for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as fear for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as joy_love',
#                                         'Highest proportion of tweets labeled as joy_love for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as joy_love for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as sadness',
#                                         'Highest proportion of tweets labeled as sadness for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as sadness for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as surprise',
#                                         'Highest proportion of tweets labeled as surprise for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as surprise for a handle in this bucket'
#                                         ]
# csv_Data_medBucket.append(csv_row_medBucket)

csv_row_medBucket = [  '~~~ medBucket ~~~',
                                        len(handle_bucket_medTweets),
                                        totalTweets_medBucket,

                                        proportion_emotion_medBucket,
                                        emotion_maxProportion_medBucket,
                                        emotion_minProportion_medBucket,
                                        medianProportion_emotion_medBucket,

                                        proportion_anger_disgust_medBucket,
                                        anger_disgust_maxProportion_medBucket,
                                        anger_disgust_minProportion_medBucket,
                                        medianProportion_anger_disgust_medBucket,

                                        proportion_fear_medBucket,
                                        fear_maxProportion_medBucket,
                                        fear_minProportion_medBucket,
                                        medianProportion_fear_medBucket,

                                        proportion_joy_love_medBucket,
                                        joy_love_maxProportion_medBucket,
                                        joy_love_minProportion_medBucket,
                                        medianProportion_joy_love_medBucket,

                                        proportion_sadness_medBucket,
                                        sadness_maxProportion_medBucket,
                                        sadness_minProportion_medBucket,
                                        medianProportion_sadness_medBucket,

                                        proportion_surprise_medBucket,
                                        surprise_maxProportion_medBucket,
                                        surprise_minProportion_medBucket,
                                        medianProportion_surprise_medBucket]


csv_Data_medBucket.append(csv_row_medBucket)

# myFile = open('print_data_medBucket.csv', 'w')
# writer = csv.writer(myFile)
# writer.writerows(csv_Data_medBucket)

# tableName_row = ['~~~ medBucket ~~~', '', '', '','','','','','','','','','','','','','','','','']
# csv_Data_medBucket.insert(0, tableName_row)
# csv_Data_medBucket.append('\n')
writer = csv.writer(comprehensive_bucket_File)
writer.writerows(csv_Data_medBucket)

# END med BUCKET ANALYTICS




# BEGIN med BUCKET ANALYTICS -  SUBSAMPLE of last 1/10
# numHandles_toSelect_medSub = len(handle_bucket_medTweets) / 10
# numHandles_toSelect_medSub = 25
# numHandles_toSelect_medSub = len(handle_bucket_medTweets)
# handle_bucket_medTweets_subset =  handle_bucket_medTweets[-numHandles_toSelect_medSub:]

emotion_tweet_sum = 0
emotion_proportionList_medBucket = []
anger_disgust_proportionList_medBucket = []
fear_proportionList_medBucket = []
joy_love_proportionList_medBucket = []
sadness_proportionList_medBucket = []
surprise_proportion_bucketList_medBucket = []

emotion_maxProportion_medBucket = 0
anger_disgust_maxProportion_medBucket = 0
fear_maxProportion_medBucket = 0
joy_love_maxProportion_medBucket = 0
sadness_maxProportion_medBucket = 0
surprise_maxProportion_medBucket = 0

emotion_minProportion_medBucket = 1
anger_disgust_minProportion_medBucket = 1
fear_minProportion_medBucket = 1
joy_love_minProportion_medBucket = 1
sadness_minProportion_medBucket = 1
surprise_minProportion_medBucket = 1

anger_disgust_proportion_handle = 0
fear_proportion_handle = 0
joy_love_proportion_handle = 0
sadness_proportion_handle = 0
surprise_proportion_handle = 0
none_proportion_handle = 0

totalTweets_medBucket_sub = 0

for handle in handle_bucket_medTweets:

    tweet_list = handle_toTweetList_dict[handle][-20:] # select last 20
    totalTweetCount, numLabelledAs_anger_disgust, numLabelledAs_fear, numLabelledAs_joy, numLabelledAs_sadness, numLabelledAs_surprise, numLabelledAs_none = calculate_handleStats( tweet_list )

    totalTweets_medBucket_sub = totalTweets_medBucket_sub + totalTweetCount

    anger_disgust_proportion_handle = numLabelledAs_anger_disgust / float ( totalTweetCount )
    fear_proportion_handle = numLabelledAs_fear / float ( totalTweetCount )
    joy_love_proportion_handle = numLabelledAs_joy / float ( totalTweetCount )
    sadness_proportion_handle = numLabelledAs_sadness / float ( totalTweetCount )
    surprise_proportion_handle = numLabelledAs_surprise / float ( totalTweetCount )
    none_proportion_handle = numLabelledAs_none / float ( totalTweetCount )

    # proportion of tweets that have emotions for this handle
    none_proportion_handle = handleTo_numLabelledAs_none[handle] / float( handle_totalTweetCount[handle] )
    emotion_proportion_handle = 1 - none_proportion_handle
    emotion_proportionList_medBucket.append(emotion_proportion_handle)

    # anger_disgust proportion list append
    anger_disgust_proportionList_medBucket.append(anger_disgust_proportion_handle)

    # fear proportion list append
    fear_proportionList_medBucket.append(fear_proportion_handle)

    # joy_love proportion list append
    joy_love_proportionList_medBucket.append(joy_love_proportion_handle)

    # sadness proportion list append
    sadness_proportionList_medBucket.append(sadness_proportion_handle)

    # surprise proportion list append
    surprise_proportion_bucketList_medBucket.append(surprise_proportion_handle)

    # update Max and Min proportions if neccesary
    if emotion_maxProportion_medBucket < emotion_proportion_handle:
        emotion_maxProportion_medBucket = emotion_proportion_handle
    if anger_disgust_maxProportion_medBucket < anger_disgust_proportion_handle:
        anger_disgust_maxProportion_medBucket = anger_disgust_proportion_handle
    if fear_maxProportion_medBucket < fear_proportion_handle:
        fear_maxProportion_medBucket = fear_proportion_handle
    if joy_love_maxProportion_medBucket < joy_love_proportion_handle:
        joy_love_maxProportion_medBucket = joy_love_proportion_handle
    if sadness_maxProportion_medBucket < sadness_proportion_handle:
        sadness_maxProportion_medBucket = sadness_proportion_handle
    if surprise_maxProportion_medBucket < surprise_proportion_handle:
        surprise_maxProportion_medBucket = surprise_proportion_handle

    if emotion_minProportion_medBucket > emotion_proportion_handle:
        emotion_minProportion_medBucket = emotion_proportion_handle
    if anger_disgust_minProportion_medBucket > anger_disgust_proportion_handle:
        anger_disgust_minProportion_medBucket = anger_disgust_proportion_handle
    if fear_minProportion_medBucket > fear_proportion_handle:
        fear_minProportion_medBucket = fear_proportion_handle
    if joy_love_minProportion_medBucket > joy_love_proportion_handle:
        joy_love_minProportion_medBucket = joy_love_proportion_handle
    if sadness_minProportion_medBucket > sadness_proportion_handle:
        sadness_minProportion_medBucket = sadness_proportion_handle
    if surprise_minProportion_medBucket > surprise_proportion_handle:
        surprise_minProportion_medBucket = surprise_proportion_handle


# proportion of tweets that have emotions for this bucket
proportion_emotion_medBucket = statistics.mean ( emotion_proportionList_medBucket )
medianProportion_emotion_medBucket = statistics.median ( emotion_proportionList_medBucket )

# proportion of anger_disgust tweets that have emotions for this bucket
proportion_anger_disgust_medBucket = statistics.mean ( anger_disgust_proportionList_medBucket )
medianProportion_anger_disgust_medBucket = statistics.median ( anger_disgust_proportionList_medBucket )

# proportion of fear tweets that have emotions for this bucket
proportion_fear_medBucket = statistics.mean ( fear_proportionList_medBucket )
medianProportion_fear_medBucket = statistics.median ( fear_proportionList_medBucket )

# proportion of joy_love tweets that have emotions for this bucket
proportion_joy_love_medBucket = statistics.mean ( joy_love_proportionList_medBucket )
medianProportion_joy_love_medBucket = statistics.median ( joy_love_proportionList_medBucket )

# proportion of sadness tweets that have emotions for this bucket
proportion_sadness_medBucket = statistics.mean ( sadness_proportionList_medBucket )
medianProportion_sadness_medBucket = statistics.median ( sadness_proportionList_medBucket )

# proportion of surprise tweets that have emotions for this bucket
proportion_surprise_medBucket = statistics.mean ( surprise_proportionList_medBucket )
medianProportion_surprise_medBucket = statistics.median ( surprise_proportionList_medBucket )

# for each bucket of handles, provide
# proportion of tweets that have emotions
    # sum counts of all tweets that have emotion, then divide by total num tweets
# distrubtions of emotions for each bucket
    # proportion of tweets belonging to each emotion
# range of distrubtions for each emotion across users in a bcuket
    # for each emotion, go through all users and record max and min proportions
    #

# write out to csv
csv_Data_medBucket_subset = []
# csv_row_medBucket_subset = [   'Number of handles in this bucket',
#                                         'Number of tweets corresponding to the handles in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as emotional',
#                                         'Highest proportion of tweets labeled as emotional for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as emotional for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as anger_disgust',
#                                         'Highest proportion of tweets labeled as anger_disgust for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as anger_disgust for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as fear',
#                                         'Highest proportion of tweets labeled as fear for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as fear for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as joy_love',
#                                         'Highest proportion of tweets labeled as joy_love for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as joy_love for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as sadness',
#                                         'Highest proportion of tweets labeled as sadness for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as sadness for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as surprise',
#                                         'Highest proportion of tweets labeled as surprise for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as surprise for a handle in this bucket'
#                                         ]
# csv_Data_medBucket_subset.append(csv_row_medBucket_subset)

csv_row_medBucket_subset = [   '~~~ medBucket last 20 tweets per handle subsample ~~~',
                                        len(handle_bucket_medTweets),
                                        totalTweets_medBucket_sub,

                                        proportion_emotion_medBucket,
                                        emotion_maxProportion_medBucket,
                                        emotion_minProportion_medBucket,
                                        medianProportion_emotion_medBucket,

                                        proportion_anger_disgust_medBucket,
                                        anger_disgust_maxProportion_medBucket,
                                        anger_disgust_minProportion_medBucket,
                                        medianProportion_anger_disgust_medBucket,

                                        proportion_fear_medBucket,
                                        fear_maxProportion_medBucket,
                                        fear_minProportion_medBucket,
                                        medianProportion_fear_medBucket,

                                        proportion_joy_love_medBucket,
                                        joy_love_maxProportion_medBucket,
                                        joy_love_minProportion_medBucket,
                                        medianProportion_joy_love_medBucket,

                                        proportion_sadness_medBucket,
                                        sadness_maxProportion_medBucket,
                                        sadness_minProportion_medBucket,
                                        medianProportion_sadness_medBucket,

                                        proportion_surprise_medBucket,
                                        surprise_maxProportion_medBucket,
                                        surprise_minProportion_medBucket,
                                        medianProportion_surprise_medBucket]

csv_Data_medBucket_subset.append(csv_row_medBucket_subset)

# myFile = open('print_data_medBucket_subset.csv', 'w')
# writer = csv.writer(myFile)
# writer.writerows(csv_Data_medBucket_subset)

# tableName_row = ['~~~ medBucket 1/10 subset ~~~', '', '', '','','','','','','','','','','','','','','','','']
# csv_Data_medBucket_subset.insert(0, tableName_row)
# csv_Data_medBucket_subset.append('\n')
writer = csv.writer(comprehensive_bucket_File)
writer.writerows(csv_Data_medBucket_subset)

# END med BUCKET ANALYTICS - SUBSAMPLE of last 1/10





# BEGIN med BUCKET ANALYTICS - RANDOM 1/10 SUBSAMPLE

# select random 10th

# use same number of tweets as in previous non random subset

# num_to_select = len(tweets_lowBucket) / 10

# tweets_medBucket_RANDOMsubset = random.sample(tweets_medBucket, totalTweets_medBucket_sub)
# tweets_medBucket_RANDOMsubset = random.sample(tweets_medBucket, num_to_select)
# handle_totalTweetCount_SUB, handleTo_numLabelledAs_anger_disgust_SUB, handleTo_numLabelledAs_fear_SUB, handleTo_numLabelledAs_joy_SUB, handleTo_numLabelledAs_sadness_SUB, handleTo_numLabelledAs_surprise_SUB, handleTo_numLabelledAs_none_SUB, unique_handleList = calculate_handleStats( tweetToHandle_dict, tweets_medBucket_RANDOMsubset )


emotion_tweet_sum = 0
emotion_proportionList_medBucket = []
anger_disgust_proportionList_medBucket = []
fear_proportionList_medBucket = []
joy_love_proportionList_medBucket = []
sadness_proportionList_medBucket = []
surprise_proportion_bucketList_medBucket = []

emotion_maxProportion_medBucket = 0
anger_disgust_maxProportion_medBucket = 0
fear_maxProportion_medBucket = 0
joy_love_maxProportion_medBucket = 0
sadness_maxProportion_medBucket = 0
surprise_maxProportion_medBucket = 0

emotion_minProportion_medBucket = 1
anger_disgust_minProportion_medBucket = 1
fear_minProportion_medBucket = 1
joy_love_minProportion_medBucket = 1
sadness_minProportion_medBucket = 1
surprise_minProportion_medBucket = 1

anger_disgust_proportion_handle = 0
fear_proportion_handle = 0
joy_love_proportion_handle = 0
sadness_proportion_handle = 0
surprise_proportion_handle = 0
none_proportion_handle = 0

# totalTweets_medBucket = 0

for handle in handle_bucket_medTweets:

    # totalTweets_medBucket = totalTweets_medBucket + handle_totalTweetCount_SUB[handle]

    tweetList = handle_toTweetList_dict[handle]
    num_to_select = len(tweetList) / 10
    tweetList_RANDOMsubset = random.sample(tweetList, num_to_select)

    totalTweetCount, numLabelledAs_anger_disgust, numLabelledAs_fear, numLabelledAs_joy, numLabelledAs_sadness, numLabelledAs_surprise, numLabelledAs_none = calculate_handleStats( tweetList_RANDOMsubset )

    totalTweets_medBucket_sub = totalTweets_medBucket_sub + totalTweetCount

    anger_disgust_proportion_handle = numLabelledAs_anger_disgust / float ( totalTweetCount )
    fear_proportion_handle = numLabelledAs_fear / float ( totalTweetCount )
    joy_love_proportion_handle = numLabelledAs_joy / float ( totalTweetCount )
    sadness_proportion_handle = numLabelledAs_sadness / float ( totalTweetCount )
    surprise_proportion_handle = numLabelledAs_surprise / float ( totalTweetCount )
    none_proportion_handle = numLabelledAs_none / float ( totalTweetCount )

    # proportion of tweets that have emotions for this handle
    # none_proportion_handle = handleTo_numLabelledAs_none_SUB[handle] / float( handle_totalTweetCount_SUB[handle] )
    emotion_proportion_handle = 1 - none_proportion_handle
    emotion_proportionList_medBucket.append(emotion_proportion_handle)

    # anger_disgust proportion list append
    anger_disgust_proportionList_medBucket.append(anger_disgust_proportion_handle)

    # fear proportion list append
    fear_proportionList_medBucket.append(fear_proportion_handle)

    # joy_love proportion list append
    joy_love_proportionList_medBucket.append(joy_love_proportion_handle)

    # sadness proportion list append
    sadness_proportionList_medBucket.append(sadness_proportion_handle)

    # surprise proportion list append
    surprise_proportion_bucketList_medBucket.append(surprise_proportion_handle)

    # update Max and Min proportions if neccesary
    if emotion_maxProportion_medBucket < emotion_proportion_handle:
        emotion_maxProportion_medBucket = emotion_proportion_handle
    if anger_disgust_maxProportion_medBucket < anger_disgust_proportion_handle:
        anger_disgust_maxProportion_medBucket = anger_disgust_proportion_handle
    if fear_maxProportion_medBucket < fear_proportion_handle:
        fear_maxProportion_medBucket = fear_proportion_handle
    if joy_love_maxProportion_medBucket < joy_love_proportion_handle:
        joy_love_maxProportion_medBucket = joy_love_proportion_handle
    if sadness_maxProportion_medBucket < sadness_proportion_handle:
        sadness_maxProportion_medBucket = sadness_proportion_handle
    if surprise_maxProportion_medBucket < surprise_proportion_handle:
        surprise_maxProportion_medBucket = surprise_proportion_handle

    if emotion_minProportion_medBucket > emotion_proportion_handle:
        emotion_minProportion_medBucket = emotion_proportion_handle
    if anger_disgust_minProportion_medBucket > anger_disgust_proportion_handle:
        anger_disgust_minProportion_medBucket = anger_disgust_proportion_handle
    if fear_minProportion_medBucket > fear_proportion_handle:
        fear_minProportion_medBucket = fear_proportion_handle
    if joy_love_minProportion_medBucket > joy_love_proportion_handle:
        joy_love_minProportion_medBucket = joy_love_proportion_handle
    if sadness_minProportion_medBucket > sadness_proportion_handle:
        sadness_minProportion_medBucket = sadness_proportion_handle
    if surprise_minProportion_medBucket > surprise_proportion_handle:
        surprise_minProportion_medBucket = surprise_proportion_handle


# proportion of tweets that have emotions for this bucket
proportion_emotion_medBucket = statistics.mean ( emotion_proportionList_medBucket )
medianProportion_emotion_medBucket = statistics.median ( emotion_proportionList_medBucket )

# proportion of anger_disgust tweets that have emotions for this bucket
proportion_anger_disgust_medBucket = statistics.mean ( anger_disgust_proportionList_medBucket )
medianProportion_anger_disgust_medBucket = statistics.median ( anger_disgust_proportionList_medBucket )

# proportion of fear tweets that have emotions for this bucket
proportion_fear_medBucket = statistics.mean ( fear_proportionList_medBucket )
medianProportion_fear_medBucket = statistics.median ( fear_proportionList_medBucket )

# proportion of joy_love tweets that have emotions for this bucket
proportion_joy_love_medBucket = statistics.mean ( joy_love_proportionList_medBucket )
medianProportion_joy_love_medBucket = statistics.median ( joy_love_proportionList_medBucket )

# proportion of sadness tweets that have emotions for this bucket
proportion_sadness_medBucket = statistics.mean ( sadness_proportionList_medBucket )
medianProportion_sadness_medBucket = statistics.median ( sadness_proportionList_medBucket )

# proportion of surprise tweets that have emotions for this bucket
proportion_surprise_medBucket = statistics.mean ( surprise_proportionList_medBucket )
medianProportion_surprise_medBucket = statistics.median ( surprise_proportionList_medBucket )


# for each bucket of handles, provide
# proportion of tweets that have emotions
    # sum counts of all tweets that have emotion, then divide by total num tweets
# distrubtions of emotions for each bucket
    # proportion of tweets belonging to each emotion
# range of distrubtions for each emotion across users in a bcuket
    # for each emotion, go through all users and record max and min proportions
    #

# write out to csv
csv_Data_medBucket_RANDOMsample = []
# csv_row_medBucket_RANDOMsample = [   'Number of handles in this bucket',
#                                         'Number of tweets corresponding to the handles in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as emotional',
#                                         'Highest proportion of tweets labeled as emotional for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as emotional for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as anger_disgust',
#                                         'Highest proportion of tweets labeled as anger_disgust for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as anger_disgust for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as fear',
#                                         'Highest proportion of tweets labeled as fear for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as fear for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as joy_love',
#                                         'Highest proportion of tweets labeled as joy_love for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as joy_love for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as sadness',
#                                         'Highest proportion of tweets labeled as sadness for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as sadness for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as surprise',
#                                         'Highest proportion of tweets labeled as surprise for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as surprise for a handle in this bucket'
#                                         ]
# csv_Data_medBucket_RANDOMsample.append(csv_row_medBucket_RANDOMsample)

csv_row_medBucket_RANDOMsample = [   '~~~ medBucket 1/10 random subset of tweets per handle ~~~',
                                        len(handle_bucket_medTweets),
                                        totalTweets_medBucket_sub,

                                        proportion_emotion_medBucket,
                                        emotion_maxProportion_medBucket,
                                        emotion_minProportion_medBucket,
                                        medianProportion_emotion_medBucket,

                                        proportion_anger_disgust_medBucket,
                                        anger_disgust_maxProportion_medBucket,
                                        anger_disgust_minProportion_medBucket,
                                        medianProportion_anger_disgust_medBucket,

                                        proportion_fear_medBucket,
                                        fear_maxProportion_medBucket,
                                        fear_minProportion_medBucket,
                                        medianProportion_fear_medBucket,

                                        proportion_joy_love_medBucket,
                                        joy_love_maxProportion_medBucket,
                                        joy_love_minProportion_medBucket,
                                        medianProportion_joy_love_medBucket,

                                        proportion_sadness_medBucket,
                                        sadness_maxProportion_medBucket,
                                        sadness_minProportion_medBucket,
                                        medianProportion_sadness_medBucket,

                                        proportion_surprise_medBucket,
                                        surprise_maxProportion_medBucket,
                                        surprise_minProportion_medBucket,
                                        medianProportion_surprise_medBucket]

csv_Data_medBucket_RANDOMsample.append(csv_row_medBucket_RANDOMsample)

# myFile = open('print_data_medBucket_RANDOMsubset.csv', 'w')
# writer = csv.writer(myFile)
# writer.writerows(csv_Data_medBucket_RANDOMsample)

# tableName_row = ['~~~ medBucket 1/10 random subset ~~~', '', '', '','','','','','','','','','','','','','','','','']
# csv_Data_medBucket_RANDOMsample.insert(0, tableName_row)
# csv_Data_medBucket_RANDOMsample.append('\n')
writer = csv.writer(comprehensive_bucket_File)
writer.writerows(csv_Data_medBucket_RANDOMsample)
# END med BUCKET ANALYTICS - RANDOM 1/10 SUBSAMPLE
















# BEGIN high BUCKET ANALYTICS
emotion_tweet_sum = 0
emotion_proportionList_highBucket = []
anger_disgust_proportionList_highBucket = []
fear_proportionList_highBucket = []
joy_love_proportionList_highBucket = []
sadness_proportionList_highBucket = []
surprise_proportionList_highBucket = []

emotion_maxProportion_highBucket = 0
anger_disgust_maxProportion_highBucket = 0
fear_maxProportion_highBucket = 0
joy_love_maxProportion_highBucket = 0
sadness_maxProportion_highBucket = 0
surprise_maxProportion_highBucket = 0

emotion_minProportion_highBucket = 1
anger_disgust_minProportion_highBucket = 1
fear_minProportion_highBucket = 1
joy_love_minProportion_highBucket = 1
sadness_minProportion_highBucket = 1
surprise_minProportion_highBucket = 1

anger_disgust_proportion_handle = 0
fear_proportion_handle = 0
joy_love_proportion_handle = 0
sadness_proportion_handle = 0
surprise_proportion_handle = 0
none_proportion_handle = 0

totalTweets_highBucket = 0

for handle in handle_bucket_highTweets:

    totalTweets_highBucket = totalTweets_highBucket + handle_totalTweetCount[handle]

    anger_disgust_proportion = 0
    fear_proportion = 0
    joy_love_proportion = 0
    sadness_proportion = 0
    surprise_proportion = 0
    none_proportion = 0

    if handle in handleTo_numLabelledAs_anger_disgust:
        anger_disgust_proportion_handle = handleTo_numLabelledAs_anger_disgust[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_fear:
        fear_proportion_handle = handleTo_numLabelledAs_fear[handle] / float ( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_joy:
        joy_love_proportion_handle = handleTo_numLabelledAs_joy[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_sadness:
        sadness_proportion_handle = handleTo_numLabelledAs_sadness[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_surprise:
        surprise_proportion_handle = handleTo_numLabelledAs_surprise[handle] / float( handle_totalTweetCount[handle] )
    if handle in handleTo_numLabelledAs_none:
        none_proportion_handle = handleTo_numLabelledAs_none[handle] / float( handle_totalTweetCount[handle] )

    # proportion of tweets that have emotions for this handle
    none_proportion_handle = handleTo_numLabelledAs_none[handle] / float( handle_totalTweetCount[handle] )
    emotion_proportion_handle = 1 - none_proportion_handle

    emotion_proportionList_highBucket.append(emotion_proportion_handle)

    # anger_disgust proportion list append
    anger_disgust_proportionList_highBucket.append(anger_disgust_proportion_handle)

    # fear proportion list append
    fear_proportionList_highBucket.append(fear_proportion_handle)

    # joy_love proportion list append
    joy_love_proportionList_highBucket.append(joy_love_proportion_handle)

    # sadness proportion list append
    sadness_proportionList_highBucket.append(sadness_proportion_handle)

    # surprise proportion list append
    surprise_proportionList_highBucket.append(surprise_proportion_handle)

    # update Max and Min proportions if neccesary
    if emotion_maxProportion_highBucket < emotion_proportion_handle:
        emotion_maxProportion_highBucket = emotion_proportion_handle
    if anger_disgust_maxProportion_highBucket < anger_disgust_proportion_handle:
        anger_disgust_maxProportion_highBucket = anger_disgust_proportion_handle
    if fear_maxProportion_highBucket < fear_proportion_handle:
        fear_maxProportion_highBucket = fear_proportion_handle
    if joy_love_maxProportion_highBucket < joy_love_proportion_handle:
        joy_love_maxProportion_highBucket = joy_love_proportion_handle
    if sadness_maxProportion_highBucket < sadness_proportion_handle:
        sadness_maxProportion_highBucket = sadness_proportion_handle
    if surprise_maxProportion_highBucket < surprise_proportion_handle:
        surprise_maxProportion_highBucket = surprise_proportion_handle

    if emotion_minProportion_highBucket > emotion_proportion_handle:
        emotion_minProportion_highBucket = emotion_proportion_handle
    if anger_disgust_minProportion_highBucket > anger_disgust_proportion_handle:
        anger_disgust_minProportion_highBucket = anger_disgust_proportion_handle
    if fear_minProportion_highBucket > fear_proportion_handle:
        fear_minProportion_highBucket = fear_proportion_handle
    if joy_love_minProportion_highBucket > joy_love_proportion_handle:
        joy_love_minProportion_highBucket = joy_love_proportion_handle
    if sadness_minProportion_highBucket > sadness_proportion_handle:
        sadness_minProportion_highBucket = sadness_proportion_handle
    if surprise_minProportion_highBucket > surprise_proportion_handle:
        surprise_minProportion_highBucket = surprise_proportion_handle



# proportion of tweets that have emotions for this bucket
proportion_emotion_highBucket = statistics.mean ( emotion_proportionList_highBucket )
medianProportion_emotion_highBucket = statistics.median ( emotion_proportionList_highBucket )

# proportion of anger_disgust tweets that have emotions for this bucket
proportion_anger_disgust_highBucket = statistics.mean ( anger_disgust_proportionList_highBucket )
medianProportion_anger_disgust_highBucket = statistics.median ( anger_disgust_proportionList_highBucket )

# proportion of fear tweets that have emotions for this bucket
proportion_fear_highBucket = statistics.mean ( fear_proportionList_highBucket )
medianProportion_fear_highBucket = statistics.median ( fear_proportionList_highBucket )

# proportion of joy_love tweets that have emotions for this bucket
proportion_joy_love_highBucket = statistics.mean ( joy_love_proportionList_highBucket )
medianProportion_joy_love_highBucket = statistics.median ( joy_love_proportionList_highBucket )

# proportion of sadness tweets that have emotions for this bucket
proportion_sadness_highBucket = statistics.mean ( sadness_proportionList_highBucket )
medianProportion_sadness_highBucket = statistics.median ( sadness_proportionList_highBucket )

# proportion of surprise tweets that have emotions for this bucket
proportion_surprise_highBucket = statistics.mean ( surprise_proportionList_highBucket )
medianProportion_surprise_highBucket = statistics.median ( surprise_proportionList_highBucket )


# for each bucket of handles, provide
# proportion of tweets that have emotions
    # sum counts of all tweets that have emotion, then divide by total num tweets
# distrubtions of emotions for each bucket
    # proportion of tweets belonging to each emotion
# range of distrubtions for each emotion across users in a bcuket
    # for each emotion, go through all users and record max and min proportions
    #

# write out to csv
csv_Data_highBucket = []
# csv_row_highBucket = [   'Number of handles in this bucket',
#                                         'Number of tweets corresponding to the handles in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as emotional',
#                                         'Highest proportion of tweets labeled as emotional for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as emotional for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as anger_disgust',
#                                         'Highest proportion of tweets labeled as anger_disgust for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as anger_disgust for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as fear',
#                                         'Highest proportion of tweets labeled as fear for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as fear for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as joy_love',
#                                         'Highest proportion of tweets labeled as joy_love for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as joy_love for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as sadness',
#                                         'Highest proportion of tweets labeled as sadness for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as sadness for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as surprise',
#                                         'Highest proportion of tweets labeled as surprise for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as surprise for a handle in this bucket'
#                                         ]
# csv_Data_highBucket.append(csv_row_highBucket)

csv_row_highBucket = [  '~~~ highBucket ~~~',
                                        len(handle_bucket_highTweets),
                                        totalTweets_highBucket,

                                        proportion_emotion_highBucket,
                                        emotion_maxProportion_highBucket,
                                        emotion_minProportion_highBucket,
                                        medianProportion_emotion_highBucket,

                                        proportion_anger_disgust_highBucket,
                                        anger_disgust_maxProportion_highBucket,
                                        anger_disgust_minProportion_highBucket,
                                        medianProportion_anger_disgust_highBucket,

                                        proportion_fear_highBucket,
                                        fear_maxProportion_highBucket,
                                        fear_minProportion_highBucket,
                                        medianProportion_fear_highBucket,

                                        proportion_joy_love_highBucket,
                                        joy_love_maxProportion_highBucket,
                                        joy_love_minProportion_highBucket,
                                        medianProportion_joy_love_highBucket,

                                        proportion_sadness_highBucket,
                                        sadness_maxProportion_highBucket,
                                        sadness_minProportion_highBucket,
                                        medianProportion_sadness_highBucket,

                                        proportion_surprise_highBucket,
                                        surprise_maxProportion_highBucket,
                                        surprise_minProportion_highBucket,
                                        medianProportion_surprise_medBucket]

csv_Data_highBucket.append(csv_row_highBucket)



# myFile = open('print_data_highBucket.csv', 'w')
# writer = csv.writer(myFile)
# writer.writerows(csv_Data_highBucket)


# tableName_row = ['~~~ highBucket ~~~', '', '', '','','','','','','','','','','','','','','','','']
# csv_Data_highBucket.insert(0, tableName_row)
# csv_Data_highBucket.append('\n')
writer = csv.writer(comprehensive_bucket_File)
writer.writerows(csv_Data_highBucket)
# END high BUCKET ANALYTICS




# BEGIN high BUCKET ANALYTICS -  SUBSAMPLE of last 1/10
# numHandles_toSelect_highSub = len(handle_bucket_highTweets) / 10
# numHandles_toSelect_highSub = 25
# numHandles_toSelect_highSub = len(handle_bucket_highTweets)
# handle_bucket_highTweets_subset =  handle_bucket_highTweets[-numHandles_toSelect_highSub:]

emotion_tweet_sum = 0
emotion_proportionList_highBucket = []
anger_disgust_proportionList_highBucket = []
fear_proportionList_highBucket = []
joy_love_proportionList_highBucket = []
sadness_proportionList_highBucket = []
surprise_proportionList_highBucket = []

emotion_maxProportion_highBucket = 0
anger_disgust_maxProportion_highBucket = 0
fear_maxProportion_highBucket = 0
joy_love_maxProportion_highBucket = 0
sadness_maxProportion_highBucket = 0
surprise_maxProportion_highBucket = 0

emotion_minProportion_highBucket = 1
anger_disgust_minProportion_highBucket = 1
fear_minProportion_highBucket = 1
joy_love_minProportion_highBucket = 1
sadness_minProportion_highBucket = 1
surprise_minProportion_highBucket = 1

anger_disgust_proportion_handle = 0
fear_proportion_handle = 0
joy_love_proportion_handle = 0
sadness_proportion_handle = 0
surprise_proportion_handle = 0
none_proportion_handle = 0

totalTweets_highBucket_sub = 0

for handle in handle_bucket_highTweets:

    tweet_list = handle_toTweetList_dict[handle][-20:] # select last 20
    totalTweetCount, numLabelledAs_anger_disgust, numLabelledAs_fear, numLabelledAs_joy, numLabelledAs_sadness, numLabelledAs_surprise, numLabelledAs_none = calculate_handleStats( tweet_list )

    totalTweets_highBucket_sub = totalTweets_highBucket_sub + totalTweetCount

    anger_disgust_proportion_handle = numLabelledAs_anger_disgust / float ( totalTweetCount )
    fear_proportion_handle = numLabelledAs_fear / float ( totalTweetCount )
    joy_love_proportion_handle = numLabelledAs_joy / float ( totalTweetCount )
    sadness_proportion_handle = numLabelledAs_sadness / float ( totalTweetCount )
    surprise_proportion_handle = numLabelledAs_surprise / float ( totalTweetCount )
    none_proportion_handle = numLabelledAs_none / float ( totalTweetCount )

    # proportion of tweets that have emotions for this handle
    none_proportion_handle = handleTo_numLabelledAs_none[handle] / float( handle_totalTweetCount[handle] )
    emotion_proportion_handle = 1 - none_proportion_handle
    emotion_proportionList_highBucket.append(emotion_proportion_handle)

    # anger_disgust proportion list append
    anger_disgust_proportionList_highBucket.append(anger_disgust_proportion_handle)

    # fear proportion list append
    fear_proportionList_highBucket.append(fear_proportion_handle)

    # joy_love proportion list append
    joy_love_proportionList_highBucket.append(joy_love_proportion_handle)

    # sadness proportion list append
    sadness_proportionList_highBucket.append(sadness_proportion_handle)

    # surprise proportion list append
    surprise_proportionList_highBucket.append(surprise_proportion_handle)

    # update Max and Min proportions if neccesary
    if emotion_maxProportion_highBucket < emotion_proportion_handle:
        emotion_maxProportion_highBucket = emotion_proportion_handle
    if anger_disgust_maxProportion_highBucket < anger_disgust_proportion_handle:
        anger_disgust_maxProportion_highBucket = anger_disgust_proportion_handle
    if fear_maxProportion_highBucket < fear_proportion_handle:
        fear_maxProportion_highBucket = fear_proportion_handle
    if joy_love_maxProportion_highBucket < joy_love_proportion_handle:
        joy_love_maxProportion_highBucket = joy_love_proportion_handle
    if sadness_maxProportion_highBucket < sadness_proportion_handle:
        sadness_maxProportion_highBucket = sadness_proportion_handle
    if surprise_maxProportion_highBucket < surprise_proportion_handle:
        surprise_maxProportion_highBucket = surprise_proportion_handle

    if emotion_minProportion_highBucket > emotion_proportion_handle:
        emotion_minProportion_highBucket = emotion_proportion_handle
    if anger_disgust_minProportion_highBucket > anger_disgust_proportion_handle:
        anger_disgust_minProportion_highBucket = anger_disgust_proportion_handle
    if fear_minProportion_highBucket > fear_proportion_handle:
        fear_minProportion_highBucket = fear_proportion_handle
    if joy_love_minProportion_highBucket > joy_love_proportion_handle:
        joy_love_minProportion_highBucket = joy_love_proportion_handle
    if sadness_minProportion_highBucket > sadness_proportion_handle:
        sadness_minProportion_highBucket = sadness_proportion_handle
    if surprise_minProportion_highBucket > surprise_proportion_handle:
        surprise_minProportion_highBucket = surprise_proportion_handle


# proportion of tweets that have emotions for this bucket
proportion_emotion_highBucket = statistics.mean ( emotion_proportionList_highBucket )
medianProportion_emotion_highBucket = statistics.median ( emotion_proportionList_highBucket )

# proportion of anger_disgust tweets that have emotions for this bucket
proportion_anger_disgust_highBucket = statistics.mean ( anger_disgust_proportionList_highBucket )
medianProportion_anger_disgust_highBucket = statistics.median ( anger_disgust_proportionList_highBucket )

# proportion of fear tweets that have emotions for this bucket
proportion_fear_highBucket = statistics.mean ( fear_proportionList_highBucket )
medianProportion_fear_highBucket = statistics.median ( fear_proportionList_highBucket )

# proportion of joy_love tweets that have emotions for this bucket
proportion_joy_love_highBucket = statistics.mean ( joy_love_proportionList_highBucket )
medianProportion_joy_love_highBucket = statistics.median ( joy_love_proportionList_highBucket )

# proportion of sadness tweets that have emotions for this bucket
proportion_sadness_highBucket = statistics.mean ( sadness_proportionList_highBucket )
medianProportion_sadness_highBucket = statistics.median ( sadness_proportionList_highBucket )

# proportion of surprise tweets that have emotions for this bucket
proportion_surprise_highBucket = statistics.mean ( surprise_proportionList_highBucket )
medianProportion_surprise_highBucket = statistics.median ( surprise_proportionList_highBucket )

# for each bucket of handles, provide
# proportion of tweets that have emotions
    # sum counts of all tweets that have emotion, then divide by total num tweets
# distrubtions of emotions for each bucket
    # proportion of tweets belonging to each emotion
# range of distrubtions for each emotion across users in a bcuket
    # for each emotion, go through all users and record max and min proportions
    #

# write out to csv
csv_Data_highBucket_subset = []
# csv_row_highBucket_subset = [   'Number of handles in this bucket',
#                                         'Number of tweets corresponding to the handles in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as emotional',
#                                         'Highest proportion of tweets labeled as emotional for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as emotional for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as anger_disgust',
#                                         'Highest proportion of tweets labeled as anger_disgust for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as anger_disgust for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as fear',
#                                         'Highest proportion of tweets labeled as fear for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as fear for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as joy_love',
#                                         'Highest proportion of tweets labeled as joy_love for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as joy_love for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as sadness',
#                                         'Highest proportion of tweets labeled as sadness for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as sadness for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as surprise',
#                                         'Highest proportion of tweets labeled as surprise for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as surprise for a handle in this bucket'
#                                         ]
# csv_Data_highBucket_subset.append(csv_row_highBucket_subset)

csv_row_highBucket_subset = [   '~~~ highBucket last 20 tweets per handle subsample ~~~',
                                        len(handle_bucket_highTweets),
                                        totalTweets_highBucket_sub,

                                        proportion_emotion_highBucket,
                                        emotion_maxProportion_highBucket,
                                        emotion_minProportion_highBucket,
                                        medianProportion_emotion_highBucket,

                                        proportion_anger_disgust_highBucket,
                                        anger_disgust_maxProportion_highBucket,
                                        anger_disgust_minProportion_highBucket,
                                        medianProportion_anger_disgust_highBucket,

                                        proportion_fear_highBucket,
                                        fear_maxProportion_highBucket,
                                        fear_minProportion_highBucket,
                                        medianProportion_fear_highBucket,

                                        proportion_joy_love_highBucket,
                                        joy_love_maxProportion_highBucket,
                                        joy_love_minProportion_highBucket,
                                        medianProportion_joy_love_highBucket,

                                        proportion_sadness_highBucket,
                                        sadness_maxProportion_highBucket,
                                        sadness_minProportion_highBucket,
                                        medianProportion_sadness_highBucket,

                                        proportion_surprise_highBucket,
                                        surprise_maxProportion_highBucket,
                                        surprise_minProportion_highBucket,
                                        medianProportion_surprise_medBucket]

csv_Data_highBucket_subset.append(csv_row_highBucket_subset)

# myFile = open('print_data_highBucket_subset.csv', 'w')
# writer = csv.writer(myFile)
# writer.writerows(csv_Data_highBucket_subset)


# tableName_row = ['~~~ highBucket 1/10 subset ~~~', '', '', '','','','','','','','','','','','','','','','','']
# csv_Data_highBucket_subset.insert(0, tableName_row)
# csv_Data_highBucket_subset.append('\n')
writer = csv.writer(comprehensive_bucket_File)
writer.writerows(csv_Data_highBucket_subset)

# END high BUCKET ANALYTICS - SUBSAMPLE of last 1/10



# BEGIN high BUCKET ANALYTICS - RANDOM 1/10 SUBSAMPLE

# select random 10th
# num_to_select = len(handle_bucket_highTweets) / 10
# handle_bucket_highTweets_RANDOMsubset = handle_bucket_highTweets[:num_to_select]
# handle_bucket_highTweets_RANDOMsubset = random.sample(handle_buck et_highTweets, num_to_select)

# tweets_highBucket_RANDOMsubset = random.sample(tweets_highBucket, numHandles_toSelect_highSub)
# tweets_highBucket_RANDOMsubset = random.sample(tweets_highBucket, num_to_select)
# handle_totalTweetCount_SUB, handleTo_numLabelledAs_anger_disgust_SUB, handleTo_numLabelledAs_fear_SUB, handleTo_numLabelledAs_joy_SUB, handleTo_numLabelledAs_sadness_SUB, handleTo_numLabelledAs_surprise_SUB, handleTo_numLabelledAs_none_SUB, unique_handleList = calculate_handleStats( tweetToHandle_dict, tweets_highBucket_RANDOMsubset )


emotion_tweet_sum = 0
emotion_proportionList_highBucket = []
anger_disgust_proportionList_highBucket = []
fear_proportionList_highBucket = []
joy_love_proportionList_highBucket = []
sadness_proportionList_highBucket = []
surprise_proportionList_highBucket = []

emotion_maxProportion_highBucket = 0
anger_disgust_maxProportion_highBucket = 0
fear_maxProportion_highBucket = 0
joy_love_maxProportion_highBucket = 0
sadness_maxProportion_highBucket = 0
surprise_maxProportion_highBucket = 0

emotion_minProportion_highBucket = 1
anger_disgust_minProportion_highBucket = 1
fear_minProportion_highBucket = 1
joy_love_minProportion_highBucket = 1
sadness_minProportion_highBucket = 1
surprise_minProportion_highBucket = 1

anger_disgust_proportion_handle = 0
fear_proportion_handle = 0
joy_love_proportion_handle = 0
sadness_proportion_handle = 0
surprise_proportion_handle = 0
none_proportion_handle = 0

# totalTweets_highBucket = 0

for handle in handle_bucket_highTweets:

    # totalTweets_highBucket = totalTweets_highBucket + handle_totalTweetCount_SUB[handle]

    tweetList = handle_toTweetList_dict[handle]
    num_to_select = len(tweetList) / 10
    tweetList_RANDOMsubset = random.sample(tweetList, num_to_select)

    totalTweetCount, numLabelledAs_anger_disgust, numLabelledAs_fear, numLabelledAs_joy, numLabelledAs_sadness, numLabelledAs_surprise, numLabelledAs_none = calculate_handleStats( tweetList_RANDOMsubset )

    totalTweets_highBucket_sub = totalTweets_highBucket_sub + totalTweetCount

    anger_disgust_proportion_handle = numLabelledAs_anger_disgust / float ( totalTweetCount )
    fear_proportion_handle = numLabelledAs_fear / float ( totalTweetCount )
    joy_love_proportion_handle = numLabelledAs_joy / float ( totalTweetCount )
    sadness_proportion_handle = numLabelledAs_sadness / float ( totalTweetCount )
    surprise_proportion_handle = numLabelledAs_surprise / float ( totalTweetCount )
    none_proportion_handle = numLabelledAs_none / float ( totalTweetCount )


    # proportion of tweets that have emotions for this handle
    # none_proportion_handle = handleTo_numLabelledAs_none_SUB[handle] / float( handle_totalTweetCount_SUB[handle] )
    emotion_proportion_handle = 1 - none_proportion_handle
    emotion_proportionList_highBucket.append(emotion_proportion_handle)

    # anger_disgust proportion list append
    anger_disgust_proportionList_highBucket.append(anger_disgust_proportion_handle)

    # fear proportion list append
    fear_proportionList_highBucket.append(fear_proportion_handle)

    # joy_love proportion list append
    joy_love_proportionList_highBucket.append(joy_love_proportion_handle)

    # sadness proportion list append
    sadness_proportionList_highBucket.append(sadness_proportion_handle)

    # surprise proportion list append
    surprise_proportionList_highBucket.append(surprise_proportion_handle)

    # update Max and Min proportions if neccesary
    if emotion_maxProportion_highBucket < emotion_proportion_handle:
        emotion_maxProportion_highBucket = emotion_proportion_handle
    if anger_disgust_maxProportion_highBucket < anger_disgust_proportion_handle:
        anger_disgust_maxProportion_highBucket = anger_disgust_proportion_handle
    if fear_maxProportion_highBucket < fear_proportion_handle:
        fear_maxProportion_highBucket = fear_proportion_handle
    if joy_love_maxProportion_highBucket < joy_love_proportion_handle:
        joy_love_maxProportion_highBucket = joy_love_proportion_handle
    if sadness_maxProportion_highBucket < sadness_proportion_handle:
        sadness_maxProportion_highBucket = sadness_proportion_handle
    if surprise_maxProportion_highBucket < surprise_proportion_handle:
        surprise_maxProportion_highBucket = surprise_proportion_handle

    if emotion_minProportion_highBucket > emotion_proportion_handle:
        emotion_minProportion_highBucket = emotion_proportion_handle
    if anger_disgust_minProportion_highBucket > anger_disgust_proportion_handle:
        anger_disgust_minProportion_highBucket = anger_disgust_proportion_handle
    if fear_minProportion_highBucket > fear_proportion_handle:
        fear_minProportion_highBucket = fear_proportion_handle
    if joy_love_minProportion_highBucket > joy_love_proportion_handle:
        joy_love_minProportion_highBucket = joy_love_proportion_handle
    if sadness_minProportion_highBucket > sadness_proportion_handle:
        sadness_minProportion_highBucket = sadness_proportion_handle
    if surprise_minProportion_highBucket > surprise_proportion_handle:
        surprise_minProportion_highBucket = surprise_proportion_handle


# proportion of tweets that have emotions for this bucket
proportion_emotion_highBucket = statistics.mean ( emotion_proportionList_highBucket )
medianProportion_emotion_highBucket = statistics.median ( emotion_proportionList_highBucket )

# proportion of anger_disgust tweets that have emotions for this bucket
proportion_anger_disgust_highBucket = statistics.mean ( anger_disgust_proportionList_highBucket )
medianProportion_anger_disgust_highBucket = statistics.median ( anger_disgust_proportionList_highBucket )

# proportion of fear tweets that have emotions for this bucket
proportion_fear_highBucket = statistics.mean ( fear_proportionList_highBucket )
medianProportion_fear_highBucket = statistics.median ( fear_proportionList_highBucket )

# proportion of joy_love tweets that have emotions for this bucket
proportion_joy_love_highBucket = statistics.mean ( joy_love_proportionList_highBucket )
medianProportion_joy_love_highBucket = statistics.median ( joy_love_proportionList_highBucket )

# proportion of sadness tweets that have emotions for this bucket
proportion_sadness_highBucket = statistics.mean ( sadness_proportionList_highBucket )
medianProportion_sadness_highBucket = statistics.median ( sadness_proportionList_highBucket )

# proportion of surprise tweets that have emotions for this bucket
proportion_surprise_highBucket = statistics.mean ( surprise_proportionList_highBucket )
medianProportion_surprise_highBucket = statistics.median ( surprise_proportionList_highBucket )

# for each bucket of handles, provide
# proportion of tweets that have emotions
    # sum counts of all tweets that have emotion, then divide by total num tweets
# distrubtions of emotions for each bucket
    # proportion of tweets belonging to each emotion
# range of distrubtions for each emotion across users in a bcuket
    # for each emotion, go through all users and record max and min proportions
    #

# write out to csv
csv_Data_highBucket_RANDOMsample = []
# csv_row_highBucket_RANDOMsample = [   'Number of handles in this bucket',
#                                         'Number of tweets corresponding to the handles in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as emotional',
#                                         'Highest proportion of tweets labeled as emotional for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as emotional for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as anger_disgust',
#                                         'Highest proportion of tweets labeled as anger_disgust for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as anger_disgust for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as fear',
#                                         'Highest proportion of tweets labeled as fear for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as fear for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as joy_love',
#                                         'Highest proportion of tweets labeled as joy_love for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as joy_love for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as sadness',
#                                         'Highest proportion of tweets labeled as sadness for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as sadness for a handle in this bucket',

#                                         'Proportion of all tweets in this bucket that are labeled as surprise',
#                                         'Highest proportion of tweets labeled as surprise for a handle in this bucket',
#                                         'Lowest proportion of tweets labeled as surprise for a handle in this bucket'
#                                         ]
# csv_Data_highBucket_RANDOMsample.append(csv_row_highBucket_RANDOMsample)

csv_row_highBucket_RANDOMsample = [   '~~~ highBucket 1/10 random subset of tweets per handle ~~~',
                                        len(handle_bucket_highTweets),
                                        totalTweets_highBucket_sub,

                                        proportion_emotion_highBucket,
                                        emotion_maxProportion_highBucket,
                                        emotion_minProportion_highBucket,
                                        medianProportion_emotion_highBucket,

                                        proportion_anger_disgust_highBucket,
                                        anger_disgust_maxProportion_highBucket,
                                        anger_disgust_minProportion_highBucket,
                                        medianProportion_anger_disgust_highBucket,

                                        proportion_fear_highBucket,
                                        fear_maxProportion_highBucket,
                                        fear_minProportion_highBucket,
                                        medianProportion_fear_highBucket,

                                        proportion_joy_love_highBucket,
                                        joy_love_maxProportion_highBucket,
                                        joy_love_minProportion_highBucket,
                                        medianProportion_joy_love_highBucket,

                                        proportion_sadness_highBucket,
                                        sadness_maxProportion_highBucket,
                                        sadness_minProportion_highBucket,
                                        medianProportion_sadness_highBucket,

                                        proportion_surprise_highBucket,
                                        surprise_maxProportion_highBucket,
                                        surprise_minProportion_highBucket,
                                        medianProportion_surprise_medBucket]

csv_Data_highBucket_RANDOMsample.append(csv_row_highBucket_RANDOMsample)

# myFile = open('print_data_highBucket_RANDOMsubset.csv', 'w')
# writer = csv.writer(myFile)
# writer.writerows(csv_Data_highBucket_RANDOMsample)

# tableName_row = ['~~~ highBucket 1/10 random subset ~~~', '', '', '','','','','','','','','','','','','','','','','']
# csv_Data_highBucket_RANDOMsample.insert(0, tableName_row)
# csv_Data_highBucket_RANDOMsample.append('\n')
writer = csv.writer(comprehensive_bucket_File)
writer.writerows(csv_Data_highBucket_RANDOMsample)
# END high BUCKET ANALYTICS - RANDOM 1/10 SUBSAMPLE



#ADD: total users

# create function to print distribution of # tweets per user?
# creat dict of handles to num_tweets, sort on dict value
# print list

# low: under 50 tweets. med: 51 - 500 tweets. high: more than 500 tweets ?
# NEED TO: remove more than 50 tweets condition

# create three buckets of users
# then go through each bucket
# and present:
#   breakdown of emotions
#   breakdown of proportions
#   number of

# for each bucket,
# DO: take a random sample
# DO: take a sample from the last X
# for these sub samples, compare to big sample, see if we are introducing variation bias




