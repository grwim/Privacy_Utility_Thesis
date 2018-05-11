import pdb
import csv
import tweet_processing
import svc_emotions
from nltk.tokenize import word_tokenize

# create tie functionality
# count how many ties I have (can break out tie counts for each pair of emotions)

# for FN of none category, look at avg # of emotional dict hits for all tweets that were actually none, but labelled as emotional
    # keep count of total dict hits for each tweet that were actually none, but labelled as emotional [sum this count over all such tweets]
    # then divide that sum by the number of such tweets

# for FN's of emotional categories, look at corresponding tweets
    # keep a list; append tweet, ascribed label and true label if true label is emotional and ascribed label is 'none'

# ---
# Look at overlap of words between each emotional dict
    # every time there is a tie, store the tweet to a corresponding list of tweets for each tie-scenario

        # make a dictionary for every tie combo that has each emotion as a key and a list to store tweets as a value

# want to re-randomize the data for the dictionary modifications

# hey. the proposed changes are few, so I don't know if they would make a big difference either way.
# I would let the data decide by getting precision/recall using the proposed modifications.
# make sure you re-randomize the the test data set before doing this, since you don't want to 'cheat' by
# making the exactly the changes the that would help the exact data you have

# @Chris Kirov Given that the DictHit method doesn't involve the training of a classifier, I'm not doing any hold-out methods;
# The algorithm merely iterates over all tweets in our corpus and ascribes labels according to dictionary hits.
# Given that a classifier isnt being trained, how would you recomend achieving such a re-randomization?

def output_list_from_csv(emotion):
    filename = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/emotion_dictionary/' + emotion + '.txt'

    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        synonym_list = list(reader)[0]
    return synonym_list

def classify_tweet(tweet):


    emotion_list = ['anger_disgust', 'fear', 'joy_love', 'sadness', 'surprise',]

    tweet = word_tokenize(tweet)

    dictionary_hit_count = {'anger_disgust': 0,
                                        'fear': 0,
                                        'joy_love': 0,
                                        'sadness': 0,
                                        'surprise': 0}
    no_dict_hit = True

    emotion_word_count = 0
    word_count = 0

    for word in tweet:
        word_count = word_count + 1
        # check word against each dictionary
        # go through each emotion dictinoary
        for emotion in emotion_list:
            snyonym_list = output_list_from_csv(emotion)
            if word in snyonym_list:
                no_dict_hit = False
                # increment count
                dictionary_hit_count[emotion] = dictionary_hit_count[emotion] + 1
                emotion_word_count = emotion_word_count + 1

    # if no hit with any dict, then increment
    label_predicted = ''
    freq_max = 0
    if no_dict_hit:
        # label as 'none'
        label_predicted = 'none'
    else:
        # use max hit count to select best fitting dictionary
        label_predicted = max(dictionary_hit_count, key=dictionary_hit_count.get)
        freq_max = dictionary_hit_count[label_predicted]

    return label_predicted, emotion_word_count, word_count, dictionary_hit_count, emotion_list



# tweets_legacy= tweet_processing.clean("tweets_full.txt")
# labels_legacy = svc_emotions.file_to_array("labels_full.txt")

# emotion_list = ['anger', 'disgust', 'fear', 'joy', 'love', 'sadness', 'surprise',]

# all_processed_tweets = []

# count_TP = {'anger': 0,
#                                     'disgust': 0,
#                                     'fear': 0,
#                                     'joy': 0,
#                                     'love': 0,
#                                     'sadness': 0,
#                                     'surprise': 0}

# count_FP = {'anger': 0,
#                                     'disgust': 0,
#                                     'fear': 0,
#                                     'joy': 0,
#                                     'love': 0,
#                                     'sadness': 0,
#                                     'surprise': 0}

# count_FN = {'anger': 0,
#                                     'disgust': 0,
#                                     'fear': 0,
#                                     'joy': 0,
#                                     'love': 0,
#                                     'sadness': 0,
#                                     'surprise': 0}

# dictionary_hit_count_TOTAL = {'anger': 0,
#                                     'disgust': 0,
#                                     'fear': 0,
#                                     'joy': 0,
#                                     'love': 0,
#                                     'sadness': 0,
#                                     'surprise': 0}

# count_emotional_CORRECT = {'anger': 0,
#                                     'disgust': 0,
#                                     'fear': 0,
#                                     'joy': 0,
#                                     'love': 0,
#                                     'sadness': 0,
#                                     'surprise': 0}

# count_tweets_per_emotion = {'anger': 0,
#                                     'disgust': 0,
#                                     'fear': 0,
#                                     'joy': 0,
#                                     'love': 0,
#                                     'sadness': 0,
#                                     'surprise': 0}

# # load in each dictionary into a list

# none_total = 0

# two_way_tie_count = 0
# three_way_tie_count = 0
# four_way_tie_count = 0

# anger_disgust_tie_count = 0
# anger_fear_tie_count = 0
# anger_joy_tie_count = 0
# anger_love_tie_count = 0
# anger_sadness_tie_count = 0
# anger_surprise_tie_count = 0
# disgust_fear_tie_count = 0
# disgust_joy_tie_count = 0
# disgust_love_tie_count = 0
# disgust_sadness_tie_count = 0
# disgust_surprise_tie_count= 0
# fear_joy_tie_count = 0
# fear_love_tie_count = 0
# fear_sadness_tie_count = 0
# fear_surprise_tie_count = 0
# joy_love_tie_count = 0
# joy_sadness_tie_count = 0
# joy_surprise_tie_count = 0
# love_sadness_tie_count= 0
# love_surprise_tie_count = 0
# sadness_surprise_tie_count = 0

# tied_tweets_anger_disgust_list = []
# tied_tweets_anger_fear_list = []
# tied_tweets_anger_joy_list = []
# tied_tweets_anger_love_list = []
# tied_tweets_anger_sadness_list = []
# tied_tweets_anger_surprise_list = []
# tied_tweets_disgust_fear_list = []
# tied_tweets_disgust_joy_list  = []
# tied_tweets_disgust_love_list = []
# tied_tweets_disgust_sadness_list = []
# tied_tweets_disgust_surprise_list = []
# tied_tweets_fear_joy_list = []
# tied_tweets_fear_love_list = []
# tied_tweets_fear_sadness_list = []
# tied_tweets_fear_surprise_list = []
# tied_tweets_joy_love_list = []
# tied_tweets_joy_sadness_list = []
# tied_tweets_joy_surprise_list = []
# tied_tweets_love_sadness_list = []
# tied_tweets_love_surprise_list =[]
# tied_tweets_sadness_surprise_list = []

# FN_dict_hits_none = 0

# dict_of_FN_emotion_tweets = {'anger': [],
#                                                 'disgust': [],
#                                                 'fear': [],
#                                                 'joy': [],
#                                                 'love': [],
#                                                 'sadness': [],
#                                                 'surprise': [],
#                                                 'emotion': []}


# # MERGE EMOTIONS HERE
# combine_joy_love = False
# combine_anger_disgust = False

# # go through each tweet
# for x in range(0, len(tweets_legacy)):

#     tweet = tweets_legacy[x]
#     tweet = word_tokenize(tweet)

#     # reintilize dict hit counts
#     dictionary_hit_count = {'anger': 0,
#                                         'disgust': 0,
#                                         'fear': 0,
#                                         'joy': 0,
#                                         'love': 0,
#                                         'sadness': 0,
#                                         'surprise': 0
#                                                             }


#     # reset bool for whether any dictionary was hit for a tweet
#     no_dict_hit = True

#     for word in tweet:
#         # check word against each dictionary
#         # go through each emotion dictinoary
#         for emotion in emotion_list:
#             snyonym_list = output_list_from_csv(emotion)
#             if word in snyonym_list:
#                 no_dict_hit = False
#                 # increment count
#                 dictionary_hit_count[emotion] = dictionary_hit_count[emotion] + 1
#                 dictionary_hit_count_TOTAL[emotion] = dictionary_hit_count_TOTAL[emotion] + 1

# # if no hit with any dict, then increment
#     label_predicted = ''
#     freq_max = 0
#     if no_dict_hit:
#         # label as 'none'
#         label_predicted = 'none'
#         none_total = none_total + 1
#     else:
#         # COMBINE EMOTIONS HERE; combined emotions are subsumed into the first of the two emotions (anger,disgust --> anger )
#         if combine_joy_love:
#             dictionary_hit_count['joy'] += dictionary_hit_count['love']
#         if combine_anger_disgust:
#             dictionary_hit_count['anger'] += dictionary_hit_count['disgust' ]

#         # use max hit count to select best fitting dictionary
#         label_predicted = max(dictionary_hit_count, key=dictionary_hit_count.get)
#         freq_max = dictionary_hit_count[label_predicted]

#     # tie checking functionality
#         # go through each emotion
#     tied_emotions_for_tweet = []
#     for emotion in emotion_list:
#         # check dictionary hits for that emotion
#         if (dictionary_hit_count[emotion] == freq_max) & (freq_max > 0):
#         # if # of hits for that dict matches current freq_max, then add to list of emotions that tied for hits with this tweet
#             tied_emotions_for_tweet.append(emotion)
#     # based on size of list, increment either count for 2-way tie, 3-way tie, or 4-way tie
#     if len(tied_emotions_for_tweet) == 2:
#         two_way_tie_count = two_way_tie_count + 1
#     elif len(tied_emotions_for_tweet) == 3:
#         three_way_tie_count = three_way_tie_count + 1
#     elif len(tied_emotions_for_tweet) == 4:
#         four_way_tie_count = four_way_tie_count + 1

#     emotion_word_list = []

#     # # keep track of tie counts between each emotion
#     if ('anger' in tied_emotions_for_tweet) & ('disgust' in tied_emotions_for_tweet):
#         anger_disgust_tie_count = anger_disgust_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_anger_disgust_list.append(emotion_word_list)
#     elif ('anger' in tied_emotions_for_tweet) & ('fear' in tied_emotions_for_tweet):
#         anger_fear_tie_count = anger_fear_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_anger_fear_list.append(emotion_word_list)
#     elif ('anger' in tied_emotions_for_tweet) & ('joy' in tied_emotions_for_tweet):
#         anger_joy_tie_count = anger_joy_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_anger_joy_list.append(emotion_word_list)
#     elif ('anger' in tied_emotions_for_tweet) & ('love' in tied_emotions_for_tweet):
#         anger_love_tie_count = anger_love_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_anger_love_list.append(emotion_word_list)
#     elif ('anger' in tied_emotions_for_tweet) & ('sadness' in tied_emotions_for_tweet):
#         anger_sadness_tie_count = anger_sadness_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_anger_sadness_list.append(emotion_word_list)
#     elif ('anger' in tied_emotions_for_tweet) & ('surprise' in tied_emotions_for_tweet):
#         anger_surprise_tie_count = anger_surprise_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_anger_surprise_list.append(emotion_word_list)
#     elif ('disgust' in tied_emotions_for_tweet) & ('fear' in tied_emotions_for_tweet):
#         disgust_fear_tie_count = disgust_fear_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_disgust_fear_list.append(emotion_word_list)
#     elif ('disgust' in tied_emotions_for_tweet) & ('joy' in tied_emotions_for_tweet):
#         disgust_joy_tie_count = disgust_joy_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_disgust_joy_list.append(emotion_word_list)
#     elif ('disgust' in tied_emotions_for_tweet) & ('love' in tied_emotions_for_tweet):
#         disgust_love_tie_count = disgust_love_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_disgust_love_list.append(emotion_word_list)
#     elif ('disgust' in tied_emotions_for_tweet) & ('sadness' in tied_emotions_for_tweet):
#         disgust_sadness_tie_count = disgust_sadness_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_disgust_sadness_list.append(emotion_word_list)
#     elif ('disgust' in tied_emotions_for_tweet) & ('surprise' in tied_emotions_for_tweet):
#         disgust_surprise_tie_count = disgust_surprise_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_disgust_surprise_list.append(emotion_word_list)

#     elif ('fear' in tied_emotions_for_tweet) & ('joy' in tied_emotions_for_tweet):
#         fear_joy_tie_count = fear_joy_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_fear_joy_list.append(emotion_word_list)
#     elif ('fear' in tied_emotions_for_tweet) & ('love' in tied_emotions_for_tweet):
#         fear_love_tie_count = fear_love_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_fear_love_list.append(emotion_word_list)
#     elif ('fear' in tied_emotions_for_tweet) & ('sadness' in tied_emotions_for_tweet):
#         fear_sadness_tie_count = fear_sadness_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_fear_sadness_list.append(emotion_word_list)
#     elif ('fear' in tied_emotions_for_tweet) & ('surprise' in tied_emotions_for_tweet):
#         fear_surprise_tie_count = fear_surprise_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_fear_surprise_list.append(emotion_word_list)

#     elif ('joy' in tied_emotions_for_tweet) & ('love' in tied_emotions_for_tweet):
#         joy_love_tie_count = joy_love_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_joy_love_list.append(emotion_word_list)
#     elif ('joy' in tied_emotions_for_tweet) & ('sadness' in tied_emotions_for_tweet):
#         joy_sadness_tie_count = joy_sadness_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_joy_sadness_list.append(emotion_word_list)
#     elif ('joy' in tied_emotions_for_tweet) & ('surprise' in tied_emotions_for_tweet):
#         joy_surprise_tie_count = joy_surprise_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_joy_surprise_list.append(emotion_word_list)

#     elif ('love' in tied_emotions_for_tweet) & ('sadness' in tied_emotions_for_tweet):
#         love_sadness_tie_count = love_sadness_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_love_sadness_list.append(emotion_word_list)
#     elif ('love' in tied_emotions_for_tweet) & ('surprise' in tied_emotions_for_tweet):
#         love_surprise_tie_count = love_surprise_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_love_surprise_list.append(emotion_word_list)

#     elif ('sadness' in tied_emotions_for_tweet) & ('surprise' in tied_emotions_for_tweet):
#         sadness_surprise_tie_count = sadness_surprise_tie_count + 1
#         for word in tweet:
#             for emotion in emotion_list:
#                 snyonym_list = output_list_from_csv(emotion)
#                 if word in snyonym_list:
#                     emotion_word_list.append(word)
#         tied_tweets_sadness_surprise_list.append(emotion_word_list)

#     # anger_tied = False
#     # disgust_tied = False
#     # fear_tied = False
#     # joy_tied = False
#     # love_tied = False
#     # sadness_tied = False
#     # surprise_tied = False
#     # for item in tied_emotions_for_tweet:
#     #     if item == 'anger':
#     #         anger_tied = True
#     #     elif item == 'disgust':
#     #         disgust_tied = True
#     #     elif item == 'fear':
#     #         fear_tied = True
#     #     elif item == 'joy':
#     #         joy_tied = True
#     #     elif item == 'love':
#     #         love_tied = True
#     #     elif item == 'sadness':
#     #         sadness_tied = True
#     #     elif item == 'surprise':
#     #         surprise_tied = True




# #   ---- tweet, hit count, predicted label, actual label
#     processed_tweet = []
#     processed_tweet.append(tweet)
#     processed_tweet.append(freq_max)
#     processed_tweet.append(label_predicted)
#     processed_tweet.append(labels_legacy[x])

#     all_processed_tweets.append(processed_tweet)

#     is_FN_none = False
#     if ((processed_tweet[2] != 'none') & (processed_tweet[3] == 'none')): # if false negative for 'none'
#         is_FN_none = True

#     is_FN_emotion = False
#     if ((processed_tweet[2] == 'none') & (processed_tweet[3] != 'none')): # if false negative for emotion
#         is_FN_emotion = True

#     for word in tweet:
#         # check word against each dictionary
#         # go through each emotion dictinoary
#         for emotion in emotion_list:
#             snyonym_list = output_list_from_csv(emotion)
#             if word in snyonym_list:
#                 if is_FN_none:
#                     FN_dict_hits_none = FN_dict_hits_none + 1


#     if is_FN_emotion:
#     # add to list of tweets that are FP for emotion based on dict hit
#         emotion = processed_tweet[3]
#         # pdb.set_trace()
#         dict_of_FN_emotion_tweets[emotion].append(processed_tweet)


# # --- meta statistics ---
# # count of total hits for each dictionary
# # co occurence stats

# # statistics for correct predictions for each emotion

# # number tweets with 1 dict hit, 2 seperate dict hits, 3, etc.

# total_tweet_count = 0
# not_calcluated = True; # used to caluclate total number of tweets

# for emotion in emotion_list:
#     emotion_tweet_count = 0
#     print
#     print
#     print emotion

#     for processed_tweet in all_processed_tweets:
#         if not_calcluated: # count total tweets on first loop through tweets
#             total_tweet_count = total_tweet_count + 1

#         if emotion == processed_tweet[3]: # print all tweets that belong to that emotion
#             emotion_tweet_count = emotion_tweet_count + 1

#         # processed_tweet[2] != 'none' & processed_tweet[3] != 'none'
#         if  ((processed_tweet[2]  != 'none' ) & (processed_tweet[3] == emotion)):
#             count_emotional_CORRECT[emotion] = count_emotional_CORRECT[emotion] + 1


#         if combine_joy_love | combine_anger_disgust:
#             if emotion == 'joy': # update count for combined catagory
#                 if processed_tweet[3] == 'love':
#                     emotion_tweet_count = emotion_tweet_count + 1

#                 if ( (processed_tweet[2] == emotion) | (processed_tweet[2] == 'love') ) & ( (processed_tweet[3] == emotion) | (processed_tweet[3] == 'love') ): # if true positive
#                     count_TP[emotion] = count_TP[emotion] + 1


#             # ascribed label is the current emotion and the real label is none
#                 elif ( (processed_tweet[2] == emotion) | (processed_tweet[2] == 'love')  )  & (processed_tweet[3] == 'none') :
#                     count_FP[emotion] = count_FP[emotion] + 1

#             # ascribed label is none and the real label is the current emotion
#                 elif (processed_tweet[2] == 'none') & ( (processed_tweet[3] == emotion) | (processed_tweet[3] == 'love') ):
#                     count_FN[emotion] = count_FN[emotion] + 1
#                     print processed_tweet[2], processed_tweet[3]

#             elif emotion == 'anger':
#                 if processed_tweet[3] == 'disgust': # update count for combied category
#                     emotion_tweet_count = emotion_tweet_count + 1

#                 if (processed_tweet[2] == emotion) | (processed_tweet[2] == 'disgust') & ( (processed_tweet[3] == emotion) | (processed_tweet[3] == 'disgust') ):# if true positive
#                     count_TP[emotion] = count_TP[emotion] + 1

#                 # ascribed label is the current emotion and the real label is none
#                 elif ( (processed_tweet[2] == emotion) | (processed_tweet[2] == 'disgust')  )  & (processed_tweet[3] == 'none') :
#                      count_FP[emotion] = count_FP[emotion] + 1

#                 # ascribed label is none and the real label is the current emotion
#                 elif (processed_tweet[2] == 'none') & ( (processed_tweet[3] == emotion) | (processed_tweet[3] == 'disgust') ):
#                      count_FN[emotion] = count_FN[emotion] + 1

#             else:
#                 if (processed_tweet[2] == processed_tweet[3]) & (processed_tweet[3] == emotion): # if true positive
#                     count_TP[emotion] = count_TP[emotion] + 1

#                 # ascribed label is the current emotion and the real label is none
#                 elif ((processed_tweet[2] == emotion) & (processed_tweet[3] == 'none') ):
#                      count_FP[emotion] = count_FP[emotion] + 1

#                 # ascribed label is none and the real label is the current emotion
#                 elif (processed_tweet[2] == 'none') & (processed_tweet[3] == emotion):
#                      count_FN[emotion] = count_FN[emotion] + 1

#         else:
#             if (processed_tweet[2] == processed_tweet[3]) & (processed_tweet[3] == emotion): # if true positive
#                 count_TP[emotion] = count_TP[emotion] + 1

#             # ascribed label is the current emotion and the real label is none
#             elif ((processed_tweet[2] == emotion) & (processed_tweet[3] == 'none') ):
#                  count_FP[emotion] = count_FP[emotion] + 1

#             # ascribed label is none and the real label is the current emotion
#             elif (processed_tweet[2] == 'none') & (processed_tweet[3] == emotion):
#                  count_FN[emotion] = count_FN[emotion] + 1

#         # for item in processed_tweet:
#         #     if isinstance(item, list):
#         #         for x in item:
#         #             print x,
#         #         print
#         #     else:
#         #         print item,
#         # print
#         # print
#     count_tweets_per_emotion[emotion] = emotion_tweet_count

#     # print total tweets for that emotion
#     print 'Total ', emotion, ' tweets: ', count_tweets_per_emotion[emotion]
#     print 'True Positives from max dict hit method: ', count_TP[emotion]
#     not_calcluated = False

# # print none
# print
# print
# print 'none'
# none_tweet_count = 0
# count_none_TP = 0
# count_none_FP = 0
# count_none_FN = 0

# total_none_dict_hits = 0
# # total_none_dict_tweets = count_none_FN

#     # keep count of total dict hits for each tweet that were actually none, but labelled as emotional [sum this count over all such tweets]
# for processed_tweet in all_processed_tweets:

#         if 'none' == processed_tweet[3]: # print all tweets that belong to that emotion
#             # total_tweet_count = total_tweet_count + 1
#             none_tweet_count = none_tweet_count + 1

#         if (processed_tweet[2] == processed_tweet[3]) & (processed_tweet[3] == 'none'): # if true positive
#             count_none_TP = count_none_TP + 1
#         if ((processed_tweet[2] == 'none') & (processed_tweet[3] != 'none')): # if false positive
#             count_none_FP = count_none_FP + 1
#         if ((processed_tweet[2] != 'none') & (processed_tweet[3] == 'none')): # if false negative
#             count_none_FN = count_none_FN + 1
#             # look at dict count

#         #     if isinstance(item, list):
#         #         for x in item:
#         #             print x,
#         #         print
#         #     else:
#         #         print item,
#         # print
#         # print


# -------------------------------------------

# print 'Total none tweets: ', none_tweet_count
# print 'TP [Correctly labelled as none]: ', count_none_TP
# print 'FP [Ascribed label is none and the real label is emotional]: ', count_none_FP
# print 'FN [Ascribed label is emotional and the real label is none]:', count_none_FN
# print

# # get total dictionary hits for each emotion
# # and average emotional hits per dictionary
# print '----- Macro Data -----'
# print 'average dictionary hits for each emotion using ', total_tweet_count, ' tweets: '
# print '(total hits for each emotion are divided by number of tweets that belong to that emotion)'
# for emotion in emotion_list:
#     print emotion, ' dictionary hits: ', (dictionary_hit_count_TOTAL[emotion] / float(count_tweets_per_emotion[emotion]))


# # (break out: all, TP, FP ) across each emotion
# for emotion in emotion_list:
#     print
#     # print emotion, 'count: ', count_tweets_per_emotion[emotion]
#     print emotion, 'count: ', count_TP[emotion] + count_FN[emotion]
#     print 'Correctly labelled as emotional, when true label was',emotion,': ', count_emotional_CORRECT[emotion]
#     print '[TP] Correctly labeled as',emotion, count_TP[emotion]
#     print '[FP] Ascribed label is ',emotion, 'and the real label is none: ', count_FP[emotion]
#     print '[FN] Ascribed label is none and the real label is',emotion,': ', count_FN[emotion]
#     print 'Precision:', count_TP[emotion] / float(count_TP[emotion] + count_FP[emotion])
#     print 'Recall:', count_TP[emotion] / float(count_TP[emotion] + count_FN[emotion])


# print
# print 'Total none tweets: ', none_tweet_count
# print 'TP [Correctly labelled as none]: ', count_none_TP
# print 'FP [Ascribed label is none and the real label is emotional]: ', count_none_FP
# print 'FN [Ascribed label is emotional and the real label is none]:', count_none_FN

# print
# print 'Tie Counts:'
# print 'two_way_tie_count: ', two_way_tie_count
# print 'three_way_tie_count: ', three_way_tie_count
# print 'four_way_tie_count: ', four_way_tie_count
# print
# print 'anger_disgust_tie_count: ', anger_disgust_tie_count
# print 'anger_fear_tie_count: ', anger_fear_tie_count
# print 'anger_joy_tie_count: ', anger_joy_tie_count
# print 'anger_love_tie_count: ', anger_love_tie_count
# print 'anger_sadness_tie_count: ', anger_sadness_tie_count
# print 'anger_surprise_tie_count: ', anger_surprise_tie_count
# print 'disgust_fear_tie_count: ', disgust_fear_tie_count
# print 'disgust_joy_tie_count: ', disgust_joy_tie_count
# print 'disgust_love_tie_count: ', disgust_love_tie_count
# print 'disgust_sadness_tie_count: ', disgust_sadness_tie_count
# print 'disgust_surprise_tie_count: ', disgust_surprise_tie_count
# print 'fear_joy_tie_count: ', fear_joy_tie_count
# print 'fear_love_tie_count: ', fear_love_tie_count
# print 'fear_sadness_tie_count: ', fear_sadness_tie_count
# print 'fear_surprise_tie_count: ', fear_surprise_tie_count
# print 'joy_love_tie_count: ', joy_love_tie_count
# print 'joy_sadness_tie_count: ', joy_sadness_tie_count
# print 'joy_surprise_tie_count: ', joy_surprise_tie_count
# print 'love_sadness_tie_count: ', love_sadness_tie_count
# print 'love_surprise_tie_count: ', love_surprise_tie_count
# print 'sadness_surprise_tie_count: ', sadness_surprise_tie_count
# print
# print 'Average Dict Hit Freq for False Positives of NONE tweets is:', (FN_dict_hits_none / float(count_none_FN))
# print
# for emotion in emotion_list:

#     print emotion, 'tweets that the dictionary of synoynms should have had a word for [duplication of a tweet indicates multiple occurences of words that should be in emotion dictionary: '
#     print
#     for processed_tweet in dict_of_FN_emotion_tweets[emotion]:
#         for element in processed_tweet[0]:
#             print element,
#         print
#         print
#     print
#     print
#     print


# print 'Tweets for each tie combination: '
# print 'tied_tweets_anger_disgust_list:'
# for tweet in tied_tweets_anger_disgust_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_anger_fear_list:'
# for tweet in tied_tweets_anger_fear_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_anger_joy_list:'
# for tweet in tied_tweets_anger_joy_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_anger_love_list:'
# for tweet in tied_tweets_anger_love_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_anger_sadness_list:'
# for tweet in tied_tweets_anger_sadness_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_anger_surprise_list:'
# for tweet in tied_tweets_anger_surprise_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_disgust_fear_list:'
# for tweet in tied_tweets_disgust_fear_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_disgust_joy_list:'
# for tweet in tied_tweets_disgust_joy_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_disgust_love_list:'
# for tweet in tied_tweets_disgust_love_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_disgust_sadness_list:'
# for tweet in tied_tweets_disgust_sadness_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_disgust_surprise_list:'
# for tweet in tied_tweets_disgust_surprise_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_fear_joy_list:'
# for tweet in tied_tweets_fear_joy_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_fear_love_list:'
# for tweet in tied_tweets_fear_love_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_fear_sadness_list:'
# for tweet in tied_tweets_fear_sadness_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_fear_surprise_list:'
# for tweet in tied_tweets_fear_surprise_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_joy_love_list:'
# for tweet in tied_tweets_joy_love_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_joy_sadness_list:'
# for tweet in tied_tweets_joy_sadness_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_joy_surprise_list:'
# for tweet in tied_tweets_joy_surprise_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_love_sadness_list:'
# for tweet in tied_tweets_love_sadness_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_love_surprise_list:'
# for tweet in tied_tweets_love_surprise_list:
#     for item in tweet:
#         print item,
#     print

# print
# print 'tied_tweets_sadness_surprise_list:'
# for tweet in tied_tweets_sadness_surprise_list:
#     for item in tweet:
#         print item,
#     print
