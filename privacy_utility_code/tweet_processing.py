import re
import numpy as np
from collections import Counter
import pdb

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import statistics
import operator

from itertools import izip

from nltk.stem import WordNetLemmatizer
from functools32 import lru_cache

import emoji

# from __future__ import print_function


# NOTE: Main tweet processing function is clean_toFile()
    # the function InsertEmojiFeatures() is the one that processes emojis


def char_is_emoji(character):
    return character in emoji.UNICODE_EMOJI

def text_has_emoji(text):
    text = text.decode('unicode-escape')
    for character in text:
        if character in emoji.UNICODE_EMOJI:

            # pdb.set_trace()
            return True
    return False



def aquire_handleStatistics ( file_name, num_tweets ):
    # median
    # mean

    infile = open(file_name)

    current_line = 1

    handle_toTweetCount_dict = {}

    for i in range(num_tweets):
        line = infile.next().strip()
        print 'processing line', current_line, 'of', num_tweets
        current_line = current_line + 1

        line = unicode(line,'utf-8')
        line = line.encode('unicode-escape')

        # remove unicode encodings of ascii characters
        line = line.decode('unicode_escape').encode('ascii','ignore')

        handle = line

        if handle in handle_toTweetCount_dict:
            handle_toTweetCount_dict[handle] = handle_toTweetCount_dict[handle] + 1
        else:
            handle_toTweetCount_dict[handle] = 1

    # aquire dict of handle to num tweets, then sort by value, then divide list in three
    handle_TweetCount_sorted = sorted(handle_toTweetCount_dict.items(), key=operator.itemgetter(1), reverse = True)


    # find first instance that has less than 25 tweets
    cutoff_index = 0
    keptValues_list = []
    for i in range ( len ( handle_TweetCount_sorted ) ):
        print handle_TweetCount_sorted[i][1]
        if handle_TweetCount_sorted[i][1] <= 20:
            break
        cutoff_index = cutoff_index + 1
        keptValues_list.append(handle_TweetCount_sorted[i][1])

    handle_TweetCount_sorted = handle_TweetCount_sorted[:cutoff_index]


    oneThird_handleIndex = len(handle_TweetCount_sorted) / 3
    twoThird_handleIndex = ( 2 * len(handle_TweetCount_sorted) ) / 3



    print 'num tweets @ 1/3rd handle: ', handle_TweetCount_sorted[oneThird_handleIndex][1]
    print 'num tweets @ 2/3rd handle: ', handle_TweetCount_sorted[twoThird_handleIndex][1]
    print 'mean tweets per handle: ', statistics.mean ( keptValues_list )
    print 'median tweets per handle: ', statistics.median ( keptValues_list )








def retweet_proportion( file_name, num_tweets ):

    N = num_tweets
    infile = open(file_name)

    retweet_count = 0

    current_line = 1

    for i in range(N):
        line= infile.next().strip()
        print 'processing line', current_line, 'of', N
        current_line = current_line + 1

        line = unicode(line,'utf-8')
        line = line.encode('unicode-escape')

        # remove unicode encodings of ascii characters
        line = line.decode('unicode_escape').encode('ascii','ignore')

        word_tokens = re.sub(r'[^\w\'] ', " ",  line).split() # convert into list of words
        # new code to remove stop words and better tokenization [1/29/18]
        stop_words = set(stopwords.words('english'))
        # word_tokens = word_tokenize(line)


        # make everything lower case
        for i in range(len(word_tokens)):
            word_tokens[i] = word_tokens[i].lower()

        if word_tokens:
            if word_tokens[0] == 'rt':
                retweet_count = retweet_count + 1

    proportion_retweet = retweet_count / float( num_tweets )
    # proportion_original = 1 - proportion_retweet

    print 'proportion retweets: ', proportion_retweet







def get_Labels_Mohammed_Format(file_name):
    file = open(file_name)
    labels = np.array([])

    for line in file:
        line = unicode(line,'utf-8')
        line = line.encode('unicode-escape')

        line = re.sub(r'\\n', "", line)

        line_listForm = []
        line_listForm = re.sub(r'[^\w\'] ', " ",  line).split() # convert into list of words

        labels = np.append( labels,  line_listForm[-1] )

    return labels


def clean(file_name):
    N= 1200 # number of lines to read (starting at top of file)

    infile = open(file_name, 'r')

    cleaned_tweets = np.array([])
    # emojiCount_perTweet = np.array([])

    emojiCount_perTweet = []

    current_line = 1

    file_size = 0
    print 'Calculating size of file to clean...'
    for i in range(N):
        file_size = file_size + 1

    for i in range(N):
        line = infile.next().strip()
        print 'cleaning line', current_line, 'of', file_size
        current_line = current_line + 1
        emojiCount = 0

        emotion_rich = False

        original_line = line

        line = unicode(line,'utf-8')
        line = line.encode('unicode-escape')

        line = re.sub(r'\\r', "", line)
        line = re.sub(r'\\n', "", line)

        # ;) ; ) : ) :) : D :D  :( : ( : / :/

        if   bool (  re.search( r';\)', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r';\)', " winkeSmile_face ", line)

        if   bool (  re.search( r'; \)', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'; \)', " winkeSmile_face ", line)

        if   bool (  re.search( r': \)', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r': \)', " smile_face ", line)



        # if   bool (  re.search( r':\)', line) ):
        #     emotion_rich = True
        #     emojiCount = emojiCount + 1
        # line = re.sub(r':\)', "smile_face", line)

        # if   bool (  re.search( r': \D', line) ):
        #     emotion_rich = True
        #     emojiCount = emojiCount + 1
        # line = re.sub(r': \D', "smile_face", line)

        # if   bool (  re.search( r':\D', line) ):
        #     emotion_rich = True
        #     emojiCount = emojiCount + 1
        # line = re.sub(r':\D', "smile_face", line)

        #:(
        if   bool (  re.search( r':\(', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r':\(', " frown_face ", line)

        #: (
        if   bool (  re.search( r': \(', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r': \(', "frown_face", line)

        # :/
        # if   bool (  re.search( r':\\', line) ):
        #     emotion_rich = True
        #     emojiCount = emojiCount + 1
        # line = re.sub(r':\D', "frown_face", line)

        #: /
        # if   bool (  re.search( r': \\'  , line) ):
        #     emotion_rich = True
        #     emojiCount = emojiCount + 1
        # line = re.sub(r':\D', "frown_face", line)

      # double left quotation mark
        line = re.sub(r'\\u201c', "", line)

###########

        # grimace: \U0001f62c
        if   bool (  re.search( r'\\U0001f62c', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f62c', " grimace_emoji ", line)

        # grimmace_face_emoji    \U0001F601
        if   bool (  re.search( r'\\U0001f601', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f601', " grinn_face_smile_eyes_emoji ", line)
        # open_mouth_smile_emoji  \U0001F603
        if   bool (  re.search( r'\\U0001f603', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f603', " open_mouth_smile_emoji ", line)

        # open_mouth_smile_eyes_emoji
        if   bool (  re.search( r'\\U0001f604', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f604', " open_mouth_smile_emoji ", line)
        # grin_face_emoji
        if   bool (  re.search( r'\\U0001f600', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f600', " smile_face ", line)
        # grin_face_emoji
        if   bool (  re.search( r'\\U0001f602', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f602', " tears_joy ", line)

        if   bool (  re.search( r'\\U0001f605', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f605', " smile_openMouth_coldSweat ", line)

        if   bool (  re.search( r'\\U0001f606', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f606', " smile_face ", line)

        if   bool (  re.search( r'\\U0001f607', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f607', " smile_face ", line)

        if   bool (  re.search( r'\\U0001f609', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f609', " wink_face ", line)

        if   bool (  re.search( r'\\U0001f60a', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f60a', " smile_face ", line)

        if   bool (  re.search( r'\\U0001f642', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f642', " smile_face ", line)

        if   bool (  re.search( r'\\U0001f643', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f643', " upside_down_face ", line)

        if   bool (  re.search( r'\\U263a', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U263a', " smile_face ", line)

        if   bool (  re.search( r'\\Ufe0f', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\Ufe0f', " smile_face ", line)

        if   bool (  re.search( r'\\U0001f60b', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f60b', " delicious_food_savor ", line)

        if   bool (  re.search( r'\\U0001f60c', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f60c', " relieved_face ", line)

        if   bool (  re.search( r'\\U0001f60d', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f60d', " smile_heartShapedEyes ", line)

        if   bool (  re.search( r'\\U000fe32c', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U000fe32c', " kiss_face ", line)

        if   bool (  re.search( r'\\U0001f617', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f617', " kiss_face ", line)

        if   bool (  re.search( r'\\U0001f619', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f619', " face_kiss ", line)

        if   bool (  re.search( r'\\U0001f61a', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f61a', " face_kiss ", line)

        if   bool (  re.search( r'\\U0001f44c', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f44c', " OK_sign ", line)

        if   bool (  re.search( r'\\U0001f44a', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f44a', " fistHand_sign ", line)

        if   bool (  re.search( r'\\U0001f629', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f629', " weary_face ", line)

        if   bool (  re.search( r'\\U0001f63d', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f63d', " kissingCat_closedEyes ", line)

        if   bool (  re.search( r'\\U0001f614', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f614', " pensive_face ", line)

        if   bool (  re.search( r'\\U0001f64c', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f64c', " both_hands_raised_celebration ", line)

        if   bool (  re.search( r'\\U0001fe44d', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001fe44d', " thumbs_up_sign ", line)

        if   bool (  re.search( r'\\U0001f44c', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f44c', " ok_hand_sign ", line)

        if   bool (  re.search( r'\\U0001f44c\\U0001f3fc', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f44c\\U0001f3fc', " ok_hand_sign ", line)

        if   bool (  re.search( r'\\U0001f44c', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f44c', " ok_hand_sign ", line)

        if   bool (  re.search( r'\\U0001f3fc', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\U0001f3fc', " ok_hand_sign ", line)

        if   bool (  re.search( r'\\U0001f44c\\U0001f3fb', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f44c\\U0001f3fb', " ok_hand_sign ", line)

        if   bool (  re.search( r'\\U0001f3fb', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        ine = re.sub(r'\\U0001f3fb', " ok_hand_sign ", line)

        if   bool (  re.search( r'\\U0001f44c', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        ine = re.sub(r'\\U0001f44c', " ok_hand_sign ", line)

        if   bool (  re.search( r'\\U0001f44c\\U0001f3ff', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f44c\\U0001f3ff', " ok_hand_sign ", line)

        if   bool (  re.search( r'\\U0001f609', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f609', " wink_face ", line)

        if   bool (  re.search( r'\\U0001f637', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f637', " face_medical_mask ", line)

        if   bool (  re.search( r'\\U0001f621', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f621', " angry_pout_face ", line)

        if   bool (  re.search( r'\\U0001F624', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001F624', " triumpLook_face ", line)

        if   bool (  re.search( r'\\U0001f622', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f622', " cry_face ", line)

        # if   bool (  re.search( r'\\u270c\ufe0f', line) ):
        #     emotion_rich = True
        #     emojiCount = emojiCount + 1
        # line = re.sub(r'\\u270c\ufe0f', " victory_hand ", line)

############

         # \U0001f914 thinking_face

        if bool (  re.search( r'\\U0001f914', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f914', " thinking_face ", line)

         # \U0001f61c wink_tongue_out_face

        if bool (  re.search( r'\\U0001f61c', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f61c', " wink_tongue_out_face ", line)

         # \U0001f917 warm_grin_hands_out_face

        if bool (  re.search( r'\\U0001f917', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f917', " warm_grin_hands_out_face ", line)

         # \U0001f644 big_eyes_looking_up_face
        if bool (  re.search( r'\\U0001f644', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f644', " big_eyes_looking_up_face ", line)

         # \U0001f622 sad_tear_face
        if bool (  re.search( r'\\U0001f622', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f622', " sad_tear_face ", line)

         # \U0001f61b tongue_out_face
        if bool (  re.search( r'\\U0001f61b', line) ):
            # emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f61b', " tongue_out_face ", line)

         # \U0001f633 blush_face
        if bool (  re.search( r'\\U0001f633', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f633', " blush_face ", line)

         # \U0001f631 surprise_face
        if bool (  re.search( r'\\U0001f631', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f631', " fear_scream_face ", line)

        # fear
        # U0001F628 fearful_face
        if bool (  re.search( r'\\U0001f628', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f628', " fearful_face ", line)

       # U0001F630 openMouth_coldSweat_face
        if bool (  re.search( r'\\U0001f630', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f630', " openMouth_coldSweat_face ", line)

       # U0001F627 anguish_face
        if bool (  re.search( r'\\U0001f627', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f627', " anguish_face ", line)

        #U0001F61F worried_face
        if bool (  re.search( r'\\U0001f61f', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f61f', " worried_face ", line)
        # end fear

        # anger_disgust
        #U0001F611 expressionless_face
        if bool (  re.search( r'\\U0001f611', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f611', " expressionless_face ", line)

        #U0001F620 angry_face
        if bool (  re.search( r'\\U0001f620', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f620', " angry_face ", line)

        #U0001F63E grumpy_cat_face
        if bool (  re.search( r'\\U0001F63E', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001F63E', " grumpy_cat_face ", line)

        #U0001F595 middle_finger
        if bool (  re.search( r'\\U0001f595', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f595', " middle_finger ", line)
        # end anger_disgust

        # sadness
        #U0001F62B weary_face
        if bool (  re.search( r'\\U0001f62B', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f62B', " weary_face ", line)

        #U0001F629 weary_face
        if bool (  re.search( r'\\U0001f629', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f629', " weary_face ", line)

        #U0001F613 cold_sweat_face
        if bool (  re.search( r'\\U0001f613', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f613', " cold_sweat_face ", line)

        #U00002639 frown_face
        if bool (  re.search( r'\\U00002639', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U00002639', " frown_face ", line)

        #U0001F641 slight_frown_face
        if bool (  re.search( r'\\U0001f641', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f641', " slight_frown_face ", line)

        #U0001F622 cry_face
        if bool (  re.search( r'\\U0001f622', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f622', " cry_face ", line)

        #U0001F62D loud_cry_face
        if bool (  re.search( r'\\U0001f62d', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f62d', " loud_cry_face ", line)

        #U0001F630 open_mouth_cold_sweat_face
        if bool (  re.search( r'\\U0001f630', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f630', " open_mouth_cold_sweat_face ", line)

        #U0001F912 thermometer_face
        if bool (  re.search( r'\\U0001f912', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f912', " thermometer_face ", line)

        #U0001F915 head_bandage_face
        if bool (  re.search( r'\\U0001f915', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f915', " head_bandage_face ", line)

        #U0001F63F cry_cat_face
        if bool (  re.search( r'\\U0001f63f', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f63f', " cry_cat_face ", line)
        #end sadness

        # joy_love
        #U0001F494 broken_heart
        if bool (  re.search( r'\\U0001f494', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f494', " broken_heart ", line)

        #U0001F63B smile_heartEyes_catFace
        if bool (  re.search( r'\\U0001f63b', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f63b', " smile_heartEyes_catFace ", line)

        #U0001F46B couple_holdHands
        if bool (  re.search( r'\\U0001f46b', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f46b', " couple_holdHands ", line)

        #U0001F46C couple_holdHands
        if bool (  re.search( r'\\U0001f46c', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f46c', " couple_holdHands ", line)

        #U0001F46D couple_holdHands
        if bool (  re.search( r'\\U0001f46d', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f46d', " couple_holdHands ", line)

        #U0001F48F kiss_emoji
        if bool (  re.search( r'\\U0001f48f', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f48f', " kiss_emoji ", line)

        #U0001F491 couple_withHeart_emoji
        if bool (  re.search( r'\\U0001f491', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f491', " couple_withHeart_emoji ", line)

        #U0001F498 heart_with_arrow_emoji
        if bool (  re.search( r'\\U0001f498', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f498', " heart_with_arrow_emoji ", line)

        #U0001F493 beating_heart
        if bool (  re.search( r'\\U0001f493', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f493', " beating_heart ", line)

        #U0001F495 two_hearts
        if bool (  re.search( r'\\U0001f495', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f495', " two_hearts ", line)

        #U0001F496 sparkle_heart
        if bool (  re.search( r'\\U0001f496', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f496', " sparkle_heart ", line)

        #U0001F497 growing_heart
        if bool (  re.search( r'\\U0001f497', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f497', " growing_heart ", line)

        #U0001F49D ribbon_heart
        if bool (  re.search( r'\\U0001f49d', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f49d', " ribbon_heart ", line)

        #U0001F49E revolving_hearts
        if bool (  re.search( r'\\U0001f49e', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f49e', " revolving_hearts ", line)

        #U0001F48C love_letter
        if bool (  re.search( r'\\U0001f48c', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f48c', " love_letter ", line)

        #U0001F61D stuckOutTongue_closedEyes_face
        if bool (  re.search( r'\\U0001f61d', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f61d', " stuckOutTongue_closedEyes_face ", line)

        #U0001F638 grin_smileEyes_catFace
        if bool (  re.search( r'\\U0001f638', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f638', " grin_smileEyes_catFace ", line)

        #U0001F639 tearsJoy_catFace
        if bool (  re.search( r'\\U0001f633', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f639', " tearsJoy_catFace ", line)

        #U0001F618 blowKiss_face
        if bool (  re.search( r'\\U0001f618', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f618', " blowKiss_face ", line)

        #U0001F389 party_popper
        if bool (  re.search( r'\\U0001f389', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f389', " party_popper ", line)

        #U0001F499 blue_heart
        if bool (  re.search( r'\\U0001f499', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f499', " heart_emoji ", line)

        #U0001F49C purple_heart
        if bool (  re.search( r'\\U0001f49c', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f49c', " heart_emoji ", line)

        #U0001F49A green_heart
        if bool (  re.search( r'\\U0001f49a', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f49a', " heart_emoji ", line)

        # end love_joy

        # surprise
        #U0001F62E openMouth_face
        if bool (  re.search( r'\\U0001f62e', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f62e', " openMouth_face ", line)

        #U0001F62F surprise_face
        if bool (  re.search( r'\\U0001f62f', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f62f', " surprise_face ", line)

        #U0001F627 anguish_face
        if bool (  re.search( r'\\U0001f627', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f627', " anguish_face ", line)

        #U0001F626 frown_openMouth_face
        if bool (  re.search( r'\\U0001f626', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f626', " frown_openMouth_face ", line)

        #U0001F631 scream_fear_face
        if bool (  re.search( r'\\U0001f631', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f631', " scream_fear_face ", line)

        #U0001F640 surprise_catFace
        if bool (  re.search( r'\\U0001f640', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f640', " surprise_catFace ", line)

        # end surprise



##############
        # remove links
        line = re.sub(r'^https?:\/\/.*[\r\n]*', '', line)
        line = re.sub(r'[a-z]*[:.]+\S+', '', line)

        #remove handles
        line = re.sub(r'@([A-Za-z0-9_]+)', '', line)


        # capitalization
        num_cap = sum(1 for c in line if c.isupper())
        num_low = sum(1 for c in line if c.islower())

        if not (num_low == 0 and num_cap == 0):
            if ( num_cap / ( num_cap + num_low )  > .80 ):  # if proportion of caps is greater than 80%
                # then add cap feature
                line = line + " high_cap"

        # exclemation points

# DISABLED FOR SERVER COMPATABILITY
        c = Counter(line)
        frequncies = reversed(c.most_common())

        for aTuple in frequncies:
            if aTuple[0] == '!':
                if int(aTuple[1]) >= 3:
                    emotion_rich = True
                    line = line + " extreme_exclemation "
                elif int(aTuple[1]) >= 1:
                    emotion_rich = True
                    line = line + " exclemation "

        line_listForm = []
        word_tokens = re.sub(r'[^\w\'] ', " ",  line).split() # convert into list of words
        # new code to remove stop words and better tokenization [1/29/18]
        stop_words = set(stopwords.words('english'))
        # word_tokens = word_tokenize(line)

                # make everything lower case
        for i in range(len(word_tokens)):
            word_tokens[i] = word_tokens[i].lower()

        # print 'before: ', word_tokens
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        # print 'after: ', filtered_sentence

        line_listForm = filtered_sentence

        words_toAppend = []
        for word in line_listForm: # double word if all caps
            if  word.isupper() and len(word) > 1:
                emotion_rich = True
                words_toAppend.append(word)

        for item in words_toAppend:
            line_listForm.append(item)

        # double hash-tagged words (remove hashtag)
        hashed = [ word for word in line_listForm if word.startswith("#") ]
        for word in hashed:
                line_listForm.append( word[1:] )

        indices_to_delete = []
        # remove hashtag version
        for i in range(len(line_listForm)):
            if line_listForm[i].startswith("#"):
                indices_to_delete.append(i)

        for index in reversed(indices_to_delete):
            del line_listForm[index]

        # double emotion key words and their synonyms if they exist
        emotions_file = open("emotions_synonyms.txt")
        lines = emotions_file.readlines()
            # build dictionary of emotions and synonmys from file

        emotions_list = dict()
        for synonym_line in lines:
            synonym_line = synonym_line.rstrip("\n")
            words = synonym_line.split(",")
            emotion_curr = words[0]
            synonyms = words[1:]
            emotions_list[emotion_curr] = synonyms
        emotions_file.close()

        for word in line_listForm:
            words_toAppend = []
            if word == "anger":
                words_toAppend.append(word)
                emotion_rich = True
            elif word == "disgust":
                words_toAppend.append(word)
                emotion_rich = True
            elif word == "fear":
                words_toAppend.append(word)
                emotion_rich = True
            elif word == "joy":
                words_toAppend.append(word)
                emotion_rich = True
            elif word == "love":
                words_toAppend.append(word)
                emotion_rich = True
            elif word == "sadness":
                words_toAppend.append(word)
                emotion_rich = True
            elif word == "surprise":
                words_toAppend.append(word)
                emotion_rich = True
            elif word in emotions_list:
                emotion_rich = True
                words_toAppend.append(word)

        for word in words_toAppend:
            line_listForm.append(word)

        if emotion_rich:
            line_listForm.append("emotion_rich")

        line = ""

        for word in line_listForm:
            line = line + " " + word

        line = line[1:] # remove empty space at begining

            # line = line + " " + "emotion_rich"

        emojiCount_perTweet.append(emojiCount)

        cleaned_tweets = np.append(cleaned_tweets,  line)

        # if emojiCount > 0:
        #     print original_line
        #     print line

    file.close()

    outfile = open("clean_test.txt", 'w')


    for line in cleaned_tweets:
        outfile.write(line + "\n" )

    outfile.close()



    return cleaned_tweets, emojiCount_perTweet




def clean_Mohammed_Format(file_name):

    file = open(file_name)

    cleaned_tweets = np.array([])

    for line_fresh in file:

        emotion_rich = False

        line_fresh = unicode(line_fresh,'utf-8')
        line_fresh = line_fresh.encode('unicode-escape')

        # ONLY FOR Mohammed data set
        line_fresh = re.sub(r'[\d]+:', "", line_fresh )
        line_fresh = re.sub(r'\\t', " ", line_fresh)
        line_fresh = re.sub(r'::', "", line_fresh)
        line_fresh = re.sub(r':', "", line_fresh)

        line_listForm = []
        line_listForm = re.sub(r'[^\w\'] ', " ",  line_fresh).split() # convert into list of words

        line_listForm = line_listForm[:-1]

        line = ""
        for word in line_listForm:
            line = line + " " + word

        line = line[1:] # remove empty space at begining
        # END only for Mohammed data set

        line = re.sub(r'\\U0001f62c', "grimace_emoji", line)

        line = re.sub(r'\\r', "", line)
        line = re.sub(r'\\n', "", line)
        line = re.sub(r'\n', "", line)
        line = re.sub(r'\r', "", line)

        # ;) ; ) : ) :) : D :D  :( : ( : / :/

        if   bool (  re.search( r';\)', line) ):
            emotion_rich = True
        line = re.sub(r';\)', "winkeSmile_face", line)

        if   bool (  re.search( r'; \)', line) ):
            emotion_rich = True
        line = re.sub(r'; \)', "winkeSmile_face", line)

        if   bool (  re.search( r': \)', line) ):
            emotion_rich = True
        line = re.sub(r': \)', "smile_face", line)

        if   bool (  re.search( r':\)', line) ):
            emotion_rich = True
        line = re.sub(r':\)', "smile_face", line)

        if   bool (  re.search( r': \D', line) ):
            emotion_rich = True
        line = re.sub(r': \D', "smile_face", line)

        if   bool (  re.search( r':\D', line) ):
            emotion_rich = True
        line = re.sub(r':\D', "smile_face", line)

        #:(
        if   bool (  re.search( r':\(', line) ):
            emotion_rich = True
        line = re.sub(r':\(', "frown_face", line)

        #: (
        if   bool (  re.search( r': \(', line) ):
            emotion_rich = True
        line = re.sub(r': \(', "frown_face", line)

        # :/
        if   bool (  re.search( r':\\', line) ):
            emotion_rich = True
        line = re.sub(r':\D', "frown_face", line)

        #: /
        if   bool (  re.search( r': \\'  , line) ):
            emotion_rich = True
        line = re.sub(r':\D', "frown_face", line)

      # double left quotation mark
        line = re.sub(r'\\u201c', "", line)


        # grimace: \U0001f62c
        if   bool (  re.search( r'\\U0001f62c', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f62c', "grimace_emoji", line)

        # grimmace_face_emoji    \U0001F601
        if   bool (  re.search( r'\\U0001f601', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f601', "grimmace_face_smile_eyes_emoji", line)
        # open_mouth_smile_emoji  \U0001F603
        if   bool (  re.search( r'\\U0001f603', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f603', "open_mouth_smile_emoji", line)

        # open_mouth_smile_eyes_emoji
        if   bool (  re.search( r'\\U0001f603', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f604', "open_mouth_smile_emoji", line)
        # grin_face_emoji
        if   bool (  re.search( r'\\U0001f600', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f600', "smile_face", line)
        # grin_face_emoji
        if   bool (  re.search( r'\\U0001f602', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f602', "tears_joy", line)

        if   bool (  re.search( r'\\U0001f605', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f605', "smile_openMouth_coldSweat", line)

        if   bool (  re.search( r'\\U0001f606', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f606', "smile_face", line)

        if   bool (  re.search( r'\\U0001f607', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f607', "smile_face", line)

        if   bool (  re.search( r'\\U0001f609', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f609', "wink_face", line)

        if   bool (  re.search( r'\\U0001f60a', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f60a', "smile_face", line)

        if   bool (  re.search( r'\\U0001f642', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f642', "smile_face", line)

        if   bool (  re.search( r'\\U0001f643', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f643', "upside_down_face", line)

        if   bool (  re.search( r'\\U263a', line) ):
            emotion_rich = True
        line = re.sub(r'\\U263a', "smile_face", line)

        if   bool (  re.search( r'\\Ufe0f', line) ):
            emotion_rich = True
        line = re.sub(r'\\Ufe0f', "smile_face", line)

        if   bool (  re.search( r'\\U0001f60b', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f60b', "delicious_food_savor", line)

        if   bool (  re.search( r'\\U0001f60c', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f60c', "relieved_face", line)

        if   bool (  re.search( r'\\U0001f60d', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f60d', "smile_heartShapedEyes", line)

        if   bool (  re.search( r'\\U000fe32c', line) ):
            emotion_rich = True
        line = re.sub(r'\\U000fe32c', "kiss_face", line)

        if   bool (  re.search( r'\\U0001f617', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f617', "kiss_face", line)

        if   bool (  re.search( r'\\U0001f619', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f619', "face_kiss", line)

        if   bool (  re.search( r'\\U0001f61a', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f61a', "face_kiss", line)

        if   bool (  re.search( r'\\U0001f44c', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f44c', "OK_sign", line)

        if   bool (  re.search( r'\\U0001f44a', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f44a', "fistHand_sign", line)

        if   bool (  re.search( r'\\U0001f629', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f629', "weary_face", line)

        if   bool (  re.search( r'\\U0001f63d', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f63d', "kissingCat_closedEyes", line)

        if   bool (  re.search( r'\\U0001f614', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f614', "pensive_face", line)

        if   bool (  re.search( r'\\U0001f64c', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f64c', "both_hands_raised_celebration", line)

        if   bool (  re.search( r'\\U0001fe44d', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001fe44d', "thumbs_up_sign", line)

        if   bool (  re.search( r'\\U0001f44c', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f44c', "ok_hand_sign", line)

        if   bool (  re.search( r'\\U0001f44c\\U0001f3fc', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f44c\\U0001f3fc', "ok_hand_sign", line)

        if   bool (  re.search( r'\\U0001f44c', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f44c', "ok_hand_sign", line)

        if   bool (  re.search( r'\\U0001f3fc', line) ):
            emotion_rich = True
        line = re.sub(r'\U0001f3fc', "ok_hand_sign", line)

        if   bool (  re.search( r'\\U0001f44c\\U0001f3fb', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f44c\\U0001f3fb', "ok_hand_sign", line)

        if   bool (  re.search( r'\\U0001f3fb', line) ):
            emotion_rich = True
        ine = re.sub(r'\\U0001f3fb', "ok_hand_sign", line)

        if   bool (  re.search( r'\\U0001f44c', line) ):
            emotion_rich = True
        ine = re.sub(r'\\U0001f44c', "ok_hand_sign", line)

        if   bool (  re.search( r'\\U0001f44c\\U0001f3ff', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f44c\\U0001f3ff', "ok_hand_sign_black", line)

        if   bool (  re.search( r'\\U0001f609', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f609', "wink_face", line)

        if   bool (  re.search( r'\\U0001f637', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f637', "face_medical_mask", line)

        if   bool (  re.search( r'\\U0001f621', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f621', "angry_pout_face", line)

        if   bool (  re.search( r'\\U0001F624', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001F624', "steam_from_nose_face", line)

        if   bool (  re.search( r'\\U0001f622', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f622', "cry_face", line)

        if   bool (  re.search( r'\\U0001f622', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f622', "cry_face", line)


        if   bool (  re.search( r'\\u270c\ufe0f', line) ):
            emotion_rich = True
        line = re.sub(r'\\u270c\ufe0f', "victory_hand", line)

        if   bool (  re.search( r'\\U0001f629', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001f629', "weary_face", line)

        if   bool (  re.search( r'\\U0001F63D', line) ):
            emotion_rich = True
        line = re.sub(r'\\U0001F63D', "kiss_face", line)

        # remove links
        line = re.sub(r'^https?:\/\/.*[\r\n]*', '', line)

        #remove handles
        line = re.sub(r'@([A-Za-z0-9_]+)', '', line)


        # capitalization

        num_cap = sum(1 for c in line if c.isupper())
        num_low = sum(1 for c in line if c.islower())

        if not (num_low == 0 and num_cap == 0):
            if ( num_cap / ( num_cap + num_low )  > .80 ):  # if proportion of caps is greater than 80%
                # then add cap feature
                line = line + " high_cap"


        # exclemation points

        # REMOVED FOR SERVER COMPATABILITY
        c = Counter(line)
        frequncies = reversed(c.most_common())

        for aTuple in frequncies:
            if aTuple[0] == '!':
                if int(aTuple[1]) >= 3:
                    emotion_rich = True
                    line = line + " extreme_exclemation"
                elif int(aTuple[1]) >= 1:
                    emotion_rich = True
                    line = line + " exclemation"

        line_listForm = []
        # line_listForm = re.sub(r'[^\w\'] ', " ",  line).split() # convert into list of words

        # new code to remove stop words and better tokenization [1/29/18]
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(line)

        # print 'before: ', word_tokens
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        # print 'after: ', filtered_sentence

        line_listForm = filtered_sentence

        words_toAppend = []
        for word in line_listForm: # double word if all caps
            if  word.isupper() and len(word) > 1:
                emotion_rich = True
                words_toAppend.append(word)

        for item in words_toAppend:
            line_listForm.append(item)

        # double hash-tagged words (remove hashtag)
        hashed = [ word for word in line_listForm if word.startswith("#") ]
        for word in hashed:
                line_listForm.append( word[1:] )
                line_listForm.append( word[1:] )

        indices_to_delete = []
        # remove hashtag version
        for i in range(len(line_listForm)):
            if line_listForm[i].startswith("#"):
                indices_to_delete.append(i)

        for index in reversed(indices_to_delete):
            del line_listForm[index]


        # make everything lower case
        for i in range(len(line_listForm)):
            line_listForm[i] = line_listForm[i].lower()

        # double emotion key words and their synonyms if they exist
        emotions_file = open("emotions_synonyms.txt")
        lines = emotions_file.readlines()
            # build dictionary of emotions and synonmys from file

        emotions_list = dict()
        for synonym_line in lines:
            synonym_line = synonym_line.rstrip("\n")
            words = synonym_line.split(",")
            emotion_curr = words[0]
            synonyms = words[1:]
            emotions_list[emotion_curr] = synonyms
        emotions_file.close()

        for word in line_listForm:
            words_toAppend = []
            if word == "anger":
                words_toAppend.append(word)
                emotion_rich = True
            elif word == "disgust":
                words_toAppend.append(word)
                emotion_rich = True
            elif word == "fear":
                words_toAppend.append(word)
                emotion_rich = True
            elif word == "joy":
                words_toAppend.append(word)
                emotion_rich = True
            elif word == "love":
                words_toAppend.append(word)
                emotion_rich = True
            elif word == "sadness":
                words_toAppend.append(word)
                emotion_rich = True
            elif word == "surprise":
                words_toAppend.append(word)
                emotion_rich = True
            elif word in emotions_list:
                emotion_rich = True
                words_toAppend.append(word)

        for word in words_toAppend:
            line_listForm.append(word)

        if emotion_rich:
            line_listForm.append("emotion_rich")

        line = ""

        for word in line_listForm:
            line = line + " " + word

        line = line[1:] # remove empty space at begining

            # line = line + " " + "emotion_rich"

        cleaned_tweets = np.append(cleaned_tweets,  line)

    file.close()

    outfile = open("clean_test.txt", 'w')

    for line in cleaned_tweets:
        outfile.write(line + "\n" )

    outfile.close()

    return cleaned_tweets



def get_handles(tweets_filename, handles_filename, num_tweets):

    N = num_tweets

    outfile_handles_original = open('handlesoriginal' + str(N) + '.txt', 'w')
    outfile_handles_retweet = open('handlesretweet' + str(N) + '.txt', 'w')

    handles_original = []
    handles_retweet = []

    with open(tweets_filename) as file_tweets, open(handles_filename) as file_handles:

        line_count = 0
        for tweet, handle in izip(file_tweets, file_handles):
            print 'processing line: ', line_count
            line_count = line_count + 1


            line = unicode(tweet,'utf-8')
            line = line.encode('unicode-escape')

            # remove unicode encodings of ascii characters
            line = line.decode('unicode_escape').encode('ascii','ignore')

            # remove commas
            # line = re.sub(r'\,', '', line)
            # line = re.sub(r'\(', '', line)
            # line = re.sub(r'\)', '', line)
            # line = re.sub(r'\"', '', line)
            # line = re.sub(r'\\r', "", line)
            # line = re.sub(r'\\n', "", line)

            handle = handle.rstrip()

            # assign value to boolean based on whether there is a match to 'RT @handle'
            if   bool (  re.search( r'RT @', line) ) or bool ( re.search( r'RT', line) ):
                # print 'retweet detected'
                handles_retweet.append(handle)
            else:
                handles_original.append(handle)

            if line_count > num_tweets:
                break


    for handle in handles_original:
        outfile_handles_original.write(handle + "\n" )

    for handle in handles_retweet:
        outfile_handles_retweet.write(handle + "\n" )

    outfile_handles_original.close()
    outfile_handles_retweet.close()



# returns modified line, emojiCount, emotion_rich
def InsertEmojiFeatures(line):

    emotion_rich = False
    emojiCount = 0


#     if   bool (  re.search( r'\\U0001f605', line) ):
#         emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f605', " smile_openMouth_coldSweat ", line)

#     if   bool (  re.search( r'\\U0001f60b', line) ):
#         emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f60b', " delicious_food_savor ", line)

#     if   bool (  re.search( r'\\U0001f60c', line) ):
#         emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f60c', " relieved_face ", line)

#     if   bool (  re.search( r'\\U0001f44c', line) ):
#         # emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f44c', " OK_sign ", line)

#     if   bool (  re.search( r'\\U0001f44a', line) ):
#         # emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f44a', " fistHand_sign ", line)

#     if   bool (  re.search( r'\\U0001f629', line) ):
#         emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f629', " weary_face ", line)

#     if   bool (  re.search( r'\\U0001f63d', line) ):
#         emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f63d', " kissingCat_closedEyes ", line)

#     if   bool (  re.search( r'\\U0001f614', line) ):
#         emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f614', " pensive_face ", line)

#     if   bool (  re.search( r'\\U0001f64c', line) ):
#         emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f64c', " both_hands_raised_celebration ", line)

#     if   bool (  re.search( r'\\U0001fe44d', line) ):
#         # emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001fe44d', " thumbs_up_sign ", line)

#     if   bool (  re.search( r'\\U0001f44c', line) ):
#         # emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f44c', " ok_hand_sign ", line)

#     if   bool (  re.search( r'\\U0001f44c\\U0001f3fc', line) ):
#         # emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f44c\\U0001f3fc', " ok_hand_sign ", line)

#     if   bool (  re.search( r'\\U0001f44c', line) ):
#         # emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f44c', " ok_hand_sign ", line)

#     if   bool (  re.search( r'\\U0001f3fc', line) ):
#         # emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\U0001f3fc', " ok_hand_sign ", line)

#     if   bool (  re.search( r'\\U0001f44c\\U0001f3fb', line) ):
#         # emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f44c\\U0001f3fb', " ok_hand_sign ", line)

#     if   bool (  re.search( r'\\U0001f3fb', line) ):
#         # emotion_rich = True
#         emojiCount = emojiCount + 1
#     ine = re.sub(r'\\U0001f3fb', " ok_hand_sign ", line)

#     if   bool (  re.search( r'\\U0001f44c', line) ):
#         # emotion_rich = True
#         emojiCount = emojiCount + 1
#     ine = re.sub(r'\\U0001f44c', " ok_hand_sign ", line)

#     if   bool (  re.search( r'\\U0001f44c\\U0001f3ff', line) ):
#         # emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f44c\\U0001f3ff', " ok_hand_sign ", line)

#     if   bool (  re.search( r'\\U0001f609', line) ):
#         emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f609', " wink_face ", line)

#     if   bool (  re.search( r'\\U0001f637', line) ):
#         emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f637', " face_medical_mask ", line)





#     # if   bool (  re.search( r'\\u270c\ufe0f', line) ):
#     #     emotion_rich = True
#     #     emojiCount = emojiCount + 1
#     # line = re.sub(r'\\u270c\ufe0f', " victory_hand ", line)

# ############

#      # \U0001f914 thinking_face

#     if bool (  re.search( r'\\U0001f914', line) ):
#         # emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f914', " thinking_face ", line)

#      # \U0001f61c wink_tongue_out_face

#     if bool (  re.search( r'\\U0001f61c', line) ):
#         # emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f61c', " wink_tongue_out_face ", line)

#      # \U0001f917 warm_grin_hands_out_face

#     if bool (  re.search( r'\\U0001f917', line) ):
#         emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f917', " warm_grin_hands_out_face ", line)

#      # \U0001f644 big_eyes_looking_up_face
#     if bool (  re.search( r'\\U0001f644', line) ):
#         # emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f644', " big_eyes_looking_up_face ", line)

#      # \U0001f61b tongue_out_face
#     if bool (  re.search( r'\\U0001f61b', line) ):
#         # emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f61b', " tongue_out_face ", line)

#      # \U0001f633 blush_face
#     if bool (  re.search( r'\\U0001f633', line) ):
#         emotion_rich = True
#         emojiCount = emojiCount + 1
#     line = re.sub(r'\\U0001f633', " blush_face ", line)


    # fear

        # grimace: \U0001f62c
    if   bool (  re.search( r'\\U0001f62c', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f62c', " grimace_emoji ", line)

    # \U0001f631 surprise_face
    if bool (  re.search( r'\\U0001f631', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f631', " fear_scream_face ", line)

    # U0001F628 fearful_face
    if bool (  re.search( r'\\U0001f628', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f628', " fearful_face ", line)

   # U0001F630 openMouth_coldSweat_face
    if bool (  re.search( r'\\U0001f630', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f630', " openMouth_coldSweat_face ", line)

   # U0001F627 anguish_face
    if bool (  re.search( r'\\U0001f627', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f627', " anguish_face ", line)

    #U0001F61F worried_face
    if bool (  re.search( r'\\U0001f61f', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f61f', " worried_face ", line)
    # end fear



    # anger_disgust
    if bool (  re.search( r'\\U0001f624', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f624', " triumpLook_face ", line)

    #U0001F611 expressionless_face
    if bool (  re.search( r'\\U0001f611', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f611', " expressionless_face ", line)

    #U0001F620 angry_face
    if bool (  re.search( r'\\U0001f620', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f620', " angry_face ", line)

    #U0001F63E grumpy_cat_face
    if bool (  re.search( r'\\U0001f63e', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U000163e', " grumpy_cat_face ", line)

    #U0001F595 middle_finger
    if bool (  re.search( r'\\U0001f595', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f595', " middle_finger ", line)

    if   bool (  re.search( r'\\U0001f621', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f621', " angry_pout_face ", line)
    # end anger_disgust



    # sadness
    if   bool (  re.search( r'\\U0001f622', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f622', " cry_face ", line)

         # \U0001f622 sad_tear_face
    if bool (  re.search( r'\\U0001f622', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f622', " sad_tear_face ", line)

    # sad face
    if bool (  re.search( r'\\u2639\\ufe0f', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\u2639\\ufe0f', " sad_face ", line)

    #U0001F62B weary_face
    if bool (  re.search( r'\\U0001f62b', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f62b', " weary_face ", line)

    #U0001F629 weary_face
    if bool (  re.search( r'\\U0001f629', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f629', " weary_face ", line)

    #U0001F613 cold_sweat_face
    if bool (  re.search( r'\\U0001f613', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f613', " cold_sweat_face ", line)

    #U00002639 frown_face
    if bool (  re.search( r'\\U00002639', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U00002639', " frown_face ", line)

    #U0001F641 slight_frown_face
    if bool (  re.search( r'\\U0001f641', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f641', " slight_frown_face ", line)

    #U0001F622 cry_face
    if bool (  re.search( r'\\U0001f622', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f622', " cry_face ", line)

    #U0001F62D loud_cry_face
    if bool (  re.search( r'\\U0001f62d', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f62d', " loud_cry_face ", line)

    #U0001F630 open_mouth_cold_sweat_face
    if bool (  re.search( r'\\U0001f630', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f630', " open_mouth_cold_sweat_face ", line)

    #U0001F912 thermometer_face
    if bool (  re.search( r'\\U0001f912', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f912', " thermometer_face ", line)

    #U0001F915 head_bandage_face
    if bool (  re.search( r'\\U0001f915', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f915', " head_bandage_face ", line)

    #U0001F63F cry_cat_face
    if bool (  re.search( r'\\U0001f63f', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f63f', " cry_cat_face ", line)
    #end sadness



    # joy_love

        # grin_face_emoji
    if   bool (  re.search( r'\\U0001f602', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f602', " tears_joy ", line)

        # grinn_face_emoji    \U0001F601
    if   bool (  re.search( r'\\U0001f601', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f601', " smile_face ", line)
    # open_mouth_smile_emoji  \U0001F603

    if   bool (  re.search( r'\\U0001f603', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f603', " smile_face ", line)

    # open_mouth_smile_eyes_emoji
    if   bool (  re.search( r'\\U0001f604', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f604', " smile_face ", line)

    # grin_face_emoji
    if   bool (  re.search( r'\\U0001f600', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f600', " smile_face ", line)

    if   bool (  re.search( r'\\U0001f606', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f606', " smile_face ", line)

    if   bool (  re.search( r'\\U0001f607', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f607', " smile_face ", line)

    if   bool (  re.search( r'\\U0001f60a', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f60a', " smile_face ", line)

    if   bool (  re.search( r'\\U0001f642', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f642', " smile_face ", line)

    if   bool (  re.search( r'\\U263a', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U263a', " smile_face ", line)

    if   bool (  re.search( r'\\Ufe0f', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\Ufe0f', " smile_face ", line)

    if   bool (  re.search( r'\\U0001f60d', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f60d', " smile_heartShapedEyes ", line)

    if   bool (  re.search( r'\\U000fe32c', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U000fe32c', " kiss_face ", line)

    if   bool (  re.search( r'\\U0001f617', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f617', " kiss_face ", line)

    if   bool (  re.search( r'\\U0001f619', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f619', " kiss_face ", line)

    if   bool (  re.search( r'\\U0001f61a', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f61a', " kiss_face ", line)

    # happy face
    if bool (  re.search( r'\\u263a\\ufe0f', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\u263a\\ufe0f', " happy_face ", line)

        # red heart
    if bool (  re.search( r'\\u2764\\ufe0f', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\u2764\\ufe0f', " heart_emoji ", line)

    #U0001F494 broken_heart
    if bool (  re.search( r'\\U0001f494', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f494', " broken_heart ", line)

    #U0001F63B smile_heartEyes_catFace
    if bool (  re.search( r'\\U0001f63b', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f63b', " smile_heartEyes_catFace ", line)

    #U0001F46B couple_holdHands
    if bool (  re.search( r'\\U0001f46b', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f46b', " couple_holdHands ", line)

    #U0001F46C couple_holdHands
    if bool (  re.search( r'\\U0001f46c', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f46c', " couple_holdHands ", line)

    #U0001F46D couple_holdHands
    if bool (  re.search( r'\\U0001f46d', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f46d', " couple_holdHands ", line)

    #U0001F48F kiss_emoji
    if bool (  re.search( r'\\U0001f48f', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f48f', " kiss_emoji ", line)

    #U0001F491 couple_withHeart_emoji
    if bool (  re.search( r'\\U0001f491', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f491', " couple_withHeart_emoji ", line)

    #U0001F498 heart_with_arrow_emoji
    if bool (  re.search( r'\\U0001f498', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f498', " heart_with_arrow_emoji ", line)

    #U0001F493 beating_heart
    if bool (  re.search( r'\\U0001f493', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f493', " beating_heart ", line)

    #U0001F495 two_hearts
    if bool (  re.search( r'\\U0001f495', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f495', " two_hearts ", line)

    #U0001F496 sparkle_heart
    if bool (  re.search( r'\\U0001f496', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f496', " sparkle_heart ", line)

    #U0001F497 growing_heart
    if bool (  re.search( r'\\U0001f497', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f497', " growing_heart ", line)

    #U0001F49D ribbon_heart
    if bool (  re.search( r'\\U0001f49f', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f49f', " ribbon_heart ", line)

    #U0001F49E revolving_hearts
    if bool (  re.search( r'\\U0001f49e', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f49e', " revolving_hearts ", line)

    #U0001F48C love_letter
    if bool (  re.search( r'\\U0001f48c', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f48c', " love_letter ", line)

    #U0001F61D stuckOutTongue_closedEyes_face
    if bool (  re.search( r'\\U0001f61d', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f61d', " stuckOutTongue_closedEyes_face ", line)

    #U0001F638 grin_smileEyes_catFace
    if bool (  re.search( r'\\U0001f638', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f638', " grin_smileEyes_catFace ", line)

    #U0001F639 tearsJoy_catFace
    if bool (  re.search( r'\\U0001f633', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f639', " tearsJoy_catFace ", line)

    #U0001F618 blowKiss_face
    if bool (  re.search( r'\\U0001f618', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f618', " blowKiss_face ", line)

    #U0001F389 party_popper
    if bool (  re.search( r'\\U0001f389', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f389', " party_popper ", line)

    #U0001F499 blue_heart
    if bool (  re.search( r'\\U0001f499', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f499', " heart_emoji ", line)

    #U0001F49C purple_heart
    if bool (  re.search( r'\\U0001f49c', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f49c', " heart_emoji ", line)

    #U0001F49A green_heart
    if bool (  re.search( r'\\U0001f49a', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001f49a', " heart_emoji ", line)

    # end love_joy


    # surprise
    #U0001F62E openMouth_face
    if bool (  re.search( r'\\U0001F62E', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
    line = re.sub(r'\\U0001F62E', " openMouth_face ", line)

    #U0001F62F surprise_face
    if bool (  re.search( r'\\U0001F62F', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001F62F', " surprise_face ", line)

    #U0001F627 anguish_face
    if bool (  re.search( r'\\U0001F627', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001F627', " anguish_face ", line)

    #U0001F626 frown_openMouth_face
    if bool (  re.search( r'\\U0001F626', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001F626', " frown_openMouth_face ", line)

    #U0001F631 scream_fear_face
    if bool (  re.search( r'\\U0001F631', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001F631', " scream_fear_face ", line)

    #U0001F640 surprise_catFace
    if bool (  re.search( r'\\U0001F640', line) ):
        emotion_rich = True
        emojiCount = emojiCount + 1
        line = re.sub(r'\\U0001F640', " surprise_catFace ", line)
    # end surpeise
    return line, emojiCount, emotion_rich



def clean_toFile(file_name, num_tweets):

    #SET UP LEMMATIZATION
    wnl = WordNetLemmatizer()
    lemmatize = lru_cache(maxsize=50000)(wnl.lemmatize)

    N = num_tweets # number of lines to read (starting at top of file)

    infile = open(file_name)

    outfile_tweets = open('clean_tweets_EMOTION' + str(N) + '.txt', 'w')
    outfile_emojis = open('emojiCount_perTweet_' + str(N) + '.txt', 'w')

    outfile_tweets_original = open('clean_tweets_original' + str(N) + '.txt', 'w')
    outfile_emojis_original = open('emojiCount_perTweet_original' + str(N) + '.txt', 'w')

    outfile_tweets_retweet = open('clean_tweets_retweet' + str(N) + '.txt', 'w')
    outfile_emojis_retweet = open('emojiCount_perTweet_retweet' + str(N) + '.txt', 'w')


    cleaned_tweets = np.array([])
    # emojiCount_perTweet = np.array([])

    emojiCount_perTweet = []

    current_line = 0

    # file_size = 0
    # print 'Calculating size of file to clean...'
    # with open(file_name) as f:
    #     for line in f:
    #         file_size = file_size + 1


    sadness_list = []
    sadness_file = open('sadness.txt')
    for line in sadness_file:
        feature_line_list = line.split(',')
        for feature in feature_line_list:
            sadness_list.append(feature)

    joy_love_list = []
    joy_love_file = open('joy_love.txt')
    for line in joy_love_file:
        feature_line_list = line.split(',')
        for feature in feature_line_list:
            joy_love_list.append(feature)

    anger_disgust_list = []
    anger_disgust_file = open('anger_disgust.txt')
    for line in anger_disgust_file:
        feature_line_list = line.split(',')
        for feature in feature_line_list:
            anger_disgust_list.append(feature)

    surprise_list = []
    surprise_file = open('surprise.txt')
    for line in surprise_file:
        feature_line_list = line.split(',')
        for feature in feature_line_list:
            surprise_list.append(feature)

    fear_list = []
    fear_file = open('fear.txt')
    for line in fear_file:
        feature_line_list = line.split(',')
        for feature in feature_line_list:
            fear_list.append(feature)

    for i in range(N):
        line= infile.next().strip()
        print 'cleaning line', current_line, 'of', N

        current_line = current_line + 1
        emojiCount = 0

        emotion_rich = False

        original_line = line

        line = line.strip('\n')
        line = line.strip('\r')

        line = unicode(line,'utf-8') # convert to unicode
        # line = line.encode('utf-8')
        line = line.encode('unicode-escape')

        # print line
        has_emoji = False
        if text_has_emoji(line): # convert to emoji_feature    --> condition for doing emoji feature insertion
            # print 'before unicode encode'
            # print str(current_line), original_line
            # print line
            has_emoji = True
            line, emojiCount, emotion_rich = InsertEmojiFeatures(line)
            # print line_new
            # pdb.set_trace()
        # remove unicode encodings of ascii characters  --> THIS LINE IS REMOVING ALL EMOJIS

        line = line.decode('unicode_escape').encode('ascii','ignore')

        # make: one with only retweets, one with only original
        is_original_tweet = True
        # assign value to boolean based on whether there is a match to 'RT @handle'
        if   bool (  re.search( r'RT @', line) ) or bool (  re.search( r'RT', line) ):
            # print 'retweet detected'
            is_original_tweet = False
            line = re.sub(r'RT @', 'rt @', line)
            line = re.sub(r'RT', 'rt', line)
        # else:
        #     print line
        #     pdb.set_trace()

        # ;) ; ) : ) :) : D :D  :( : ( : / :/

        if   bool (  re.search( r';\) ', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
            line = re.sub(r';\)', " smile_face ", line)

        if   bool (  re.search( r'; \) ', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
            line = re.sub(r'; \)', " smile_face ", line)

        if   bool (  re.search( r': \) ', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
            line = re.sub(r': \)', " smile_face ", line)

        if   bool (  re.search( r':\) ', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
            line = re.sub(r':\)', " smile_face ", line)

        if   bool (  re.search( r':D ', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
            line = re.sub(r':D', " smile_face ", line)

        #:(
        if   bool (  re.search( r':\( ', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
            line = re.sub(r':\(', " frown_face ", line)

        #: (
        if   bool (  re.search( r': \( ', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
            line = re.sub(r': \(', " frown_face ", line)

        # : /
        if   bool (  re.search( r':\\ ', line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
            line = re.sub(r':\\', " frown_face ", line)

        #: /
        if   bool (  re.search( r': \\ '  , line) ):
            emotion_rich = True
            emojiCount = emojiCount + 1
            line = re.sub(r': \\', " frown_face ", line)

      # double left quotation mark
        line = re.sub(r'\\u201c', "", line)


        # remove commas
        line = re.sub(r'\,', '', line)

        line = re.sub(r'\(', '', line)
        line = re.sub(r'\)', '', line)
        line = re.sub(r'\"', '', line)

        line = re.sub(r'\\r', "", line)
        line = re.sub(r'\\n', "", line)


        # if has_emoji:
        #     print line
        #     pdb.set_trace()

        # remove links
        line = re.sub(r'^https?:\/\/.*[\r\n]*', '', line)
        line = re.sub(r'[a-z]*[:.]+\S+', '', line)

        #remove handles
        line = re.sub(r'@([A-Za-z0-9_]+)', '', line)

        # remove one letter words
        line = re.sub(r'^[a-zA-Z1-9]{1}$', '', line)

       # remove numbers
        line = re.sub(r'[1-9]+', '', line)

        # remove comma seperated numbers
        line = re.sub(r'[0-9]+(,[0-9]+)*', '', line)

        # contractions
        line = re.sub(r'can\'t', 'can_not', line)
        line = re.sub(r'don\'t', 'do_not', line)
        line = re.sub(r'doesn\'t', 'does_not', line)
        line = re.sub(r'didn\'t', 'did_not', line)
        line = re.sub(r'isn\'t', 'is_not', line)
        line = re.sub(r'won\'t', 'will_not', line)
        line = re.sub(r'aren\'t', 'are_not', line)

        line = re.sub(r'http', '', line)
        line = re.sub(r'https', '', line)

        # line = re.sub(r'\b\S{1}\b', '', line)


        # capitalization
        num_cap = sum(1 for c in line if c.isupper())
        num_low = sum(1 for c in line if c.islower())

        if not (num_low == 0 and num_cap == 0):
            if ( num_cap / ( num_cap + num_low )  > .80 ):  # if proportion of caps is greater than 80%
                # then add cap feature
                line = line + " high_cap "

        # exclemation points


# DISABLED FOR SERVER COMPATABILITY
        c = Counter(line)
        frequncies = reversed(c.most_common())

        for aTuple in frequncies:
            if aTuple[0] == '!':
                if int(aTuple[1]) >= 3:
                    emotion_rich = True
                    line = line + " extreme_exclamation "
                elif int(aTuple[1]) >= 1:
                    emotion_rich = True
                    line = line + " exclamation "

        #  exclamation point, hahstag

        # remove punctuation

        # remove stopword

        line_listForm = []
        word_tokens = re.sub(r'[^\w\'] ', " ",  line).split() # convert into list of words
        # new code to remove stop words and better tokenization [1/29/18]
        stop_words = set(stopwords.words('english'))
        # word_tokens = word_tokenize(line)

        # make everything lower case, and LEMMATIZE
        for i in range(len(word_tokens)):
            word_tokens[i] = lemmatize( word_tokens[i].lower() )

        # print 'before: ', word_tokens
        filtered_sentence = [w for w in word_tokens if not w in stop_words]

        # pdb.set_trace()
        # print 'after: ', filtered_sentence

        line_listForm = filtered_sentence

        words_toAppend = []
        for word in line_listForm: # double word if all caps
            if  word.isupper() and len(word) > 1:
                emotion_rich = True
                words_toAppend.append(word)

        for item in words_toAppend:
            line_listForm.append(item)

        # double hash-tagged words (remove hashtag)
        hashed = [ word for word in line_listForm if word.startswith("#") ]
        for word in hashed:
                line_listForm.append( word[1:] )

        indices_to_delete = []
        # # remove hashtag version
        # for i in range(len(line_listForm)):
        #     if line_listForm[i].startswith("#"):
        #         indices_to_delete.append(i)

        for index in reversed(indices_to_delete):
            del line_listForm[index]

        # double emotion key words and their synonyms if they exist
        # emotions_file = open("emotions_synonyms.txt")
        # lines = emotions_file.readlines()
        #     # build dictionary of emotions and synonmys from file

        # emotions_list = dict()
        # for synonym_line in lines:
        #     synonym_line = synonym_line.rstrip("\n")
        #     words = synonym_line.split(",")
        #     emotion_curr = words[0]
        #     synonyms = words[1:]
        #     emotions_list[emotion_curr] = synonyms
        # emotions_file.close()

        line_listForm = [str(item) for item in line_listForm]

        emotion_insert = False
        # check if hits on emotional synonym lists. if so, add that emotion
        for word in line_listForm:
            if word in sadness_list:
                emotion_insert = True
                line_listForm.append('sadness_emotionality')
            elif word in joy_love_list:
                emotion_insert = True
                line_listForm.append('joy_love_emotionality')
            elif word in anger_disgust_list:
                emotion_insert = True
                line_listForm.append('anger_disgust_emotionality')
            elif word in surprise_list:
                emotion_insert = True
                line_listForm.append('surprise_emotionality')
            elif word in fear_list:
                emotion_insert = True
                line_listForm.append('fear_emotionality')

        # if emotion_insert:
        #     pdb.set_trace()
        # emotions_file = open("emotions_synonyms.txt")

        # emotion_feature_list = []
        # for line in emotions_file:
        #     feature_line_list = line.split(',')
        #     for feature in feature_line_list:
        #         if feature not in emotion_feature_list:
        #             emotion_feature_list.append(feature)

        # print 'emotion_feature_list_size = ', str(len(emotion_feature_list))

        # words_toAppend = []
        # for word in line_listForm:
        #     if word == "anger":
        #         words_toAppend.append(word)
        #         emotion_rich = True
        #     elif word == "disgust":
        #         words_toAppend.append(word)
        #         emotion_rich = True
        #     elif word == "fear":
        #         words_toAppend.append(word)
        #         emotion_rich = True
        #     elif word == "joy":
        #         words_toAppend.append(word)
        #         emotion_rich = True
        #     elif word == "love":
        #         words_toAppend.append(word)
        #         emotion_rich = True
        #     elif word == "sadness":
        #         words_toAppend.append(word)
        #         emotion_rich = True
        #     elif word == "surprise":
        #         words_toAppend.append(word)
        #         emotion_rich = True
        #     elif word in emotions_list:
        #         emotion_rich = True
        #         words_toAppend.append(word)

        # for word in words_toAppend:
        #     line_listForm.append(word)

        # if emotion_rich:
        #     line_listForm.append("emotion_rich")



        line = ""

        for word in line_listForm:
            line = line + ' ' + word

                # remove punctuation
        line = re.sub(r'[^\w\s]', '', line)
        line = re.sub(r'_+', '', line)

        line = line[1:] # remove empty space at begining
        line = re.sub(r'\b\S{1}\b', '', line)

        # if has_emoji:
        #     print line

            # line = line + " " + "emotion_rich"

        # emojiCount_perTweet.append(emojiCount)

        # cleaned_tweets = np.append(cleaned_tweets,  line)

        # print line

        if is_original_tweet:
            outfile_tweets_original.write(line + "\n" )
            outfile_emojis_original.write(str(emojiCount) + "\n" )
        else:
            outfile_tweets_retweet.write(line + "\n" )
            outfile_emojis_retweet.write(str(emojiCount) + "\n" )

        outfile_emojis.write(str(emojiCount) + "\n" )
        outfile_tweets.write(line + "\n" )

        # if emojiCount > 0:
        #     print original_line
        #     print line

    infile.close()

    # outfile_tweets = open('clean_tweets_' + N + '.txt', 'w')
    # for line in cleaned_tweets:
    #     outfile_tweets.write(line + "\n" )

    outfile_emojis.close()
    outfile_emojis_original.close()
    outfile_emojis_retweet.close()

    return

    # return cleaned_tweets, emojiCount_perTweet


if __name__ == '__main__':

    tweetFile = 'tweets.txt'
    handleFile = 'handles.txt'

    num_tweets = 4000000
    # num_tweets = 100000

    clean_toFile(tweetFile, num_tweets)

    # get_handles(tweetFile, handleFile, num_tweets)

    # get_handles(tweetFile, handleFile, num_tweets )

    # retweet_proportion(tweetFile, num_tweets)

    # aquire_handleStatistics(handleFile, 4000000)

