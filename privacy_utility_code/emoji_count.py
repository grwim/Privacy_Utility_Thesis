import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import emoji
import pdb
import re

# accrues a count for each emoji from all tweets; put good,
# top emojis that are missing in our dictionaries into corresponding dictionaries

filename_tweets = '/Users/Konrad/Desktop/Development/Emotional_Profiles/fwdrecode/tweets.txt'

tweetList = []

N= 10000
f=open(filename_tweets)
for i in range(N):
    line=f.next().strip()
    line = unicode(line,'utf-8')
    line = line.encode('unicode-escape')
    tweetList.append(line)
f.close()

countForEmoji = {}
emojiList_unique = []

for tweet in tweetList:
    # print tweet

#     print tweet.encode('utf-8')
    emoji_pattern = r'\\U[\w\d]{8}'
    list_emojis = re.findall(emoji_pattern, tweet)

    # print tweet
    # print list_emojis

    for character in list_emojis:

        if character == '\U0001f3fchttps':
            print tweet


        if character not in emojiList_unique:
            emojiList_unique.append(character)

        if character in countForEmoji:
            countForEmoji[character] = countForEmoji[character] + 1
        else:
            countForEmoji[character] = 1

    # if len(emojis) > 1:
    #     print tweet
    #     print emojis
#     emojis = re.findall(ru'[\U0001f600-\U0001f650]', s)



d_view = [ (v,k) for k,v in countForEmoji.iteritems() ]
d_view.sort(reverse=True) # natively sort tuples by first element
for v,k in d_view:
    encoding = k
    k = unicode(k,'utf-8')
    k = k.decode('unicode-escape')
    # print k.decode()

    print k,v,encoding
    # print "%s: %d" % (k,v)

# for tweet in tweetList:
#     # pdb.set_trace()
#     list_emojis = extract_emojis(tweet)

#     for character in list_emojis:
#         if character not in emojiList_unique:
#             emojiList_unique.append(character)

#         if character in countForEmoji:
#             countForEmoji[character] = countForEmoji[character] + 1
#         else:
#             countForEmoji[character] = 1

# for character in emojiList_unique:
#     print character.encode('unicode-escape'), countForEmoji[character]
#     # print tweet
