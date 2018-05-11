import re
import requests
import sys
import psycopg2
import math
import random
# from hashtagemotion import HashtagEmotion
import scipy.stats as stats
from scipy.stats import rankdata
import operator
import numpy as np

def main():
	reload(sys)
	sys.setdefaultencoding('utf8')

	print("Running new version")

	# Connect to DB
	try:
		conn = psycopg2.connect(database='webfootprint_result', user='kristen', host='localhost', password='kristen')
	except:
		print('Unable to connect to database.')

	# # Set up emotions and synonmys lists
	# #emotions = [("happiness", "happy"), ("anger", "angry"), ("sadness", "sad"), ("vigilance", "vigilant"), ("rage","rageful"), ("loathing","hate"), ("grief", "bereaved"), ("amazement", "amazed"), ("terror", "terrified"), ("admiration", "admire"), ("ecstacy", "ecstatic")]
	emotions = [("anger", "angry"), ("disgust", "disgusted"), ("fear", "scared"), ("joy", "happy"), ("love", "love"), ("sadness", "sad"), ("surprise", "surprised")]
	#emotions = [("happiness", "happy"), ("sadness", "sad"), ("calm", "serenity"), ("excitement", "excited"), ("alertness", "alert"), ("oblivious", "unaware"), ("confidence", "assured"), ("vitality", "vigorous"), ("apathy", "apathetic"), ("kindness", "kind"), ("meanness", "mean")]


	#emotions_synonyms = load_emotion_synonyms(emotions)
	#print(emotions_synonyms)
	emotions_synonyms = {'love': ['passion','beloved','amour','love','enjoy','heart'], 'surprise': ['surprise', 'surprised', 'amazed','astonished', 'astounded', 'dumbfounded', 'dumbstricken', 'dumbstruck', 'dumfounded', 'flabbergasted', 'gobsmacked', 'goggle-eyed', 'jiggered', 'openmouthed', 'popeyed', 'startled', 'stunned', 'stupefied', 'thunderstruck'], 'sadness': ['sadness', 'unhappiness', 'sorrow', 'sorrowfulness', 'gloominess', 'lugubriousness', 'feeling', 'uncheerfulness', 'sad', 'deplorable', 'distressing', 'lamentable', 'pitiful', 'sorry', 'bad', 'bittersweet', 'depressing', 'depressive', 'doleful', 'gloomy', 'heavyhearted', 'melancholic', 'melancholy', 'mournful', 'pensive', 'saddening', 'sorrowful', 'tragic', 'tragical', 'tragicomic', 'tragicomical', 'wistful'], 'disgust': ['disgust', 'dislike', 'gross out', 'revolt', 'repel', 'nauseate', 'sicken', 'churn up', 'repulse', 'disgusted', 'fed up', 'sick', 'sick of', 'tired of', 'displeased'], 'anger': ['anger', 'choler', 'ire', 'angriness', 'wrath', 'angry', 'furious', 'raging', 'tempestuous', 'aggravated', 'angered', 'choleric', 'enraged', 'infuriated', 'irascible', 'irate', 'ireful', 'livid', 'mad', 'maddened', 'outraged', 'smoldering', 'smouldering', 'wrathful', 'wroth', 'wrothful'], 'fear': ['fear', 'fearfulness', 'fright', 'dread', 'worry', 'scared', 'frightened', 'afraid'], 'joy': ['happiness', 'felicity', 'happy', 'felicitous', 'glad', 'cheerful', 'content', 'contented', 'elated', 'euphoric', 'joyful', 'joyous', 'blessed', 'blissful', 'bright', 'fortunate', 'laughing', 'prosperous', 'riant', 'willing']}
	#emotions_synonyms = {'confidence': ['confidence', 'assurance', 'self-assurance', 'self-confidence', 'authority', 'sureness', 'trust', 'certainty', 'security', 'assured', 'confident', 'secure'], 'meanness': ['meanness', 'beastliness', 'malevolence', 'malevolency', 'malice','mean',  'hateful', 'meanspirited', 'awful', 'contemptible', 'ignoble', 'nasty'], 'apathy': ['apathy', 'indifference', 'numbness', 'spiritlessness', 'passiveness', 'passivity', 'apathetic', 'indifferent', 'spiritless', 'uninterested'], 'vitality': ['vitality', 'verve', 'energy', 'vim', 'animation', 'aliveness', 'animateness',  'liveness', 'vigor', 'vigour', 'vigorous', 'energetic', 'robust'], 'alertness': ['alertness', 'watchfulness', 'wakefulness', 'vigilance', 'attention', 'attentiveness', 'alert', 'wary', 'watchful', 'aware', 'cognisant', 'cognizant', 'conscious',  'vigilant', 'preparedness', 'readiness'], 'excitement': ['excitement', 'exhilaration', 'excitation', 'agitation', 'turmoil', 'upheaval', 'disturbance', 'excited', 'worked up', 'delirious', 'frantic', 'agitated','overexcited','stimulated'], 'sadness': ['sadness', 'unhappiness', 'sorrow', 'sorrowfulness', 'gloominess', 'lugubriousness', 'feeling', 'uncheerfulness', 'sad', 'deplorable', 'distressing', 'lamentable', 'pitiful', 'sorry', 'depressing', 'depressive', 'gloomy', 'heavyhearted', 'melancholic', 'melancholy', 'mournful', 'pensive', 'saddening', 'sorrowful', 'tragic', 'tragical', 'tragicomic', 'tragicomical', 'wistful'], 'calm': ['calm', 'unagitated', 'serene', 'tranquil', 'easygoing', 'placid', 'peaceful', 'composed', 'content', 'contented', 'quiet', 'settled', 'smooth', 'windless', 'composure', 'calmness', 'equanimity',  'tranquilize', 'tranquillize', 'tranquillise', 'quieten', 'lull', 'steady', 'becalm', 'cool off', 'chill out', 'simmer down', 'settle down', 'cool it', 'serenity', 'repose', 'quiet', 'placidity', 'tranquillity', 'tranquility', 'peace', 'peacefulness', 'peace of mind', 'heartsease', 'calm', 'calmness', 'quietness', 'quietude'], 'oblivious': ['oblivious', 'forgetful', 'unmindful', 'inattentive', 'incognizant', 'unaware', 'unaware', 'incognizant', 'unwitting', 'insensible', 'unconscious', 'unmindful', 'unsuspecting'], 'kindness': ['kindness', 'forgivingness', 'benignity', 'good', 'goodness', 'mercifulness', 'mercy', 'kind', 'genial', 'tolerant', 'considerate', 'good-natured', 'merciful', 'benevolent', 'charitable', 'forgiving', 'gentle', 'good-hearted', 'gracious', 'hospitable', 'kind-hearted', 'kindhearted', 'kindly', 'large-hearted', 'openhearted', 'sympathetic'], 'happiness': ['happiness', 'felicity', 'spirit', 'happy', 'felicitous', 'glad', 'cheerful', 'content', 'contented', 'elated', 'euphoric', 'joyful', 'joyous', 'blessed', 'blissful', 'bright', 'fortunate', 'laughing', 'prosperous', 'riant']}

	#Create a cursor to execute database queries
	cur = conn.cursor()
	select_tweets = 'SELECT twitter_handle, tweet FROM tweet ORDER BY random() LIMIT 500000;'
	cur.execute(select_tweets)

	# Select tweets by user
	tweets_list = []
	tweets_by_user = dict()

	rows = cur.fetchall()
	for tweet in rows:
		tweets_list.append(tweet[1])
		# if tweet[0] not in tweets_by_user:
		# 	tweets_by_user[tweet[0]] = list(tweet[1])
		# else:
		# 	tweets_by_user[tweet[0]].append(tweet[1])

	#compute_user_emotion_stdev(tweets_by_user, emotions, emotions_synonyms)

	get_tweets_with_emotions(tweets_list, emotions, emotions_synonyms)


	# rand_users_sample = random.sample(tweets_by_user, 100)
	# rand_tweets_dict = dict()
	# for user in rand_users_sample:
	# 	rand_tweets_dict[user] = tweets_by_user[user]
	#test_user_distances(tweets_by_user, emotions, emotions_synonyms)

	#conn.commit()

	cur.close()
	conn.close()

# def getEmotions(tweets):
# 	emotions = [("vigilance", "vigilant"), ("rage","rageful"), ("loathing","hate"), ("grief", "bereaved"), ("amazement", "amazed"), ("terror", "terrified"), ("admiration", "admire"), ("ecstacy", "ecstatic")]
# 	emotionsCount = dict()
# 	synonmysMap = dict()
# 	for emotion in emotions:
# 		emotionsCount[emotion[0]] = 0
# 		emotions_synonyms[emotion[0]] = emotion[0]
# 		for synonym in callThesaurus(emotion[0]):
# 			synonmysMap[synonym] = emotion[0]
# 		emotions_synonyms[emotion[1]] = emotion[0]
# 		for synonym in callThesaurus(emotion[1]):
# 			synonmysMap[synonym] = emotion[0]
# 	emotionsCount["total"] = 0
# 	for tweet in tweets:
# 		for emotion in synonmysMap.keys():
# 			if emotion in tweet:
# 				emotionsCount[synonmysMap[emotion]] += 1
# 				emotionsCount["total"] += 1
# 	#print emotionsCount

def compute_user_emotion_stdev(tweets, emotion_list, emotions_synonyms):

	user_emotions_vectors = []
	for user in tweets:
		emotion_vector = compute_user_emotion_vector(tweets[user], emotion_list, emotions_synonyms)
		user_emotions_vectors.append(emotion_vector)

	output_file = open('emotion_stats_gpoms.txt', 'w')

	emotion_list.append(("total", "total"))

	for emotion in emotion_list:
		frequency_values = [user[emotion[0]] for user in user_emotions_vectors]
		stddev = pstdev(frequency_values)
		meanval = mean(frequency_values)
		output_file.write("Mean of frequency for " + emotion[0] + ": " + str(meanval) + "\n")
		output_file.write("SD of frequency for " + emotion[0] + ": " + str(stddev) + "\n")


		# # compute difference in means & statistical significance
		# frequency_values_1 = [user[emotion] for user in users_group_1]
		# frequency_values_2 = [user[emotion] for user in users_group_2]
		# stddev1 = pstdev(frequency_values_1)
		# stddev2 = pstdev(frequency_values_2)
		# mean1 = mean(frequency_values_1)
		# mean2 = mean(frequency_values_2)
		# if mean1 is not 0.0 and mean2 is not 0.0:
		# 	se = math.sqrt((math.pow(stddev1, 2)/mean1)+(math.pow(stddev2, 2)/mean2))
		# 	t = (mean1 - mean2) / se
		# else:
		# 	t = 0
		# print("t stat for " + emotion + ": " + str(t) + ", with sample 1 n = " + str(len(frequency_values_1)) + " and sample 2 n = " + str(len(frequency_values_2)))

	output_file.close()

def test_user_distances(tweets, emotion_list, emotions_synonyms):

	print("kendall tau v 2")

	distance_to_self = []
	distance_to_other = []
	output_file = open('kendall_tau_2.txt', 'w')

	num_users = 0
	num_self_users = 0
	num_concordant_self = 0
	num_discordant_other = 0
	num_discordant_average = 0

	average_order = ["happiness", "sadness", "jealousy", "fear", "disgust", "surprise", "shame", "confusion", "anger", "nervousness"]
	average_ranks = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

	for user1 in tweets:
		# print("working on " + str(user1))
		emotions_vector_1 = compute_user_emotion_vector(tweets[user1], emotion_list, emotions_synonyms)
		frequencies_1 = list()
		taus = list()
		for item in average_order:
			frequencies_1.append(emotions_vector_1[item])
		ranks_1_a = rankdata(np.asarray(frequencies_1))
		ranks_1 = (ranks_1_a).astype(int)
		if checkEqual2(ranks_1) is not True:
			num_users += 1
			tau, p_value = stats.kendalltau(np.asarray(average_ranks), np.asarray(ranks_1))
			taus.append(tau)
			if tau < 0.2:
				num_discordant_average += 1
				output_file.write("User " + str(user1) + " discordant with AVERAGE: " + str(tau) + "\n")

			# for user2 in tweets:
			# 	if user1 is not user2:
			# 		emotions_vector_2 = compute_user_emotion_vector(tweets[user2], emotion_list, emotions_synonyms)
			# 		if checkEqual2(emotions_vector_2) is not True:
			# 			frequencies_2 = list()
			# 			for item in average_order:
			# 				frequencies_2.append(emotions_vector_2[item])
			# 			ranks_2_a = rankdata(frequencies_2)
			# 			ranks_2 = (ranks_2_a).astype(int)
			# 			tau, p_value = stats.kendalltau(np.asarray(ranks_1), np.asarray(ranks_2))
			# 			if tau < 0.2 and tau > -0.2:
			# 				num_discordant_other += 1
			# 			output_file.write("OTHER: " + str(tau) + "\n")

			# 	else:

			# user1_control = []
			# user1_test = []
			# for tweet in tweets[user1]:
			# 	randint = random.random()
			# 	if randint < 0.5:
			# 		user1_control.append(tweet)
			# 	else:
			# 		user1_test.append(tweet)
			# emotions_vec = compute_user_emotion_vector(user1_control, emotion_list, emotions_synonyms)
			# frequencies_3 = list()
			# for item in average_order:
			# 	frequencies_3.append(emotions_vec[item])
			# ranks_3_a = rankdata(np.asarray(frequencies_3))
			# ranks_3 = (ranks_3_a).astype(int)
			# emotions_vector_test = compute_user_emotion_vector(user1_test, emotion_list, emotions_synonyms)
			# frequencies_4 = list()
			# for item in average_order:
			# 	frequencies_4.append(emotions_vector_test[item])
			# ranks_4_a = rankdata(frequencies_4)
			# ranks_4 = (ranks_4_a).astype(int)
			# # print("freq3: " + str(emotions_vec) + " freq4: " + str(emotions_vector_test))
			# # print("ranks_3: " + str(ranks_3) + " ranks 4: " + str(ranks_4))
			# if checkEqual2(ranks_3) is not True and checkEqual2(ranks_4) is not True:
			# 	num_self_users += 1
			# 	tau, p_value = stats.kendalltau(np.asarray(ranks_3), np.asarray(ranks_4))
			# 	if tau > .08:
			# 		num_concordant_self += 1
			# 	else:
			# 		output_file.write("SELF " + str(user1) + str(tau) + "\n")


	output_file.write(" of " + str(num_users) + " users was discordant with average " + str(num_discordant_average) + " times. \n")
	mean = np.mean(np.array(taus))
	p25 = np.percentile(np.array(taus), 25)
	p75 = np.percentile(np.array(taus), 75)
	output_file.write(" mean of tau value: " + str(mean) + " p25 " + str(p25) + " p75 " + str(p75))

	output_file.close()


def checkEqual2(iterator):
     return len(set(iterator)) <= 1

	# stddev_self = pstdev(distance_to_self)
	# stddev_other = pstdev(distance_to_other)
	# mean_self = mean(distance_to_self)
	# mean_other = mean(distance_to_other)
	# if mean_self is not 0.0 and mean_other is not 0.0:
	# 	se = math.sqrt((math.pow(stddev_self, 2)/mean_self)+(math.pow(stddev_other, 2)/mean_other))
	# 	t = (mean_self - mean_other) / se
	# else:
	# 	t = 0
	# output_file.write("t stat for difference between self and other distances = " + str(t) + ", with self-sample n = " + str(len(distance_to_self)) + " and other-sample n = " + str(len(distance_to_other)) + "\n")


def compute_emotion_vector_distance(vector1, vector2, emotion_list):
	distance_sum = 0.0
	for emotion in emotion_list:
		distance_sum += math.pow(vector1[emotion[0]] - vector2[emotion[0]], 2)
	return math.sqrt(distance_sum)

def compute_user_emotion_vector(tweets, emotion_list, emotions_synonyms):
	num_emotions = get_emotion_counts(tweets, emotion_list, emotions_synonyms)
	num_tweets = float(len(tweets))
	emotions_vector = dict.fromkeys(num_emotions.keys(), 0.0)
	for key in emotions_vector:
		emotions_vector[key] = num_emotions[key] / num_tweets

	return emotions_vector

def get_tweets_with_emotions(tweets, emotion_list, emotions_synonyms):

	print("Started emotion search\n")


	output_file = open('emotionaltweets3.txt', 'w')

	for tweet in tweets:
		for emotion in emotion_list:
			emotionFound = False
			if emotion[0] is not "total":
				for synonym in emotions_synonyms[emotion[0]]:
					string_pat = "[^\\w]" + synonym.lower() + "[^\\w]"
					synonym_pattern = re.compile(string_pat)
					if synonym_pattern.search(unicode(tweet, errors='ignore').lower()) is not None and emotionFound is not True:
						output_file.write(emotion[0] + "," + tweet + "\n")
						emotionFound = True

						# if emotion[0] == "anger":
						# 	print(synonym + " found in " + tweet)
						# num_emotions["total"] += 1.0
						# num_emotions[emotion[0]] += 1.0
						# emotionFound = True
	print("Finished emotion search\n")

	output_file.close

def get_emotion_counts(tweets, emotion_list, emotions_synonyms):

	num_emotions = dict()
	for emotion in emotion_list:
		num_emotions[emotion[0]] = 0.0
	num_emotions["total"] = 0.0

	for tweet in tweets:
		for emotion in emotion_list:
			emotionFound = False
			if emotion[0] is not "total":
				for synonym in emotions_synonyms[emotion[0]]:
					string_pat = "[^\\w]" + synonym.lower() + "[^\\w]"
					synonym_pattern = re.compile(string_pat)
					if synonym_pattern.search(unicode(tweet, errors='ignore').lower()) is not None and emotionFound is not True:
						# if emotion[0] == "anger":
						# 	print(synonym + " found in " + tweet)
						num_emotions["total"] += 1.0
						num_emotions[emotion[0]] += 1.0
						emotionFound = True

	return num_emotions

def matchHashtagsToEmotion(tweets, cur):
	emotions = ["happy", "sad", "angry", "scared", "excited"]
	emotionSynonyms = dict.fromkeys(emotions, list())
	for emotion in emotions:
		synonym_list = list()
		for synonym in callThesaurus(emotion):
			synonym_list.append(synonym)
		emotionSynonyms[emotion] = synonym_list

	hashtag_emotion_dictionary = dict()

	num_hashtags = 0
	num_emotions = dict.fromkeys(emotions, 0)
	num_emotions["total"] = 0

	hashtagPattern = re.compile('#\\w+')
	for tweet in tweets:
		hashtagsInTweet = hashtagPattern.findall(tweet)
		for hashtag in hashtagsInTweet:
			num_hashtags += 1
			for emotion in emotions:
				emotionFound = False
				for synonym in emotionSynonyms[emotion]:
					if synonym in tweet and emotionFound is not True:
						num_emotions["total"] += 1
						num_emotions[emotion] += 1
						if hashtag.lower() not in hashtag_emotion_dictionary:
							hashtag_emotion_dictionary[hashtag.lower()] = HashtagEmotion(hashtag.lower())
							hashtag_emotion_dictionary[hashtag.lower()].increment_emotion(emotion)
						else:
							hashtag_emotion_dictionary[hashtag.lower()].increment_emotion(emotion)
						emotionFound = True

	print "Total number of tweets: " + str(num_hashtags)
	for k, v in num_emotions.iteritems():
		print "Occurences of emotion " + k + ": " + str(v)

	clear_table = 'DELETE FROM tweets_hashtags_emotions;'
	cur.execute(clear_table)

	for item in hashtag_emotion_dictionary.values():
		insert_row = 'INSERT INTO tweets_hashtags_emotions VALUES (' + item.sql_string() + ');'
		cur.execute(insert_row)

def load_emotion_synonyms(emotions):

	emotions_synonyms = dict()

	for emotion in emotions:
		emotions_synonyms[emotion[0]] = list()
		emotions_synonyms[emotion[0]].append(emotion[0])
		for synonym in callThesaurus(emotion[0]):
			emotions_synonyms[emotion[0]].append(synonym)
		emotions_synonyms[emotion[0]].append(emotion[1])
		for synonym in callThesaurus(emotion[1]):
			emotions_synonyms[emotion[0]].append(synonym)

	return emotions_synonyms

def getHashtags(tweets):
	hashtags = dict()
	hashtagPattern = re.compile('#\\w+')
	for tweet in tweets:
		hashtagsInTweet = hashtagPattern.findall(tweet)
		for hashtag in hashtagsInTweet:
			if hashtag not in hashtags:
				hashtags[hashtag] = 1
			else:
				hashtags[hashtag] += 1
	print hashtags.items()

def callThesaurus(word):
	requestUrl = "http://words.bighugelabs.com/api/2/81f84f749a5f18e85dcbdc3014040e89/" + word + "/"
	result = requests.get(requestUrl)
	synonyms = []
	if result.status_code == 200:
		resultLines = result.text.split('\n')
		for line in resultLines:
			splitLine = line.split('|')
			if len(splitLine) > 1:
				if splitLine[1] == "syn" or splitLine[1] == "rel" or splitLine[1] == "sim":
					if splitLine[2] not in synonyms:
						synonyms.append(splitLine[2])
	return synonyms


def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/float(n) # in Python 2 use sum(data)/float(n)

def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

def pstdev(data):
    """Calculates the population standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/n # the population variance
    return pvar**0.5

if  __name__ =='__main__':
    main()


