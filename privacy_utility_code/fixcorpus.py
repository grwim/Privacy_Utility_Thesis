#python script to clean incoming Sanders data-- gets rid of first three words and quotation marks, only includes if not irrelevant
import svc_emotions

tweets = svc_emotions.file_to_array("newercorpus.txt")

#create outfile to put new stuff into
outfile_name = "quoteless.txt"
outfile = open(outfile_name, 'w')

for tweet in tweets:
	#part that first gets rid of the tabs and then parses based on spaces and removes irrelevant tweets
	# newList = tweet.split(" ")
	# newString = ""
	# for smallString in newList[4:]:
	# 	newString = newString + " " + smallString
	# if newList[2] != "irrelevant":
	# 	outfile.write("\n" + newString)

	#part that takes the quotes away
	newTweet = tweet[4:]
	newTweet = newTweet[:-3]

	outfile.write("\n" + newTweet)

outfile.close()