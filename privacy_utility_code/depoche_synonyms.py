
import svc_emotions
import pdb
import sys
import re

#  extract synynoms

# be able to pass paramaters --> choose entropy or not, choose threshold otherwise, choose type of file to draw from

# create alternate synonym list out of their lexicon
# use entropy mechanism to select words for a given emotion that have a particularly high score
    # for that one emotion relative to the scores for all other emotions.
# create a list of words that have a generally uniform likelihood across all emotions;
    # potentialy remove these words from tweets as they are just noise


# file_type = freq, normFreq, tfidf

# get the following emotions: afraid, angry, happy, sad

def create_synonymFile( file_Type , threshold ) :

    fileName = ''


    if (file_Type == 'freq'):
        fileName = 'DepecheMood_freq.txt'
    elif (file_Type == 'normFreq'):
        fileName = 'DepecheMood_normfreq.txt'
    elif (file_Type == 'tfidf'):
        fileName = 'DepecheMood_tfidf.txt'
    else:
        fileName = 'DepecheMood_normfreq.txt'

    file_array = []

    # create dictionary --> term as key, list of scores as value

    with open(fileName) as file:
        for line_terminated in file:
            line = line_terminated.rstrip('\n')
            line = re.split(r'\t+', line.rstrip('\t'))
            line[0] = re.sub(r'#[\w]', '', line[0])
            file_array.append(line)

    column_names = file_array[0]
    del file_array[0]

    # save indices that have desired emotions
    relevant_emotion_indices = []

    for i in range(len(column_names)):
        if ( ( column_names[i] == 'AFRAID' ) or (column_names[i] == 'ANGRY') or (column_names[i] == 'HAPPY') or (column_names[i] == 'SAD') ):
            relevant_emotion_indices.append(i)

    # create dictionary to hold all synonyms for relevant emotions
    emotion_synonym_dict = {}
    for i in relevant_emotion_indices:
        emotion_synonym_dict[ column_names[i] ] = []

    # adjust to match wording with kristien's lexicon / emotions --> have first entry for each emotion list be the proper wording
    # afraid --> fear     angry --> anger      happy -->  joy     sad --> sadness
    emotion_synonym_dict['AFRAID'].append('fear')
    emotion_synonym_dict['ANGRY'].append('anger')
    emotion_synonym_dict['HAPPY'].append('joy')
    emotion_synonym_dict['SAD'].append('sadness')

    # for each entry in file, go through scores for all relevent emotions and add entry to corresponding synonym list
    for entry in file_array:
        for i in relevant_emotion_indices:
            if float(entry[i]) > threshold:
                emotion_synonym_dict[ column_names[i] ].append(entry[0])



    # read in synoynms for emotions that i'm not changed the s
    # output to file
    emotions_file = open("emotions_synonyms_depoche.txt", 'w')
    for key, value in emotion_synonym_dict.iteritems():
        line = ''
        for i in range(len(value) - 1) : # add all synonyms but the last one with commas
            line += (value[i] + ',')
        # add last synonym
        line += (value[  (len(value) - 1) ] + '\n')
        emotions_file.write(line)




if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf8')
    create_synonymFile('normFreq', .8)









