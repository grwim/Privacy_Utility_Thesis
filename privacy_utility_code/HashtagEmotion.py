class HashtagEmotion:

	def __init__(self, hashtag):
		self.hashtag = hashtag
		self.total_emotion_count = 0
		self.happy_count = 0
		self.sad_count = 0
		self.angry_count = 0
		self.scared_count = 0
		self.excited_count = 0

	def increment_emotion(self, emotion):
		self.total_emotion_count += 1
		if emotion == "happy":
			self.happy_count += 1
		elif emotion == "sad":
			self.sad_count += 1
		elif emotion == "angry":
			self.angry_count += 1
		elif emotion  == "scared":
			self.scared_count += 1
		elif emotion == "excited":
			self.excited_count += 1

	def sql_string(self):
		return "'" + self.hashtag + "'" + ", " + str(self.total_emotion_count) + ", " + str(self.happy_count) + ", " + str(self.sad_count) + ", " + str(self.angry_count) + ", " + str(self.scared_count) + ", " + str(self.excited_count)

	def __str__(self):
		return self.hashtag + ": " + str(self.total_emotion_count) + ", " + str(self.happy_count) + ", " + str(self.sad_count) + ", " + str(self.angry_count) + ", " + str(self.scared_count) + ", " + str(self.excited_count)
