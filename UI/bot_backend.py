import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
import json
import random
from os.path import isfile
from model import Model

class bot_response:
    def __init__(self):
        self.intents = json.loads(open('data/intents.json').read())
        if isfile('data/words.pkl') != True or isfile('data/classes.pkl') != True or isfile('data/chatbot_model.h5') != True:
            s = Model()
            s.main()
        self.words = pickle.load(open('data/words.pkl','rb')) 
        self.classes = pickle.load(open('data/classes.pkl','rb'))
        self.model = load_model('data/chatbot_model.h5')

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    
    def bow(self, sentence, show_details=True):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(self.words)
        for s in sentence_words:
            for i,w in enumerate(self.words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)
        return(np.array(bag))

    def predict_class(self, sentence):
        # filter out predictions below a threshold
        p = self.bow(sentence,show_details=False)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list
    
    def getResponse(self, ints):
        tag = ints[0]['intent']
        list_of_intents = self.intents['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result
    
    def chatbot_response(self, msg):
        ints = self.predict_class(msg)
        res = self.getResponse(ints)
        return res


#Creating GUI with tkinter
