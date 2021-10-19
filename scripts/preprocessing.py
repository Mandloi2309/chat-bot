from innit import chatBot
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle

class dataPP():
    def __init__(self):
        self.CB = chatBot()
    def preprocessing(self):            
        for intent in self.CB.intents['intents']:
            for pattern in intent['patterns']:
                ## tokenize each word
                word = nltk.word_tokenize(pattern)
                self.CB.words.extend(word)        
                ## add documents in the corpus
                self.CB.documents.append((word, intent['tag']))
                ## add to our classes list
                if intent['tag'] not in self.CB.classes:
                    self.CB.classes.append(intent['tag'])
        #print(self.CB.documents)

    def lemmantize(self):
        ## lemmaztize and lower each word and remove duplicates
        self.CB.words = [lemmatizer.lemmatize(w.lower()) for w in self.CB.words if w not in self.CB.ignore_words]
        self.CB.words = sorted(list(set(self.CB.words)))
        # sort classes
        self.CB.classes = sorted(list(set(self.CB.classes)))
        ## documents = combination between patterns and intents
        # print (len(self.CB.documents), "documents")
        ## classes = intents
        # print (len(self.CB.classes), "classes", self.CB.classes)
        ## words = all words, vocabulary
        # print (len(self.CB.words), "unique lemmatized words", self.CB.words)
        
        pickle.dump(self.CB.words,open('data/words.pkl','wb'))
        pickle.dump(self.CB.classes,open('data/classes.pkl','wb'))
        
    def main(self):
        self.preprocessing()
        self.lemmantize()
    
