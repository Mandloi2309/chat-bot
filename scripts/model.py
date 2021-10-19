from preprocessing import dataPP

import numpy as np
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import random

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD 

class Model(dataPP):
    def __init__(self):
        self.data = dataPP()
        self.data.main()
        
    def training(self):
        # create the training data
        training = []
        # create empty array for the output
        output_empty = [0] * len(self.data.CB.classes)
        # training set, bag of words for every sentence
        for doc in self.data.CB.documents:
            # initializing bag of words
            bag = []
            # list of tokenized words for the pattern
            word_patterns = doc[0]
            # lemmatize each word - create base word, in attempt to represent related words
            word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            # create the bag of words array with 1, if word is found in current pattern
            for word in self.data.CB.words:
                bag.append(1) if word in word_patterns else bag.append(0)
            # output is a '0' for each tag and '1' for current tag (for each pattern)
            output_row = list(output_empty)
            output_row[self.data.CB.classes.index(doc[1])] = 1
            training.append([bag, output_row])
        # shuffle the features and make numpy array
        random.shuffle(training)
        training = np.array(training, dtype=object)
        # create training and testing lists. X - patterns, Y - intents
        self.train_x = list(training[:,0])
        self.train_y = list(training[:,1])
        print("Training data is created")
        
    def model_prep(self):
        # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
        # equal to number of intents to predict output intent with softmax
        model = Sequential()
        model.add(Dense(128, input_shape=(len(self.train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.train_y[0]), activation='softmax'))
        
        # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        #fitting and saving the model
        hist = model.fit(np.array(self.train_x), np.array(self.train_y), epochs=200, batch_size=5, verbose=0)
        model.save('data/chatbot_model.h5', hist)
        
        print("model created")
    def main(self):
        self.training()
        self.model_prep()
        