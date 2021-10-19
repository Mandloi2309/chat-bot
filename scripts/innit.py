import json

class chatBot:
    def __init__(self, filepath='data/intents.json'):
        self.words=[]
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!', ',', '.']
        ## Training data import
        self.intents = json.loads(open(filepath).read())
        