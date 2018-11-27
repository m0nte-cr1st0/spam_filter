import re
import math
import copy


# Naïve Bayes class
class NaiveBayes(object):
    def __init__(self):
        '''Initialization method.
        Initialize required properties of the class instance.'''
        self.data = None
        self.total_docs = None
        self.class_priors = {}
        self.corpus = {}
        self.vocab = {}
        self.tokens = {}        
        
    def train_model(self, data):
        '''This method train model by input data
        :param data: - input data as a list of strings'''
        # Set parameters
        self.data = data
        
        # Calc total docs
        self.set_total_docs()
        
        # Calc num and priors of classes
        self.calc_classes_priors()

        # Load corpus
        self.loadCorpus() 

        # tokenize text
        self.tokenize_text()

        # Set vocabulary for corpus and classes
        self.set_vocabulary()

        # Set frequences
        self.set_frequences()

    def classify(self, text):
        '''This method claccify input text using by trained model
        :param text: - classified text'''        
        
        # Check out the text is not empty
        assert text != '', 'Input text cannot be empty string'
        
        # Tokenize text
        # Convert input text string into list of the words
        words = self.tokenize_text(text)
        
        # Calc word' frequences
        # for each word in list of splitted words
        # Get word' frequency for each class from vocab property
        # If word was present in the train data
        # Else calc word' frequency for each class
        # And add it to the list of frequences
        frequences = []
        for word in words:
            if word not in self.vocab["corpus"]:
                frequences.append(self.set_frequences([word]))
            else: frequences.append(self.vocab["corpus"][word])
        
        # Calc class prediction
        # For each class calc prediction value
        # How likely the input text belongs to the given class
        predicted = {}
        for i in frequences:
            for cat in self.class_priors:
                predicted[cat] = predicted.get(cat, 1.) + math.log(i[cat])

        # Adjust estimations by class' priors
        for key in predicted.keys(): predicted[key] += math.log(self.class_priors[key])

        # Print out calculated estimations
        print('Estimations for each class: {}'.format(predicted))

        # Sort out preditcion by estimations
        estim = sorted(predicted, key=predicted.get, reverse=True)        
        
        return (text, estim[0][0],)
        
    def set_total_docs(self):
        '''Set number of total docs.
        For given test total_docs must be equal to 4.
        total_docs can be calculated as num of items in input data'''        
        self.total_docs = len(data)
        
    def calc_classes_priors(self):
        '''Calc number of classes and their priors.
        1. Number of classes can be calculated as a number of distinct categories in input data
        2. Priors for classes can be calculated as number of docs per class divided by total number of docs'''
        self.num_classes = {}
        for entry in self.data:
            text = entry[0]
            category = entry[1]            
            if category not in self.num_classes:
                self.num_classes[category] = 1
            else:
                self.num_classes[category] += 1

        # Calc priors
        for category_key, category_value in self.num_classes.items():            
            self.class_priors[category_key] = category_value / self.total_docs

    def set_vocabulary(self):
        '''Set corpus and classes vocabulary.
        1. Corpus key should contains set of words from input text with theirs number of occuriences.
        2. Classes keys should contains set of words from input text belonging to the given class with theirs number of occuriences'''
        for key, value in self.tokens.items(): self.vocab[key] = self.makeDictionary(value)

    def set_frequences(self, corpus=None):
        '''Set frequences for words from corpus.
        For each word in input corpus calc its frequency
        :param corpus: - the text corpus to be calculated'''
        if not corpus: corpus = self.vocab['corpus']
        for word in corpus:
            freq = {}
            for cat in self.class_priors:                
                freq[cat] = (self.vocab[cat].get(word, 0) + 1) / (sum(self.vocab[cat].values())+ len(self.vocab['corpus']))
            corpus[word] = freq
        return freq
            
    def get_total_docs(self):
        '''Get number of total docs
        Must return value from total_docs'''        
        return self.total_docs

    def get_num_classes(self):
        '''Get number of classes
        Must return len of classes properties'''        
        return len(self.class_priors)
                    
    def get_priors(self):
        '''Get number of priors for detected classes
        Must return value from classes properties as a dictionary'''        
        return self.class_priors

    def get_corpus(self):
        '''Get text corpus
        Must return string of words from the input data set'''        
        return self.corpus['corpus']
                    
    def get_class_corpus(self):
        '''Get text corpus per class
        Must return string of words for each class as a dictionary'''        
        self.copy_corpus = self.corpus.copy()
        self.copy_corpus.pop('corpus')
        return self.copy_corpus
    
    def get_class_tokens(self):
        '''Get list of tokens for detected classes
        Must return list of tokens for each class as a dictionary'''        
        return self.tokens
    
    def get_class_vocabulary(self):
        '''Get vocabulary for detected classes
        Must return vocabulary for each class as a dictionary'''        
        return self.vocab
   
    def loadCorpus(self):
        '''Load corpus text by categories'''
        for text, category in self.data:
            if 'corpus' not in self.corpus: self.corpus['corpus'] = []            
            self.corpus['corpus'].append(text)

            if category not in self.corpus: 
                self.corpus[category] = []            
            self.corpus[category].append(text)

#         # Convert list of text to the one text string
        for key in self.corpus:            
            self.corpus[key] = ' '.join(self.corpus[key])          

    def tokenize_text(self, text=None):
        '''Tokenize text
        For given text split it by space delimiter
        and return it as a list if text parameter is not None
        Else add this list to the dictionary of tokens.
        :param text: - the text to be tokenized'''
        if text:
            return text.split()
        for key, value in self.corpus.items():            
            self.tokens[key] = value.split()

    @staticmethod
    def makeDictionary(text):
        '''Make dictionary of words from input text.
        key: word
        value: frequence of word in the input text
        :param text: - the text to be tokenized
        :return: - dictionary
        '''
        D = {}
        for k in text:
            if k not in D:
                D[k] = 1
            else:
                D[k] += 1
        return D
		
		
# Test
data = [["Chinese Beijing Chinese","0"],
        ["Chinese Chinese Shanghai","0"], 
        ["Chinese Macao","0"],
        ["Tokyo Japan Chinese","1"]]

# Create instance of NaiveBayes class
nb = NaiveBayes()

nb.train_model(data)

nb.classify("Chinese Chinese Chinese Tokyo Japan")