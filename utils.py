from nltk.tokenize import regexp_tokenize
import numpy as np

# Here is a default pattern for tokenization, can substitue it
default_pattern =  r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                        |(?:[.,;"'?():-_`])    
                    """

def tokenize(text, pattern = default_pattern):
    """Tokenize senten with specific pattern
    
    Arguments:
        text {str} -- sentence to be tokenized, such as "I love NLP"
    
    Keyword Arguments:
        pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})
    
    Returns:
        list -- list of tokenized words, such as ['I', 'love', 'nlp']
    """
    text = text.lower()
    return regexp_tokenize(text, pattern)


class FeatureExtractor(object):
    """Base class for feature extraction.
    """
    def __init__(self):
        pass
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass



class UnigramFeature(FeatureExtractor):
    """Example code for unigram feature extraction
    """
    def __init__(self):
        self.unigram = {}
        
    def fit(self, text_set: list):
        """Fit a feature extractor based on given data 
        
        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.unigram:
                    self.unigram[text_set[i][j].lower()] = index
                    index += 1
                else:
                    continue
        
                    
    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        feature = np.zeros(len(self.unigram))
        for i in range(0, len(text)):
            if text[i].lower() in self.unigram:
                feature[self.unigram[text[i].lower()]] += 1
        
        return feature
    
    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        
        Returns:
            array -- unigram feature arrays, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)
            

class BigramFeature(FeatureExtractor):
    """Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self):
        # Add your code here!
        raise Exception("Must be implemented")
    def fit(self, text_set):
        # Add your code here!
        raise Exception("Must be implemented")
    def transform(self, text):
        # Add your code here!
        raise Exception("Must be implemented")
    def transform_list(self, text_set):
        # Add your code here!
        raise Exception("Must be implemented")

class CustomFeature(FeatureExtractor):
    """customized feature extractor, such as TF-IDF
    """
    def __init__(self):
        # Add your code here!
        self.unigram = {}
        self.docWordCount = {}
        self.stopWords = ["a", "about", "above", "after", "all", "also", "am", "an", "and", "any", "are", "as",  "at", "be", "because", "been", "before", "being", "between", "both", "but", "by",                   
                          "could", "did", "do", "does", "doing", "down", "during", "each", "few", "first",                   
                          "for", "from", "further", "get", "had", "hadn't", "has", "hasn't", "have", "haven't",                   
                          "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "however", "i",                   
                          "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "just", "let", "many",                  
                          "may", "me", "might", "must", "my", "myself", "now", "of", "off", "often", "on", "once",                   
                          "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "said", "same",                   
                          "see", "should", "since", "so", "some", "still", "such", "than", "that", "that's", "the",                   
                          "their", "theirs", "them", "themselves", "then", "there", "these", "they", "third", "this",                   
                          "those", "through", "thus", "to", "toward", "two", "under", "until", "up", "us", "was",                   
                          "wasn't", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "whose",                   
                          "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves"]

        # self.stopWords = []
        #raise Exception("Must be implemented")

    def fit(self, text_set):
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.unigram and text_set[i][j].lower() not in self.stopWords:
                    self.unigram[text_set[i][j].lower()] = index
                    self.docWordCount[text_set[i][j]] = 0
                    index += 1
                else:
                    continue
        # Add your code here!
        #raise Exception("Must be implemented")
        
    def transform(self, text):
        # Add your code here!
        # number of documents that contain the term t
        encounteredWords = []
        feature = np.zeros(len(self.unigram))
        
        for i in range(0, len(text)):
            if text[i].lower() in self.unigram:
                if(text[i].lower() not in encounteredWords):
                    encounteredWords.append(text[i].lower())
                feature[self.unigram[text[i].lower()]] += 1
                
        # /////// implementing TF /////
        # feature /= len(text)

        # /////// implementing IDF /////////

        # count how many times the word shows up for every sentence for each word
        for i in encounteredWords:
            self.docWordCount[i] += 1
        return feature
        # raise Exception("Must be implemented")

    def transform_list(self, text_set):
        # Add your code here!
        # for x,y in zip(self.unigram.keys(),self.docWordCount.keys()):
        #     if x !=y:
        #         print(x,y)
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        # IDF = log( len(text_set)/(1+docWordCount))
        # dictArr = np.array(list(self.docWordCount.values()))
        # idf = np.log(len(text_set) / (1 + dictArr))
        # for i in range(len(features)):
        #     features[i] *= idf
        return np.array(features)
        # raise Exception("Must be implemented")


        
