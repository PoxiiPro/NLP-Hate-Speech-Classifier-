import numpy as np
import utils 
# build own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# use the models form sklearn packages to check the performance of own models

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the number of sentences
        """
        pass


class AlwaysPreditZero(HateSpeechClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# class NaiveBayesClassifier(HateSpeechClassifier):
#     """Naive Bayes Classifier
#     """
#     def __init__(self):
#         # Add your code here!

#         # for smoothing
#         alpha = 1
#         self.alpha = alpha

#         # words selected as features for training
#         self.wordFeatures = None

#         self.priorPHate = None
#         self.priorPNotHate = None

#         self.conditionalProbability = None

#         self.isHateCount = None
#         self.notHateCount = None

#         self.hateWordList = []
#         self.notHateWordList = []

#         self.conProbHate = []
#         self.conProbNotHate = []
        
#         # each example is labeled as 1 (hatespeech) or 0 (Non-hatespeech)
#         self.label = None

#         #creates an object to call functions from UnigramDeature class 
#         self.unigramObject = utils.UnigramFeature()
#         self.top10Hate = None
#         self.top10NotHate = None
        

#     def fit(self, X, Y):

#         # Count of total positive/negative labels    
#         # each example is labeled as 1 (hatespeech) or 0 (Non-hatespeech)
#         #self.isHateCount = sum([y for y in Y.array if y == 1])
#         #self.notHateCount = len(Y) - self.isHateCount

#         # Set up lists for counting the number of occurences in both hate and non
#         # hate speech data
#         self.hateWordList = np.array([0.0 for _ in range(len(X[0]))])
#         self.notHateWordList = np.array([0.0 for _ in range(len(X[0]))])

#         # Calculate the conditional probability 
#         self.conProbHate = np.array([0.0 for _ in range(len(X[0]))])
#         self.conProbNotHate = np.array([0.0 for _ in range(len(X[0]))])


#         # Set up lists to calculate prior probability


#         # For each feature word, count the number of occurences in both hate and non
#         # hate speech data
#         y_index = 0 
#         for x in X:
#                 #if the X list is labeled as a hatespeech then add the word to hateWordCount
#             if(Y.array[y_index] == 1):
#                 self.hateWordList += x
#             elif(Y.array[y_index] == 0):
#                 self.notHateWordList += x
#             y_index += 1
        
#         self.isHateCount = sum(self.hateWordList)
#         self.notHateCount = sum(self.notHateWordList)
#         self.priorPHate = self.isHateCount / len(X[0])  
#         self.priorPNotHate =  self.notHateCount / len(X[0])

#         # print(self.hateWordList)

#         # add alpha into each count
#         for i in range(len(self.hateWordList)):
#             self.hateWordList[i] = self.hateWordList[i] + self.alpha

#         # add alpha into each count
#         for j in range(len(self.notHateWordList)):
#             self.notHateWordList[j] = self.notHateWordList[j] + self.alpha

#         # print(self.hateWordList)

#         # Calculate the conditional probability
#         # self.conProbHate = self.hateWordList/ (self.isHateCount * self.alpha)
#         # self.conProbNotHate = self.notHateWordList/ (self.notHateCount * self.alpha)

#         # Calculate the conditional probability
#         self.conProbHate = (self.hateWordList + self.alpha) / (self.isHateCount + self.alpha * len(self.hateWordList))
#         self.conProbNotHate = (self.notHateWordList + self.alpha) / (self.notHateCount + self.alpha * len(self.notHateWordList))

        
#         # Ratio P(w|1)/P(w|0) for deliberable  
#         self.top10Hate = (-np.log(self.conProbHate)/np.log(self.conProbNotHate)).argsort()[:10]
#         self.top10NotHate = (np.log(self.conProbHate)/np.log(self.conProbNotHate)).argsort()[:10]
#         print(self.top10Hate)
#         print(self.top10NotHate)
    
#     def predict(self, X):
#         # Add your code here!
#         #print(X)
#         prediction = []
#         for i in range(len(X)):
#             indices = np.nonzero(X[i])
#             t = np.take(self.conProbHate,indices=indices)
#             tt = self.priorPHate * np.product((np.take(t,np.nonzero(t))))
#             y = np.take(self.conProbNotHate,indices=indices)
#             yy = self.priorPNotHate * np.product((np.take(y,np.nonzero(y))))
#             ttt = np.log(tt)/np.log(tt+yy)
#             yyy = np.log(yy)/np.log(tt+yy)
#             toAppend = 0 if yyy > ttt else 1
#             prediction.append(toAppend)
#         #print(prediction)
#         return prediction

class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        # Add your code here!

        # for smoothing
        alpha = 1
        self.alpha = alpha

        # words selected as features for training
        self.wordFeatures = None

        self.priorPHate = None
        self.priorPNotHate = None

        self.conditionalProbability = None

        self.isHateCount = None
        self.notHateCount = None

        self.hateWordFrequency = []
        self.notHateWordFrequency = []
        self.hateSentences = []
        self.notHateSentences = []
        self.conProbHate = []
        self.conProbNotHate = []
        
        # each example is labeled as 1 (hatespeech) or 0 (Non-hatespeech)
        self.label = None

        #creates an object to call functions from UnigramDeature class 
        self.unigramObject = utils.UnigramFeature()
        self.top10Hate = None
        self.top10NotHate = None
        

    def fit(self, X, Y):

        # Count of total positive/negative labels    
        # each example is labeled as 1 (hatespeech) or 0 (Non-hatespeech)
        #self.isHateCount = sum([y for y in Y.array if y == 1])
        #self.notHateCount = len(Y) - self.isHateCount

        # Set up lists for counting the number of occurences in both hate and non
        # hate speech data
        self.hateWordFrequency = np.array([0.0 for _ in range(len(X[0]))])
        self.notHateWordFrequency = np.array([0.0 for _ in range(len(X[0]))])

        # Calculate the conditional probability 
        self.conProbHate = np.array([0.0 for _ in range(len(X[0]))])
        self.conProbNotHate = np.array([0.0 for _ in range(len(X[0]))])


        # Set up lists to calculate prior probability


        # For each feature word, count the number of occurences in both hate and non
        # hate speech data
        y_index = 0 
        for x in X:
                #if the X list is labeled as a hatespeech then add the word to hateWordCount
            if(Y.array[y_index] == 1):
                self.hateWordFrequency += x
                self.hateSentences.append(x)
            elif(Y.array[y_index] == 0):
                self.notHateWordFrequency += x
                self.notHateSentences.append(x)
            y_index += 1
        self.hateSentences = np.array(self.hateSentences)
        self.notHateSentences = np.array(self.notHateSentences)

        self.isHateCount = sum(self.hateWordFrequency)
        self.notHateCount = sum(self.notHateWordFrequency)
        self.priorPHate = np.log(len(self.hateSentences) / len(X[0]))
        self.priorPNotHate =  np.log(len(self.notHateSentences) / len(X[0]))

        # add alpha into each count
        for i in range(len(self.hateWordFrequency)):
            self.hateWordFrequency[i] = self.hateWordFrequency[i] + self.alpha

        # add alpha into each count
        for j in range(len(self.notHateWordFrequency)):
            self.notHateWordFrequency[i] = self.notHateWordFrequency[i] + self.alpha

        # Calculate the conditional probability
        self.conProbHate = self.hateWordFrequency/ (self.isHateCount + len(X[0]))
        self.conProbNotHate = self.notHateWordFrequency/ (self.notHateCount + len(X[0]))
        
        # Ratio P(w|1)/P(w|0) for deliberable  
        self.top10Hate = (-np.log(self.conProbHate)/np.log(self.conProbNotHate)).argsort()[:10]
        self.top10NotHate = (np.log(self.conProbHate)/np.log(self.conProbNotHate)).argsort()[:10]
        print(self.top10Hate)
        print(self.top10NotHate)
    
    def predict(self, X):
        # Add your code here!
        #print(X)
        prediction = []
        for i in range(len(X)):
            indices = np.nonzero(X[i])
            t = np.take(self.conProbHate,indices=indices)
            tt = self.priorPHate + (np.log(t))*np.take(X[i],indices)
            y = np.take(self.conProbNotHate,indices=indices)
            yy = self.priorPNotHate + (np.log(y))* np.take(X[i],indices)
            tt = np.sum(tt)
            yy = np.sum(yy)
            toAppend = 0 if yy > tt else 1
            
            prediction.append(toAppend)
        #print(prediction)
        return prediction

# TODO: Implement this
class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self,lamba=0.01):
        # Add your code here!

        # the weights aka beta
        self.beta = None
        
        # x in the formula, count of occurances of vocab showing up
        self.countX = None

        # learning rate, we decided to go with 1 
        self.alpha = 0.01
        self.lamba = lamba

    def fit(self, X, Y):
        # λ = {0.0001, 0.001, 0.01, 0.1, 1, 10}
        lamba = self.lamba
        # print(X, Y)
        #print(X.shape)
        #initialize all the values of the weights and count to be 0
        self.beta = np.zeros(len(X[0])).T
        iterations = 1000
        #print(beta.shape)
        for _ in range(iterations):
                pHat = self.sigmoid(X@self.beta.T)
                if(lamba == 0):
                    self.beta = self.beta + ((self.alpha) * ( (np.dot(Y.array - pHat,X))) )

                else:
                    self.beta = self.beta + ((self.alpha) * ( (np.dot(Y.array - pHat,X))) - (lamba*self.beta)) 

                #self.beta = self.beta + ((self.alpha) * ( (np.dot(Y.array - pHat,X))) )
        # Perform gradient descent to initialize values of beta        
        #print(self.beta)
    
    def predict(self, X):
        # Not regularized
        predict = self.sigmoid(np.dot(X,self.beta))
        #print(predict)
        return predict.round()
    
    # helper function to help compute p hat
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

# OUTPUT for report later:
# before r2 regularization: alpha = .01 iterations = 1000
# ===== Train Accuracy =====
# Accuracy: 1412 / 1413 = 0.9993 
# ===== Test Accuracy =====
# Accuracy: 188 / 250 = 0.7520

# OUTPUT for report later:
# with r2 regularization: alpha = .01 iterations = 1000 lambda = 0.8
# ===== Train Accuracy =====
# Accuracy: 1385 / 1413 = 0.9802 
# ===== Test Accuracy =====
# Accuracy: 188 / 250 = 0.7520 
# ===== Dev Accuracy =====
# Accuracy: 181 / 250 = 0.7240 
# Time for training and test: 2.87 seconds

# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
# implemented custom features: Removing stopwords and TF-IDF 
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
