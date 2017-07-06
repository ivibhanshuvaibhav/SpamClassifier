import sys
reload(sys)
sys.setdefaultencoding("latin-1")

import os
from nltk.tokenize import word_tokenize
import pickle
import nltk
from nltk import NaiveBayesClassifier
import random

enron_path = "Enron Spam"

ham_list = []
spam_list = []

def create_word_features(words):
    my_dict = dict([(word, True) for word in words])
    return my_dict

for dir, subdir, files in os.walk(enron_path):

    if os.path.split(dir)[1] == "ham":
        for filename in files:
            with open(os.path.join(dir, filename)) as f:
                data = f.read()
                words = word_tokenize(data)
                ham_list.append((create_word_features(words),"ham"))


    if os.path.split(dir)[1] == "spam":
        for filename in files:
            with open(os.path.join(dir, filename)) as f:
                data = f.read()
                words = word_tokenize(data)
                spam_list.append((create_word_features(words),"spam"))

combined_list = ham_list + spam_list

random.shuffle(combined_list)

testing_size = int(len(combined_list) * 0.7)

testing_data = combined_list[testing_size:]
training_data = combined_list[:testing_size]

print "Combined list size: ", len(combined_list)
print "Testing data size: ", len(testing_data)
print "Training data size: ", len(training_data)

classifier = NaiveBayesClassifier.train(training_data)
print("Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_data)) * 100)
classifier.show_most_informative_features(20)

save_classifier = open("naive_bayes_classifier.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()