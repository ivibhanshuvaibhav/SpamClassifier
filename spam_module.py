import pickle
from nltk import word_tokenize

save_classifier = open("naive_bayes_classifier.pickle", "rb")
classifier = pickle.load(save_classifier)
save_classifier.close()


def create_word_features(words):
    my_dict = dict([(word, True) for word in words])
    return my_dict

def analysis(text):
    words = word_tokenize(text)
    print classifier.classify(create_word_features(words))
