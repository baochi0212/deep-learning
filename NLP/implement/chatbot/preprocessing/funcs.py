import numpy as np 
import re
import csv
import codecs
import nltk
import spacy
import string
import pandas as pd
import time 
nltk.download('stopwords')

class Vocab:
    def __init__(self, name=None):
        self.name = name
        self.word2idx = {"PAD": 0, "SOS": 1, "EOS": 2}
        self.idx2word = {1: "SOS", 0: "PAD", 2: "EOS"} #3 special keywords
        self.num_words = 3
        self.word_counts = {} #SOS, EOS, PAD

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
    def add_word(self, word):
        if word in self.word2idx.keys():
            self.word_counts[word] += 1
        else:
            self.word2idx[word] = self.num_words
            self.idx2word[self.num_words] = word
            self.word_counts[word] = 1 
            self.num_words += 1

class normalize_funcs:
    def __init__(self, rareword=None):
        self.punctuation = string.punctuation
        self.rareword = rareword
    def lower_case(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', self.punctuation))

    def remove_rareword(self, text):
        return ' '.join(word for word in text.split() if word not in self.rareword)
    def trimming(self, input, target, min_freq=3):
        counter = {}
        for arg in [input, target]:
            for line in arg:
                for word in line.split():
                    if word in counter:
                        counter[word] += 1
                    else:
                        counter[word] = 1

        #min freq = 3

        rare_words = []
        for key, value in counter.items():
            if value < min_freq:
                rare_words.append(key)
        #trimming
        new_input = []
        new_target = []
        for line1, line2 in zip(input, target):
            if word in line1 or word in line2:
                continue
            else:
                new_input.append(line1)
                new_target.append(line2)
        return new_input, new_target
                


if __name__ == '__main__':
    normalize = normalize_funcs()
    text = 'A good day'
    print(normalize.lower_case(text))
    input = ['a b c', 'c d e ']
    target = ['a a a', 'a a a aaaaaaa aa a a a']
    normalize.trimming(input, target)
