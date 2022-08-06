import numpy as np 
import re
import nltk
import string
import pandas as pd
import time 


import torch
import torch.nn.functional as F




nltk.download('stopwords')

#attention function
def AttentionWeight(q, k, v, mask, type="dot"):
    '''
    - q, k, v : b x n x d (batch x n_seq x dim) <k, v same>
    query (q)  infer the attention weight from keys (k) -> w (n_seq_q x n_seq_k) 
    weight (w) infer the value for query -> n_seq_q x n_seq_v
    - the scale 1/(sqrt(dk)) to reduce the variance (large values -> small gradients for SoftMax)
    - mask for Autoregressive Decoding (Masked Attention)
    '''
    if type == "dot":
        dot_product = torch.bmm(q, k.permute(0, 2, 1))
        mask_weight = F.softmax(dot_product * mask, dim=-1) 

    return torch.bmm(mask_weight/k.shape[-1]**0.5, v)

#vocab prepare class
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

#normalize words class 
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
    #trimming pair of source and target
    def trimming(self, input, target, min_freq=1):
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
        print("RARE", len(rare_words), len(counter.keys())) 
        #trimming
        new_input = []
        new_target = []
        for line1, line2 in zip(input, target):
            # for word in rare_words:
            #     if word in line1:
            #         line1 = line1.replace(word, "")
            #     if word in line2:
            #         line2 = line2.replace(word, "")
                    
    
            new_input.append(line1.strip())
            new_target.append(line2.strip())
        return new_input, new_target

if __name__ == '__main__':
    q, k, v = torch.rand(32, 3, 10), torch.rand(32, 3, 10), torch.rand(32, 3, 10)
    mask = torch.tril(torch.ones(q.shape[1], k.shape[1]), diagonal=0) #take into account current pos 
    print("attention", AttentionWeight(q, k, v, mask).shape)

    normalize = normalize_funcs()
    text = 'A good day'
    print(normalize.lower_case(text))
    input = ['i love myself', 'myself']
    target = ['myself is best version of mine', 'i love love i']
    print("trimming", normalize.trimming(input, target))

