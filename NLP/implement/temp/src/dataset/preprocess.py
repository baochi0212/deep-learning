import numpy as np 
import re
import csv
import codecs
import nltk
import spacy
import string
import pandas as pd
import sys
PATH = os.environ['dir']
sys.path.append(PATH + "/src")
from utils import Vocab, normalize_funcs
import os
from pathlib import Path

#basename is the name of file, dirname is dirpath to file
PATH = os.path.join(Path(os.path.dirname(__file__)).parent, 'data')

#get the sentences by id
def extract_lines(file):
    dir = os.path.join(PATH, file)
    with open(dir, 'r', encoding='iso-8859-1') as f:
        lines = f.readlines()
    save = {}
    for line in lines:
        id = line.split(' +++$+++ ')[0]
        sentence = line.split(' +++$+++ ')[4]
        save[id] = sentence
    return save 
#input and target pair in conversations 
def extract_pairs(file1, file2):
    input = []
    target = []
    dir = os.path.join(PATH, file1)
    sentences = extract_lines(file2)
    with open(dir, 'r', encoding='iso-8859-1') as f:
        lines = f.readlines()
    #get id of sentence
    id_pattern = 'L[0-9]+'
    for line in lines:
        id = re.findall(id_pattern, line)
        #get the pairs
        for i in range(len(id)):
            if i % 2 == 0 and i + 1 != len(id):
                input.append(sentences[id[i]].strip())
                target.append(sentences[id[i + 1]].strip())
    return input, target
#turn to format csv
def extract_csv(input, target, path_to_save, MAX_LEN=10):
    normalize = normalize_funcs()
    for i in range(len(input)):
        for line in [input, target]:
            line[i] = line[i].lower()
            line[i] = normalize.remove_punctuation(normalize.lower_case(line[i]))

    input, target = normalize.trimming(input, target)
    data = pd.DataFrame({'input': input, 'target': target})
    drop_rows = []
    for i in range(len(data)):
        input, target = data.iloc[i, :]
        if len(input)*len(target) == 0 or (len(input.split()) > MAX_LEN) or (len(target.split()) > MAX_LEN):
            drop_rows.append(i)

    data.drop(drop_rows, inplace=True)
    data.reset_index(inplace=True, drop=True)
    data.to_csv(os.path.join(PATH, path_to_save), index=False, sep='@')
    return data
    



if __name__ == '__main__':
    #get data
    normalizer = normalize_funcs()
    PATH = os.environ['dir']
    line_path = os.path.join(PATH, 'data/raw/cornell movie-dialogs corpus/movie_lines.txt')
    conversation_path = os.path.join(PATH, 'data/raw/cornell movie-dialogs corpus/movie_conversations.txt')
    csv_path = os.path.join(PATH, 'data/raw/cornell movie-dialogs corpus/pair_df.csv')
    print(extract_lines(line_path)['L104'])
    input, target = extract_pairs(conversation_path, line_path)
    print("----INPUT, TARGET", input[0], target[0])
    # print("----Trimming", normalizer.trimming(input[:10000], target[:10000]))
    print("")
    print("----data", extract_csv(input, target, csv_path))
    #build the vocab and futher processing
    voc = Vocab('movie')
    data = pd.read_csv(os.path.join(PATH, csv_path), sep='@')
    print('OFFICIAL DATA', data)
    print('DF', len(data))
    new_input = []
    new_target = []
    for i in range(len(data)):
        input, target = data.iloc[i, :]
        voc.add_sentence(input)
        voc.add_sentence(target)
        new_input.append(input)
        new_target.append(target)
    print('VOCAB', voc.name)
    print('NUMWORDS', voc.num_words)
    # counter = {}
    # for arg in [new_input, new_target]:
    #     for line in arg:
    #         for word in line.split():
    #             if word in counter:
    #                 counter[word] += 1
    #             else:
    #                 counter[word] = 1
    # print('WTF', len(counter.keys()))