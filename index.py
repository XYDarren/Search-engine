import os
import sys
import json
import re
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from collections import defaultdict

def has_punctuation(word):
    return any(char in punctuation_list for char in word)

def is_decimal(s):
    try:
        float(s)
        return '.' in s
    except ValueError:
        return False

def is_ambiguous(term):
    has_noun = bool(wn.synsets(term, pos=wn.NOUN))
    has_verb = bool(wn.synsets(term, pos=wn.VERB))
    return has_noun and has_verb

def normalize_word(word, pos, is_abbreviation):
    if is_abbreviation:
        return {word.lower()}
    word = word.lower()
    if is_ambiguous(word):
        return {lemmatizer.lemmatize(word, pos='n'), lemmatizer.lemmatize(word, pos='v')}
    if pos.startswith('N'):
        return {lemmatizer.lemmatize(word, pos='n')}
    if pos.startswith('V'):
        return {lemmatizer.lemmatize(word, pos='v')}
    return {word}

if __name__ == "__main__":
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    punctuation_list = [char for char in string.punctuation if char not in '. ,']
    lemmatizer = WordNetLemmatizer()
    
    document_folder = sys.argv[1]
    indexes_folder = sys.argv[2]
    os.makedirs(indexes_folder, exist_ok=True)
    
    inverted_index = defaultdict(lambda: defaultdict(list))
    doc_list = sorted(int(doc) for doc in os.listdir(document_folder))

    for doc_id in doc_list:
        position = 0
        line_number = 0
        file_name = os.path.join(document_folder, str(doc_id))
        with open(file_name, 'r') as doc:
            for line in doc:
                tokens = word_tokenize(line)
                # print(tokens)
                for token, pos in pos_tag(tokens):
                    if pos == 'POS' or (len(token) == 1 and token in string.punctuation):
                        continue
                    if is_decimal(token):
                        continue
                    if has_punctuation(token):
                        words = re.split(r'\W+', token)
                        for word in words:
                            if word:
                                normalized_tokens = normalize_word(word, pos ,False)
                                for element in normalized_tokens:
                                    inverted_index[element][doc_id].append((line_number, position))
                        position += 1
                        continue
                    is_abbreviation = len(token) > 1 and ('.' in token or ',' in token)
                    token = token.replace('.', '').replace(',', '')
                    normalized_tokens = normalize_word(token, pos, is_abbreviation)
                    for element in normalized_tokens:
                        inverted_index[element][doc_id].append((line_number, position))
                    position += 1
                line_number += 1

    # Write entire index to file at once
    inverted_index['document_path'] = document_folder
    index_file_name = os.path.join(indexes_folder, 'index.json')
    with open(index_file_name, 'w') as index_file:
        json.dump(inverted_index, index_file)
    
    # print(inverted_index)
