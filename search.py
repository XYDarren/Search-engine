import os
import sys
import string
import json
import nltk
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.metrics import edit_distance
from nltk.corpus import words


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

def normalize_word(word, pos, key):
    if(key == 1):
        return {word.lower()} 
    word = word.lower()
    if is_ambiguous(word):
        noun_form = lemmatizer.lemmatize(word, pos='n')
        verb_form = lemmatizer.lemmatize(word, pos='v')
        return {noun_form, verb_form} if noun_form != verb_form else {noun_form}
    if pos.startswith('N'):
        return {lemmatizer.lemmatize(word, pos='n')}
    if pos.startswith('V'):
        return {lemmatizer.lemmatize(word, pos='v')}
    return {word}

def minimum_distance_with_path(positionlist):
    # print(positionlist)
    dp = [(pos, 0, [pos]) for pos in positionlist[0]]
    
    for i in range(1, len(positionlist)):
        next_dp = []
        
        for pos2 in positionlist[i]:
            min_distance = float('inf')
            best_path = []
            for pos1, total_distance, path in dp:
                distance = abs(pos1 - pos2)
                current_distance = total_distance + distance
                if current_distance < min_distance:
                    min_distance = current_distance
                    best_path = path + [pos2]
            
            next_dp.append((pos2, min_distance, best_path))
        
        dp = next_dp
    
    _, shortest_distance, shortest_path = min(dp, key=lambda x: x[1])
    return shortest_path, shortest_distance

    

def ranking(allpositionlist, doclist):
    result = []
    for i, positionlist in enumerate(allpositionlist):
        shortest_path, shortest_path_distance = minimum_distance_with_path(positionlist)
        result.append((doclist[i], shortest_path_distance, shortest_path))
    
    result = sorted(result, key=lambda x: (x[1], x[0]))
    # print(result)
    
    return result
                    


def find_candidates(term, max_distance=2):
    candidates = []
    for indexed_term in index_terms:
        if edit_distance(term.lower(), indexed_term.lower()) <= max_distance:
            candidates.append(indexed_term)
    return candidates

def find_best_candidate(term):
    candidates = find_candidates(term)
    if candidates:
        return sorted(candidates, key=lambda x: (edit_distance(term, x), x))[0]
    else:
        return None
    
    

def read_specific_line(file_path, line_number):
    with open(file_path, 'r') as file:
        lines = file.readlines()  
        if line_number <= len(lines): 
            return lines[line_number].rstrip()  
        else:
            return None  
        
def clean_word(word):
    word = re.sub(r'(.)\1*$', r'\1', word)
    word = re.sub(r'(e*s*)$', '', word)
    return word

if __name__ == "__main__":
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('words', quiet=True)
    english_words = set(words.words())
    index_folder = sys.argv[1]
    index_file_name = os.path.join(index_folder, "index.json")
    index_file = open(index_file_name)
    index_dict = json.load(index_file)
    index_terms = set(index_dict.keys())
    punctuation_list = [char for char in string.punctuation if char != '.' and char !=',']
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    document_folder = index_dict['document_path']
    # print(document_folder)
    while(True):
        try:
            user_input = input()
            if user_input == "": 
                break
        except EOFError:
            break

        tokens = word_tokenize(user_input)
        query = []
        key = 0
        cout = 0
        if(user_input.startswith('>')):
            cout = 1
        for token,pos in pos_tag(tokens):
            if (pos == 'POS'):
                continue
            if (len(token) == 1 and token in string.punctuation):
                continue
            if (is_decimal(token)):
                continue
            if (has_punctuation(token)):
                words = re.split(r'\W+', token)
                for word in words:
                    token_list = normalize_word(word,pos,key)
                    for element in token_list:
                        query.append(element)
                continue
            if (len(token) > 1 and '.' in token):
                token = token.replace('.', '')
                key = 1
            if (len(token) > 1 and ',' in token):
                token = token.replace(',', '')
            token_list = normalize_word(token,pos,key)
            key = 0
            for element in token_list:
                query.append(element)
        corrected_query = []
        for term in query:
            if term in index_terms:
                corrected_query.append(term)
            else:
                corrected = find_best_candidate(term)
                if corrected:
                    corrected_query.append(corrected)
                else:
                    corrected = find_best_candidate(clean_word(term))
                    if corrected:  
                        corrected_query.append(corrected) 
                    else:
                        corrected_query.append(term)
        query = corrected_query
        # print(query)
        first = query[0]
        postinglist = index_dict[first]
        result = postinglist.keys()
        positionlist = []
        allpositionlist = []
        elementpositionlist = []
        comparelist = []
        for element in query[1:]:
            postinglist = index_dict[element]
            temp = postinglist.keys()
            result = list(set(result) & set(temp))
        if(len(query)==1 and cout ==0):
            for doc in result:
                print(doc)
            continue
        if(len(query)==1 and cout == 1):
            element = query[0]
            for doc in result:
                print('> ' + doc)
                data_path = document_folder + '/' + doc
                postinglist = index_dict[element]
                for position in postinglist[doc]:
                    line = read_specific_line(data_path,position[0])
                    break
                print(line)
            continue
        for doc in result:
            for element in query:
                postinglist = index_dict[element]
                for position in postinglist[doc]:
                    positionlist.append(position[1])
                elementpositionlist.append(positionlist)
                positionlist = []
            allpositionlist.append(elementpositionlist)
            elementpositionlist = []
        result = ranking(allpositionlist,result)
        if(cout == 0):
            for j in result:
                print(j[0])
        if(cout == 1):
            lineset = set()
            # print("1")
            for j in result:
                print('> ' + j[0])
                data_path = document_folder + '/' + j[0]
                for element in query:
                    postinglist = index_dict[element]
                    # print(postinglist)
                    for position in postinglist[j[0]]:
                        # print(position)
                        if(position[1] in j[2]):
                            lineset.add(position[0])
                # print(lineset)
                lineset = sorted(lineset)
                # lineset = set(lineset)
                for l in lineset:
                    # data_path = document_folder + '/' + j[0]
                    line = read_specific_line(data_path,l)
                    print(line)
                lineset.clear()
                lineset = set(lineset)
                # break
                    
        # print(result)
            






