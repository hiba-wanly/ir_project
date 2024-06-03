from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words
from nltk.stem import PorterStemmer ,WordNetLemmatizer
import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math


class TestProcessing:
    def dictionary(lines):
        doc_set = {}
        for l in lines:
                doc_set[l[0]] = l[1]
        # Print something to see the dictionary structure, etc.
        print(f"Number of documents = {len(doc_set)}" + ".\n")
        # print(doc_set["343638"])
        processed_set={}
        proc_token_id=""
        proc_token_text=""
        for i in doc_set:
            doc_token_id=i
            processed_set[doc_token_id]=TestProcessing.preprocess(doc_set[str(i)])
        print("done")
        # print(processed_set["343638"])
        return processed_set
    
    def convert_lower_case(data):
        return np.char.lower(data)
    
    
    def remove_stop_words(data):
        stop_words = stopwords.words('english')
        words = word_tokenize(str(data))
        new_text = ""
        for w in words:
            if w not in stop_words and len(w) > 1:
                new_text = new_text + " " + w
        return new_text
    
    def remove_punctuation(data):
        # حذف علامات الترقيم فواصل نقاط 
        # print('  -    -  here remove_punctuations   -    -     -')
        symbols = string.punctuation
        for i in range(len(symbols)):
            data = np.char.replace(data, symbols[i], ' ')
            data = np.char.replace(data, "  ", " ")
        data = np.char.replace(data, ',', '')
        return data
    
    def remove_apostrophe(data):
        return np.char.replace(data, "'", "")
    def remove_special_chars(data):
         return re.sub(r'[^a-zA-Z0-9\s]', '', data)    
   def remove_urls(data):
      url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_+.~#?&/=]*)'
    
      if isinstance(data, bytes):
          data = data.decode('utf-8')
    
      return re.sub(url_pattern, '', data, flags=re.IGNORECASE)    
    
    def stemming(data):
        stemmer= PorterStemmer()
        
        tokens = word_tokenize(str(data))
        new_text = ""
        for w in tokens:
            new_text = new_text + " " + stemmer.stem(w)
        return new_text
    
    def wordNet_lemmatizer(data):
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(str(data))
        new_text = ""
        for w in tokens:
            new_text = new_text + " " + lemmatizer.lemmatize(w)
        return new_text
        
    def convert_numbers(data):
        tokens = word_tokenize(str(data))
        new_text = ""
        for w in tokens:
            try:
                w = num2words(int(w))
            except:
                a = 0
            new_text = new_text + " " + w
        new_text = np.char.replace(new_text, "-", " ")
        return new_text
    
    
    def preprocess(data):
        # print(data)
#         print(type(data))
        data = remove_urls(data)
        data = convert_lower_case(data)
#         data = remove_urls(data)
        data = remove_punctuation(data) #remove comma seperately
        data = remove_apostrophe(data)
        data = remove_stop_words(data)
        data = remove_special_chars(data)
        data = convert_numbers(data)
        data = wordNet_lemmatizer(data)
        
        data = remove_punctuation(data)
        data = convert_numbers(data)
        data = wordNet_lemmatizer(data)
        data = remove_punctuation(data)
        data = remove_stop_words(data)
        return data
    
