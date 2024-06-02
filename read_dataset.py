import csv
import nltk
from nltk.stem import PorterStemmer ,WordNetLemmatizer
# from nltk.tokenize import word_tokenize
from nltk.tokenize import  word_tokenize,sent_tokenize
from nltk import pos_tag,ne_chunk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
# from spellchecker import SpellChecker 
from nltk.tokenize import word_tokenize
from typing import List  # Import the List type from the typing module
import string
import pandas as pd
import os
from save_index_model import SaveLoad

class ReadDataset:
    
    def readDataset_lotte():
        # cur = mysql.connection.cursor()
        lines = []
        file_path = os.path.join("C:", "Users", "yamen", "OneDrive", "Desktop", "DATASET", "lotte", "lotte", "science", "dev", "collection.tsv")
        with open(r"C:\\Users\\yamen\\OneDrive\\Desktop\\DATASET\\lotte\\lotte\\science\\dev\\collection.tsv",  errors='ignore' ) as file:
            # tsv_file = pd.read_csv(file, sep='\t')
            tsv_file = csv.reader(file, delimiter="\t")
            # data = list(tsv_file)
            print('- - - - - - - - - here tsv_file - - - - - - - - -') 
            # print(data)
            
            print('- - - - - - - - - here data set - - - - - - - - -') 
            for line in tsv_file:
                # print(line)
                lines.append(line) 
            # print(lines)  
            # for l in lines:
            #     # print(l[0])
            #     # print(l[1])
            #     SaveLoad.apply_query(f"INSERT INTO data_set_lotte (documents_id,documents) VALUES (?,?)",(l[0],l[1]))
            # # cur.close()
            return lines  

    def readDataset_antique():
        # cur = mysql.connection.cursor()
        lines = []
        file_path = os.path.join("C:", "Users", "yamen", "OneDrive", "Desktop", "DATASET", "antique", "antique", "collectionan.tsv")
        with open(r"C:\Users\yamen\OneDrive\Desktop\DATASET\\antique\\antique\collection.tsv" , errors='ignore') as file:
            # tsv_file = pd.read_csv(file, sep='\t')
            tsv_file = csv.reader(file, delimiter="\t")
            # data = list(tsv_file)
            print('- - - - - - - - - here tsv_file - - - - - - - - -') 
            # print(data)
            
            print('- - - - - - - - - here data set - - - - - - - - -') 
            for line in tsv_file:
                # print(line)
                lines.append(line) 
            # print(lines)  
            # for l in lines:
                # print(l[0])
                # print(l[1])
                # SaveLoad.apply_query(f"INSERT INTO data_set_antique (documents_id,documents) VALUES (?,?)",(l[0],l[1]))
            # cur.close()
            return lines  

