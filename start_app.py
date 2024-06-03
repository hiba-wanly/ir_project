import csv
import json
import sqlite3
import nltk
from nltk.stem import PorterStemmer 
# from nltk.tokenize import word_tokenize
from nltk.tokenize import  word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import stopwords
# from spellchecker import SpellChecker 
from nltk.tokenize import word_tokenize
from typing import List  # Import the List type from the typing module
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from clustering import Clustering
# from doc_2_vec import DOC2VEC
from new.calculating import Calculating
from new.cosine_2 import CosineSimilarity
from new.precision_and_recall import PrecisionandRecall
from new.reading_ground import ReadingGround
from new.test_pro_2 import TestProcessing
from new.text_to_tokens import TestToToken
from save_index_model import SaveLoad 
from read_dataset import ReadDataset 
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import math
from _init_ import create_app
from flask import Blueprint, jsonify, request
from sklearn.metrics.pairwise import cosine_similarity
from _init_ import database_path


app  = create_app()


# read data set from tsv
read_data = [
            ['343638', 'Yes, everything does have an infinitesimal amount of uncertainty. See this post for some discussion on it. As you say, there is always the possibility for events to randomly play out in an unexpected way. Even if, for a particular event, the laws of physics demand the event must play out in a certain way - we can never be certain of the laws of physics, nor can we ever be fully certain that our calculations using those laws were correct. Because our brains are flawed and stochastic machines, to be rational we must be slightly uncertain of everything - including even the degree of uncertainty.'],
            ['343639', "Actually, thermodynamics all by itself does not provide an arrow of time, although it is often erroneously believed to do so. Let's do a thought experiment: At time t = 0, place an ice cube (with random initial conditions) on a table in a room that is kept at room temperature. Now let the laws of physics run normally in positive time. The ice cube will almost certainly melt, as we expect. Now go back to t = 0 and instead turn on the laws of physics in negative time. With overwhelming certainty, the ice cube will melt just as it did in positive time. It is only the combination of the Second Law of thermodynamics with the low-entropy initial conditions that we observe (for instance, stars that radiate in the positive direction of time) that results in entropy's increasing."],
            ['343640', 'Is what is pious loved by the gods because it is pious, or is it pious because it is loved by the gods? - Plato stating the Euthyphro dilemma. The flaw is both god/s & reality are likely vastly beyond our understanding - see the Kurzgesacht video The Egg for an illustrative example. Judaism & Islam insist directly god is unimaginable, divinely incorporeal, & aniconic because any representation inc imagination must tragically fall short. Only Christian philosophy makes a mental toy of god'],
            ['343641', 'Let\'s take a cold, hard (quasi-Kantian) look at what you\'re suggesting. So, if there is a one-in-a-billion chance that one might end up in unbearable suffering, that means that in a world of seven billion people, seven people (give or take) will end up in unbearable suffering. So your question is whether all seven billion people on the planet should kill themselves out of fear that they might end up as one of those seven. Does that seem reasonable? We can play with these numbers all we like, but unless we invoke an absurdly terrifying world we\'d still be asking billions of people who would otherwise lead long, happy, healthy, wonderful lives to kill themselves out of fear they\'ll end up in the wrong group. And no, I\'m not trying to appeal to numerical absurdity here; I\'m merely following this down to two deeper questions: Why would we focus on the risk of terrible suffering when we could instead focus on the risk of joy, comfort, and ease? A risk is a risk is a risk...: statistics doesn\'t care, so why do we? Are we thinking universally or collapsing into selfishness? I mean, some people go into terrible spirals of suffering when they get a hangnail or fail to get a promotion at work; others face starvation, wounds, diseases, etc with composure and grace, experiencing the pain without suffering from it. Which are we? There\'s a sense to this question as though we are saying: "It\'s ok if other people suffer, because someone has to draw the short straw, but I won\'t take the risk for myself." But why is that? Is there something they have which we lack, like composure and grace? Or is there something we have that they lack, like meaningfulness or a soul? The latter seems sociopathic or narcissistic, while the former seems self-defeatist, assuming that composure and grace are out of our reach. But what is the basis for this distinction? As Abraham Lincoln once said: "Most people are about as happy as they make up their minds to be." If we stop making up our minds to be miserable, suffering wretches, that\'s half the battle.']
        ]

def Data_lotte():
     print("DONE 1")
     id = 1
     read_data2 = ReadDataset.readDataset_lotte()
     print("1")
     test_Processing = TestProcessing.dictionary(read_data2)
     print("2")
     test_token = TestToToken.testToToken(test_Processing,id)
     print("3")
     # print("test_Processing")
     # print(test_Processing)
     calculate_tf = Calculating.CalculatingTF(test_token)
     # # print("CalculatingTF")
     # # print(calculate_tf)
     calculate_idf = Calculating.CalculatingIDF(test_token)
     # print("CalculatingIDF")
     # print(cc1)
     
     calculate_tf_idf = Calculating.CalculatingTFIDF(test_Processing, id)
     # print("CalculatingTFIDF")
     # print(cc2)
     print("DONE Data_lotte")

# Data_lotte()

def Data_antique():
     print("DONE 2")
     id = 2
     read_data2 = ReadDataset.readDataset_antique()
     test_Processing = TestProcessing.dictionary(read_data2)
     test_token = TestToToken.testToToken(test_Processing,id)
     # print("test_Processing")
     # print(test_Processing)
     calculate_tf = Calculating.CalculatingTF(test_token)
     # print("CalculatingTF")
     # print(calculate_tf)
     calculate_idf = Calculating.CalculatingIDF(test_token)
     # print("CalculatingIDF")
     # print(cc1)
     
     calculate_tf_idf = Calculating.CalculatingTFIDF(test_Processing, id)
     # print("CalculatingTFIDF")
     # print(cc2)
     print("DONE Data_antique")     
  

# Data_antique()

def Evaluation():
     id = 1               
     ReadingGround.Precision_and_Recall(id)

# Evaluation() 



def Clustering_fun():
     id = 1 
     # read_data2 = ReadDataset.readDataset_antique()
     # test_Processing = TestProcessing.dictionary(read_data2)
     co = Clustering.clustering(id)

# Clustering_fun()
# Evaluation()   

def Clustering_new_data():
     id = 1
     # new_data = 'Let\'s take a cold, hard (quasi-Kantian) look at what you\'re suggesting. So, if there is a one-in-a-billion chance that one might end up in unbearable suffering, that means that in a world of seven billion people, seven people (give or take) will end up in unbearable suffering. So your question is whether all seven billion people on the planet should kill themselves out of fear that they might end up as one of those seven. Does that seem reasonable? We can play with these numbers all we like, but unless we invoke an absurdly terrifying world we\'d still be asking billions of people who would otherwise lead long, happy, healthy, wonderful lives to kill themselves out of fear they\'ll end up in the wrong group. And no, I\'m not trying to appeal to numerical absurdity here; I\'m merely following this down to two deeper questions: Why would we focus on the risk of terrible suffering when we could instead focus on the risk of joy, comfort, and ease? A risk is a risk is a risk...: statistics doesn\'t care, so why do we? Are we thinking universally or collapsing into selfishness? I mean, some people go into terrible spirals of suffering when they get a hangnail or fail to get a promotion at work; others face starvation, wounds, diseases, etc with composure and grace, experiencing the pain without suffering from it. Which are we? There\'s a sense to this question as though we are saying: "It\'s ok if other people suffer, because someone has to draw the short straw, but I won\'t take the risk for myself." But why is that? Is there something they have which we lack, like composure and grace? Or is there something we have that they lack, like meaningfulness or a soul? The latter seems sociopathic or narcissistic, while the former seems self-defeatist, assuming that composure and grace are out of our reach. But what is the basis for this distinction? As Abraham Lincoln once said: "Most people are about as happy as they make up their minds to be." If we stop making up our minds to be miserable, suffering wretches, that\'s half the battle.'
     new_data =  "are alpha and beta glucose geometric isomers?"
     co = Clustering.new_data_clustering(id,new_data)

# Clustering_new_data()


def cos_with_without_clus():
     id = 2
     # documents = ReadDataset.readDataset_lotte()
     get_query = 'What causes severe swelling and pain in the knees?'
     # get_query = 'what is the enthalpy change for the reverse reaction?'
     ranked_results = CosineSimilarity.cosine_similarity(10,get_query,id)
     # ranked_results = CosineSimilarity.cosine_similarity_cluster(10,get_query,id,documents)########
     for doc in ranked_results:
         print(doc)

# cos_with_without_clus()





# read_data2 = ReadDataset.readDataset_lotte()




@app.route("/add_query", methods=["POST"])
def add_query():
     data = request.get_json()
     get_query = data.get("query")
     get_id = data.get("id")
     print(get_query)
     # id = 1

     c3 = CosineSimilarity.cosine_similarity(10,get_query,get_id)
     print("_______________")
     print(c3[0])
     if  get_id == 1:
          read_data2 = ReadDataset.readDataset_lotte()
     if get_id == 2 :
          read_data2 = ReadDataset.readDataset_antique()
     doc_set = {}
     for l in read_data2:
          doc_set[l[0]] = l[1]
          
     top_k_doc_ids = [list(doc_set.keys())[idx] for idx in c3]
     # Print the top k document IDs
     print("Top k document IDs:")
     sorted_docIds = np.array(list(doc_set.keys()))[c3]       
     for doc_id in top_k_doc_ids:
         print(doc_id)  
            
     result = []
     db = sqlite3.connect(database_path)
     cursor = db.cursor()
     if get_id ==  1:
          for i in c3:
               sql2 = f"""select documents from data_set_lotte WHERE  documents_id LIKE '{i}%'"""
               cursor.execute(sql2)
               dd = cursor.fetchone()
               result.append(dd)
     if get_id == 2:
          for i in c3:
               sql2 = f"""select documents from data_set_antique WHERE  documents_id LIKE '{i}%'"""
               cursor.execute(sql2)
               dd = cursor.fetchone()
               result.append(dd)
                    
     print(result)     

     return jsonify({"mes":result}) 



if __name__ == "__main__":   
     # with app.app_context():
     #    db.create_all()
     app.run(debug=True,host='0.0.0.0',port=8000)
