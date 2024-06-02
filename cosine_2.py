import joblib
from clustering import Clustering
from new.calculating import Calculating
from new.test_pro_2 import TestProcessing
from save_index_model import SaveLoad
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
import math
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd


class CosineSimilarity:
    def cosine_sim(a, b):
        cos_sim = cosine_similarity(a, b)
        # print("cos")
        # print(cos_sim)
        return cos_sim
    

    def cosine_similarity( k, query,id):
        tf_idf_load=SaveLoad.load_model(id)
        loaded_matrix  = tf_idf_load
        if id == 1 :
            loaded_vectorizer = joblib.load("C:\\Users\\yamen\\OneDrive\\Desktop\\IRRRRRR\\IR_Project\\vectorizer_lotte.pkl")
        if id == 2 :   
            loaded_vectorizer = joblib.load("C:\\Users\\yamen\\OneDrive\\Desktop\\IRRRRRR\\IR_Project\\vectorizer_antique.pkl")
        
        preprocessed_query = TestProcessing.preprocess(query)
        print(preprocessed_query)
        tokens = word_tokenize(preprocessed_query)
        # print(tokens)
        #print("\nQuery:", query)
        #########
        query_documents = [preprocessed_query]
        print(query_documents)
        query_vec = loaded_vectorizer.transform(query_documents)
        print("query_vec")
        print(query_vec)
        cosine_sim = CosineSimilarity.cosine_sim(query_vec, loaded_matrix).flatten()
        print("cosine_sim")
        print(cosine_sim)
        
        top_k_indices = cosine_sim.argsort()[-k:][::-1]
        print("top_k_indices")
        print(top_k_indices)

        
        print("Most similar Dpocuments-IDs : ")


        return top_k_indices
    
    # def gen_vector(tokens,total_vocab,DF,N):

        # Q = np.zeros((len(total_vocab)))
        
        # counter = Counter(tokens)
        # words_count = len(tokens)
        
        # query_weights = {}
        
        # for token in np.unique(tokens):
            
        #     tf = counter[token]/words_count
        #     df = Calculating.doc_freq(DF,token)
        #     idf = math.log((N+1)/(df+1))
        
        #     try:
        #         ind = total_vocab.index(token)
        #         Q[ind] = tf*idf
        #     except:
        #         pass  
  
        # return Q
    

    def cosine_similarity_cluster( k, query,id,documents):
        tf_idf_load=SaveLoad.load_model(id)
    
        centroids,labels = Clustering.clustering(id)
        if id == 1 :
            loaded_vectorizer = joblib.load("vectorizer_lotte.pkl")
        if id == 2 :   
            loaded_vectorizer = joblib.load("vectorizer_antique.pkl")
        
        preprocessed_query = TestProcessing.preprocess(query)
        print(preprocessed_query)
        tokens = word_tokenize(preprocessed_query)
        # print(tokens)
        #print("\nQuery:", query)
        #########
        query_documents = [preprocessed_query]
        print(query_documents)
        query_vec = loaded_vectorizer.transform(query_documents)
        print("query_vec")
        print(query_vec)
        cosine_sim = CosineSimilarity.cosine_sim(query_vec, centroids).flatten()
        print("cosine_sim")
        print(cosine_sim)
        top_cluster_ids = np.argsort(cosine_sim)[::-1]
        print(top_cluster_ids)
        relevant_tfidf = tf_idf_load[top_cluster_ids[0]]
        cosine_sim2 = cosine_similarity(query_vec, [relevant_tfidf]).flatten()
        print("cosine_sim22222222")
        print(cosine_sim2)
        top_k_indices2 = cosine_sim2.argsort()[-k:][::-1]
        print("top_k_indices22222222")
        print(top_k_indices2)
        ################
        # cosine_sim22 = CosineSimilarity.cosine_sim(query_vec, centroids[top_cluster_ids[0]]).flatten()
        # print("cosine_sim22")
        # print(cosine_sim22)
        # # Get the document IDs for the closest cluster
        # closest_cluster_docs = [documents[i][0] for i, label in enumerate(labels) if label == top_cluster_ids[0]]
        # Sort the documents by cosine similarity to the query
        # sorted_docs = sorted(zip(closest_cluster_docs, cosine_sim[labels == top_cluster_ids]), key=lambda x: x[1], reverse=True)
        
        # # Get the top 10 most relevant documents
        # top_10_docs = [doc[0] for doc in sorted_docs[:10]]
        
        # print(f"Top 10 most relevant documents: {', '.join(top_10_docs)}")
        
        # print(f"The new query is closest to cluster {top_cluster_ids[0]}")
        # print(f"Cosine similarity to cluster {top_cluster_ids[0]}: {cosine_sim[top_cluster_ids[0]]:.2f}")
        # print(f"Documents in the closest cluster: {', '.join(closest_cluster_docs)}")
        #################
        # Get the document IDs for the top 10 closest clusters
        # top_10_docs = []
        # for cluster_id in top_cluster_ids:
        #     closest_cluster_docs = [documents[i][0] for i, label in enumerate(labels) if label == cluster_id]
        #     cluster_cosine_sims = cosine_sim[labels == cluster_id]
        #     sorted_docs = sorted(zip(closest_cluster_docs, cluster_cosine_sims), key=lambda x: x[1], reverse=True)
        #     top_10_docs.extend([doc[0] for doc in sorted_docs[:min(10, len(sorted_docs))]])
        
        # print(f"Top 10 most relevant documents: {', '.join(top_10_docs[:10])}")
        ###############
        # Retrieve documents from the relevant clusters
        # top_documents = []
        # for cluster_id in top_cluster_ids:
        #     top_documents.extend([documents[idx] for idx in cluster_index[cluster_id]])
        #  # Rank the documents within the relevant clusters
        # ranked_documents = sorted(top_documents, key=lambda x: cosine_similarity(loaded_vectorizer.transform([x]), query_vec), reverse=True)
        # print("ranked_documents")
        # print(ranked_documents)
        # return ranked_documents


        # top_documents = []
        # for cluster_id in set(labels):
        #     cluster_docs = [doc for i, doc in enumerate(documents) if labels[i] == cluster_id]
        #     cluster_vecs = loaded_vectorizer.transform(cluster_docs)[0]

        #     similarities = [cosine_sim(query_vec, doc_vec) for doc_vec in cluster_vecs]
        #     top_indices = np.argsort(similarities)[::-1][:k]
        #     top_cluster_docs = [cluster_docs[i] for i in top_indices]
        #     top_documents.extend(top_cluster_docs)
        # print(top_documents[:k])
        


        
      



       
    
    # def gen_vector(tokens,total_vocab,DF,N):

        # Q = np.zeros((len(total_vocab)))
        
        # counter = Counter(tokens)
        # words_count = len(tokens)
        
        # query_weights = {}
        
        # for token in np.unique(tokens):
            
        #     tf = counter[token]/words_count
        #     df = Calculating.doc_freq(DF,token)
        #     idf = math.log((N+1)/(df+1))
        
        #     try:
        #         ind = total_vocab.index(token)
        #         Q[ind] = tf*idf
        #     except:
        #         pass  
  
        # return Q
    