import numpy as np
from collections import Counter
from collections import defaultdict
from save_index_model import SaveLoad
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class Calculating:
    def CalculatingTF(tokens_set):
        tf_dict = {}
        for doc_id, doc_text in tokens_set.items():
            tf = {}
            for term in doc_text:
                tf[term] =(doc_text.count(term) / len(doc_text))
            tf_dict[doc_id] = tf
        return tf_dict
    
    def CalculatingIDF(tokens_set):
        DF = defaultdict(int)
        for doc_id, doc_text in tokens_set.items():
            for term in doc_text:
                DF[term] += 1
        n_samples = len(tokens_set)
        idf = {term: np.log(float(n_samples) / DF[term]) + 1.0 for term in DF}
        
        return idf           
  
    
    # def total_vocab_size( df,tf_idf):
    #     total_vocab_size = len(df)
    #     total_vocab = [x for x in df]
    #     # print("total_vocab")
    #     # print(total_vocab)
    #     N=len(total_vocab) 
    #     D = np.zeros((N, total_vocab_size)) #total_vocab_size is the length of DF
    #     # print("D")
    #     # print(tf_idf)    
    #     for i in tf_idf:
    #         # print(tf_idf[i])
    #         try:
    #             ind = total_vocab.index(i[1])
    #             D[i[0]][ind] = tf_idf[i]
    #         except:
    #             pass
    #     # print("D")
    #     # print(D)    
    #     return D, total_vocab,df,N    

        
    # def doc_freq( DF,word):    
    #     c = 0
    #     try:
    #         c = DF[word] 
    #     except:
    #         pass
    #     return c
    
    def CalculatingTFIDF( corpus,id):
        vectorizer = TfidfVectorizer()
        documents = list(corpus.values())
        # print("documents")
        # print(documents)
        tfidf_matrix = vectorizer.fit_transform(documents)
        print("tf-idf done")
        # print(tfidf_matrix)
        if id == 1 :
            joblib.dump(vectorizer, "vectorizer_lotte.pkl")
            SaveLoad.save_model(tfidf_matrix,id)
        if id == 2 :   
            joblib.dump(vectorizer, "vectorizer_antique.pkl")
            SaveLoad.save_model(tfidf_matrix,id) 
        
        return tfidf_matrix



