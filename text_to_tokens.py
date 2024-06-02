from nltk.tokenize import word_tokenize
import numpy as np
from collections import defaultdict

from save_index_model import SaveLoad

class TestToToken:
    def testToToken( processed_set, id ):
        tokens_set={}
        doc_token_id=""
        doct_token_text=""
        
        for i in processed_set:
            doc_token_id=i
            tokens_set[doc_token_id]=word_tokenize(processed_set[str(i)])
        print("done TestToToken")
        # print(tokens_set)

        inverted_index = defaultdict(list)
        for docId, doc in processed_set.items():
            doc_terms = set(word_tokenize(doc))
            for term in doc_terms:
                inverted_index[term].append(docId)
        # print(inverted_index)        
        dict(inverted_index)
        SaveLoad.save_index(inverted_index, id)
        

        return tokens_set
    

        
    
