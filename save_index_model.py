import sqlite3
import numpy as np
from _init_ import database_path
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz


class SaveLoad:
    def apply_query(query, values):
        db = sqlite3.connect(database_path)
        cursor = db.cursor()
        cursor.execute(query, values)
        db.commit()

    def save_index(index,id):
        if id == 1:
            filename = 'index\save_index_lotte.npy'
            np.save(filename, index)
        if id == 2:    
            filename = 'index\save_index_antique.npy'
            np.save(filename, index)
        print(f"Term index saved to {filename}")
    
    
    def load_index(id):
        if id == 1:
            filename = 'index\save_index_lotte.npy'
            loaded_index = np.load(filename,allow_pickle=True)
        if id == 2:    
            filename = 'index\save_index_antique.npy'
            loaded_index = np.load(filename,allow_pickle=True)
        # print(f"Term index loaded  {loaded_index}")
        return loaded_index
    
    
    def save_model(model,id):
        if id == 1:
            # filename = 'model\save_model_lotte.npy'
            # vectorizer_config = "vectorizer_config.npy"
            # np.save(filename, model.toarray())
            filename = 'model/save_model_lotte.npz'
            save_npz(filename, csr_matrix(model))
        if id == 2:
            filename = 'model/save_model_antique.npz'
            save_npz(filename, csr_matrix(model))    
            # filename = 'model\save_model_antique.npy'
            # vectorizer_config = "vectorizer_config.npy"
            # np.save(filename, model.toarray())
        
        print(f"Term model saved to {filename}")
    
    
    def load_model(id):
        if id == 1:
            filename = 'C:\\Users\\yamen\\OneDrive\\Desktop\\IRRRRRR\\IR_Project\\model\\save_model_lotte.npz'
            tfidf_matrix = load_npz(filename)
            # filename = 'model\save_model_lotte.npy'
            # vectorizer_config = "vectorizer_config.npy"
            # loaded_model = np.load(filename)
        if id == 2:    
            filename = 'C:\\Users\\yamen\\OneDrive\\Desktop\\IRRRRRR\\IR_Project\\model\\save_model_antique.npz'
            tfidf_matrix = load_npz(filename)
            # filename = 'model\save_model_antique.npy'
            # vectorizer_config = "vectorizer_config.npy"
            # loaded_model = np.load(filename)
        
        # print(f"Term model loaded  {loaded_model}") 
        return tfidf_matrix 
    