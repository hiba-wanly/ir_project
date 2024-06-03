import joblib
# from new.cosine_2 import CosineSimilarity
from new.test_pro_2 import TestProcessing
from save_index_model import SaveLoad
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


class Clustering:
    # def clustering2(id):
    #     X=SaveLoad.load_model(id)
    #     kmeans = KMeans(n_clusters=5, random_state=42)
    #     cluster_labels = kmeans.fit_predict(X)
    #     cluster_centers = kmeans.cluster_centers_
    #     cluster_index = {}
    #     for i, label in enumerate(cluster_labels):
    #         if label not in cluster_index:
    #             cluster_index[label] = []
    #         cluster_index[label].append(i)
    #     # Use t-SNE to project the documents into a 2D space
    #     tsne = TSNE(n_components=2, random_state=42)
    #     X_tsne = tsne.fit_transform(X)
        
    #     # Plot the documents with their cluster labels
    #     plt.figure(figsize=(10, 8))
    #     scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels)
    #     plt.legend(scatter.legend_elements()[0], [f"Cluster {i}" for i in range(5)], loc="upper left")
    #     plt.title("Document Clustering Visualization")
    #     plt.xlabel("t-SNE Dimension 1")
    #     plt.ylabel("t-SNE Dimension 2")
    #     plt.show()    
    #     return cluster_centers  ,cluster_index

    def clustering(id):
        print("clustering here")
        tf_idf_load=SaveLoad.load_model(id)
        print("1")
        loaded_matrix  = tf_idf_load
        if id == 1 :
            loaded_vectorizer = joblib.load("vectorizer_lotte.pkl")
        if id == 2 :   
            loaded_vectorizer = joblib.load("vectorizer_antique.pkl")

        # Plot the documents before clustering
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(loaded_matrix)
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1])#, s=50)
        plt.legend()
        plt.title("Documents Before Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.savefig("documents_before_clustering.png")
        plt.show()
        
        print("2")
        num_clusters = 3
        print("3")
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        print("4")
        # Fit the model to the data
        kmeans.fit(loaded_matrix)##
        print("5")

        # Get the cluster labels for each data point
        labels = kmeans.labels_ #kmeans.labels_ 
        print("6")
        # Get the cluster centroids
        centroids = kmeans.cluster_centers_
        
        # Print the results
        print("Cluster labels:")
        print(labels)
        print("\nCluster centroids:")
        print(centroids)
        # Calculate the silhouette score
        # silhouette_avg = silhouette_score(loaded_matrix, kmeans.labels_)
        
        # print(f"The average silhouette score is: {silhouette_avg:.2f}")
        # Use PCA to project the data into 2D space
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(loaded_matrix)
        
        # Create a scatter plot of the data points, colored by their cluster assignments
        plt.figure(figsize=(10, 8))##
        # for i in labels:
            # plt.scatter(X_pca[cluster_labels == i , 0] , X_pca[cluster_labels == i , 1] , cluster_labels = i)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
        plt.scatter(centroids[:,0], centroids[:,1],color='red',marker='+',s=200,label='centroid')
        plt.legend()
        plt.title('K-Means Clustering Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig("documents_after_clustering.png")
        plt.show()
        return centroids,labels

        





    def new_data_clustering(id, new_data):
        print("Clustering here")
        tf_idf_load = SaveLoad.load_model(id)
        print("1")
        loaded_matrix = tf_idf_load
        if id == 1:
            loaded_vectorizer = joblib.load("vectorizer_lotte.pkl")
        elif id == 2:
            loaded_vectorizer = joblib.load("vectorizer_antique.pkl")

        # Plot the documents before clustering
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(loaded_matrix)
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1])
        plt.legend()
        plt.title("Documents Before Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.savefig("documents_before_clustering.png")

        print("2")
        num_clusters = 3
        print("3")
        kmeans = KMeans(n_clusters=num_clusters, n_init=100, random_state=52)
        print("4")
        # Fit the model to the data
        cluster_labels = kmeans.fit(loaded_matrix)
        print("5")

        # Get the cluster labels for each data point
        labels = cluster_labels.labels_
        print("6")
        # Get the cluster centroids
        centroids = kmeans.cluster_centers_

        # Print the results
        print("Cluster labels:")
        print(labels)
        print("\nCluster centroids:")
        print(centroids)

        # Use PCA to project the data into 2D space
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(loaded_matrix)

        # Create a scatter plot of the data points, colored by their cluster assignments
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
        plt.scatter(pca.components_[0], pca.components_[1], s=100, c='red', label='Cluster Centroids')
        plt.legend()
        plt.title('K-Means Clustering Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig("documents_after_clustering.png")
        plt.show()

        # Preprocess the new data
        new_data_pro = TestProcessing.preprocess(new_data)
        new_data_vec = loaded_vectorizer.transform([new_data_pro])

        # Find the top k most similar centroids
        k = 3
        cosine_sims = [cosine_similarity([new_data_vec[0]], [centroid])[0][0] for centroid in centroids]
        top_k_indices = np.argsort(cosine_sims)[-k:][::-1]
        top_k_centroids = centroids[top_k_indices]

        # Assign the new document to the cluster with the most similar centroid
        new_labels = [labels[i] for i in top_k_indices]
        print(f"The new document has been assigned to clusters {new_labels}")

        # Find the most similar documents in the top k clusters
        relevant_tfidf = []
        for i in top_k_indices:
            relevant_tfidf.extend(loaded_matrix[np.where(labels == i)[0]])
        cosine_sims = cosine_similarity(new_data_vec, relevant_tfidf).flatten()
        top_k_indices = cosine_sims.argsort()[-10:][::-1]
        print(cosine_sims)
        print("DDDDDDDDDDDDDDDDDDDDD")
        print(top_k_indices)

        return None
        # print(similarities) 
       
        
        
    def cosine_similarity11(doc1, doc2, cluster_labels,cluster_similarities,similarity_threshold):
        cluster_x = cluster_labels[0]
        cluster_y = cluster_labels[1]
        
        # Check if clusters are similar enough to warrant further comparison
        if cluster_similarities[cluster_x, cluster_y] < similarity_threshold:
            return 0.0
        
        # Compute cosine similarity between documents
        return cosine_similarity(doc1, doc2)       