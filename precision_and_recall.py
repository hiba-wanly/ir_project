import sqlite3
from new.cosine_2 import CosineSimilarity
import numpy as np
from _init_ import database_path
import csv

class PrecisionandRecall:
    def precision_and_recall(doc_set,qry_set,rel_set,id):
        Q = len(qry_set)  
        cumulative_reciprocal = 0 
        k = 9
        precision_list=[]
        recall_list=[]
        accuracy_list=[]
        k1 = [1,4,5]
        ap = []
        for i in range(1,len(doc_set)):
            try:
                
                result_from_cosine= CosineSimilarity.cosine_similarity(10 ,qry_set[str(i)],id)
                print("result_from_cosine")
                print(result_from_cosine)
                result_from_ground_truth=rel_set[str(i)][0]
                print("result_from_ground_truth")
                print(result_from_ground_truth)
                reciprocal = 1 / result_from_ground_truth[0]
                cumulative_reciprocal += reciprocal
                # ap_num = 0
                # for x in range(k):
                #     act_set = result_from_ground_truth
                #     result_from_cosine= result_from_cosine
                #     pred_set = set(result_from_cosine)
                #     precision_at_k = len(act_set & result_from_cosine) / (x+1)
                #     if result_from_cosine[x] in act_set:
                #         rel_k = 1
                #     else:
                #         rel_k = 0
                #     # calculate numerator value for ap
                #     ap_num += precision_at_k * rel_k
                # ap_q = ap_num / len(act_set)
                # print(f"AP@{k}_{i+1} = {round(ap_q,2)}")
                # ap.append(ap_q)

#It computes the intersection between the sets of results obtained from the cosine similarity method (result_from_cosine) and the ground truth (expected) results (result_from_ground_truth).
                true_Positive=len(set(result_from_cosine) & set(result_from_ground_truth)) #set(a) & set(b) gives us intersection between a and b
#It computes the set difference (elements in result_from_cosine but not in result_from_ground_truth).
                false_Positive=len(np.setdiff1d(result_from_cosine , result_from_ground_truth))
#It computes the set difference (elements in result_from_ground_truth but not in result_from_cosine)                
                false_Negative=len(np.setdiff1d(result_from_ground_truth , result_from_cosine))
#It subtracts the sum of true positives, false negatives, and false positives from the total number of documents (len(doc_set))                
                true_negative= ( len(doc_set) -  (true_Positive + false_Negative + false_Positive) )

                print("true psotive",true_Positive)
                print("false negative",false_Negative)
                print("false psotive",false_Positive)
                print("true negative",true_negative)
                
                try:
                    precission= (true_Positive) / ( true_Positive + false_Positive )
                    recall= (true_Positive) / (true_Positive + false_Negative)
                    
                    accuracy= ( true_negative + true_Positive ) / (  true_negative + true_Positive + false_Negative +false_Positive)
                   
                except ZeroDivisionError:
                    print("error002")
                    pass
        
                precision_list.append(precission)
                recall_list.append(recall)
                accuracy_list.append(accuracy)
                # #MAP
                # num_relevant_docs = len(precision_list)
                # calculate_ap = sum(precision_list) / num_relevant_docs
                # print(f"Average Precision (AP): {calculate_ap:.2f}")



            except KeyError:
                print("error")
                pass   

        average_precision=sum(precision_list)   
        average_recall=sum(recall_list) 
        Accuracy= sum(accuracy_list)
        mrr = 1/Q * cumulative_reciprocal
        # if average_precision + average_recall == 0:
        #      average_precision = 1
        F_Measure = (2 * average_precision * average_recall) / (average_precision + average_recall)
        print("Average Precision is : ", average_precision)
        print("Average Recall is : ", average_recall)
        print("MRR =", round(mrr,2))
        map_at_k = sum(ap) / Q
        
        # generate results
        # print(f"mAP@{k} = {round(map_at_k, 2)}") 

#         print("F-score is : " ,F_Measure)
#         print("Accuracy : " ,Accuracy)
        
        # MAP ALL
        k = 10
        ap = []
        k1 = [1,4,5]
        for q in k1:
            ap_num = 0
            for x in range(k):
                act_set = set(rel_set[str(q)][0])
                result_from_cosine= CosineSimilarity.cosine_similarity(x+1 ,qry_set[str(q)],id)
                pred_set = set(result_from_cosine)
                precision_at_k = len(act_set & pred_set) / (x+1)
                if result_from_cosine[x] in act_set:
                    rel_k = 1
                else:
                    rel_k = 0
                # calculate numerator value for ap
                ap_num += precision_at_k * rel_k
            # now we calculate the AP value as the average of AP
            # numerator values
            ap_q = ap_num / len(act_set)
            print(f"AP@{k}_{q+1} = {round(ap_q,2)}")
            ap.append(ap_q)
        # now we take the mean of all ap values to get mAP
        map_at_k = sum(ap) / Q
        
        # generate results
        print(f"mAP@{k} = {round(map_at_k, 2)}")   
#         AP@10_6 = 0.12
# mAP@10 = 0.16 
        # num_queries = len(precision_list)
        # mean_average_precision = sum(precision_list) / num_queries
        # print(f"Mean Average Precision (MAP): {mean_average_precision:.2f}")


        
