import csv
import json

from new.precision_and_recall import PrecisionandRecall



class ReadingGround:
    def Precision_and_Recall(id):
        if id == 1:
             # get all queries
             with open("C:\\Users\\yamen\\OneDrive\\Desktop\\DATASET\\lotte\\lotte\\science\\dev\\questions.search.tsv", errors='ignore') as file:
                 tsv_query = csv.reader(file, delimiter="\t")
                 data1 = list(tsv_query)
             qry_set = {}
             qry_id = ""  
             for d1 in data1:
                  qry_id = d1[0]
                  qry_set[qry_id] = d1[1]  
                  qry_id = ""
             print(f"Number of queries = {len(qry_set)}" + ".\n")
             # print("Query # 2 : ", qry_set["2"])  
             
             
             rel_set = {}
             with open("C:\\Users\\yamen\\OneDrive\\Desktop\\DATASET\\lotte\\lotte\\science\\dev\\qas.search.jsonl", 'r') as file:
                 json_list = list(file) 
             #     tsv_qas = csv.reader(file, delimiter="\t")
                 for d2 in json_list:
                       result = json.loads(d2)
                       # print(result)
                       # print(result["qid"])
                       # print(result["answer_pids"])
                       qry_id = str(result["qid"])
                       doc_id = result["answer_pids"]
                       if qry_id in rel_set:
                            rel_set[qry_id].append(doc_id)
                       else:
                            rel_set[qry_id] = []
                            rel_set[qry_id].append(doc_id)
             # print(rel_set["466"])  
             
             doc_set = {}
             doc_id = ""
             doc_text = ""
             with open("C:\\Users\\yamen\\OneDrive\\Desktop\\DATASET\\lotte\\lotte\\science\\dev\\collection.tsv", errors='ignore') as file:
                         tsv_file = csv.reader(file, delimiter="\t")
                         data3 = list(tsv_file)
                       #   print(data3)
                         for d3 in data3:
                               doc_id = d3[0]
                               doc_set[doc_id] = d3[1]
                               doc_id = ""
                               doc_text = ""
             print(f"Number of documents = {len(doc_set)}" + ".\n")
             # print(doc_set["3"])
             
             PrecisionandRecall.precision_and_recall(doc_set,qry_set,rel_set,id)
        
        if id == 2 :
             # get all queries
             with open("C:\\Users\\yamen\\OneDrive\\Desktop\\DATASET\\antique\\antique\\train\\queries.txt", errors='ignore') as file:
                 tsv_query = csv.reader(file, delimiter="\t")
                 data1 = list(tsv_query)
             qry_set = {}
             qry_id = ""  
             for d1 in data1:
                  qry_id = d1[0]
                  qry_set[qry_id] = d1[1]  
                  qry_id = ""
             print(f"Number of queries = {len(qry_set)}" + ".\n")
             # print("Query # 2 : ", qry_set["2"])  
              
             
             doc_set = {}
             doc_id = ""
             doc_text = ""
             with open("C:\\Users\\yamen\\OneDrive\\Desktop\\DATASET\\antique\\antique\\collection.tsv", errors='ignore') as file:
                         tsv_file = csv.reader(file, delimiter="\t")
                         data3 = list(tsv_file)
                       #   print(data3)
                         for d3 in data3:
                               doc_id = d3[0]
                               doc_set[doc_id] = d3[1]
                               doc_id = ""
                               doc_text = ""
             print(f"Number of documents = {len(doc_set)}" + ".\n")
             # print(doc_set["3"])
             
             PrecisionandRecall.precision_and_recall(doc_set,qry_set,rel_set,id)

            
                 
            
            
            