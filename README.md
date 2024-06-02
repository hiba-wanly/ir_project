# ir_project


download the datasets from thr linkes below :
antique/train : https://ir-datasets.com/antique.html#antique/train
lotte/ science : https://ir-datasets.com/lotte.html#lotte/science/dev

The program begins from the `start_app.py` file. Initially, we execute the `Data_lotte` and `Data_antique` functions. 
These functions are responsible for reading the required data and conducting `test_processing` operations.
They also perform the necessary TF-IDF calculations and save the resulting matrix and index in the model and index folders.
Following this, we can execute the `Evaluation` function, which performs the required computational operations. 
Additionally, we can run the `Clustering_fun` function to create the desired clustering.
The `Clustering_new_data` function is used to determine which cluster a new file belongs to. 
The `add_query` function is responsible for receiving a query from the user interface and returning the results.
the repository for the user interface can be found at https://gitlab.com/hnry/ir_front.git
