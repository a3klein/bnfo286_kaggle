# bnfo286_kaggle

This repository contains the code for Kaggle Competition for BNFO 286 final project developed by Amit Klein, Maggie Lin, and Zach Wallace. The code creates communities for two separate datasets: protein interaction data and protein co-elution data. The datasets are supplied by the Kaggle competition found at https://www.kaggle.com/competitions/bnfo286coelution and https://www.kaggle.com/competitions/bnfo286interaction

## Generating Protein Interaction Communities

To generate an example of communities from the protein interaction data, run the following:

    python embedding_clusters.py --network_edgelist [Kaggle Data] --num_neighbors [num_neighbors for KNN graph]

After running this code, a KNN graph will be stored as node2vec_knn_graph_[NUM_NEIGHBORS].tsv. To finish creating the clusters, run the following:

    python hidef_clustering.py --input_graph [input KNN graph] --maxres 50 --max_size 25 --cluster_out [output_file]

The results clusters will be saved to the output file, that of which was used in the Kaggle submission.

## Generating Protein Coelution Communities

To generate an example of communities from the protein coelution data, run the following:

    python coelution_clusters.py --coelution_data [Kaggle Data] --correlate --factorize --num_neighbors [num_neighbors for KNN graph]

After running this code, a KNN graph will be stored as factorized_knn_graph_[NUM_NEIGHBORS].tsv. To finish creating the clusters, run the following:

    python hidef_clustering.py --input_graph [input KNN graph] --maxres 100 --max_size 40 --cluster_out [output_file]