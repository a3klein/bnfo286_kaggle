import argparse
import pandas as pd
import numpy as np
import ndex2
import cdapsutil
import networkx as nx

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input a source, target, weight graph
    parser.add_argument('--input_graph', dest='input_graph', type=str, default="factorized_knn_graph_10.tsv")

    # HiDef Clustering Hyperparameters
    parser.add_argument('--maxres', dest='maxres', type=str, default='25.0')
    parser.add_argument('--alg', default='leiden', choices=['louvain', 'leiden'])
    parser.add_argument('--tau', dest='tau', type=str, default='0.75')
    parser.add_argument('--chi', dest='chi', type=str, default='5')
    parser.add_argument('--p', dest='p', type=str, default='0.75')

    # Additional parameters
    parser.add_argument('--max_size', dest='max_size', type=int, default=100)

    # File for saving cluster results 
    parser.add_argument('--cluster_out', dest='cluster_out', type=str, default="hidef_clusters.csv")

    args = parser.parse_args()

    edgelist = pd.read_csv(args.input_graph, sep = "\t")

    # Create a directed graph from edgelist
    graph = nx.DiGraph()
    for _, row in edgelist.iterrows():
        graph.add_edge(row['source'], row['target'], weight=row['weight'])

    # Create ndex2 object
    graph_cx = ndex2.create_nice_cx_from_networkx(graph)

    # Run HiDef on the graph object
    cd = cdapsutil.CommunityDetection()
    print("Run HiDef Communitity Detection ...")
    hidef_results = cd.run_community_detection(graph_cx, algorithm='hidefv1.1beta', 
                                               arguments={'--maxres': args.maxres,
                                                          '--alg': args.alg
                                                          #'--t': args.tau,
                                                          #'--k': args.chi,
                                                          #'--p': args.p
                                                         }
                                              )
    
    hidef_results = hidef_results.to_networkx(mode='default')
    hidef_results = dict(hidef_results.nodes(data=True))

    clusters = []
    id_col = 0
    clust_num = 1
    for comm in hidef_results.keys():
        comm_nodes = hidef_results[comm]['CD_MemberList'].split()
        if len(comm_nodes) <= args.max_size:
            for node in comm_nodes:
                cluster = pd.DataFrame({"id": [id_col], "xxx": [node], "prediction": ["clust"+str(clust_num)]})
                clusters.append(cluster)
                id_col += 1
            clust_num += 1

    clusters = pd.concat(clusters, ignore_index = True)

    print('\n')
    print("Number of clusters:", len(list(set(list(clusters["prediction"])))))
    print("Total number of nodes:", len(list(set(list(clusters["xxx"])))))
    print('\n')

    # Save cluster results to csv
    clusters.to_csv(args.cluster_out, index = False)



        