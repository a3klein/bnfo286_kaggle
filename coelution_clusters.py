import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

import igraph as ig
import leidenalg as la

import torch.nn.functional as F
from torch_geometric.nn.pool import knn
from torch_geometric.nn import Node2Vec
from scipy.sparse import coo_matrix

class ResolutionGraph(object):
    def __init__(self, corr_matrix, z_score=2.0):

        # Binarize the co-elution correlation matrix according to a z-score threshold
        mean_val = np.nanmean(corr_matrix)
        std_val = np.nanstd(corr_matrix)
        z_matrix = (corr_matrix - mean_val) / std_val
        binarized = (z_matrix >= z_score).astype(int)
        graph_matrix = np.asarray(binarized)
        np.fill_diagonal(graph_matrix, val = 0)
        
        adjacency = pd.DataFrame(graph_matrix, index = list(corr_matrix.index), columns = list(corr_matrix.index))

        self.adjacency = adjacency

    def get_adjacency(self):

        return(self.adjacency)

class MatrixFactEmbeddings(object):
    def __init__(self, corr_matrix, embedding_dim, lr, max_iters):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.N = len(corr_matrix)
        self.P = torch.nn.Parameter(torch.randn(self.N, embedding_dim, device=self.device))
        self.M = torch.from_numpy(np.asarray(corr_matrix.values)).to(self.device)
        self.max_iters = max_iters
        self.node_names = list(corr_matrix.index)
        self.optimizer = torch.optim.Adam([self.P], lr=lr)
        
    def factorized_knn_graph(self, num_neighbors, embedding_out):

        for it in range(self.max_iters):
            self.optimizer.zero_grad()
            M_approx = torch.matmul(self.P, self.P.T)
            loss = torch.mean((self.M - M_approx) ** 2)
            loss.backward()
            self.optimizer.step()

            if (it + 1) % 500 == 0:
                print(f"Iteration {it+1}/{self.max_iters}, loss = {loss.item():.6f}")

        # Build KNN graph
        row, col = knn(x=self.P, y=self.P, k=num_neighbors, cosine=True)
        cos_sim = F.cosine_similarity(self.P[row], self.P[col], dim=1)
        cos_sim = torch.clamp(cos_sim, min=0.0, max=1.0)
        row_cpu = row.detach().cpu().numpy()
        col_cpu = col.detach().cpu().numpy()
        sim_cpu = cos_sim.detach().cpu().numpy()
        knn_dist_mat = coo_matrix((sim_cpu, (row_cpu, col_cpu)), shape=(self.N, self.N))
        self.save_knn_to_edgelist(knn_dist_mat, self.node_names, num_neighbors)
        graph = ig.Graph.Weighted_Adjacency(knn_dist_mat, mode="directed", loops=False)

        # Save the node2vec embeddings to pickle
        embeddings = self.P.detach().cpu().numpy()
        with open(embedding_out, "wb") as handle:
            pickle.dump(embeddings, handle)
        
        return(graph, self.node_names)

    def save_knn_to_edgelist(self, coo_mat, node_names, num_neighbors):
        
        rows, cols, weights = coo_mat.row, coo_mat.col, coo_mat.data
        
        sources = np.array(node_names)[rows]
        targets = np.array(node_names)[cols]
        
        edgelist_df = pd.DataFrame({'source': sources, 'target': targets,'weight': weights})
        edgelist_df = edgelist_df[(edgelist_df["weight"] != 1.0)]
        edgelist_df.to_csv("factorized_knn_graph_"+str(num_neighbors)+".tsv", sep = '\t', index = False)

class Node2VecEmbeddings(object):
    def __init__(self, adjacency, embedding_dim, walk_length, context_size, walks_per_node, epochs, lr, batch_size):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        network = adjacency.stack().reset_index()
        network.columns = ['source', 'target', 'weight']
        network = network[network['weight'] != 0].reset_index(drop=True)
        
        # Create int IDs for the protein label and create edge_index (needed for node2vec)
        unique_labels = pd.unique(network[['source','target']].values.ravel())
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        network['source'] = network['source'].map(label_to_id)
        network['target'] = network['target'].map(label_to_id)
        edge_index = torch.tensor(network[['source', 'target']].values, dtype=torch.long).t()

        # Instantiate the node2vec model
        self.model = Node2Vec(edge_index=edge_index, 
                              embedding_dim=embedding_dim,
                              walk_length=walk_length, # walks to sample for each node
                              context_size=context_size, # skip-gram window size, increases sampling rate
                              walks_per_node=walks_per_node, # walks to sample for each node
                              p=1.0,
                              q=1.0,
                              num_negative_samples=1,
                              sparse=True).to(self.device)
        
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=lr)
        self.id_to_label = {idx: label for label, idx in label_to_id.items()}
        self.num_nodes = len(unique_labels)
        self.batch_size = batch_size
        self.epochs = epochs

    # Train node2vec over range of epochs to acquire node embeddings
    def node2vec_knn_graph(self, num_neighbors, embedding_out):

        # Train node2vec model
        for epoch in range(self.epochs):
            loss = self.train()
            print(f"Epoch: {epoch}, Loss: {loss:.4f}")

        # Detach embeddings for GPU
        self.model.eval()
        embeddings = self.model().detach().cpu().numpy()

        # Convert embeddings to a DataFrame indexed by the node name
        embeddings = pd.DataFrame(embeddings, index=[self.id_to_label[i] for i in range(self.num_nodes)])
        node_names = embeddings.index.tolist()

        # Save the node2vec embeddings to pickle
        with open(embedding_out, "wb") as handle:
            pickle.dump(embeddings, handle)

        # Create KNN graph based on cosine similarity
        N = len(embeddings)
        embeddings = torch.from_numpy(embeddings.values).to(device=self.device)
        row, col = knn(x=embeddings, y=embeddings, k=num_neighbors, cosine=True)
        cos_sim = F.cosine_similarity(embeddings[row], embeddings[col], dim=1)
        cos_sim = torch.clamp(cos_sim, min=0.0, max=1.0)
        row_cpu = row.detach().cpu().numpy()
        col_cpu = col.detach().cpu().numpy()
        sim_cpu = cos_sim.detach().cpu().numpy()
        knn_dist_mat = coo_matrix((sim_cpu, (row_cpu, col_cpu)), shape=(N, N))
        graph = ig.Graph.Weighted_Adjacency(knn_dist_mat, mode="directed", loops=False)

        return(graph, node_names)

    # Training function
    def train(self):
        
        self.model.train()
        total_loss = 0
        loader = self.model.loader(batch_size=self.batch_size, shuffle=True)

        # The model provides a loader that returns pairs (pos_rw, neg_rw)
        #   pos_rw: random walks for positive edges
        #   neg_rw: random walks for negative edges
        for pos_rw, neg_rw in loader:
            pos_rw, neg_rw = pos_rw.to(self.device), neg_rw.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw, neg_rw)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
    
        return(total_loss / len(loader))

class ClusterGraph(object):
    def __init__(self, graph, node_names, weighted=False):

        self.graph = graph
        self.node_names = node_names
        self.weighted = "weight" if weighted else None

    def MultiResLeiden(self, resolution_min=0.001, resolution_max=25, tau=0.75, chi=2, p=0.75, seed=42):
    
        # Run the Leiden algorithm for increasing gamma resolution
        partitions = {}
        gamma_values = self.sample_gamma_values(gamma_min=resolution_min, gamma_max=resolution_max)
        print("Running Leiden for", len(gamma_values), "resolutions")
        for resolution in gamma_values:
            partition = la.find_partition(self.graph, la.RBConfigurationVertexPartition, weights=self.weighted,
                                          resolution_parameter=resolution, seed=seed)

            cluster_dict = defaultdict(set)
            for node, cluster in zip(self.node_names, list(partition.membership)):
                cluster_dict[cluster].add(node)
            
            partitions[resolution] = list(cluster_dict.values())
            print("Done with Leiden at resolution", resolution)

        # ------ Everything after this is for 'persistent communities' ------ #

        gamma_values_sorted = sorted(gamma_values)
        gamma_proximal_pairs = []
        for i in range(len(gamma_values_sorted)):
            for j in range(i+1, len(gamma_values_sorted)):
                g1 = gamma_values_sorted[i]
                g2 = gamma_values_sorted[j]
                diff = abs(np.log10(g1) - np.log10(g2))
                if diff < 0.1:
                    gamma_proximal_pairs.append((g1, g2))
                else:
                    # If your gamma_values are sorted and you only move
                    # in small steps (0.1), you can break as soon as diff >= 0.1.
                    break

        persistent_graph = nx.Graph()

        # Add one node for each community at each gamma. Label them uniquely (e.g., (gamma, idx)) 
        for gamma in gamma_values_sorted:
            communities = partitions[gamma]
            for idx, comm in enumerate(communities):
                persistent_graph.add_node((gamma, idx), nodeset=comm)
    
        for (g1, g2) in gamma_proximal_pairs:
            comms_g1 = partitions[g1]
            comms_g2 = partitions[g2]
            for i, c1 in enumerate(comms_g1):
                for j, c2 in enumerate(comms_g2):
                    if self.jaccard_similarity(c1, c2) > tau:
                        persistent_graph.add_edge((g1, i), (g2, j))

        families = []
        components = list(nx.connected_components(persistent_graph))
        for comp in components:
            # comp is a set of node labels: {(gamma1, i1), (gamma2, i2), ...}
            # collect the distinct gamma values & the underlying sets
            gamma_set = set()
            community_sets = []
    
            for (g, i) in comp:
                gamma_set.add(g)
                comm_nodes = persistent_graph.nodes[(g, i)]['nodeset'] # the original set of node IDs
                community_sets.append(comm_nodes)
    
            # The "persistence" is # distinct gamma's in this component
            persistence = len(gamma_set)
            families.append((comp, community_sets, persistence))

        persistent_families = [fam for fam in families if fam[2] >= chi]

        final_persistent_communities = []
        for comp, comm_sets, persistence in persistent_families:
            k = len(comm_sets)  # how many community sets are in this connected component
            count_threshold = int(np.ceil(p * k))

            # Flatten or union all node IDs to see the "universe" of nodes that ever appear
            all_nodes_in_comp = set().union(*comm_sets)

            # Count how often each node appears
            node_counts = {node:0 for node in all_nodes_in_comp}
            for cset in comm_sets:
                for node in cset:
                    node_counts[node] += 1
    
            # Keep nodes that appear in at least 'count_threshold' sets
            persistent_nodeset = {node for node, ct in node_counts.items() if ct >= count_threshold}
    
            final_persistent_communities.append({
                'component': comp,  # the G_C nodes
                'persistence': persistence,
                'persistent_nodeset': persistent_nodeset})
            
        # ------ End code for 'persistent communiteis' detection -------- #

        # Write the cluster results to Kaggle submission format
        table_rows = []
        for i, fam_dict in enumerate(final_persistent_communities, start=1):
            nodes_in_fam = fam_dict['persistent_nodeset']
            for node in nodes_in_fam:
                table_rows.append({'xxx': node, 'prediction': "clust"+str(i)})

        cluster_results = pd.DataFrame(table_rows)
        cluster_results = pd.DataFrame({"id": list(cluster_results.index), 
                                        "xxx": list(cluster_results['xxx']),
                                        "prediction": list(cluster_results['prediction'])})
        
        return(cluster_results)

    def sample_gamma_values(self, gamma_min=0.001, gamma_max=20, log_step=0.1):
        """
        Generate gamma values logarithmically from gamma_min to gamma_max
        with increments of 0.1 in log10 space.
        """
        # Compute log10 boundaries
        log_min = np.log10(gamma_min)
        log_max = np.log10(gamma_max)
    
        # Generate an array of equally spaced points in [log_min, log_max] with step log_step
        log_values = np.arange(log_min, log_max + log_step, log_step)
    
        # Convert back from log space to linear space
        gamma_values = 10**(log_values)
    
        # Make sure we do not exceed gamma_max too much due to floating-point
        gamma_values = gamma_values[gamma_values <= gamma_max]
    
        return(gamma_values)

    def jaccard_similarity(self, set_a, set_b):
        
        inter = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        
        return(inter / union if union > 0 else 0.0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Co-elution data from Kaggle OR the pre-computed correlation matrix
    parser.add_argument('--coelution_data', dest='coelution_data', type=str, default='data/coelution/repl1_repl2_combined.tsv')
    parser.add_argument('--corr_matrix', dest='corr_matrix', type=str, default='data/coelution/correlation_matrix.pkl')
    parser.add_argument('--one_cluster', dest='one_cluster', type=str, default='coelution_onecluster.csv')
    parser.add_argument('--correlate', action='store_true', default=False)
    parser.add_argument('--z_score', dest='z_score', type=float, default=2.0)

    # Matrix Factorization Specific Hyperparameters
    parser.add_argument('--factorize', action='store_true', default=False)
    parser.add_argument('--max_iters', dest='max_iters', type=int, default=5000)

    # Node2Vec Specific Hyperparameters
    parser.add_argument('--node2vec', action='store_true', default=False)
    parser.add_argument('--walk_length', dest='walk_length', type=int, default=80)
    parser.add_argument('--context_size', dest='context_size', type=int, default=10)
    parser.add_argument('--walks_per_node', dest='walks_per_node', type=int, default=5)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)
    parser.add_argument('--epochs', dest='epochs', type=int, default=50)

    # Hyperparameters used by Matrix Factorization and Node2Vec
    parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=128)
    parser.add_argument('--lr', dest='lr', type=float, default=0.01)
    parser.add_argument('--use_embeddings', action='store_true', default=False)

    # KNN Graph Hyperparameter
    parser.add_argument('--num_neighbors', dest='num_neighbors', type=int, default=10)

    # Leiden Clustering Hyperparameters
    parser.add_argument('--resolution_min', dest='resolution_min', type=float, default=0.001)
    parser.add_argument('--resolution_max', dest='resolution_max', type=float, default=25)
    parser.add_argument('--tau', dest='tau', type=float, default=0.75)
    parser.add_argument('--chi', dest='chi', type=int, default=2)
    parser.add_argument('--p', dest='p', type=float, default=0.75)

    # File for saving node2vec embeddings and cluster results 
    parser.add_argument('--embedding_out', dest='embedding_out', type=str, default="coelution_embeddings.pkl")
    parser.add_argument('--cluster_out', dest='cluster_out', type=str, default="coelution_clusters.csv")
    
    args = parser.parse_args()

    # Only correlate co-elution if not already
    if args.correlate:
        coelution = pd.read_csv(args.coelution_data, sep = '\t', index_col = 0)
        onecluster = pd.read_csv(args.one_cluster)
        oneclust_proteins = list(onecluster["xxx"])
        coelution = coelution[(coelution.index.isin(oneclust_proteins))]
        coelution = coelution.transpose()
        print("Correlating the coelution data ...")
        corr_matrix = coelution.corr(min_periods=3)
        corr_matrix = corr_matrix.fillna(0)
        with open("data/coelution/correlation_matrix.pkl", "wb") as handle:
            pickle.dump(corr_matrix, handle)
    else:
        with open(args.corr_matrix, "rb") as handle:
            corr_matrix = pickle.load(handle)

    if args.use_embeddings:
        print("Using trained embeddings ...")
        with open("coelution_embeddings.pkl", "rb") as handle:
            embeddings = pickle.load(handle)
        node_names = corr_matrix.index.tolist()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        N = len(embeddings)
        embeddings = torch.from_numpy(embeddings).to(device=device)
        row, col = knn(x=embeddings, y=embeddings, k=args.num_neighbors, cosine=True)
        cos_sim = F.cosine_similarity(embeddings[row], embeddings[col], dim=1)
        cos_sim = torch.clamp(cos_sim, min=0.0, max=1.0)
        row_cpu = row.detach().cpu().numpy()
        col_cpu = col.detach().cpu().numpy()
        sim_cpu = cos_sim.detach().cpu().numpy()
        knn_dist_mat = coo_matrix((sim_cpu, (row_cpu, col_cpu)), shape=(N, N))
        graph = ig.Graph.Weighted_Adjacency(knn_dist_mat, mode="directed", loops=False)
        weighted = True
    elif args.factorize:
        print("Learning Factorization Embeddings and building KNN graph ...")
        mf = MatrixFactEmbeddings(corr_matrix=corr_matrix,
                                  embedding_dim=args.embedding_dim,
                                  lr=args.lr,
                                  max_iters=args.max_iters)
        graph, node_names = mf.factorized_knn_graph(args.num_neighbors, args.embedding_out)
        weighted = True       
    elif args.node2vec:
        print("Learning Node2Vec Embeddings and building KNN graph ...")
        adjacency = ResolutionGraph(corr_matrix=corr_matrix, z_score=args.z_score).get_adjacency()
        n2v = Node2VecEmbeddings(adjacency=adjacency,
                                 embedding_dim=args.embedding_dim,
                                 walk_length=args.walk_length,
                                 context_size=args.context_size,
                                 walks_per_node=args.walks_per_node,
                                 epochs=args.epochs,
                                 lr=args.lr,
                                 batch_size=args.batch_size)
        graph, node_names = n2v.node2vec_knn_graph(args.num_neighbors, args.embedding_out)
        weighted = True
    else:
        adjacency = ResolutionGraph(corr_matrix=corr_matrix, z_score=args.z_score).get_adjacency()
        node_names = list(adjacency.index)
        graph = ig.Graph.Adjacency(adjacency, mode="undirected", loops=False)
        weighted = False
    
    print('\n')

    # Compute clusters from the nod2vec embeddings (here with Leiden)
    print("Clustering the Established graph ...")
    ce = ClusterGraph(graph=graph, node_names=node_names, weighted=weighted)
    clusters = ce.MultiResLeiden(resolution_min=args.resolution_min, 
                                 resolution_max=args.resolution_max, 
                                 tau=args.tau, 
                                 chi=args.chi, 
                                 p=args.p)

    print("Number of clusters:", len(list(set(list(clusters["prediction"])))))
    print("Total number of nodes:", len(list(set(list(clusters["xxx"])))))

    # Save cluster results to csv
    clusters.to_csv(args.cluster_out, index = False)