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

# Learn the node2vec embeddings the the proteins in the PPI
class Node2VecEmbeddings(object):
    def __init__(self, network_edgelist, confidence, embedding_dim, walk_length, context_size, walks_per_node, epochs, lr, batch_size):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load in network and filter edges based on confidence (if any)
        network = pd.read_csv(network_edgelist, names=['source', 'target', 'weight'], sep = '\t')
        network = network[(network["weight"] > confidence)]
        
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
    def train_node2vec(self):

        # Train node2vec model
        for epoch in range(self.epochs):
            loss = self.train()
            print(f"Epoch: {epoch}, Loss: {loss:.4f}")

        # Detach embeddings for GPU
        self.model.eval()
        embeddings = self.model().detach().cpu().numpy()

        # Convert embeddings to a DataFrame indexed by the node name
        embeddings = pd.DataFrame(embeddings, index=[self.id_to_label[i] for i in range(self.num_nodes)])

        return(embeddings)

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


# Cluster the embeddings learned from node2vec. For now, we will cluster with Leiden, but the embeddings could
#   be clustered with any method, hence we can expand additional clustering methods into the class.
class ClusterEmbeddings(object):
    def __init__(self, embeddings):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.node_names = embeddings.index.tolist()
        self.embeddings = torch.from_numpy(embeddings.values).to(device=self.device)

    def MultiResLeiden(self, num_neighbors=5, resolution_min=0.001, resolution_max=20, 
               tau=0.25, chi=2, p=0.5, seed=42):

        # Create KNN graph based on cosine similarity
        N = len(self.embeddings)
        row, col = knn(x=self.embeddings, y=self.embeddings, k=num_neighbors, cosine=True)
        cos_sim = F.cosine_similarity(self.embeddings[row], self.embeddings[col], dim=1)
        cos_sim = torch.clamp(cos_sim, min=0.0, max=1.0)
        row_cpu = row.detach().cpu().numpy()
        col_cpu = col.detach().cpu().numpy()
        sim_cpu = cos_sim.detach().cpu().numpy()
        knn_dist_mat = coo_matrix((sim_cpu, (row_cpu, col_cpu)), shape=(N, N))
                 
        # Convert to igraph object and run the Leiden algorithm for increasing gamma resolution
        partitions = {}
        graph = ig.Graph.Weighted_Adjacency(knn_dist_mat, mode="directed", loops=False)
        gamma_values = self.sample_gamma_values(gamma_min=resolution_min, gamma_max=resolution_max)
        print("Running Leiden for", len(gamma_values), "resolutions")
        for resolution in gamma_values:
            partition = la.find_partition(graph, la.RBConfigurationVertexPartition, weights="weight",
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

    # Network edgelist from Kaggle and confidence filter
    parser.add_argument('--network_edgelist', dest='network_edgelist', type=str, default='data/interaction/network_with_weights.edgelist')
    parser.add_argument('--confidence', dest='confidence', type=float, default=0.5)

    # Node2Vec Hyperparameters
    parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', dest='walk_length', type=int, default=80)
    parser.add_argument('--context_size', dest='context_size', type=int, default=10)
    parser.add_argument('--walks_per_node', dest='walks_per_node', type=int, default=5)
    parser.add_argument('--epochs', dest='epochs', type=int, default=50)
    parser.add_argument('--lr', dest='lr', type=float, default=0.01)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)

    # Leiden Clustering Hyperparameters
    parser.add_argument('--num_neighbors', dest='num_neighbors', type=int, default=5)
    parser.add_argument('--resolution_min', dest='resolution_min', type=float, default=0.001)
    parser.add_argument('--resolution_max', dest='resolution_max', type=float, default=20)
    parser.add_argument('--tau', dest='tau', type=float, default=0.25)
    parser.add_argument('--chi', dest='chi', type=int, default=2)
    parser.add_argument('--p', dest='p', type=float, default=0.5)

    # File for saving node2vec embeddings and cluster results 
    parser.add_argument('--embedding_out', dest='embedding_out', type=str, default="node2vec_embeddings.pkl")
    parser.add_argument('--cluster_out', dest='cluster_out', type=str, default="protein_clusters.csv")
    args = parser.parse_args()

    # Compute node2vec embeddings
    print("Learning Node2Vec Embeddings ...")
    embeddings = Node2VecEmbeddings(network_edgelist=args.network_edgelist,
                                    confidence=args.confidence,
                                    embedding_dim=args.embedding_dim,
                                    walk_length=args.walk_length,
                                    context_size=args.context_size,
                                    walks_per_node=args.walks_per_node,
                                    epochs=args.epochs,
                                    lr=args.lr,
                                    batch_size=args.batch_size).train_node2vec()

    print('\n')

    # Save the node2vec embeddings to pickle
    with open(args.embedding_out, "wb") as handle:
        pickle.dump(embeddings, handle)

    # Compute clusters from the nod2vec embeddings (here with Leiden)
    print("Clustering the Node2Vec Embeddings ...")
    ce = ClusterEmbeddings(embeddings=embeddings)
    clusters = ce.MultiResLeiden(num_neighbors=args.num_neighbors, resolution_min=args.resolution_min,
                                 resolution_max=args.resolution_max, tau=args.tau, chi=args.chi, p=args.p)

    print("Number of clusters:", len(list(set(list(clusters["prediction"])))))

    # Save cluster results to csv
    clusters.to_csv(args.cluster_out, index = False)

    
        